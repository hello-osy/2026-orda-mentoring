"""Record real Gazebo images + visible boxes, then extract a YOLO dataset.

Ground truth is used only here, never by the driving answer. Bounding boxes
come from Gazebo's visible-2D renderer, so hidden signs are not hallucinated.
"""

import bisect
import json
import math
import random
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml


CLASSES = ["traffic_red", "traffic_yellow", "traffic_green", "left", "right",
           "pedestrian", "slow", "parking"]


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    config["_root"] = str(path.parent.parent)
    return config


def project_path(config, value):
    path = Path(value)
    return path if path.is_absolute() else Path(config["_root"]) / path


def stamp_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def decode_image(message):
    """Respect row padding; Gazebo/ROS images are not necessarily contiguous."""
    encoding = message.encoding.lower()
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}
    if encoding not in channels:
        raise ValueError("Unsupported image encoding: " + encoding)
    count = channels[encoding]
    if message.step < message.width * count:
        raise ValueError("Image row is shorter than width * channels")
    rows = np.frombuffer(bytes(message.data), dtype=np.uint8).reshape(message.height, message.step)
    image = rows[:, :message.width * count].reshape(message.height, message.width, count)
    if encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if encoding == "mono8":
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def normalized_box(box, width, height, class_count, min_pixels=5, min_fraction=0.5):
    """Clip a visible renderer box and return one normalized YOLO row."""
    class_id = int(box["class_id"])
    if class_id < 0 or class_id >= class_count:
        raise ValueError("Unknown Gazebo class label: " + str(class_id))
    values = [float(value) for value in box["xyxy"]]
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError("Non-finite or malformed bounding box")
    x1, y1, x2, y2 = values
    original_area = max(0, x2 - x1) * max(0, y2 - y1)
    x1, x2 = max(0.0, x1), min(float(width), x2)
    y1, y2 = max(0.0, y1), min(float(height), y2)
    box_width, box_height = x2 - x1, y2 - y1
    if original_area <= 0 or min(box_width, box_height) < min_pixels:
        return None
    if box_width * box_height / original_area < min_fraction:
        return None
    return [class_id, (x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
            box_width / width, box_height / height]


def nearest_stamp(stamps, target, tolerance_ns):
    """Require an actual matching GT packet; missing GT is never a negative."""
    index = bisect.bisect_left(stamps, target)
    candidates = stamps[max(0, index - 1):index + 1]
    if not candidates:
        return None
    closest = min(candidates, key=lambda stamp: abs(stamp - target))
    return closest if abs(closest - target) <= tolerance_ns else None


def visible_traffic_lamp(crop, class_name, settings):
    """A red-state panel whose red lamp is off-screen is not a red example.

    Renderer labels describe the whole signal state, even when another sign
    hides the lit lamp. Require the actual filled lamp to remain observable.
    This only rejects ambiguous GT samples; it never invents a class label.
    """
    if class_name not in settings["hue_ranges"]:
        return True
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    if np.mean(hsv[:, :, 2] < settings["dark_value_max"]) < settings["minimum_dark_fraction"]:
        return False
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for low, high in settings["hue_ranges"][class_name]:
        mask |= cv2.inRange(hsv, (low, settings["saturation_min"], settings["value_min"]), (high, 255, 255))
    count, _, statistics, _ = cv2.connectedComponentsWithStats(mask)
    for index in range(1, count):
        _, _, width, height, area = statistics[index]
        if area / mask.size >= settings["minimum_color_fraction"] and area / (width * height) >= settings["minimum_blob_fill"]:
            return True
    return False


def split_runs(run_names, fraction, seed):
    """Keep every image from the same bag on the same side of the split."""
    names = sorted(set(run_names))
    if len(names) < 2:
        raise ValueError("At least two independent rosbag runs are required for train/val.")
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    random.Random(seed).shuffle(names)
    count = max(1, min(len(names) - 1, round(len(names) * fraction)))
    validation = set(names[:count])
    return {name: "val" if name in validation else "train" for name in names}


def collection_group(bag_path):
    """Repeated automatic collections with the same seed stay in one split."""
    metadata = bag_path / "collection.json"
    if metadata.exists():
        values = json.loads(metadata.read_text())
        if values.get("automatic_views"):
            return "automatic_seed_" + str(values["seed"])
    return bag_path.name


def _set_pose(transport, world, name, pose, timeout_ms=5000, retries=1):
    from google.protobuf.text_format import MessageToString
    from gz.msgs10.pose_pb2 import Pose
    request = Pose()
    request.name = name
    request.position.x, request.position.y, request.position.z = pose[:3]
    roll, pitch, yaw = pose[3:]
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    request.orientation.w = cr * cp * cy + sr * sp * sy
    request.orientation.x = sr * cp * cy - cr * sp * sy
    request.orientation.y = cr * sp * cy + sr * cp * sy
    request.orientation.z = cr * cp * sy - sr * sp * cy
    # gz-transport Python subscriptions and blocking requests can contend for
    # the GIL. An isolated CLI request leaves the camera callback free to run.
    command = ["gz", "service", "-s", "/world/" + world + "/set_pose",
               "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
               "--timeout", str(timeout_ms), "--req", MessageToString(request, as_one_line=True)]
    detail = ""
    for _ in range(retries):
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=timeout_ms / 1000 + 2)
            if result.returncode == 0 and "data: true" in result.stdout:
                return
            detail = result.stdout.strip() + " " + result.stderr.strip()
        except subprocess.TimeoutExpired:
            detail = "service process timed out"
    raise RuntimeError("Gazebo set_pose failed for " + name + ": " + detail)


def record(config):
    """Write standard ROS 2 sqlite3 bags; optionally collect randomized views."""
    import rclpy
    import rosbag2_py
    from ackermann_msgs.msg import AckermannDriveStamped
    from gz.msgs10.annotated_axis_aligned_2d_box_v_pb2 import AnnotatedAxisAligned2DBox_V
    from gz.transport13 import Node as TransportNode
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from rclpy.serialization import serialize_message
    from rclpy.signals import SignalHandlerOptions
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import Image, LaserScan
    from std_msgs.msg import Bool, String

    settings = config["dataset"]
    topics = config["topics"]
    collection = settings["collection"]
    bag_root = project_path(config, config["paths"]["bags"])
    bag_root.mkdir(parents=True, exist_ok=True)
    labels_topic = topics["labels"]
    lock = threading.Lock()
    state = {"writer": None, "counts": Counter(), "class_counts": Counter(),
             "accept": False, "first_sim": None}
    # Keep the context alive through Ctrl+C so the collection lock and stop
    # command can be delivered in finally before shutting DDS down.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Node("orda_dataset_recorder", parameter_overrides=[Parameter("use_sim_time", value=True)])
    transport = TransportNode()
    active_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                           durability=DurabilityPolicy.TRANSIENT_LOCAL)
    active = node.create_publisher(Bool, topics["collection_active"], active_qos)
    drive = node.create_publisher(AckermannDriveStamped, topics["drive"], 10)
    subscriptions = []
    manifest = None

    def write(topic, message, timestamp):
        with lock:
            if state["writer"] is not None and state["accept"]:
                state["writer"].write(topic, serialize_message(message), int(timestamp))
                state["counts"][topic] += 1

    def receive(topic, message):
        if hasattr(message, "header"):
            timestamp = stamp_ns(message.header.stamp)
            if timestamp == 0:
                timestamp = node.get_clock().now().nanoseconds
        elif hasattr(message, "clock"):
            timestamp = stamp_ns(message.clock)
        else:
            timestamp = node.get_clock().now().nanoseconds
        if topic == topics["image"]:
            state["first_sim"] = timestamp
        write(topic, message, timestamp)

    specifications = [(topics["image"], Image, "sensor_msgs/msg/Image"),
                      (topics["scan"], LaserScan, "sensor_msgs/msg/LaserScan"),
                      (topics["odometry"], Odometry, "nav_msgs/msg/Odometry"),
                      (topics["drive"], AckermannDriveStamped, "ackermann_msgs/msg/AckermannDriveStamped"),
                      ("/clock", Clock, "rosgraph_msgs/msg/Clock")]
    for topic, message_type, _ in specifications:
        subscriptions.append(node.create_subscription(message_type, topic,
                             lambda message, name=topic: receive(name, message), qos_profile_sensor_data))

    def boxes_callback(message):
        boxes = []
        for item in message.annotated_box:
            if item.label == 0:
                continue
            boxes.append({"class_id": int(item.label) - 1,
                          "xyxy": [item.box.min_corner.x, item.box.min_corner.y,
                                   item.box.max_corner.x, item.box.max_corner.y]})
        timestamp = int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nsec)
        packet = String()
        packet.data = json.dumps({"stamp_ns": timestamp, "boxes": boxes}, separators=(",", ":"))
        write(labels_topic, packet, timestamp)
        if state["writer"] is not None and state["accept"]:
            for box in boxes:
                if 0 <= box["class_id"] < len(CLASSES):
                    state["class_counts"][CLASSES[box["class_id"]]] += 1

    if not transport.subscribe(AnnotatedAxisAligned2DBox_V, topics["boxes"], boxes_callback):
        raise RuntimeError("Could not subscribe to Gazebo visible bounding boxes")

    def publish_stop():
        message = AckermannDriveStamped()
        message.header.stamp = node.get_clock().now().to_msg()
        drive.publish(message)

    def spin_for(seconds):
        start_sim = node.get_clock().now().nanoseconds
        start_wall = time.monotonic()
        last_command = 0.0
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            if collection["enabled"] and time.monotonic() - last_command >= 1 / collection.get("command_rate", 10.0):
                publish_stop()
                last_command = time.monotonic()
            if node.get_clock().now().nanoseconds - start_sim >= seconds * 1e9:
                return
            if time.monotonic() - start_wall > settings["max_wall_seconds"]:
                raise TimeoutError("No simulation progress; start Gazebo with physics running.")

    runs = collection["runs"] if collection["enabled"] else 1
    created = []
    shown_groups = {}
    try:
        if collection["enabled"]:
            manifest_path = project_path(config, config["paths"]["generated"]) / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            active.publish(Bool(data=True))
        start = time.monotonic()
        while state["first_sim"] is None:
            rclpy.spin_once(node, timeout_sec=0.1)
            if time.monotonic() - start > settings["startup_timeout"]:
                raise TimeoutError("No camera image received. Start scripts/simulator.sh first.")
        for run_index in range(runs):
            seed = int(collection["seed"]) + run_index
            run_name = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%f") + "_" + str(seed)
            bag_path = bag_root / run_name
            writer = rosbag2_py.SequentialWriter()
            writer.open(rosbag2_py.StorageOptions(uri=str(bag_path), storage_id=settings["storage_id"]),
                        rosbag2_py.ConverterOptions("", ""))
            all_specs = specifications + [(labels_topic, String, "std_msgs/msg/String")]
            for index, (topic, _, type_name) in enumerate(all_specs):
                writer.create_topic(rosbag2_py.TopicMetadata(id=index, name=topic, type=type_name,
                                                           serialization_format="cdr"))
            with lock:
                state["counts"] = Counter()
                state["class_counts"] = Counter()
                state["writer"] = writer
            print("Recording", bag_path, flush=True)
            run_completed = False
            try:
                if collection["enabled"]:
                    rng = random.Random(seed)
                    signs = list(manifest["signs"])
                    requested = collection.get("classes", [])
                    targets = [sign for sign in signs if not requested or sign["class_name"] in requested]
                    if not targets:
                        raise ValueError("No manifest signs match dataset.collection.classes")
                    order = [(sign, False) for sign in targets for _ in range(collection["views_per_class"])]
                    order += [(rng.choice(targets), True) for _ in range(collection["negative_views"])]
                    rng.shuffle(order)
                    for view_index, (sign, negative) in enumerate(order):
                        state["accept"] = False
                        previous = shown_groups.get(sign["group"])
                        if previous != sign["name"]:
                            for other in signs:
                                if other["group"] != sign["group"]:
                                    continue
                                # First encounter synchronizes the entire group;
                                # later changes need only hide/show two panels.
                                if previous is not None and other["name"] not in (previous, sign["name"]):
                                    continue
                                pose = list(other["pose"])
                                if other["name"] != sign["name"]:
                                    pose[2] = -10
                                _set_pose(transport, manifest["world"], other["name"], pose,
                                          collection.get("pose_timeout_ms", 5000), collection.get("pose_retries", 3))
                            shown_groups[sign["group"]] = sign["name"]
                        x, y, _, _, _, yaw = sign["pose"]
                        distance = rng.uniform(*collection["distances"])
                        lateral = rng.uniform(-collection["lateral_range"], collection["lateral_range"])
                        body_yaw = yaw + rng.uniform(-collection["yaw_range"], collection["yaw_range"])
                        if negative:
                            body_yaw += math.pi
                        pose = [x - distance * math.cos(yaw) - lateral * math.sin(yaw),
                                y - distance * math.sin(yaw) + lateral * math.cos(yaw),
                                config["robot"]["spawn"][2], 0, 0, body_yaw]
                        _set_pose(transport, manifest["world"], manifest["robot"], pose,
                                  collection.get("pose_timeout_ms", 5000), collection.get("pose_retries", 3))
                        spin_for(collection["settle_seconds"])
                        state["accept"] = True
                        spin_for(collection["capture_seconds"])
                        if (view_index + 1) % 16 == 0 or view_index + 1 == len(order):
                            print(f"Run {run_index + 1}/{runs}: {view_index + 1}/{len(order)} views, "
                                  f"{state['counts'][topics['image']]} RGB frames", flush=True)
                else:
                    state["accept"] = True
                    spin_for(settings["record_seconds"])
                run_completed = True
            finally:
                with lock:
                    state["accept"] = False
                    state["writer"] = None
                writer.close()
                summary = {"seed": seed, "automatic_views": collection["enabled"], "complete": run_completed,
                           "topics": dict(state["counts"]), "class_counts": dict(state["class_counts"]),
                           "classes": settings["classes"]}
                (bag_path / "collection.json").write_text(json.dumps(summary, indent=2))
            if not state["counts"][topics["image"]] or not state["counts"][labels_topic]:
                raise RuntimeError("Bag has no RGB/ground-truth pair; check boundingbox camera sensor.")
            created.append(str(bag_path))
            print("Saved", dict(state["counts"]), flush=True)
            print("Visible class counts:", dict(state["class_counts"]), flush=True)
    finally:
        if manifest is not None:
            for sign in manifest["signs"]:
                pose = list(sign["pose"])
                if not sign["active"]:
                    pose[2] = -10
                try:
                    _set_pose(transport, manifest["world"], sign["name"], pose)
                except RuntimeError:
                    pass
            if "spawn_pose" in manifest:
                try:
                    _set_pose(transport, manifest["world"], manifest["robot"], manifest["spawn_pose"])
                except RuntimeError:
                    pass
        if rclpy.ok():
            active.publish(Bool(data=False))
            publish_stop()
            rclpy.spin_once(node, timeout_sec=0.05)
        transport.unsubscribe(topics["boxes"])
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return created


def _reader(path, storage_id, topic=None):
    import rosbag2_py
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id=storage_id),
                rosbag2_py.ConverterOptions("", ""))
    if topic is not None:
        reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    return reader


def extract(config):
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Image
    from std_msgs.msg import String

    settings = config["dataset"]
    classes = settings["classes"]
    if classes != CLASSES:
        raise ValueError("dataset.classes must match Gazebo label order: " + str(CLASSES))
    bags = []
    for metadata in sorted(project_path(config, config["paths"]["bags"]).glob("*/metadata.yaml")):
        collection_metadata = metadata.parent / "collection.json"
        if collection_metadata.exists() and not json.loads(collection_metadata.read_text()).get("complete", True):
            print("Skipping incomplete collection:", metadata.parent.name, flush=True)
            continue
        bags.append(metadata.parent)
    groups = {path.name: collection_group(path) for path in bags}
    group_splits = split_runs(groups.values(), settings["validation_fraction"], settings["split_seed"])
    splits = {name: group_splits[group] for name, group in groups.items()}
    destination = project_path(config, config["paths"]["dataset"])
    if destination.exists():
        raise FileExistsError("Dataset already exists. Set paths.dataset to a new directory in config.yaml.")
    destination.mkdir(parents=True)
    for split in ("train", "val"):
        (destination / "images" / split).mkdir(parents=True)
        (destination / "labels" / split).mkdir(parents=True)
    (destination / "previews").mkdir()
    report = {"source": "Gazebo RGB + visible_2d boxes, synchronized by simulation timestamp",
              "quality_parameters": {key: settings[key] for key in
                  ("min_box_pixels", "min_clipped_fraction", "minimum_object_contrast", "traffic_quality", "sync_tolerance")},
              "splits": splits, "collection_groups": groups,
              "images": Counter(), "class_counts": {"train": Counter(), "val": Counter()},
              "negative_images": Counter(), "unmatched_images": 0, "filtered_boxes": 0,
              "texture_rejected_images": 0,
              "ambiguous_signal_images": 0,
              "samples": []}
    interval = int(1e9 / settings["sample_hz"])
    tolerance = int(settings["sync_tolerance"] * 1e9)
    preview_count = 0
    for bag_path in bags:
        label_reader = _reader(bag_path, settings["storage_id"], config["topics"]["labels"])
        packets = {}
        while label_reader.has_next():
            _, data, _ = label_reader.read_next()
            packet = json.loads(deserialize_message(data, String).data)
            packets[int(packet["stamp_ns"])] = packet["boxes"]
        stamps = sorted(packets)
        image_reader = _reader(bag_path, settings["storage_id"], config["topics"]["image"])
        last_stamp = -interval
        split = splits[bag_path.name]
        while image_reader.has_next():
            _, data, _ = image_reader.read_next()
            message = deserialize_message(data, Image)
            timestamp = stamp_ns(message.header.stamp)
            if timestamp - last_stamp < interval:
                continue
            match = nearest_stamp(stamps, timestamp, tolerance)
            if match is None:
                report["unmatched_images"] += 1
                continue
            last_stamp = timestamp
            image = decode_image(message)
            height, width = image.shape[:2]
            labels = []
            for box in packets[match]:
                label = normalized_box(box, width, height, len(classes), settings["min_box_pixels"],
                                       settings["min_clipped_fraction"])
                if label is None:
                    report["filtered_boxes"] += 1
                else:
                    labels.append(label)
            # An image with an ignored tiny object is not an honest negative.
            if packets[match] and not labels:
                continue
            # All signs in this world contain a symbol or traffic lamp. A flat
            # crop indicates missing material/texture, not useful training data.
            texture_ok = True
            signal_ok = True
            for class_id, cx, cy, bw, bh in labels:
                x1, y1 = max(0, int((cx-bw/2)*width)), max(0, int((cy-bh/2)*height))
                x2, y2 = min(width, int((cx+bw/2)*width)), min(height, int((cy+bh/2)*height))
                crop = image[y1:y2, x1:x2]
                if crop.size == 0 or cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).std() < settings.get("minimum_object_contrast", 6.0):
                    texture_ok = False
                    break
                if not visible_traffic_lamp(crop, classes[class_id], settings["traffic_quality"]):
                    signal_ok = False
                    break
            if not texture_ok:
                report["texture_rejected_images"] += 1
                continue
            if not signal_ok:
                report["ambiguous_signal_images"] += 1
                continue
            name = bag_path.name + "_" + str(timestamp)
            if not cv2.imwrite(str(destination / "images" / split / (name + ".jpg")), image):
                raise OSError("Could not save dataset image")
            rows = [str(label[0]) + " " + " ".join(f"{value:.8f}" for value in label[1:]) for label in labels]
            (destination / "labels" / split / (name + ".txt")).write_text("\n".join(rows) + ("\n" if rows else ""))
            report["images"][split] += 1
            if not labels:
                report["negative_images"][split] += 1
            for label in labels:
                report["class_counts"][split][classes[label[0]]] += 1
            report["samples"].append({"image": "images/" + split + "/" + name + ".jpg",
                                      "run": bag_path.name, "image_stamp_ns": timestamp, "label_stamp_ns": match})
            if preview_count < settings["preview_count"]:
                for label in labels:
                    class_id, cx, cy, bw, bh = label
                    x1, y1 = int((cx - bw / 2) * width), int((cy - bh / 2) * height)
                    x2, y2 = int((cx + bw / 2) * width), int((cy + bh / 2) * height)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(image, classes[class_id], (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 255), 1)
                cv2.imwrite(str(destination / "previews" / (name + ".jpg")), image)
                preview_count += 1
        print("Extracted", bag_path.name, "->", split, flush=True)
    dataset_yaml = {"path": str(destination), "train": "images/train", "val": "images/val",
                    "names": dict(enumerate(classes))}
    (destination / "data.yaml").write_text(yaml.safe_dump(dataset_yaml, sort_keys=False))
    (destination / "manifest.json").write_text(json.dumps(report, indent=2))
    if not report["images"]["train"] or not report["images"]["val"]:
        raise RuntimeError("No synchronized images in one split; see dataset manifest.json.")
    missing = {split: [name for name in classes if report["class_counts"][split][name] == 0]
               for split in ("train", "val")}
    if settings["require_all_classes"] and any(missing.values()):
        raise RuntimeError("Missing classes; collect more runs and use a new dataset path: " + str(missing))
    print("Dataset", destination, "images:", dict(report["images"]), "classes:", report["class_counts"], flush=True)
    return destination


def main(action="extract"):
    if len(sys.argv) != 2:
        raise SystemExit("Pass only configs/config.yaml")
    config = load_config(sys.argv[1])
    return record(config) if action == "record" else extract(config)


if __name__ == "__main__":
    main()
