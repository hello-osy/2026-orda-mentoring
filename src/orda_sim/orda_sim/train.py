"""Train YOLOv8 using rosbag-extracted frames and export a traceable checkpoint."""

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from .dataset import CLASSES, load_config, project_path


def train(config):
    settings = config["train"]
    values = settings["augmentation"]
    augmentation = {}
    # Mirroring swaps left/right semantics; hue shifts swap light colors.
    for key in ("fliplr", "flipud", "hsv_h"):
        augmentation[key] = float(values.get(key, 0.0))
        if augmentation[key] != 0.0:
            raise ValueError("train.augmentation." + key + " must be 0: flips or hue shifts change sign/light meanings.")
    for key in ("hsv_s", "hsv_v", "degrees", "shear", "perspective", "mixup"):
        augmentation[key] = float(values[key])

    from ultralytics import YOLO
    import ultralytics
    import torch

    torch.set_num_threads(int(settings.get("cpu_threads", 2)))
    dataset = project_path(config, config["paths"]["dataset"])
    manifest = json.loads((dataset / "manifest.json").read_text())
    splits = manifest["splits"]
    if set(splits.values()) != {"train", "val"}:
        raise ValueError("Both train and val must contain independent recorded runs.")
    missing = {split: [name for name in CLASSES if manifest["class_counts"][split].get(name, 0) == 0]
               for split in ("train", "val")}
    if config["dataset"]["require_all_classes"] and any(missing.values()):
        raise ValueError("Missing classes in training/validation: " + str(missing))
    model_name = str(settings["model"])
    if not Path(model_name).name.startswith("yolov8"):
        raise ValueError("This mentoring project expects a YOLOv8 model, e.g. yolov8n.pt.")
    local_model = project_path(config, model_name)
    model = YOLO(str(local_model) if local_model.exists() else model_name)
    # Ultralytics resets the CPU thread count during device selection, so apply
    # the YAML value again through its supported lifecycle callbacks.
    def limit_threads(_):
        torch.set_num_threads(int(settings.get("cpu_threads", 2)))
    model.add_callback("on_train_start", limit_threads)
    model.add_callback("on_val_start", limit_threads)
    project = project_path(config, settings["project"])
    name = datetime.now(timezone.utc).strftime("yolov8_%Y%m%dT%H%M%S_%f")
    result = model.train(data=str(dataset / "data.yaml"), epochs=settings["epochs"],
                         imgsz=settings["image_size"], batch=settings["batch"],
                         device=settings["device"], workers=settings["workers"],
                         seed=settings["seed"], deterministic=True, patience=settings["patience"],
                         optimizer=settings["optimizer"], lr0=settings["learning_rate"], pretrained=settings["pretrained"],
                         project=str(project), name=name, exist_ok=False,
                         plots=True, **augmentation)
    output = Path(model.trainer.save_dir)
    checkpoint = output / "weights" / "best.pt"
    if not checkpoint.is_file():
        raise RuntimeError("Training ended without best.pt; inspect " + str(output))
    # Validate the saved best checkpoint, not the last in-memory training model.
    best = YOLO(str(checkpoint))
    best.add_callback("on_val_start", limit_threads)
    metrics = best.val(data=str(dataset / "data.yaml"), split="val", imgsz=settings["image_size"],
                       batch=settings["batch"], device=settings["device"], workers=settings["workers"],
                       project=str(output), name="validation", plots=True)
    models = project_path(config, config["paths"]["models"])
    models.mkdir(parents=True, exist_ok=True)
    target = models / "best.pt"
    shutil.copy2(checkpoint, target)
    summary = {
        "model": "YOLOv8", "ultralytics_version": ultralytics.__version__,
        "checkpoint": str(target), "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "training_directory": str(output), "dataset": str(dataset),
        "dataset_quality": manifest.get("quality_parameters", {}),
        "data_source": "Gazebo camera images recorded into rosbag2 and extracted with visible-2D GT",
        "split_unit": "independent collection run", "classes": CLASSES,
        "epochs_requested": settings["epochs"],
        "epochs_completed": int(model.trainer.epoch) + 1,
        "metrics": {key: float(value) for key, value in metrics.results_dict.items()},
        "per_class": [{key: value.item() if hasattr(value, "item") else value
                       for key, value in row.items()} for row in metrics.summary()],
        "limits": "Simulation-only validation; does not establish real-track accuracy or mission completion."
    }
    (models / "training_report.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return target


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Pass only configs/config.yaml")
    train(load_config(sys.argv[1]))


if __name__ == "__main__":
    main()
