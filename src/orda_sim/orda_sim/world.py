"""Build a local-only Gazebo world from the supplied CAD, without Fuel downloads.

CAD paint geometry is measured, facilities/vehicle dynamics are practice assumptions.
The robot's model origin is the rear axle's ground projection; +x forward, +y left.
"""
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

import cv2
import numpy as np


def numbers(values):
    return " ".join(str(float(value)) for value in values)


def material(color):
    value = numbers(color)
    return f"<material><ambient>{value}</ambient><diffuse>{value}</diffuse><specular>0 0 0 1</specular></material>"


def inertia(mass, x, y, z):
    return f"<inertial><mass>{mass}</mass><inertia><ixx>{mass*(y*y+z*z)/12}</ixx><iyy>{mass*(x*x+z*z)/12}</iyy><izz>{mass*(x*x+y*y)/12}</izz></inertia></inertial>"


def box_geometry(size):
    return f"<geometry><box><size>{numbers(size)}</size></box></geometry>"


def label_plugin(label):
    return f'<plugin filename="gz-sim-label-system" name="gz::sim::systems::Label"><label>{label}</label></plugin>'


def mesh_file(path, vertices, texture):
    """One textured quad. Vertices are counterclockwise when seen from the front."""
    positions = numbers([number for vertex in vertices for number in vertex])
    normal = np.cross(np.array(vertices[1]) - vertices[0], np.array(vertices[2]) - vertices[0])
    normal = numbers(normal / np.linalg.norm(normal))
    path.write_text(f'''<?xml version="1.0"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
<asset><unit name="meter" meter="1"/><up_axis>Z_UP</up_axis></asset>
<library_images><image id="image"><init_from>{escape(str(texture))}</init_from></image></library_images>
<library_effects><effect id="effect"><profile_COMMON>
<newparam sid="surface"><surface type="2D"><init_from>image</init_from></surface></newparam>
<newparam sid="sampler"><sampler2D><source>surface</source></sampler2D></newparam>
<technique sid="common"><phong><ambient><color>1 1 1 1</color></ambient><diffuse><texture texture="sampler" texcoord="UVSET0"/></diffuse></phong></technique>
</profile_COMMON></effect></library_effects>
<library_materials><material id="material"><instance_effect url="#effect"/></material></library_materials>
<library_geometries><geometry id="quad"><mesh>
<source id="positions"><float_array id="positions-array" count="12">{positions}</float_array><technique_common><accessor source="#positions-array" count="4" stride="3"><param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>
<source id="uv"><float_array id="uv-array" count="8">0 0 1 0 1 1 0 1</float_array><technique_common><accessor source="#uv-array" count="4" stride="2"><param name="S" type="float"/><param name="T" type="float"/></accessor></technique_common></source>
<source id="normals"><float_array id="normals-array" count="3">{normal}</float_array><technique_common><accessor source="#normals-array" count="1" stride="3"><param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>
<vertices id="vertices"><input semantic="POSITION" source="#positions"/></vertices>
<triangles count="2" material="mat"><input semantic="VERTEX" source="#vertices" offset="0"/><input semantic="TEXCOORD" source="#uv" offset="1" set="0"/><input semantic="NORMAL" source="#normals" offset="2"/><p>0 0 0 1 1 0 2 2 0 0 0 0 2 2 0 3 3 0</p></triangles>
</mesh></geometry></library_geometries>
<library_visual_scenes><visual_scene id="scene"><node id="node"><instance_geometry url="#quad"><bind_material><technique_common><instance_material symbol="mat" target="#material"><bind_vertex_input semantic="UVSET0" input_semantic="TEXCOORD" input_set="0"/></instance_material></technique_common></bind_material></instance_geometry></node></visual_scene></library_visual_scenes>
<scene><instance_visual_scene url="#scene"/></scene></COLLADA>''')


def texture_material(path):
    uri = escape(path.resolve().as_uri())
    return f"<material><ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse><pbr><metal><albedo_map>{uri}</albedo_map><metalness>0</metalness><roughness>1</roughness></metal></pbr></material>"


def mesh_geometry(path):
    return f"<geometry><mesh><uri>{escape(path.resolve().as_uri())}</uri></mesh></geometry>"


def assets_dir():
    local = Path(__file__).resolve().parents[1] / "assets"
    if local.exists():
        return local
    from ament_index_python.packages import get_package_share_directory
    return Path(get_package_share_directory("orda_sim")) / "assets"


def draw_track(config, output):
    """Render measured DXF paths. Fill narrow enclosed paint strips, not road areas."""
    source = json.loads((assets_dir() / "cad_paths.json").read_text())
    ppm = int(config["track_pixels_per_meter"])
    mask = np.zeros((6 * ppm, 10 * ppm), np.uint8)
    # First four paths are the physical mat perimeter, not lane markings.
    for path in source["paths"][4:]:
        points = np.array([[round(x * ppm), round((6 - y) * ppm)]
                           for x, y in path["points"]], dtype=np.int32)
        cv2.polylines(mask, [points], False, 255, max(1, round(ppm * .004)), cv2.LINE_AA)
    # CAD contains outlines of 20 mm paint strips and zebra blocks. Filling
    # bounded narrow regions restores their interiors without changing coordinates.
    background = (mask <= 100).astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(background)
    distance = cv2.distanceTransform(background, cv2.DIST_L2, 3)
    for component in range(1, count):
        x, y, width, height, area = stats[component]
        if x == 0 or y == 0 or x + width >= 10 * ppm or y + height >= 6 * ppm:
            continue
        # Preserve the dark squares inside the source checker start line.
        if .9 * ppm < x < 1.1 * ppm and y < .75 * ppm:
            continue
        selected = components[y:y+height, x:x+width] == component
        _, hierarchy = cv2.findContours(selected.astype(np.uint8), cv2.RETR_CCOMP,
                                         cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None and np.any(hierarchy[0, :, 3] >= 0):
            continue
        maximum = distance[y:y+height, x:x+width][selected].max()
        if maximum < ppm * config["lane_fill_width_m"]:
            mask[y:y+height, x:x+width][selected] = 255
    image = np.empty((*mask.shape, 3), np.uint8)
    image[:] = config["road_color"]
    image[mask > 100] = config["lane_color"]
    texture = output / "track.png"
    cv2.imwrite(str(texture), image)
    mesh = output / "track.dae"
    mesh_file(mesh, [(0, 0, .001), (10, 0, .001), (10, 6, .001), (0, 6, .001)], texture)
    return mesh


def sign_texture(kind, path):
    im = np.full((256, 256, 3), 245, np.uint8)
    blue, white, black = (200, 70, 10), (250, 250, 250), (15, 15, 15)
    if kind.startswith("traffic_"):
        im[:] = black
        active = {"traffic_red": 0, "traffic_yellow": 1, "traffic_green": 2}[kind]
        colors = [(25, 25, 255), (0, 220, 255), (25, 255, 25)]
        for index in range(3):
            cv2.circle(im, (128, 44 + 84 * index), 34, colors[index] if active == index else (40, 40, 40), -1)
    elif kind in ("left", "right"):
        cv2.circle(im, (128, 128), 118, blue, -1)
        start, end = ((202, 128), (53, 128)) if kind == "left" else ((53, 128), (202, 128))
        cv2.arrowedLine(im, start, end, white, 30, tipLength=.48)
    elif kind == "parking":
        im[:] = blue
        cv2.putText(im, "P", (55, 205), cv2.FONT_HERSHEY_SIMPLEX, 6, white, 16, cv2.LINE_AA)
    elif kind == "slow":
        cv2.fillPoly(im, [np.array([[128, 12], [13, 236], [243, 236]])], (0, 190, 255))
        cv2.polylines(im, [np.array([[128, 12], [13, 236], [243, 236]])], True, (20, 20, 230), 15)
        cv2.putText(im, "!", (104, 166), cv2.FONT_HERSHEY_SIMPLEX, 3, black, 12, cv2.LINE_AA)
        cv2.putText(im, "SLOW", (53, 216), cv2.FONT_HERSHEY_SIMPLEX, 1.1, black, 3, cv2.LINE_AA)
    elif kind == "pedestrian":
        im[:] = blue
        cv2.fillPoly(im, [np.array([[128, 15], [20, 232], [236, 232]])], white)
        cv2.circle(im, (130, 89), 16, black, -1)
        for a, b in [((130, 105), (115, 154)), ((121, 130), (159, 148)),
                     ((120, 130), (86, 150)), ((115, 154), (91, 201)), ((115, 154), (156, 199))]:
            cv2.line(im, a, b, black, 13)
        for x in range(45, 215, 33):
            cv2.line(im, (x, 218), (x + 20, 205), black, 8)
    else:
        # Three moving advertisement frames deliberately contain signal colors.
        index = int(kind.rsplit("_", 1)[1])
        im[:] = [(70, 25, 120), (130, 100, 10), (25, 100, 90)][index]
        cv2.putText(im, "ORDA", (15, 73), cv2.FONT_HERSHEY_SIMPLEX, 1.8, white, 4)
        cv2.putText(im, "2026", (22, 222), cv2.FONT_HERSHEY_SIMPLEX, 1.7, white, 3)
        cv2.circle(im, (60 + index * 65, 135), 35, [(0, 0, 255), (0, 255, 0), (0, 220, 255)][index], -1)
    cv2.imwrite(str(path), im)


def sign_model(name, kind, pose, size, label, active, output, post_radius):
    width, height = size
    texture = output / f"{kind}.png"
    sign_texture(kind, texture)
    mesh = output / f"{kind}.dae"
    # This winding presents a front face toward -X. +Y is left in camera view.
    mesh_file(mesh, [(-.012, width/2, -height/2), (-.012, -width/2, -height/2),
                     (-.012, -width/2, height/2), (-.012, width/2, height/2)], texture)
    actual = list(pose)
    if not active:
        actual[2] = -10
    board = box_geometry([.02, width, height])
    tag = label_plugin(label) if label else ""
    # The post is intentionally not included in the semantic class box.
    post_height = max(.05, pose[2] - height/2)
    post_z = -(height/2 + post_height/2)
    return f'''<model name="{name}"><static>true</static><pose>{numbers(actual)}</pose>
<link name="panel"><collision name="plate_collision">{board}</collision>
<visual name="back">{board}{material([.05,.05,.05,1])}</visual>
<visual name="face">{mesh_geometry(mesh)}{texture_material(texture)}{tag}<cast_shadows>false</cast_shadows></visual>
<collision name="post_collision"><pose>0 0 {post_z} 0 0 0</pose><geometry><cylinder><radius>{post_radius}</radius><length>{post_height}</length></cylinder></geometry></collision>
<visual name="post"><pose>0 0 {post_z} 0 0 0</pose><geometry><cylinder><radius>{post_radius}</radius><length>{post_height}</length></cylinder></geometry>{material([.25,.25,.25,1])}</visual>
</link></model>'''


def parked_car(name, pose, robot, scene):
    length, width = robot["length"], robot["width"]
    size = [length, width, scene["obstacle_body_height"]]
    body_z = scene["obstacle_ground_clearance"] + size[2] / 2
    geometry = box_geometry(size)
    base = f'<collision name="body_collision"><pose>0 0 {body_z} 0 0 0</pose>{geometry}</collision><visual name="body"><pose>0 0 {body_z} 0 0 0</pose>{geometry}{material([.12,.25,.42,1])}</visual>'
    for x in [-.10, .10]:
        for y in [-width/2, width/2]:
            base += f'<visual name="wheel_{x}_{y}"><pose>{x} {y} .05 1.57079632679 0 0</pose><geometry><cylinder><radius>.05</radius><length>.025</length></cylinder></geometry>{material([.025,.025,.025,1])}</visual>'
    return f'<model name="{name}"><static>true</static><pose>{numbers(pose)}</pose><link name="body">{base}</link></model>'


def cone_model(index, xy, scene):
    radius, height, mass = scene["cone_radius"], scene["cone_height"], scene["cone_mass"]
    # A real dynamic cone mesh/collision is knockable. No wall collision proxy.
    geometry = f'<geometry><cone><radius>{radius}</radius><length>{height}</length></cone></geometry>'
    return f'''<model name="cone_{index}"><pose>{xy[0]} {xy[1]} {height/2+.003} 0 0 0</pose><link name="cone">
{inertia(mass,2*radius,2*radius,height)}<collision name="cone_collision">{geometry}</collision><visual name="cone">{geometry}{material([1,.35,.025,1])}</visual>
</link></model>'''


def robot_model(robot, topics):
    name = robot["name"]
    wheelbase, width = robot["wheelbase"], robot["width"]
    radius, wheel_width = robot["wheel_radius"], robot["wheel_width"]
    length, height, mass = robot["length"], robot["body_height"], robot["mass"]
    body_z = robot["ground_clearance"] + height/2
    parts = [f'<model name="{name}"><pose>{numbers(robot["spawn"])}</pose>']
    body_geometry = box_geometry([length, width-wheel_width, height])
    parts.append(f'<link name="base_link"><pose>{wheelbase/2} 0 {body_z} 0 0 0</pose>{inertia(mass,length,width,height)}<collision name="body_collision">{body_geometry}</collision><visual name="body">{body_geometry}{material([.1,.45,.7,1])}</visual></link>')
    for axle, x in [("rear", 0), ("front", wheelbase)]:
        for side, y in [("left", width/2), ("right", -width/2)]:
            wheel = f"{axle}_{side}"
            parent = "base_link"
            if axle == "front":
                parent = wheel + "_steer"
                parts.append(f'<link name="{parent}"><pose>{x} {y} {radius} 0 0 0</pose>{inertia(.02,.02,.02,.02)}</link>')
                parts.append(f'<joint name="{wheel}_steering" type="revolute"><parent>base_link</parent><child>{parent}</child><axis><xyz>0 0 1</xyz><limit><lower>-.8</lower><upper>.8</upper><effort>30</effort><velocity>4</velocity></limit></axis></joint>')
            geometry = f'<geometry><cylinder><radius>{radius}</radius><length>{wheel_width}</length></cylinder></geometry>'
            parts.append(f'<link name="{wheel}"><pose>{x} {y} {radius} 1.57079632679 0 0</pose>{inertia(.12,2*radius,2*radius,wheel_width)}<collision name="wheel_collision">{geometry}<surface><friction><ode><mu>1</mu><mu2>1</mu2></ode></friction></surface></collision><visual name="wheel">{geometry}{material([.025,.025,.025,1])}</visual></link>')
            parts.append(f'<joint name="{wheel}_joint" type="revolute"><parent>{parent}</parent><child>{wheel}</child><axis><xyz>0 0 -1</xyz><limit><lower>-1e16</lower><upper>1e16</upper></limit></axis></joint>')
    camera = robot["camera"]
    pose = [camera["x"], camera["y"], camera["z"], 0, camera["pitch"], 0]
    camera_spec = f'<horizontal_fov>{math.radians(camera["hfov_deg"])}</horizontal_fov><image><width>{camera["width"]}</width><height>{camera["height"]}</height><format>R8G8B8</format></image><clip><near>{camera["near"]}</near><far>{camera["far"]}</far></clip>'
    parts.append(f'<link name="camera_link"><pose>{numbers(pose)}</pose>{inertia(.01,.01,.01,.01)}')
    parts.append(f'<sensor name="rgb" type="camera"><topic>{topics["image"]}</topic><always_on>true</always_on><update_rate>{camera["rate_hz"]}</update_rate><camera>{camera_spec}</camera></sensor>')
    parts.append(f'<sensor name="boxes" type="boundingbox_camera"><topic>{topics["boxes"]}</topic><always_on>true</always_on><update_rate>{camera["rate_hz"]}</update_rate><camera><box_type>visible_2d</box_type>{camera_spec}</camera></sensor></link>')
    parts.append('<joint name="camera_fixed" type="fixed"><parent>base_link</parent><child>camera_link</child></joint>')
    lidar = robot["lidar"]
    parts.append(f'''<link name="lidar_link"><pose>{lidar['x']} {lidar['y']} {lidar['z']} 0 0 0</pose>{inertia(.01,.01,.01,.01)}
<sensor name="lidar" type="gpu_lidar"><topic>{topics['scan']}</topic><always_on>true</always_on><update_rate>{lidar['rate_hz']}</update_rate><lidar><scan><horizontal><samples>{lidar['samples']}</samples><resolution>1</resolution><min_angle>{-math.pi}</min_angle><max_angle>{math.pi}</max_angle></horizontal></scan><range><min>{lidar['range_min']}</min><max>{lidar['range_max']}</max><resolution>.01</resolution></range></lidar></sensor></link>
<joint name="lidar_fixed" type="fixed"><parent>base_link</parent><child>lidar_link</child></joint>''')
    parts.append(f'''<plugin filename="gz-sim-ackermann-steering-system" name="gz::sim::systems::AckermannSteering">
<left_joint>rear_left_joint</left_joint><right_joint>rear_right_joint</right_joint>
<left_steering_joint>front_left_steering</left_steering_joint><right_steering_joint>front_right_steering</right_steering_joint>
<kingpin_width>{width}</kingpin_width><wheel_separation>{width}</wheel_separation><wheel_base>{wheelbase}</wheel_base><wheel_radius>{radius}</wheel_radius>
<steering_limit>{robot['max_steering']}</steering_limit><steer_p_gain>8</steer_p_gain>
<min_velocity>{-robot['max_speed']}</min_velocity><max_velocity>{robot['max_speed']}</max_velocity>
<min_acceleration>{-robot['max_acceleration']}</min_acceleration><max_acceleration>{robot['max_acceleration']}</max_acceleration>
<topic>{topics["cmd_vel"]}</topic><odom_topic>/model/{name}/odometry</odom_topic><odom_publish_frequency>30</odom_publish_frequency><frame_id>odom</frame_id><child_frame_id>base_link</child_frame_id>
</plugin></model>''')
    return "\n".join(parts)


def generate_world(config, output_dir):
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    world, robot, scene = config["world"], config["robot"], config["scene"]
    track_mesh = draw_track(world, output)
    xml = [f'<sdf version="1.9"><world name="{world["name"]}">']
    xml.append(f'<physics name="physics" type="ignored"><max_step_size>{world["step_size"]}</max_step_size><real_time_factor>{world["real_time_factor"]}</real_time_factor></physics>')
    for name in ["Physics", "UserCommands", "SceneBroadcaster", "Sensors"]:
        filename = {"Physics": "physics", "UserCommands": "user-commands", "SceneBroadcaster": "scene-broadcaster", "Sensors": "sensors"}[name]
        body = '<render_engine>ogre2</render_engine>' if name == "Sensors" else ""
        xml.append(f'<plugin filename="gz-sim-{filename}-system" name="gz::sim::systems::{name}">{body}</plugin>')
    xml.append(f'''<gui fullscreen="0">
<plugin filename="MinimalScene" name="3D View"><gz-gui><title>ORDA 2026</title><property type="bool" key="showTitleBar">false</property><property type="string" key="state">docked</property></gz-gui><engine>ogre2</engine><scene>scene</scene><ambient_light>.8 .8 .8 1</ambient_light><background_color>.3 .3 .35 1</background_color><camera_pose>{numbers(world["gui_camera_pose"])}</camera_pose></plugin>
<plugin filename="GzSceneManager" name="Scene Manager"><gz-gui><property key="state" type="string">floating</property><property key="width" type="double">5</property><property key="height" type="double">5</property><property key="showTitleBar" type="bool">false</property></gz-gui></plugin>
<plugin filename="InteractiveViewControl" name="Interactive view control"><gz-gui><property key="state" type="string">floating</property><property key="width" type="double">5</property><property key="height" type="double">5</property><property key="showTitleBar" type="bool">false</property></gz-gui></plugin>
<plugin filename="CameraTracking" name="Camera Tracking"><gz-gui><property key="state" type="string">floating</property><property key="width" type="double">5</property><property key="height" type="double">5</property><property key="showTitleBar" type="bool">false</property></gz-gui></plugin>
<plugin filename="WorldControl" name="World control"><gz-gui><title>Run / pause</title><property type="string" key="state">floating</property><property type="double" key="width">200</property><property type="double" key="height">72</property><anchors target="3D View"><line own="left" target="left"/><line own="bottom" target="bottom"/></anchors></gz-gui><play_pause>true</play_pause><step>true</step><start_paused>false</start_paused><use_event>true</use_event></plugin>
</gui>''')
    xml.append(f'<scene><ambient>{numbers(world["ambient"])}</ambient><background>.3 .3 .35 1</background><shadows>false</shadows></scene>')
    xml.append('<light name="sun" type="directional"><pose>0 0 10 0 0 0</pose><diffuse>.8 .8 .8 1</diffuse><direction>0.1 0.2 -1</direction><cast_shadows>false</cast_shadows></light>')
    xml.append(f'<model name="cad_track"><static>true</static><link name="mat"><collision name="ground"><geometry><plane><normal>0 0 1</normal><size>30 30</size></plane></geometry></collision><visual name="cad_paint">{mesh_geometry(track_mesh)}{texture_material(output / "track.png")}</visual></link></model>')
    manifest = {"world": world["name"], "robot": robot["name"], "signs": [], "track_extent_m": [10, 6], "coordinate_origin": "CAD mat lower left", "spawn_pose": robot["spawn"]}
    specs = [("slow", "slow", scene["slow_pose"], True, 7),
             ("pedestrian", "pedestrian", scene["pedestrian_pose"], True, 6),
             ("parking", "parking", scene["parking_pose"], True, 8)]
    for index, kind in enumerate(["traffic_red", "traffic_yellow", "traffic_green"]):
        specs.append(("traffic", kind, scene["traffic_pose"], index == 0, index + 1))
    for kind, label in [("left", 4), ("right", 5)]:
        specs.append(("fork", kind, scene["fork_pose"], scene["fork_initial"] == kind, label))
    for index in range(3):
        specs.append(("advertisement", f"advertisement_{index}", scene["advertisement_pose"], index == 0, 0))
    for group, kind, pose, active, label in specs:
        name = "sign_" + kind
        size = scene["signal_size"] if group == "traffic" else scene["advertisement_size"] if group == "advertisement" else [scene["sign_size"]] * 2
        xml.append(sign_model(name, kind, pose, size, label, active, output, scene["post_radius"]))
        manifest["signs"].append({"name": name, "class_name": kind, "label": label, "pose": pose, "active": active, "group": group, "size": size})
    bump = scene["bump"]
    geometry = f'<geometry><cylinder><radius>{bump["radius"]}</radius><length>{bump["width"]}</length></cylinder></geometry>'
    pose = list(bump["pose"])
    pose[3] = math.pi / 2
    xml.append(f'<model name="speed_bump"><static>true</static><pose>{numbers(pose)}</pose><link name="bump"><collision name="bump_collision">{geometry}</collision><visual name="bump">{geometry}{material([.9,.65,.05,1])}</visual></link></model>')
    reference = scene["parking_reference"]
    if reference["enabled"]:
        geometry = box_geometry([reference["length"], reference["width"], .001])
        xml.append(f'<model name="parking_reference_tape"><static>true</static><pose>{numbers(reference["pose"])}</pose><link name="tape"><visual name="tape">{geometry}{material([.95,.95,.95,1])}<cast_shadows>false</cast_shadows></visual></link></model>')
    for index, pose in enumerate(scene["obstacles"]):
        xml.append(parked_car(f"obstacle_{index}", pose, robot, scene))
    for index, pose in enumerate(scene["parking_obstacles"]):
        xml.append(parked_car(f"parking_obstacle_{index}", pose, robot, scene))
    for index, xy in enumerate(scene["cones"]):
        xml.append(cone_model(index, xy, scene))
    xml.append(robot_model(robot, config["topics"]))
    xml.append('</world></sdf>')
    path = output / "orda.sdf"
    path.write_text("\n".join(xml))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return path
