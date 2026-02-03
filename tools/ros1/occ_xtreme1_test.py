#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _quat_wxyz_to_rot(q: List[float]) -> List[List[float]]:
    w, x, y, z = q
    return [
        [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
    ]


def _se3_from_quat_tran(q: List[float], t: List[float]) -> List[List[float]]:
    rot = _quat_wxyz_to_rot(q)
    return [
        [rot[0][0], rot[0][1], rot[0][2], float(t[0])],
        [rot[1][0], rot[1][1], rot[1][2], float(t[1])],
        [rot[2][0], rot[2][1], rot[2][2], float(t[2])],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _matmul_4x4(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    return [
        [
            sum(a[i][k] * b[k][j] for k in range(4))
            for j in range(4)
        ]
        for i in range(4)
    ]


def _find_xtreme1_image(img_root: Path, timestamp: str, ext: Optional[str]) -> Path:
    if ext:
        candidate = img_root / f"{timestamp}{ext}"
        if candidate.exists():
            return candidate
    else:
        matches = list(img_root.glob(f"{timestamp}.*"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Cannot find image for {timestamp} under {img_root}")


def _load_xtreme1(
    scene_root: Path,
    timestamp: str,
    img_ext: Optional[str],
    keep_cams: Optional[List[str]],
) -> List[Dict]:
    lidar_cfg_path = scene_root / "lidar_config" / f"{timestamp}.json"
    cam_cfg_path = scene_root / "camera_config" / f"{timestamp}.json"
    if not lidar_cfg_path.exists():
        raise FileNotFoundError(f"Missing lidar config: {lidar_cfg_path}")
    if not cam_cfg_path.exists():
        raise FileNotFoundError(f"Missing camera config: {cam_cfg_path}")

    lidar_cfg = json.loads(lidar_cfg_path.read_text())
    cam_cfg = json.loads(cam_cfg_path.read_text())

    ego_pose = _se3_from_quat_tran(
        lidar_cfg["ego_pose"]["rotation"], lidar_cfg["ego_pose"]["translation"]
    )
    lidar_sensor = _se3_from_quat_tran(
        lidar_cfg["calibrated_sensor"]["rotation"],
        lidar_cfg["calibrated_sensor"]["translation"],
    )
    ego2global = _matmul_4x4(ego_pose, lidar_sensor)
    camera_list = []
    for cam_name, cam_vals in cam_cfg.items():
        if keep_cams and cam_name not in keep_cams:
            continue
        cam_sensor = _se3_from_quat_tran(cam_vals["rotation"], cam_vals["translation"])
        img_path = _find_xtreme1_image(scene_root / cam_name, timestamp, img_ext)
        camera_list.append(
            {
                "name": cam_name,
                "img_path": str(img_path),
                "intrinsic": cam_vals["camera_intrinsic"],
                "extrinsic": cam_sensor,
                "ego2global": ego2global,
            }
        )
    if not camera_list:
        raise ValueError("No cameras loaded from Xtreme1 config")
    return camera_list


def _discover_xtreme1_groups(root: Path) -> List[Tuple[str, Path]]:
    def is_scene(path: Path) -> bool:
        return (path / "camera_config").is_dir() and (path / "lidar_config").is_dir()

    if is_scene(root):
        return [(root.name, root)]

    groups = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        if is_scene(sub):
            groups.append((sub.name, sub))
    if not groups:
        raise FileNotFoundError(
            f"No Xtreme1 scenes found under {root}; expected camera_config/ and lidar_config/"
        )
    return groups


def _list_xtreme1_timestamps(scene_root: Path) -> List[str]:
    cam_dir = scene_root / "camera_config"
    lidar_dir = scene_root / "lidar_config"
    timestamps = []
    for cam_json in cam_dir.glob("*.json"):
        ts = cam_json.stem
        if (lidar_dir / f"{ts}.json").exists():
            timestamps.append(ts)
    return sorted(timestamps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Publish Xtreme1 frames as ROS camera topics to simulate on-robot inputs."
        )
    )
    parser.add_argument("--xtreme1-root", required=True, help="Xtreme1 scene root")
    parser.add_argument("--timestamp", help="Timestamp to publish (without extension)")
    parser.add_argument("--run-all", action="store_true", help="Publish all timestamps")
    parser.add_argument("--img-ext", default=None, help="Image extension override")
    parser.add_argument("--cameras", nargs="*", default=None, help="Camera subset")
    parser.add_argument("--camera-left", help="Camera name to map to left")
    parser.add_argument("--camera-front", help="Camera name to map to front")
    parser.add_argument("--camera-right", help="Camera name to map to right")
    parser.add_argument(
        "--topic-left", default="/camera_image_left", help="ROS topic for left camera"
    )
    parser.add_argument(
        "--topic-front", default="/camera_image_front", help="ROS topic for front camera"
    )
    parser.add_argument(
        "--topic-right", default="/camera_image_right", help="ROS topic for right camera"
    )
    parser.add_argument(
        "--ros-rate",
        type=float,
        default=5.0,
        help="Publish rate (Hz)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop over timestamps when publishing to ROS",
    )
    parser.add_argument(
        "--use-timestamp-stamp",
        action="store_true",
        help="Use dataset timestamp as ROS header.stamp if possible",
    )
    return parser.parse_args()


def _select_camera_by_hint(cameras: List[Dict], hints: Iterable[str]) -> Optional[Dict]:
    lower_map = {cam["name"].lower(): cam for cam in cameras}
    for hint in hints:
        for name, cam in lower_map.items():
            if hint in name:
                return cam
    return None


def _resolve_camera_triplet(
    cameras: List[Dict],
    left: Optional[str],
    front: Optional[str],
    right: Optional[str],
) -> Tuple[Dict, Dict, Dict]:
    cam_by_name = {cam["name"]: cam for cam in cameras}
    if left and front and right:
        missing = [name for name in (left, front, right) if name not in cam_by_name]
        if missing:
            raise ValueError(f"Missing cameras in dataset: {missing}")
        return cam_by_name[left], cam_by_name[front], cam_by_name[right]

    left_cam = _select_camera_by_hint(cameras, ("left", "lf", "l"))
    front_cam = _select_camera_by_hint(cameras, ("front", "front_cam", "fc", "f"))
    right_cam = _select_camera_by_hint(cameras, ("right", "rf", "r"))

    if left_cam and front_cam and right_cam:
        return left_cam, front_cam, right_cam

    if len(cameras) < 3:
        raise ValueError("Need at least 3 cameras to build left/front/right triplet")
    return cameras[0], cameras[1], cameras[2]


def _find_scene_roots(xtreme1_root: Path) -> List[Path]:
    if (xtreme1_root / "camera_config").is_dir():
        return [xtreme1_root]
    groups = _discover_xtreme1_groups(xtreme1_root)
    return [scene_root for _, scene_root in groups]


def _load_xtreme1_camera_config(scene_root: Path, timestamp: str) -> Dict:
    cam_cfg_path = scene_root / "camera_config" / f"{timestamp}.json"
    if not cam_cfg_path.exists():
        raise FileNotFoundError(f"Missing camera config: {cam_cfg_path}")
    return json.loads(cam_cfg_path.read_text())


def _load_xtreme1_compat(
    scene_root: Path,
    timestamp: str,
    img_ext: Optional[str],
    keep_cams: Optional[List[str]],
) -> List[Dict]:
    """Load Xtreme1 cameras, falling back when lidar_config is missing."""
    try:
        return _load_xtreme1(scene_root, timestamp, img_ext, keep_cams)
    except FileNotFoundError:
        cam_cfg = _load_xtreme1_camera_config(scene_root, timestamp)
        camera_list = []
        for cam_name, cam_vals in cam_cfg.items():
            if keep_cams and cam_name not in keep_cams:
                continue
            img_path = _find_xtreme1_image(scene_root / cam_name, timestamp, img_ext)
            camera_list.append(
                {
                    "name": cam_name,
                    "img_path": str(img_path),
                    "intrinsic": cam_vals["camera_intrinsic"],
                    "extrinsic": _se3_from_quat_tran(
                        cam_vals["rotation"], cam_vals["translation"]
                    ),
                    "ego2global": _se3_from_quat_tran(
                        [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
                    ),
                }
            )
        if not camera_list:
            raise ValueError("No cameras loaded from Xtreme1 config")
        return camera_list


def _stamp_from_timestamp(timestamp: str):
    try:
        ts_float = float(timestamp)
    except ValueError:
        return None
    return ts_float


def publish_for_timestamp(
    scene_root: Path,
    timestamp: str,
    img_ext: Optional[str],
    keep_cams: Optional[List[str]],
    camera_left: Optional[str],
    camera_front: Optional[str],
    camera_right: Optional[str],
    topic_left: str,
    topic_front: str,
    topic_right: str,
    use_timestamp_stamp: bool,
    bridge,
    publishers: Dict[str, "rospy.Publisher"],
) -> None:
    import cv2
    import rospy

    camera_list = _load_xtreme1_compat(
        scene_root, timestamp, img_ext, keep_cams
    )
    left_cam, front_cam, right_cam = _resolve_camera_triplet(
        camera_list, camera_left, camera_front, camera_right
    )
    image_paths = {
        topic_left: Path(left_cam["img_path"]),
        topic_front: Path(front_cam["img_path"]),
        topic_right: Path(right_cam["img_path"]),
    }
    stamp_value = _stamp_from_timestamp(timestamp) if use_timestamp_stamp else None
    ros_stamp = rospy.Time.from_sec(stamp_value) if stamp_value else rospy.Time.now()
    for topic, path in image_paths.items():
        cv_image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if cv_image is None:
            raise RuntimeError(f"Failed to read image: {path}")
        msg = bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        msg.header.stamp = ros_stamp
        msg.header.frame_id = Path(topic).name
        publishers[topic].publish(msg)
    print(f"[ROS] Published {scene_root.name}/{timestamp} to {topic_left}, {topic_front}, {topic_right}")


def main() -> None:
    args = parse_args()
    import rospy
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image

    scene_roots = _find_scene_roots(Path(args.xtreme1_root))
    if not scene_roots:
        raise ValueError("No scenes found to send")

    rospy.init_node("occ_xtreme1_publisher", anonymous=True)
    bridge = CvBridge()
    publishers = {
        args.topic_left: rospy.Publisher(args.topic_left, Image, queue_size=5),
        args.topic_front: rospy.Publisher(args.topic_front, Image, queue_size=5),
        args.topic_right: rospy.Publisher(args.topic_right, Image, queue_size=5),
    }
    rate = rospy.Rate(args.ros_rate)
    while not rospy.is_shutdown():
        for scene_root in scene_roots:
            if args.run_all:
                timestamps = _list_xtreme1_timestamps(scene_root)
            elif args.timestamp:
                timestamps = [args.timestamp]
            else:
                raise ValueError("Provide --timestamp or --run-all")

            if not timestamps:
                raise ValueError(f"No timestamps found for scene {scene_root}")

            for ts in timestamps:
                publish_for_timestamp(
                    scene_root=scene_root,
                    timestamp=ts,
                    img_ext=args.img_ext,
                    keep_cams=args.cameras,
                    camera_left=args.camera_left,
                    camera_front=args.camera_front,
                    camera_right=args.camera_right,
                    topic_left=args.topic_left,
                    topic_front=args.topic_front,
                    topic_right=args.topic_right,
                    use_timestamp_stamp=args.use_timestamp_stamp,
                    bridge=bridge,
                    publishers=publishers,
                )
                rate.sleep()
        if not args.loop:
            break


if __name__ == "__main__":
    main()
