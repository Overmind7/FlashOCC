#!/usr/bin/env python
import argparse
import json
import mimetypes
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(REPO_ROOT / "tools"))
import custom_infer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send Xtreme1 frames to the ROS1 occ server and validate responses"
    )
    parser.add_argument("--xtreme1-root", required=True, help="Xtreme1 scene root")
    parser.add_argument("--timestamp", help="Timestamp to send (without extension)")
    parser.add_argument("--run-all", action="store_true", help="Send all timestamps")
    parser.add_argument("--img-ext", default=None, help="Image extension override")
    parser.add_argument("--cameras", nargs="*", default=None, help="Camera subset")
    parser.add_argument("--camera-left", help="Camera name to map to left")
    parser.add_argument("--camera-front", help="Camera name to map to front")
    parser.add_argument("--camera-right", help="Camera name to map to right")
    parser.add_argument(
        "--server-url",
        default="http://localhost:5801/infer",
        help="Inference server URL",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
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


def _build_request_payload(
    left_cam: Dict,
    front_cam: Dict,
    right_cam: Dict,
) -> Tuple[Dict, Dict]:
    image_keys = {
        left_cam["name"]: "image_left",
        front_cam["name"]: "image_front",
        right_cam["name"]: "image_right",
    }
    cameras = []
    for cam, key in (
        (left_cam, "image_left"),
        (front_cam, "image_front"),
        (right_cam, "image_right"),
    ):
        cam_meta = {
            k: v for k, v in cam.items() if k not in {"img_path"}
        }
        cam_meta["image_key"] = key
        cameras.append(cam_meta)
    metadata = {"cameras": cameras, "image_keys": image_keys}
    return metadata, image_keys


def _find_scene_root(xtreme1_root: Path) -> Path:
    if (xtreme1_root / "camera_config").is_dir():
        return xtreme1_root
    groups = custom_infer.discover_xtreme1_groups(xtreme1_root)
    if len(groups) != 1:
        names = [name for name, _ in groups]
        raise ValueError(
            "xtreme1-root points to multiple scenes, please provide a scene folder: "
            f"{names}"
        )
    return groups[0][1]


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
        return custom_infer.load_xtreme1(str(scene_root), timestamp, img_ext, keep_cams)
    except FileNotFoundError:
        cam_cfg = _load_xtreme1_camera_config(scene_root, timestamp)
        camera_list = []
        for cam_name, cam_vals in cam_cfg.items():
            if keep_cams and cam_name not in keep_cams:
                continue
            img_path = custom_infer.find_xtreme1_image(
                scene_root / cam_name, timestamp, img_ext
            )
            camera_list.append(
                {
                    "name": cam_name,
                    "img_path": str(img_path),
                    "intrinsic": cam_vals["camera_intrinsic"],
                    "extrinsic": custom_infer.se3_from_quat_tran(
                        cam_vals["rotation"], cam_vals["translation"]
                    ),
                    "ego2global": custom_infer.se3_from_quat_tran(
                        [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
                    ),
                }
            )
        if not camera_list:
            raise ValueError("No cameras loaded from Xtreme1 config")
        return camera_list


def _load_file_bytes(path: Path) -> Tuple[bytes, str]:
    data = path.read_bytes()
    content_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return data, content_type


def _send_request(
    server_url: str,
    metadata: Dict,
    image_paths: Dict[str, Path],
    timeout: float,
) -> Dict:
    files = {}
    for key, path in image_paths.items():
        data, content_type = _load_file_bytes(path)
        files[key] = (path.name, data, content_type)
    response = requests.post(
        server_url,
        files=files,
        data={"json": json.dumps(metadata)},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def _validate_response(result: Dict, expected_cameras: int) -> None:
    status = result.get("status")
    if status != "ok":
        raise RuntimeError(f"Server error: {result}")
    if result.get("num_cameras") != expected_cameras:
        raise RuntimeError(
            f"Unexpected num_cameras={result.get('num_cameras')} (expected {expected_cameras})"
        )
    if not result.get("occ_shape"):
        raise RuntimeError("Missing occ_shape in response")


def run_for_timestamp(
    scene_root: Path,
    timestamp: str,
    img_ext: Optional[str],
    keep_cams: Optional[List[str]],
    camera_left: Optional[str],
    camera_front: Optional[str],
    camera_right: Optional[str],
    server_url: str,
    timeout: float,
) -> None:
    camera_list = _load_xtreme1_compat(
        scene_root, timestamp, img_ext, keep_cams
    )
    left_cam, front_cam, right_cam = _resolve_camera_triplet(
        camera_list, camera_left, camera_front, camera_right
    )
    metadata, image_keys = _build_request_payload(left_cam, front_cam, right_cam)
    image_paths = {
        image_keys[left_cam["name"]]: Path(left_cam["img_path"]),
        image_keys[front_cam["name"]]: Path(front_cam["img_path"]),
        image_keys[right_cam["name"]]: Path(right_cam["img_path"]),
    }
    result = _send_request(server_url, metadata, image_paths, timeout)
    _validate_response(result, expected_cameras=len(metadata["cameras"]))
    print(
        f"[OK] {scene_root.name}/{timestamp} -> "
        f"occ_shape={result.get('occ_shape')} elapsed={result.get('elapsed_sec'):.3f}s"
    )


def main() -> None:
    args = parse_args()
    scene_root = _find_scene_root(Path(args.xtreme1_root))
    if args.run_all:
        timestamps = custom_infer.list_xtreme1_timestamps(scene_root)
    elif args.timestamp:
        timestamps = [args.timestamp]
    else:
        raise ValueError("Provide --timestamp or --run-all")

    if not timestamps:
        raise ValueError("No timestamps found to send")

    for ts in timestamps:
        run_for_timestamp(
            scene_root=scene_root,
            timestamp=ts,
            img_ext=args.img_ext,
            keep_cams=args.cameras,
            camera_left=args.camera_left,
            camera_front=args.camera_front,
            camera_right=args.camera_right,
            server_url=args.server_url,
            timeout=args.timeout,
        )


if __name__ == "__main__":
    main()
