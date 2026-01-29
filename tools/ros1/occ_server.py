#!/usr/bin/env python
import argparse
import io
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from flask import Flask, jsonify, request
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))
import custom_infer  # noqa: E402

app = Flask(__name__)

MODEL = None
MODEL_DEVICE = "cpu"
MODEL_INPUT = (256, 704)
MODEL_RESIZE_TEST = 0.0
DEFAULT_CAMERAS: Optional[List[Dict]] = None


def load_image(file_storage):
    if file_storage is None:
        return None
    image_bytes = io.BytesIO(file_storage.read())
    return Image.open(image_bytes).convert('RGB')


def _default_image_map() -> Dict[str, str]:
    return {
        "left": "image_left",
        "right": "image_right",
        "front": "image_front",
    }


def _load_camera_list(metadata: Dict) -> List[Dict]:
    if "camera_list" in metadata:
        return metadata["camera_list"]
    if "cameras" in metadata:
        return metadata["cameras"]
    if DEFAULT_CAMERAS is not None:
        return DEFAULT_CAMERAS
    raise ValueError("Missing camera calibration (metadata.cameras or --camera-json)")


def _prepare_frame(
    camera_list: List[Dict],
    images_by_key: Dict[str, Image.Image],
    input_size: Tuple[int, int],
    resize_test: float,
    image_keys: Dict[str, str],
) -> Tuple[
    torch.Tensor,
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
    torch.Tensor,
]:
    imgs = []
    sensor2egos = []
    ego2globals = []
    intrins = []
    post_rots = []
    post_trans = []
    lidar2imgs = []

    for cam in camera_list:
        cam_name = cam.get("name")
        req_key = cam.get("image_key") or image_keys.get(cam_name, cam_name)
        img = images_by_key.get(req_key)
        if img is None:
            raise ValueError(f"Missing image for camera '{cam_name}' (key '{req_key}')")
        img, post_rot, post_tran = custom_infer.deterministic_resize_crop(
            img, input_size, resize_test
        )
        imgs.append(custom_infer.mmlab_normalize(np.array(img)))

        sensor2egos.append(torch.tensor(cam["extrinsic"], dtype=torch.float32))
        ego2globals.append(torch.tensor(cam.get("ego2global", np.eye(4)), dtype=torch.float32))
        intrins.append(torch.tensor(cam["intrinsic"], dtype=torch.float32))
        post_rots.append(post_rot.float())
        post_trans.append(post_tran.float())

        extr = np.asarray(cam["extrinsic"], dtype=np.float32)
        rot = extr[:3, :3]
        tran = extr[:3, 3]
        cam2lidar = np.eye(4, dtype=np.float32)
        cam2lidar[:3, :3] = rot.T
        cam2lidar[:3, 3] = -rot.T @ tran
        lidar2img = intrins[-1].numpy() @ cam2lidar[:3]
        lidar2imgs.append(torch.tensor(lidar2img, dtype=torch.float32))

    imgs = torch.stack(imgs, dim=0)
    sensor2egos = torch.stack(sensor2egos, dim=0)
    ego2globals = torch.stack(ego2globals, dim=0)
    intrins = torch.stack(intrins, dim=0)
    post_rots = torch.stack(post_rots, dim=0)
    post_trans = torch.stack(post_trans, dim=0)
    lidar2imgs = torch.stack(lidar2imgs, dim=0)
    return (
        imgs,
        sensor2egos,
        ego2globals,
        intrins,
        post_rots,
        post_trans,
        lidar2imgs,
    )


def run_inference(images_by_key, metadata):
    if MODEL is None:
        return {"status": "error", "message": "Model not initialized"}
    camera_list = _load_camera_list(metadata)
    image_keys = _default_image_map()
    image_keys.update(metadata.get("image_keys", {}))
    (
        imgs,
        sensor2ego,
        ego2global,
        intrins,
        post_rots,
        post_trans,
        lidar2imgs,
    ) = _prepare_frame(camera_list, images_by_key, MODEL_INPUT, MODEL_RESIZE_TEST, image_keys)
    bda = torch.eye(3).unsqueeze(0)

    imgs_b = imgs.unsqueeze(0).to(MODEL_DEVICE)
    sensor2ego_b = sensor2ego.unsqueeze(0).to(MODEL_DEVICE)
    ego2global_b = ego2global.unsqueeze(0).to(MODEL_DEVICE)
    intrins_b = intrins.unsqueeze(0).to(MODEL_DEVICE)
    post_rots_b = post_rots.unsqueeze(0).to(MODEL_DEVICE)
    post_trans_b = post_trans.unsqueeze(0).to(MODEL_DEVICE)
    bda_b = bda.to(MODEL_DEVICE)

    with torch.no_grad():
        occ_pred = MODEL(
            [
                imgs_b,
                sensor2ego_b,
                ego2global_b,
                intrins_b,
                post_rots_b,
                post_trans_b,
                bda_b,
            ]
        )
        probs = torch.softmax(occ_pred, dim=-1)
        occ_map = probs.argmax(dim=-1).squeeze(0).cpu().numpy().astype(np.uint8)

    class_hist = np.bincount(occ_map.reshape(-1), minlength=occ_pred.shape[-1]).tolist()
    return {
        "status": "ok",
        "num_cameras": len(camera_list),
        "occ_shape": list(occ_map.shape),
        "class_hist": class_hist,
        "metadata": metadata,
        "lidar2img": lidar2imgs.cpu().numpy().tolist(),
        "occ": occ_map.tolist() if metadata.get("return_occ") else None,
    }


@app.route('/infer', methods=['POST'])
def infer():
    start = time.time()
    metadata = {}
    if 'json' in request.form:
        try:
            metadata = json.loads(request.form['json'])
        except json.JSONDecodeError:
            metadata = {'raw': request.form['json']}

    images_by_key = {
        "image_left": load_image(request.files.get("image_left")),
        "image_right": load_image(request.files.get("image_right")),
        "image_front": load_image(request.files.get("image_front")),
    }

    try:
        result = run_inference(images_by_key, metadata)
    except Exception as exc:
        result = {"status": "error", "message": str(exc)}
    result['elapsed_sec'] = time.time() - start
    return jsonify(result)


def _init_model(args: argparse.Namespace):
    global MODEL, MODEL_DEVICE, MODEL_INPUT, MODEL_RESIZE_TEST, DEFAULT_CAMERAS
    MODEL_DEVICE = args.device
    if args.camera_json:
        DEFAULT_CAMERAS = custom_infer.load_calib(args.camera_json)
        for cam in DEFAULT_CAMERAS:
            cam.pop("img_path", None)
    MODEL, cfg = custom_infer.build_model(args.config, args.device)
    custom_infer.load_checkpoint(MODEL, args.checkpoint, args.device)
    data_cfg = cfg.get("data_config", {})
    MODEL_INPUT = tuple(data_cfg.get("input_size", (256, 704)))
    MODEL_RESIZE_TEST = data_cfg.get("resize_test", 0.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FlashOcc ROS1 inference server")
    parser.add_argument("--config", required=True, help="Model config file")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--camera-json", help="Camera calibration JSON")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--port", type=int, default=5801)
    args = parser.parse_args()
    _init_model(args)
    app.run(host='0.0.0.0', port=args.port)
