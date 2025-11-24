import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from mmcv import Config
from mmcv.image.photometric import imnormalize

from projects.mmdet3d_plugin.models.backbones.resnet import CustomResNet
from projects.mmdet3d_plugin.models.necks.fpn import CustomFPN
from projects.mmdet3d_plugin.models.necks.lss_fpn import FPN_LSS
from projects.mmdet3d_plugin.models.necks.view_transformer import LSSViewTransformer
from projects.mmdet3d_plugin.models.dense_heads.bev_occ_head import BEVOCCHead2D
from mmdet.models.backbones.resnet import ResNet


def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal standalone FlashOcc inference (pure PyTorch)")
    parser.add_argument(
        "--config",
        required=True,
        help="Config file that defines the model components (e.g. projects/configs/flashocc/flashocc-r50.py)",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the trained checkpoint (loaded with torch.load)",
    )
    parser.add_argument(
        "--camera-json",
        help="JSON file describing camera images and calibration (see docs)",
    )
    parser.add_argument(
        "--xtreme1-root",
        help=(
            "Root of an Xtreme1-style scene folder containing camera_config/ "
            "and per-camera image folders. Overrides --camera-json when set."
        ),
    )
    parser.add_argument(
        "--timestamp",
        help="Timestamp (without extension) to load from the Xtreme1 folders",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run inference on every timestamp discovered under the Xtreme1 root",
    )
    parser.add_argument(
        "--img-ext",
        default=None,
        help="Image extension to use for Xtreme1 samples (e.g., .png or .jpg)",
    )
    parser.add_argument(
        "--cameras",
        nargs="*",
        default=None,
        help="Subset of cameras to load from an Xtreme1 camera_config JSON",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference",
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help="Directory to store npz/png results",
    )
    parser.add_argument(
        "--score-axis",
        default="z",
        choices=["z", "none"],
        help="How to collapse 3D grid for PNG visualization",
    )
    return parser.parse_args()


def load_calib(camera_json: str) -> List[Dict]:
    with open(camera_json, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Camera JSON must be a list of camera dicts")
    required = {"name", "img_path", "intrinsic", "extrinsic"}
    for entry in data:
        missing = required - set(entry.keys())
        if missing:
            raise ValueError(f"Missing keys {missing} in camera entry {entry}")
    return data


def quat_wxyz_to_rot(q: List[float]) -> np.ndarray:
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
    ], dtype=np.float32)
    return R


def se3_from_quat_tran(q: List[float], t: List[float]) -> np.ndarray:
    R = quat_wxyz_to_rot(q)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float32)
    return T


def find_xtreme1_image(img_root: Path, timestamp: str, ext: Optional[str]) -> Path:
    if ext:
        candidate = img_root / f"{timestamp}{ext}"
        if candidate.exists():
            return candidate
    else:
        matches = list(img_root.glob(f"{timestamp}.*"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Cannot find image for {timestamp} under {img_root}")


def load_xtreme1(camera_root: str, timestamp: str, img_ext: Optional[str],
                 keep_cams: Optional[List[str]]) -> List[Dict]:
    scene_root = Path(camera_root)
    lidar_cfg_path = scene_root / "lidar_config" / f"{timestamp}.json"
    cam_cfg_path = scene_root / "camera_config" / f"{timestamp}.json"
    if not lidar_cfg_path.exists():
        raise FileNotFoundError(f"Missing lidar config: {lidar_cfg_path}")
    if not cam_cfg_path.exists():
        raise FileNotFoundError(f"Missing camera config: {cam_cfg_path}")

    lidar_cfg = json.loads(lidar_cfg_path.read_text())
    cam_cfg = json.loads(cam_cfg_path.read_text())

    ego_pose = se3_from_quat_tran(
        lidar_cfg["ego_pose"]["rotation"], lidar_cfg["ego_pose"]["translation"]
    )
    lidar_sensor = se3_from_quat_tran(
        lidar_cfg["calibrated_sensor"]["rotation"],
        lidar_cfg["calibrated_sensor"]["translation"],
    )
    camera_list = []
    for cam_name, cam_vals in cam_cfg.items():
        if keep_cams and cam_name not in keep_cams:
            continue
        cam_sensor = se3_from_quat_tran(cam_vals["rotation"], cam_vals["translation"])
        img_path = find_xtreme1_image(scene_root / cam_name, timestamp, img_ext)
        camera_list.append(
            {
                "name": cam_name,
                "img_path": str(img_path),
                "intrinsic": cam_vals["camera_intrinsic"],
                "extrinsic": cam_sensor,
                "ego2global": ego_pose @ lidar_sensor,
            }
        )
    if not camera_list:
        raise ValueError("No cameras loaded from Xtreme1 config")
    return camera_list


def discover_xtreme1_groups(root: Path) -> List[Tuple[str, Path]]:
    """Return all scene folders that contain camera/lidar configs."""

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


def list_xtreme1_timestamps(scene_root: Path) -> List[str]:
    cam_dir = scene_root / "camera_config"
    lidar_dir = scene_root / "lidar_config"
    timestamps = []
    for cam_json in cam_dir.glob("*.json"):
        ts = cam_json.stem
        if (lidar_dir / f"{ts}.json").exists():
            timestamps.append(ts)
    return sorted(timestamps)


def mmlab_normalize(img: np.ndarray) -> torch.Tensor:
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    img = imnormalize(img.astype(np.float32), mean, std, to_rgb=True)
    return torch.from_numpy(img).permute(2, 0, 1).contiguous()


def deterministic_resize_crop(
    pil_img: Image.Image,
    input_size: Tuple[int, int],
    resize_test: float = 0.0,
) -> Tuple[Image.Image, torch.Tensor, torch.Tensor]:
    """Mirror the test-time branch of PrepareImageInputs.

    Returns the resized/cropped image and its post_rot/post_tran matrices.
    """
    fH, fW = input_size
    W, H = pil_img.size
    resize = float(fW) / float(W) + resize_test
    resize_dims = (int(W * resize), int(H * resize))
    newW, newH = resize_dims
    crop_h = int(newH - fH)
    crop_w = int(max(0, newW - fW) / 2)
    crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)

    post_rot = torch.eye(2)
    post_tran = torch.zeros(2)

    img = pil_img.resize(resize_dims)
    img = img.crop(crop)

    post_rot *= resize
    post_tran -= torch.tensor(crop[:2])
    return img, post_rot, post_tran


def prepare_single_frame(
    camera_list: List[Dict],
    input_size: Tuple[int, int],
    resize_test: float,
) -> Tuple[
    torch.Tensor,
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
]:
    imgs = []
    sensor2egos = []
    ego2globals = []
    intrins = []
    post_rots = []
    post_trans = []
    lidar2imgs = []

    for cam in camera_list:
        img = Image.open(cam["img_path"]).convert("RGB")
        img, post_rot, post_tran = deterministic_resize_crop(img, input_size, resize_test)
        imgs.append(mmlab_normalize(np.array(img)))

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
    return imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, lidar2imgs


def _strip_type(cfg: Dict) -> Dict:
    """Return a shallow copy of a config dict without the MMCV ``type`` key.

    The standalone inference path directly instantiates PyTorch modules instead
    of using MMCV's registry/build mechanism, so passing the ``type`` field
    triggers ``__init__`` argument errors. Removing it keeps the configs
    compatible with the original training files (e.g.,
    ``projects/configs/flashocc/flashocc-r50.py``).
    """

    cleaned = cfg.copy()
    cleaned.pop("type", None)
    return cleaned


class FlashOccInfer(nn.Module):
    def __init__(self, model_cfg: Dict):
        super().__init__()
        img_backbone_cfg = _strip_type(model_cfg["img_backbone"])
        self.img_backbone = ResNet(**img_backbone_cfg)
        self.img_neck = CustomFPN(**_strip_type(model_cfg["img_neck"]))
        self.img_view_transformer = LSSViewTransformer(**_strip_type(model_cfg["img_view_transformer"]))
        self.img_bev_encoder_backbone = CustomResNet(**_strip_type(model_cfg["img_bev_encoder_backbone"]))
        self.img_bev_encoder_neck = FPN_LSS(**_strip_type(model_cfg["img_bev_encoder_neck"]))
        self.occ_head = BEVOCCHead2D(**_strip_type(model_cfg["occ_head"]))

    def encode_images(self, imgs: torch.Tensor) -> torch.Tensor:
        b, n, c, h, w = imgs.shape
        imgs = imgs.view(b * n, c, h, w)
        feats = self.img_backbone(imgs)
        if isinstance(feats, (list, tuple)):
            feats = feats[1:]  # discard C1
        feats = self.img_neck(feats)
        if isinstance(feats, (list, tuple)):
            feats = feats[0]
        _, c_out, h_out, w_out = feats.shape
        feats = feats.view(b, n, c_out, h_out, w_out)
        return feats

    def lift_to_bev(self, x: torch.Tensor, meta_tensors: List[torch.Tensor]) -> torch.Tensor:
        bev, _ = self.img_view_transformer([x] + meta_tensors)
        bev = self.img_bev_encoder_backbone(bev)
        bev = self.img_bev_encoder_neck(bev)
        if isinstance(bev, (list, tuple)):
            bev = bev[0]
        return bev

    def forward(self, img_inputs: List[torch.Tensor]):
        imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = img_inputs
        b, n, _, _, _ = imgs.shape
        keyego2global = ego2globals[:, :1]
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = (global2keyego @ ego2globals.double() @ sensor2egos.double()).float()

        encoded = self.encode_images(imgs)
        bev = self.lift_to_bev(
            encoded,
            [sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda],
        )
        occ = self.occ_head(bev)
        return occ


def build_model(cfg_path: str, device: str) -> Tuple[FlashOccInfer, Dict]:
    cfg = Config.fromfile(cfg_path)
    model_cfg = cfg.model
    net = FlashOccInfer(model_cfg)
    net.to(device)
    net.eval()
    return net, cfg


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("[Warn] Missing keys:", missing)
    if unexpected:
        print("[Warn] Unexpected keys:", unexpected)


def visualize_occ(occ_grid: np.ndarray, out_path: Path, axis: str = "z"):
    if axis == "none":
        return
    import matplotlib.pyplot as plt

    if axis == "z":
        projection = occ_grid.max(axis=2)
    else:
        projection = occ_grid
    plt.figure(figsize=(6, 6))
    plt.imshow(projection.T, origin="lower", cmap="nipy_spectral")
    plt.colorbar(label="occupancy class")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main():
    args = parse_args()
    if args.xtreme1_root:
        if args.timestamp is None and not args.run_all:
            raise ValueError("Provide --timestamp or --run-all when using --xtreme1-root")
        groups = discover_xtreme1_groups(Path(args.xtreme1_root))
    elif args.camera_json:
        groups = []
    else:
        raise ValueError("Provide either --camera-json or --xtreme1-root")

    model, cfg = build_model(args.config, args.device)
    load_checkpoint(model, args.checkpoint, args.device)

    data_cfg = cfg.get("data_config", {})
    input_size = tuple(data_cfg.get("input_size", (256, 704)))
    resize_test = data_cfg.get("resize_test", 0.0)

    def run_once(camera_list: List[Dict], out_dir: Path):
        (
            imgs,
            sensor2ego,
            ego2global,
            intrins,
            post_rots,
            post_trans,
            lidar2imgs,
        ) = prepare_single_frame(camera_list, input_size, resize_test)
        bda = torch.eye(3).unsqueeze(0)

        imgs_b = imgs.unsqueeze(0).to(args.device)
        sensor2ego_b = sensor2ego.unsqueeze(0).to(args.device)
        ego2global_b = ego2global.unsqueeze(0).to(args.device)
        intrins_b = intrins.unsqueeze(0).to(args.device)
        post_rots_b = post_rots.unsqueeze(0).to(args.device)
        post_trans_b = post_trans.unsqueeze(0).to(args.device)
        lidar2imgs_b = lidar2imgs.unsqueeze(0).to(args.device)
        bda_b = bda.to(args.device)

        with torch.no_grad():
            occ_pred = model(
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

        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_dir / "occ_pred.npz",
            occ=occ_map,
            lidar2img=lidar2imgs.numpy(),
        )
        visualize_occ(occ_map, out_dir / "occ_topdown.png", axis=args.score_axis)
        print(f"Saved occupancy to {out_dir}")

    if args.xtreme1_root:
        for group_name, scene_root in groups:
            ts_list = [args.timestamp] if args.timestamp else list_xtreme1_timestamps(scene_root)
            if not ts_list:
                raise ValueError(f"No timestamps found in {scene_root}")
            for ts in ts_list:
                print(f"[Xtreme1] Running {group_name}/{ts}")
                camera_list = load_xtreme1(scene_root, ts, args.img_ext, args.cameras)
                run_once(camera_list, Path(args.output) / group_name / ts)
    else:
        camera_list = load_calib(args.camera_json)
        run_once(camera_list, Path(args.output))


if __name__ == "__main__":
    main()
