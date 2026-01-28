#!/usr/bin/env python3
from typing import Dict, List, Tuple

import numpy as np
import rospy
import torch
import torch.nn as nn
from cv_bridge import CvBridge
from mmcv import Config
from mmcv.image.photometric import imnormalize
from PIL import Image
from sensor_msgs.msg import Image as RosImage
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
import message_filters

from mmdet.models.backbones.resnet import ResNet
from projects.mmdet3d_plugin.models.backbones.resnet import CustomResNet
from projects.mmdet3d_plugin.models.dense_heads.bev_occ_head import BEVOCCHead2D
from projects.mmdet3d_plugin.models.necks.fpn import CustomFPN
from projects.mmdet3d_plugin.models.necks.lss_fpn import FPN_LSS
from projects.mmdet3d_plugin.models.necks.view_transformer import LSSViewTransformer


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
    fH, fW = input_size
    W, H = pil_img.size
    resize = float(fW) / float(W) + resize_test
    resize_dims = (int(W * resize), int(H * resize))
    newW, newH = resize_dims
    crop_h = int(newH - fH)
    crop_w = int(max(0, newW - fW) / 2)
    crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)

    post_rot = torch.eye(3, dtype=torch.float32)
    post_tran = torch.zeros(3, dtype=torch.float32)

    img = pil_img.resize(resize_dims)
    img = img.crop(crop)

    post_rot[:2, :2] *= resize
    post_tran[:2] -= torch.tensor(crop[:2], dtype=torch.float32)
    return img, post_rot, post_tran


def prepare_frame(
    images: List[Image.Image],
    camera_configs: List[Dict],
    input_size: Tuple[int, int],
    resize_test: float,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    imgs = []
    sensor2egos = []
    ego2globals = []
    intrins = []
    post_rots = []
    post_trans = []
    lidar2imgs = []

    for pil_img, cam_cfg in zip(images, camera_configs):
        img, post_rot, post_tran = deterministic_resize_crop(pil_img, input_size, resize_test)
        imgs.append(mmlab_normalize(np.array(img)))

        sensor2egos.append(torch.tensor(cam_cfg["extrinsic"], dtype=torch.float32))
        ego2globals.append(torch.tensor(cam_cfg.get("ego2global", np.eye(4)), dtype=torch.float32))
        intrins.append(torch.tensor(cam_cfg["intrinsic"], dtype=torch.float32))
        post_rots.append(post_rot.float())
        post_trans.append(post_tran.float())

        extr = np.asarray(cam_cfg["extrinsic"], dtype=np.float32)
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
    cleaned = cfg.copy()
    cleaned.pop("type", None)
    return cleaned


class FlashOccInfer(nn.Module):
    def __init__(self, model_cfg: Dict):
        super().__init__()
        self.img_backbone = ResNet(**_strip_type(model_cfg["img_backbone"]))
        self.img_neck = CustomFPN(**_strip_type(model_cfg["img_neck"]))
        self.img_view_transformer = LSSViewTransformer(**_strip_type(model_cfg["img_view_transformer"]))
        self.img_bev_encoder_backbone = CustomResNet(
            **_strip_type(model_cfg["img_bev_encoder_backbone"])  # noqa: W503
        )
        self.img_bev_encoder_neck = FPN_LSS(**_strip_type(model_cfg["img_bev_encoder_neck"]))
        self.occ_head = BEVOCCHead2D(**_strip_type(model_cfg["occ_head"]))

    def encode_images(self, imgs: torch.Tensor) -> torch.Tensor:
        b, n, c, h, w = imgs.shape
        imgs = imgs.view(b * n, c, h, w)
        feats = self.img_backbone(imgs)
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


def build_model(cfg_path: str, device: str) -> Tuple[FlashOccInfer, Config]:
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
        rospy.logwarn("[FlashOcc] Missing keys: %s", missing)
    if unexpected:
        rospy.logwarn("[FlashOcc] Unexpected keys: %s", unexpected)


def occ_to_pointcloud(
    occ_map: np.ndarray,
    grid_config: Dict,
    empty_label: int,
) -> Tuple[np.ndarray, np.ndarray]:
    mask = occ_map != empty_label
    indices = np.argwhere(mask)
    if indices.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.uint32)

    dx, dy, dz = occ_map.shape
    x_min, x_max, _ = grid_config["x"]
    y_min, y_max, _ = grid_config["y"]
    z_min, z_max, _ = grid_config["z"]
    x_step = (x_max - x_min) / float(dx)
    y_step = (y_max - y_min) / float(dy)
    z_step = (z_max - z_min) / float(dz)

    coords = np.empty((indices.shape[0], 3), dtype=np.float32)
    coords[:, 0] = x_min + (indices[:, 0] + 0.5) * x_step
    coords[:, 1] = y_min + (indices[:, 1] + 0.5) * y_step
    coords[:, 2] = z_min + (indices[:, 2] + 0.5) * z_step
    labels = occ_map[indices[:, 0], indices[:, 1], indices[:, 2]].astype(np.uint32)
    return coords, labels


class FlashOccRosNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.device = rospy.get_param("~device", "cuda" if torch.cuda.is_available() else "cpu")
        self.config_path = rospy.get_param("~config")
        self.checkpoint_path = rospy.get_param("~checkpoint")
        self.sync_slop = float(rospy.get_param("~sync_slop", 0.05))
        self.queue_size = int(rospy.get_param("~queue_size", 10))
        self.frame_id = rospy.get_param("~frame_id", "base_link")
        self.empty_label = int(rospy.get_param("~empty_label", 17))

        self.topic_left = rospy.get_param("~topic_left", "/camera_image_left")
        self.topic_right = rospy.get_param("~topic_right", "/camera_image_right")
        self.topic_front = rospy.get_param("~topic_front", "/camera_image_front")
        self.occ_topic = rospy.get_param("~occ_topic", "/occ")

        camera_configs_param = rospy.get_param("~camera_configs", [])
        if not camera_configs_param:
            raise ValueError("~camera_configs must be provided with intrinsic/extrinsic matrices")
        self.camera_configs = self._parse_camera_configs(camera_configs_param)

        self.model, self.cfg = build_model(self.config_path, self.device)
        load_checkpoint(self.model, self.checkpoint_path, self.device)

        data_cfg = self.cfg.get("data_config", {})
        self.input_size = tuple(data_cfg.get("input_size", (256, 704)))
        self.resize_test = data_cfg.get("resize_test", 0.0)
        self.grid_config = self.cfg.get("grid_config") or self.cfg.model["img_view_transformer"]["grid_config"]

        self.publisher = rospy.Publisher(self.occ_topic, PointCloud2, queue_size=1)

        left_sub = message_filters.Subscriber(self.topic_left, RosImage)
        right_sub = message_filters.Subscriber(self.topic_right, RosImage)
        front_sub = message_filters.Subscriber(self.topic_front, RosImage)
        sync = message_filters.ApproximateTimeSynchronizer(
            [left_sub, right_sub, front_sub],
            queue_size=self.queue_size,
            slop=self.sync_slop,
        )
        sync.registerCallback(self._on_images)

        rospy.loginfo("[FlashOcc] ROS1 wrapper initialized")

    @staticmethod
    def _parse_camera_configs(camera_configs_param: List[Dict]) -> List[Dict]:
        configs = []
        for entry in camera_configs_param:
            intrinsic = np.array(entry["intrinsic"], dtype=np.float32)
            extrinsic = np.array(entry["extrinsic"], dtype=np.float32)
            ego2global = np.array(entry.get("ego2global", np.eye(4)), dtype=np.float32)
            if intrinsic.shape != (3, 3):
                raise ValueError("Camera intrinsic must be 3x3")
            if extrinsic.shape != (4, 4):
                raise ValueError("Camera extrinsic must be 4x4")
            if ego2global.shape != (4, 4):
                raise ValueError("ego2global must be 4x4")
            configs.append(
                {
                    "intrinsic": intrinsic,
                    "extrinsic": extrinsic,
                    "ego2global": ego2global,
                }
            )
        return configs

    def _convert_ros_image(self, msg: RosImage) -> Image.Image:
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        rgb = cv_img[..., ::-1]
        return Image.fromarray(rgb)

    def _on_images(self, left_msg: RosImage, right_msg: RosImage, front_msg: RosImage):
        images = [
            self._convert_ros_image(left_msg),
            self._convert_ros_image(right_msg),
            self._convert_ros_image(front_msg),
        ]
        (
            imgs,
            sensor2ego,
            ego2global,
            intrins,
            post_rots,
            post_trans,
            _lidar2imgs,
        ) = prepare_frame(images, self.camera_configs, self.input_size, self.resize_test)
        bda = torch.eye(3).unsqueeze(0)

        imgs_b = imgs.unsqueeze(0).to(self.device)
        sensor2ego_b = sensor2ego.unsqueeze(0).to(self.device)
        ego2global_b = ego2global.unsqueeze(0).to(self.device)
        intrins_b = intrins.unsqueeze(0).to(self.device)
        post_rots_b = post_rots.unsqueeze(0).to(self.device)
        post_trans_b = post_trans.unsqueeze(0).to(self.device)
        bda_b = bda.to(self.device)

        with torch.no_grad():
            occ_pred = self.model(
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

        points, labels = occ_to_pointcloud(occ_map, self.grid_config, self.empty_label)
        header = left_msg.header
        header.frame_id = self.frame_id

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="label", offset=12, datatype=PointField.UINT32, count=1),
        ]
        cloud = point_cloud2.create_cloud(
            header,
            fields,
            np.column_stack([points, labels]).tolist(),
        )
        self.publisher.publish(cloud)


def main():
    rospy.init_node("flashocc_ros1_wrapper", anonymous=False)
    FlashOccRosNode()
    rospy.spin()


if __name__ == "__main__":
    main()
