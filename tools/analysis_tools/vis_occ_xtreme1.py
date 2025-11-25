import argparse
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import torch

from tools.analysis_tools.vis_occ import FREE_LABEL, VOXEL_SIZE, show_occ


CAM_LOOK_AT = np.array([0.085, 0.513, 2.485])
CAM_FRONT = np.array([0.1, -0.055, 0.221])
CAM_UP = np.array([0.221, 0.014, 0.975])
CAM_ZOOM = np.array([0.3])
CAMERA_FILENAMES = [
    'camera_image_left',
    'camera_image_front',
    'camera_image_right',
]
CAMERA_TARGET_SIZE = (640, 480)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Visualize xtreme1 occupancy predictions')
    parser.add_argument(
        '--pred-root',
        required=True,
        help='Root directory containing scene folders with occupancy predictions (occ_pred.npz)',
    )
    parser.add_argument(
        '--data-root',
        required=True,
        help='Root directory containing scene folders with original camera images',
    )
    parser.add_argument('--save-path', required=True, help='Output directory for visualizations')
    parser.add_argument(
        '--format', choices=['image', 'video'], default='image',
        help='Save per-frame images or videos per scene'
    )
    parser.add_argument('--fps', type=int, default=10, help='FPS for output videos')
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def read_image_if_exists(base_path: str) -> Optional[np.ndarray]:
    if os.path.exists(base_path):
        return cv2.imread(base_path)

    for ext in ('.png', '.jpg', '.jpeg'):
        candidate = f'{base_path}{ext}'
        if os.path.exists(candidate):
            return cv2.imread(candidate)

    return None


def letterbox_image(img: Optional[np.ndarray], target_size: Tuple[int, int]) -> np.ndarray:
    target_w, target_h = target_size
    if img is None:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


def build_combined_frame(
    camera_imgs: List[Optional[np.ndarray]],
    occ_canvas: np.ndarray,
) -> np.ndarray:
    processed_cams = [letterbox_image(img, CAMERA_TARGET_SIZE) for img in camera_imgs]
    camera_strip = np.concatenate(processed_cams, axis=1)

    gap_y = 10
    target_width = max(camera_strip.shape[1], occ_canvas.shape[1])
    combined = np.zeros(
        (camera_strip.shape[0] + gap_y + occ_canvas.shape[0], target_width, 3), dtype=np.uint8
    )

    cam_x = (target_width - camera_strip.shape[1]) // 2
    combined[:camera_strip.shape[0], cam_x:cam_x + camera_strip.shape[1]] = camera_strip

    occ_y = camera_strip.shape[0] + gap_y
    occ_x = (target_width - occ_canvas.shape[1]) // 2
    combined[occ_y:occ_y + occ_canvas.shape[0], occ_x:occ_x + occ_canvas.shape[1]] = occ_canvas

    return combined


def setup_visualizer() -> o3d.visualization.VisualizerWithKeyCallback:
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(visible=False)
    return vis


def render_occ_frame(
    vis: o3d.visualization.VisualizerWithKeyCallback,
    occ_data: np.ndarray,
    voxel_size: Tuple[float, float, float],
) -> np.ndarray:
    voxel_show = occ_data != FREE_LABEL
    vis = show_occ(
        torch.from_numpy(occ_data),
        torch.from_numpy(voxel_show),
        voxel_size=voxel_size,
        vis=vis,
        offset=[0, 0, 0],
    )

    view_control = vis.get_view_control()
    view_control.set_lookat(CAM_LOOK_AT)
    view_control.set_front(CAM_FRONT)
    view_control.set_up(CAM_UP)
    view_control.set_zoom(CAM_ZOOM)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])
    opt.line_width = 5

    vis.poll_events()
    vis.update_renderer()

    occ_canvas = vis.capture_screen_float_buffer(do_render=True)
    occ_canvas = np.asarray(occ_canvas)
    occ_canvas = (occ_canvas * 255).astype(np.uint8)[..., [2, 1, 0]]
    vis.clear_geometries()
    return occ_canvas


def process_scene(
    pred_scene_path: str,
    data_scene_path: str,
    output_dir: str,
    fmt: str,
    fps: int,
) -> None:
    frame_ids = [d for d in os.listdir(pred_scene_path) if os.path.isdir(os.path.join(pred_scene_path, d))]
    frame_ids.sort()
    if not frame_ids:
        return

    ensure_dir(output_dir)
    vis = setup_visualizer()
    video_writer = None

    for frame_id in frame_ids:
        pred_frame_dir = os.path.join(pred_scene_path, frame_id)
        data_frame_dir = os.path.join(data_scene_path, frame_id)
        occ_path = os.path.join(pred_frame_dir, 'occ_pred.npz')
        if not os.path.exists(occ_path):
            continue

        occ_data = np.load(occ_path)['occ']
        occ_canvas = render_occ_frame(vis, occ_data, VOXEL_SIZE)

        camera_imgs = [read_image_if_exists(os.path.join(data_frame_dir, name)) for name in CAMERA_FILENAMES]
        combined = build_combined_frame(
            camera_imgs=camera_imgs,
            occ_canvas=occ_canvas,
        )

        if fmt == 'image':
            out_dir = os.path.join(output_dir, frame_id)
            ensure_dir(out_dir)
            cv2.imwrite(os.path.join(out_dir, 'occ.png'), occ_canvas)
            cv2.imwrite(os.path.join(out_dir, 'combined.png'), combined)
        else:
            if video_writer is None:
                height, width = combined.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(
                    os.path.join(output_dir, f'{os.path.basename(pred_scene_path)}.mp4'), fourcc, fps, (width, height)
                )
            video_writer.write(combined)

    if video_writer is not None:
        video_writer.release()
    vis.destroy_window()


def main() -> None:
    args = parse_args()
    scenes = [d for d in os.listdir(args.pred_root) if os.path.isdir(os.path.join(args.pred_root, d))]
    scenes.sort()
    if not scenes:
        return

    for scene in scenes:
        pred_scene_path = os.path.join(args.pred_root, scene)
        data_scene_path = os.path.join(args.data_root, scene)
        if not os.path.isdir(data_scene_path):
            continue
        scene_output = os.path.join(args.save_path, scene)
        process_scene(pred_scene_path, data_scene_path, scene_output, args.format, args.fps)


if __name__ == '__main__':
    main()
