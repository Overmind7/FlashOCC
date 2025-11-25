import argparse
import os
from typing import Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import torch

from tools.analysis_tools.vis_occ import FREE_LABEL, VOXEL_SIZE, show_occ


CAM_LOOK_AT = np.array([0.085, 0.513, 2.485])
CAM_FRONT = np.array([0.1, -0.055, 0.221])
CAM_UP = np.array([0.221, 0.014, 0.975])
CAM_ZOOM = np.array([0.3])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Visualize xtreme1 occupancy predictions')
    parser.add_argument('--root-path', required=True, help='Root directory containing scene folders')
    parser.add_argument('--save-path', required=True, help='Output directory for visualizations')
    parser.add_argument(
        '--format', choices=['image', 'video'], default='image',
        help='Save per-frame images or videos per scene'
    )
    parser.add_argument('--fps', type=int, default=10, help='FPS for output videos')
    parser.add_argument(
        '--topdown-name', default='occ_topdown.png',
        help='Optional topdown image filename inside each frame folder'
    )
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


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


def combine_topdown(
    occ_canvas: np.ndarray,
    topdown: Optional[np.ndarray],
    target_topdown_size: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
    if topdown is None and target_topdown_size is None:
        return occ_canvas, None

    if topdown is not None:
        topdown = cv2.cvtColor(topdown, cv2.COLOR_BGR2RGB)
        topdown = cv2.resize(topdown, (occ_canvas.shape[1], occ_canvas.shape[0]), interpolation=cv2.INTER_LINEAR)
        target_topdown_size = (topdown.shape[1], topdown.shape[0])
    elif target_topdown_size is not None:
        width, height = target_topdown_size
        topdown = np.zeros((height, width, 3), dtype=np.uint8)

    combined = np.concatenate([occ_canvas, topdown], axis=1) if topdown is not None else occ_canvas
    return combined, target_topdown_size


def process_scene(
    scene_path: str,
    output_dir: str,
    fmt: str,
    fps: int,
    topdown_name: str,
) -> None:
    frame_ids = [d for d in os.listdir(scene_path) if os.path.isdir(os.path.join(scene_path, d))]
    frame_ids.sort()
    if not frame_ids:
        return

    ensure_dir(output_dir)
    vis = setup_visualizer()
    video_writer = None
    target_topdown_size: Optional[Tuple[int, int]] = None
    topdown_expected = any(os.path.exists(os.path.join(scene_path, d, topdown_name)) for d in frame_ids)

    for frame_id in frame_ids:
        frame_dir = os.path.join(scene_path, frame_id)
        occ_path = os.path.join(frame_dir, 'occ_pred.npz')
        if not os.path.exists(occ_path):
            continue

        occ_data = np.load(occ_path)['occ']
        occ_canvas = render_occ_frame(vis, occ_data, VOXEL_SIZE)

        if topdown_expected and target_topdown_size is None:
            target_topdown_size = (occ_canvas.shape[1], occ_canvas.shape[0])

        topdown_path = os.path.join(frame_dir, topdown_name)
        topdown_img = cv2.imread(topdown_path) if os.path.exists(topdown_path) else None
        combined, target_topdown_size = combine_topdown(occ_canvas, topdown_img, target_topdown_size)

        if fmt == 'image':
            out_dir = os.path.join(output_dir, frame_id)
            ensure_dir(out_dir)
            cv2.imwrite(os.path.join(out_dir, 'occ.png'), occ_canvas)
            if combined is not occ_canvas:
                cv2.imwrite(os.path.join(out_dir, 'combined.png'), combined)
        else:
            if video_writer is None:
                height, width = combined.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(os.path.join(output_dir, f'{os.path.basename(scene_path)}.mp4'), fourcc, fps, (width, height))
            video_writer.write(combined)

    if video_writer is not None:
        video_writer.release()
    vis.destroy_window()


def main() -> None:
    args = parse_args()
    scenes = [d for d in os.listdir(args.root_path) if os.path.isdir(os.path.join(args.root_path, d))]
    scenes.sort()
    if not scenes:
        return

    for scene in scenes:
        scene_path = os.path.join(args.root_path, scene)
        scene_output = os.path.join(args.save_path, scene)
        process_scene(scene_path, scene_output, args.format, args.fps, args.topdown_name)


if __name__ == '__main__':
    main()
