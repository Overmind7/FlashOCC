# FlashOcc ROS1 Wrapper

This node subscribes to three camera image topics, performs approximate time
synchronization, runs FlashOcc inference, and publishes a `sensor_msgs/PointCloud2`
message on `/occ`.

## Parameters

- `~config` (string, required): Path to the FlashOcc config (e.g. `projects/configs/flashocc/flashocc-r50.py`).
- `~checkpoint` (string, required): Path to the checkpoint `.pth`.
- `~device` (string, optional): Torch device, default auto-detect (`cuda` if available).
- `~sync_slop` (float, optional): Approximate time sync threshold in seconds (default `0.05`).
- `~queue_size` (int, optional): Sync queue size (default `10`).
- `~frame_id` (string, optional): Frame ID for the output pointcloud (default `base_link`).
- `~empty_label` (int, optional): Occupancy class index to ignore (default `17`, which maps to `free`).
- `~topic_left` (string, optional): Left image topic (default `/camera_image_left`).
- `~topic_right` (string, optional): Right image topic (default `/camera_image_right`).
- `~topic_front` (string, optional): Front image topic (default `/camera_image_front`).
- `~occ_topic` (string, optional): Output pointcloud topic (default `/occ`).
- `~camera_configs` (list, required): Camera intrinsics/extrinsics in the same order as the topics.

Example `camera_configs` YAML (three cameras):

```yaml
camera_configs:
  - intrinsic: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    extrinsic: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
  - intrinsic: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    extrinsic: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
  - intrinsic: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    extrinsic: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
```

## Run

```bash
rosrun flashocc flashocc_ros1_wrapper.py \
  _config:=projects/configs/flashocc/flashocc-r50.py \
  _checkpoint:=/path/to/flashocc-r50-256x704.pth \
  _camera_configs:="$(cat /path/to/camera_configs.yaml)"
```

Alternatively, load parameters from a roslaunch file or `rosparam`.
```
rosparam load /path/to/flashocc_params.yaml
rosrun flashocc flashocc_ros1_wrapper.py
```
