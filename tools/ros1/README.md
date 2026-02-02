# ROS1 FlashOcc Inference Server

This folder provides a simple ROS1 client/server bridge that runs FlashOcc
occupancy inference behind a Flask endpoint.

## Server

Start the server by pointing it at a FlashOcc config and checkpoint. Optionally
provide a camera calibration JSON (see `tools/custom_infer.py` for the expected
schema).

```bash
python tools/ros1/occ_server.py \
  --config projects/configs/flashocc/flashocc-r50.py \
  --checkpoint /path/to/checkpoint.pth \
  --camera-json /path/to/camera.json \
  --device cuda \
  --port 5801
```

### Request format

The `/infer` endpoint expects multipart form data with:

- `image_left`, `image_right`, `image_front`: JPEG/PNG images.
- `json`: Optional JSON string. Useful fields:
  - `cameras`: List of camera dicts (same schema as `camera-json`).
  - `image_keys`: Mapping from camera name to uploaded file key.
  - `return_occ`: Boolean flag to include the full occupancy grid in the response.

If `--camera-json` is provided, its calibration is used when `cameras` is
omitted in the request.

## Client (ROS1)

Run the ROS client node to collect synchronized camera frames and send them to
the server:

```bash
rosrun <your_package> occ_client.py \
  _server_url:=http://localhost:5801/infer \
  _camera_left:=/camera_image_left \
  _camera_right:=/camera_image_right \
  _camera_front:=/camera_image_front \
  _occ_topic:=/occ \
  _max_rate_hz:=5.0
```

## Xtreme1 ROS publisher

To simulate on-robot publishing, the helper script can publish Xtreme1 frames as
ROS `sensor_msgs/Image` topics. This lets you feed `occ_client.py` with dataset
frames and test the end-to-end flow with `occ_server.py`.

```bash
python tools/ros1/occ_xtreme1_test.py \
  --xtreme1-root /data/xtreme1/scene01 \
  --run-all \
  --topic-left /camera_image_left \
  --topic-front /camera_image_front \
  --topic-right /camera_image_right \
  --ros-rate 5.0
```

Notes:
- `--xtreme1-root` can point to a parent folder containing multiple scenes, as
  long as it only contains one scene.
- The script supports camera folder names such as `camera_image_left` and
  `camera_image_right` that match the Xtreme1 `camera_config` keys.
