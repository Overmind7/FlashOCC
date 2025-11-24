# Custom FlashOcc Inference (Pure PyTorch)

`tools/custom_infer.py` provides a minimal inference path without relying on the
MMDetection3D runner. It directly rebuilds the FlashOcc backbone/neck/view
transformer/occupancy head with PyTorch modules, loads weights via
`torch.load`, and runs a single forward pass on user-provided camera frames.

## 1. Prepare camera inputs
Create a JSON file that lists every camera you want to use. Each entry should
point to an image and its calibration matrices:

```json
[
  {
    "name": "CAM_FRONT",
    "img_path": "./demo/front.jpg",
    "intrinsic": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    "extrinsic": [[r00, r01, r02, t0], [r10, r11, r12, t1], [r20, r21, r22, t2], [0, 0, 0, 1]],
    "ego2global": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
  }
]
```

- `intrinsic` – camera intrinsics (float 3×3).
- `extrinsic` – camera-to-ego (lidar) transform (float 4×4). The script will
  invert this matrix to compute `lidar2img` and uses it directly as the
  `sensor2ego` pose for the view transformer.
- `ego2global` – optional ego pose; use identity when unavailable.
- `img_path` – RGB image. The script resizes/crops it to the test-time
  `input_size` defined in your config and applies the same normalization as
  training (`mean=[123.675,116.28,103.53]`, `std=[58.395,57.12,57.375]`).

## 2. Run inference
```bash
python tools/custom_infer.py \
    --config projects/configs/flashocc/flashocc-r50.py \
    --checkpoint ckpts/flashocc-r50.pth \
    --camera-json demo/cams.json \
    --output demo/output
```

### Xtreme1-style folders
If your data follow the Xtreme1 export layout
(`sceneXX/{camera_config,camera_image_*}`, filenames are timestamps), you can
skip writing a custom JSON and point the script to the scene root:

```bash
python tools/custom_infer.py \
    --config projects/configs/flashocc/flashocc-r50.py \
    --checkpoint ckpts/flashocc-r50.pth \
    --xtreme1-root /data/0911data/scene16 \
    --timestamp 1757403006752814054 \
    --img-ext .png \
    --output demo/output_xtreme1
```

- `--timestamp` selects the frame (shared name used by `camera_config/` and
  `lidar_config/`).
- The script reads `ego_pose` and `calibrated_sensor` from `lidar_config` to
  fill `ego2global`, assumes camera `translation`/`rotation` describe a
  camera-to-ego transform, and locates images under each `camera_image_*`
  folder.
- Use `--cameras camera_image_front camera_image_back ...` to restrict the
  set of cameras when your config contains more keys than you need.

#### Run every timestamp at once

When your Xtreme1 root contains many frames, add `--run-all` to iterate over
every timestamp JSON found in `camera_config/` (only those that also exist in
`lidar_config/` are used). Results are organized by scene folder and timestamp:

```bash
python tools/custom_infer.py \
    --config projects/configs/flashocc/flashocc-r50.py \
    --checkpoint ckpts/flashocc-r50.pth \
    --xtreme1-root /data/0911data \  # either a single scene or a parent folder
    --run-all \
    --img-ext .png \
    --output outputs/xtreme1_batch
```

- If `--xtreme1-root` itself has `camera_config/` and `lidar_config/`, that
  folder is treated as one scene. Otherwise, the script scans its immediate
  subfolders and processes every child that contains those directories.
- Outputs land in `outputs/xtreme1_batch/<scene>/<timestamp>/occ_pred.npz` and
  `occ_topdown.png`, preserving scene groups and chronological timestamp order.
- You can still pair `--run-all` with `--cameras ...` to drop unwanted views.

Key behaviors:
- Builds the model skeleton from the config (ResNet backbone, CustomFPN, LSS
  view transformer, BEV encoder, occupancy head) and loads weights using
  `torch.load`.
- Computes per-camera `lidar2img` matrices (saved to the npz) for quick visual
  checks or downstream projection utilities.
- Exports `occ_pred.npz` (integer occupancy grid) and a `occ_topdown.png`
  visualization that collapses the height axis.

## 3. Adjusting preprocessing
- The script pulls `input_size` and `resize_test` from the config's
  `data_config` section. Override those fields in the config if your training
  setup used different resizing.
- Use `--score-axis none` if you only need the raw 3D grid without PNG
  rendering.

With these ingredients you can validate FlashOcc on custom camera rigs without
invoking the MMDetection3D dataset/evaluation stack.
