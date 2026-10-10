# Camera-aware WAM experiment on server02

This branch starts from the existing GigaWorld Policy 0.5 MoT checkpoint and
uses the SONIC G1 pipette episodes. The goal is to test whether the action/video
joint objective tolerates small changes of the head-mounted D435i view.

## Data and geometry

- The source is 58 pipette episodes (81,277 frames) from
  `/home/ykj/project/SONICMJ/GR00T-WholeBodyControl/outputs/onboard/8_18/8_18_use_pipette_1`.
- `scripts/convert_sonic_wrist_delta.py` first makes the existing 30-D
  wrist-local delta action dataset. `scripts/convert_sonic_camera_actions.py`
  then changes the two wrist translation/rotation deltas into camera axes;
  the 12 hand deltas are unchanged. The action remains **relative**, so camera
  translation does not enter this conversion.
- The current `/data/ykj_gigaworld_policy/datasets/pipette_camera_delta_nominal`
  uses an **uncalibrated nominal optical-to-base rotation**. It is a training
  smoke-test asset, not a measured D435i extrinsic. Replace it with a measured
  camera-to-base rotation for any physical evaluation.
- D435i color intrinsics in the augmentation config come from SONIC's camera
  tuner (640x480). Confirm the actual video stream's intrinsics before a full
  experiment.

## Model changes

- `WACamVLATransforms` samples up to 5 degrees of virtual camera rotation,
  applies the same homography to the full 5-frame clip, and conjugates both
  wrist rotation deltas while rotating their translations. It also samples up
  to 8 pixels of image shift to approximate a nearby camera position. The
  shift has no depth-aware parallax and is **not** exact geometric simulation.
- The existing video-flow and action-flow heads remain intact. A 3-DoF
  camera-view auxiliary head sees only the observed reference latent and
  predicts the sampled virtual rotation. It is not a calibrated camera-pose
  estimator and is not yet used for test-time camera correction.
- On server02, the 5B MoT backbone trains with rank-8 attention LoRA while
  the expanded SONIC state/action projectors and camera head stay trainable.
  Gradient checkpointing, bf16 VAE and no EMA keep memory within a 4090.

## server02 layout and smoke run

The Git checkout is
`/home/user/ykj/project/giga-world-policy-camvla`; the Conda environment,
dataset, initial transformer, Wan VAE, caches, and runs are all under
`/data/ykj_gigaworld_policy`. `scripts/compute_camvla_t5_embedding.py` caches
the 29-token, 4096-D Wan UMT5 embedding for the single task prompt.

Run from the checkout:

```bash
bash scripts/run_camvla_wam_server02.sh
```

The script defaults to GPU 0, batch size 1, two optimizer steps and a checkpoint
at step 2. The run directory includes the start date. It accepts
`GWP_GPU_IDS`, `GWP_BATCH_SIZE_PER_GPU`, `GWP_MAX_STEPS`, and
`GWP_CHECKPOINT_INTERVAL` as overrides. Never use the nominal extrinsic for
robot deployment; estimate the actual head-camera rotation and validate the
new action conversion first.
