# Codebase Guide: Semantic-Aware ROI Video Encoding

This repository implements an offline reinforcement-learning pipeline that
chooses a bitrate ratio, output resolution, and ROI (region-of-interest)
encoding strength for every video frame. YOLO detections describe which image
regions are semantically important, a bandwidth trace represents the network,
an offline FFmpeg/VMAF grid approximates encoder behavior, and a Double DQN
learns the encoding policy.

This document describes the system as the code currently works on branch
`fix/add-roi-encoding`. It complements `rl_env_offline/README.md`, which focuses
more on the RL formulation and command examples.

## 1. System map

```text
source video
   |
   +-- extract_yolo_metadata.py ----------------> yolo_metadata.jsonl
   |                                                   |
   |                                                   +-- yolo_metadata_to_rl.py
   |                                                   |          |
   |                                                   |          v
   |                                                   |   semantic_trace.jsonl
   |                                                   |          |
   |                                                   |          +-- merge_traces()
   |                                                   |                    ^
   |                                                   |                    |
   |                                                   |   bandwidth_trace.jsonl
   |                                                   |                    ^
   |                                                   |                    |
   |                                                   |   gen_bandwidth_trace.py
   |                                                   |
   |                                                   +-- encode_grid.py
   |                                                              |
   |                                                              v
   |                                                     vcu_encode_grid.json
   |
   +------------------------------ rl_states.jsonl ----------------+
                                                                  |
                                                                  v
                                                         VCUSimEnv (env.py)
                                                                  |
                                           vcu_encode_grid.json ---+
                                                                  |
                                                                  v
                                                       Double DQN (train.py)
                                                                  |
                                                                  v
                                                           dqn_policy.pt
                                                           /             \
                                                          v               v
                                                 evaluate.py   export_encoded_video.py
                                                                      |
                                                                      v
                                                              encoded MP4 +
                                                              segment report
```

There are two separate video-related entry points at the repository root:

- `infer_yolov5s.py` is a general inference/visualization utility. It can run
  YOLO on images or videos, save annotated results, and combine frames into a
  video. Its output is not consumed by the RL pipeline.
- `extract_yolo_metadata.py` is the pipeline entry point. It writes the raw
  per-frame detections that semantic scoring, ROI grid generation, and final
  export consume.

## 2. Source files and ownership

| File | Code-level responsibility |
|---|---|
| `extract_yolo_metadata.py` | Loads a YOLOv5 model and writes one detection record per video frame. |
| `infer_yolov5s.py` | Standalone image/video inference and annotated-output helper. |
| `rl_env_offline/yolo_metadata_to_rl.py` | Converts detections to content features and `semantic_score`. |
| `rl_env_offline/gen_bandwidth_trace.py` | Generates or resamples bandwidth and merges it with semantic data. |
| `rl_env_offline/encode_grid.py` | Encodes the source at all resolution/bitrate/ROI combinations and measures per-frame rate-distortion data. |
| `rl_env_offline/env.py` | Defines the Gymnasium environment, action sanitation, encoder surrogate, and reward. |
| `rl_env_offline/train.py` | Defines the Q-network, replay buffer, Double-DQN loop, and checkpoint format. |
| `rl_env_offline/evaluate.py` | Compares the greedy learned policy with random and fixed-CBR baselines. |
| `rl_env_offline/export_encoded_video.py` | Applies a trained policy using real FFmpeg encodes and concatenates the resulting segments. |

Generated videos, traces, metadata, and checkpoints are intentionally ignored
by `.gitignore`; they are runtime artifacts, not source files.

## 3. Data contracts

### 3.1 Raw YOLO metadata

`extract_yolo_metadata.run()` reads the source with OpenCV and calls
`extract_yolo_metadata()` once per frame. The output is JSON Lines:

```json
{
  "frame_idx": 0,
  "width": 1280,
  "height": 720,
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.91,
      "bbox": [100.0, 80.0, 260.0, 620.0],
      "centroid": [180.0, 350.0],
      "area_ratio": 0.09375
    }
  ]
}
```

`bbox` uses `[x1, y1, x2, y2]` pixel coordinates in the original frame.
`area_ratio` is bounding-box area divided by full-frame area. Overlapping
boxes are not de-duplicated later, so summed ROI area can exceed the geometric
union area.

### 3.2 Semantic trace

`yolo_metadata_to_rl.compute_semantic_scores()` derives:

```text
semantic_score = alpha * num_objects
               + beta  * roi_area
               + gamma * motion
               + delta * priority
```

The default weights are `alpha=1`, `beta=10`, `gamma=5`, and `delta=2`.
`priority` is the sum of class weights in `PRIORITY_WEIGHTS`; unknown classes
use `DEFAULT_PRIORITY=0.3`.

Motion is an approximation, not tracking. `match_and_compute_motion()` greedily
matches current detections to the nearest unused previous detection of the same
class, normalizes displacement by the frame diagonal, and rejects implausibly
distant matches. The saved `motion` is the sum of accepted normalized motions
divided by the current number of objects.

Each JSONL row contains at least:

```json
{
  "frame_idx": 0,
  "semantic_score": 4.1375,
  "num_objects": 1,
  "roi_area": 0.0938,
  "motion": 0.0,
  "motion_sum": 0.0,
  "priority": 1.0
}
```

### 3.3 Bandwidth and RL trace

`gen_bandwidth_trace.py` supports two bandwidth sources:

- `gen_synthetic_bandwidth()` creates sticky bandwidth levels with optional
  congestion periods and per-frame noise.
- `load_real_trace_csv()` plus `resample_real_trace()` linearly resamples a
  `time_s,bandwidth_kbps` CSV at video-frame timestamps.

`merge_traces()` joins semantic and bandwidth JSONL records by `frame_idx` and
writes `rl_states.jsonl`. Semantic rows with no matching bandwidth row are
silently skipped. A normal environment input row is therefore:

```json
{
  "frame_idx": 0,
  "semantic_score": 4.1375,
  "num_objects": 1,
  "roi_area": 0.0938,
  "motion": 0.0,
  "priority": 1.0,
  "bandwidth": 2500.0
}
```

### 3.4 Encode grid

`encode_grid.build_grid()` probes the source video, divides it into fixed
`segment_frames` ranges, and constructs the Cartesian product of:

- `RESOLUTIONS`: `640x480`, `1280x720`, `1920x1080`;
- configurable bitrate levels, defaulting to 600, 900, 1500, 2500, 4000, and
  6000 kbps;
- `ROI_QOFFSET_LEVELS`: `0.0`, `-0.3`, and `-0.6`.

For each segment, `union_roi_for_segment()` takes all YOLO boxes in that
segment, computes one enclosing union rectangle, scales it to the selected
resolution, clamps it to the frame, and aligns it for YUV 4:2:0. This static
rectangle is necessary because the FFmpeg `addroi` invocation used here does
not change coordinates frame by frame within one encode.

`_grid_segment_worker()` invokes `encode_segment()` and collects actual packet
bitrate, whole-frame VMAF, ROI-crop VMAF, and optional PSNR. When a segment has
no detections, ROI levels 1 and 2 reuse ROI level 0 rather than running
equivalent encodes.

The output is one JSON object. Grid arrays are addressed by key and frame
position:

```json
{
  "source_video": "...",
  "fps": 25.0,
  "num_frames": 1500,
  "resolutions": [[640, 480], [1280, 720], [1920, 1080]],
  "bitrate_levels_kbps": [600.0, 900.0, 1500.0, 2500.0, 4000.0, 6000.0],
  "roi_qoffset_levels": [0.0, -0.3, -0.6],
  "grid": {
    "res1_br3_roi2": [
      {
        "bitrate_kbps": 2412.31,
        "target_bitrate_kbps": 2500.0,
        "vmaf": 87.42,
        "roi_vmaf": 91.18
      }
    ]
  }
}
```

The array index is the frame index; cells do not repeat `frame_idx`.

## 4. Environment behavior (`env.py`)

### 4.1 Trace selection

`VCUSimEnv.__init__()` chooses its state trace in this order:

1. non-empty `trace_path`;
2. `outputs/metadata/yolo_metadata.jsonl`, converted in memory and paired with
   synthetic bandwidth by `load_yolo_metadata_as_trace()`;
3. the fully synthetic trace from `build_synthetic_trace()`.

The default paths are relative to the current working directory, not to
`env.py`. Commands in `rl_env_offline/README.md` therefore use explicit `../`
paths when run from `rl_env_offline/`.

### 4.2 Observation

`STATE_DIM` is 3. `_get_obs()` returns a normalized `float32` vector:

```text
[
  clip(bandwidth / 8000, 0, 1),
  clip(previous_actual_bitrate / 8000, 0, 1),
  clip(semantic_score / 30, 0, 1)
]
```

`8000` is the default `max_bitrate_kbps`; callers can replace it. At reset,
the previous bitrate is initialized to `min(0.7 * first_bandwidth,
max_bitrate)`.

### 4.3 Action space and index ordering

`EncoderAction` has three fields:

```python
EncoderAction(bitrate_ratio, resolution_idx, roi_idx)
```

The dimensions are:

- bitrate ratio: `[0.50, 0.75, 0.95]`;
- resolution index: `[0, 1, 2]`;
- ROI index: `[0, 1, 2]`, mapping to qoffset `[0.0, -0.3, -0.6]`.

`ACTIONS` is built with `itertools.product()` in that order, so ROI changes
fastest, then resolution, then bitrate. The exact mapping is:

```text
action_idx = bitrate_ratio_idx * 9 + resolution_idx * 3 + roi_idx
```

For example, index 0 is `(0.50, 480p, no ROI)`, index 14 is
`(0.75, 720p, qoffset=-0.6)`, and index 26 is
`(0.95, 1080p, qoffset=-0.6)`. The DQN output has 27 Q-values.

### 4.4 Action sanitation

`step()` decodes the raw action, then `_sanitize_action()` applies hard
constraints before encoding:

- below 300 kbps, resolution is capped at 720p and ratio at 0.75;
- from 300 through 899.999 kbps, resolution is capped at 720p;
- above semantic score 25.5 with bandwidth above 2500 kbps, resolution is at
  least 720p;
- except when `t % segment_len == 0`, resolution and ROI remain at their
  current values; bitrate ratio may still change every frame.

Both the requested `raw_action` and applied `safe_action` are returned in the
`info` dictionary.

### 4.5 Encoder paths

If an encode grid is supplied, `_encode_from_grid()` uses the selected
resolution and ROI curve. `_interpolate_grid_vmaf()` collects the measured
points across all bitrate levels for the current frame, sorts and de-duplicates
them by measured bitrate, and linearly interpolates both full-frame and ROI
VMAF. Values outside the measured range are clamped to the nearest endpoint.

The combined quality is:

```text
semantic_norm = clip(semantic_score / 30, 0, 1)
roi_weight    = semantic_norm * clip(3 * roi_area, 0, 0.75)
quality_vmaf  = (1 - roi_weight) * global_vmaf + roi_weight * roi_vmaf
```

Grid lookup uses `frame_idx % len(curve)`. This prevents an index error but
also means a longer or mismatched trace silently wraps around the grid. The
source video, frame numbering, FPS, and frame count should be validated before
training.

If no grid is supplied, `_simulate_encode()` estimates bitrate, VMAF, latency,
and power from resolution, target bitrate, motion, ROI area, semantic score,
and ROI strength. This path is useful for smoke tests, but it is not a
replacement for measured rate-distortion data.

### 4.6 Reward and transition

The target bitrate is `safe_action.bitrate_ratio * bandwidth`. With a modern
grid, `actual_bitrate` is this target clipped to `[50, max_bitrate]`; measured
grid bitrates define the interpolation axis rather than the value returned as
actual bitrate. The simulator instead caps actual bitrate by its estimated
content demand.

`_compute_reward()` implements:

```text
reward = 0.60 * clip(vmaf / 100, 0, 1)
       - 0.40 * (actual_bitrate / max_bitrate)
```

Latency and power are diagnostic values in `info`; neither affects reward.
After each step, the environment updates previous bitrate/VMAF/latency and the
current resolution/ROI, advances to the next trace row, and terminates at the
end unless `loop=True`.

## 5. Training (`train.py`)

`QNetwork` is a multilayer perceptron:

```text
3 inputs -> Linear(64) -> ReLU -> Linear(64) -> ReLU -> Linear(27)
```

`train()` seeds Python, NumPy, and PyTorch, then runs one full trace per
episode. Action selection is epsilon-greedy. Each transition is stored in a
20,000-entry `ReplayBuffer`; optimization begins once at least `batch_size`
transitions exist.

The update is Double DQN:

```text
best_next_action = argmax online_net(next_state)
target           = reward + gamma * target_net(next_state, best_next_action)
```

The loss is Smooth L1, gradients are clipped to norm 5, epsilon decays once
per episode, and the target network is copied from the online network every
`target_update_every` episodes.

`save_training_checkpoint()` writes a dictionary containing both network
states, optimizer state, `state_dim`, `num_actions`, the decoded action table,
episode and epsilon values, reward history, and the serialized replay buffer.
It also writes `<checkpoint-stem>.rewards.json`. Because the ROI branch changed
the output from 9 to 27 actions, older 9-action checkpoints fail the explicit
shape check and must be retrained.

## 6. Evaluation (`evaluate.py`)

`evaluate.py` loads the checkpoint and checks `state_dim` and `num_actions`
against the current environment. `run_episode()` gathers reward, VMAF,
bitrate, latency, power, bandwidth-overflow rate, action counts, and applied
ROI counts.

Three policies are compared:

- learned: greedy `argmax` over the DQN output;
- random: uniform random action index;
- fixed CBR baseline: the nearest action to `(0.75, resolution_idx=1,
  roi_idx=0)`, which is 720p without ROI.

`check_convergence()` reads the reward JSON and can save a moving-average plot
when Matplotlib is installed. Evaluation must use the same trace/grid semantics
and environment normalization as training for meaningful results.

## 7. Real export (`export_encoded_video.py`)

The exporter loads the same `QNetwork`, `ACTIONS`, sanitation logic, and trace
normalization used during training. `build_decision_stream()` asks the greedy
policy for one decision per trace row and stores the sanitized action.

`build_and_encode_segments()` groups consecutive identical sanitized
`(bitrate_ratio, resolution_idx, roi_idx)` tuples. For each group it:

1. averages bandwidth over the group and multiplies it by the bitrate ratio;
2. builds a static union ROI box from YOLO detections when `roi_idx > 0`;
3. invokes `encode_segment_abr()` with libx264 ABR, `maxrate=1.2 * target`,
   `bufsize=2 * target`, no B-frames, and one GOP spanning the group;
4. measures the encoded packet bitrate with FFprobe;
5. records the requested and measured settings in the optional segment report.

`concat_segments()` scales all encoded groups to the configured common output
size and re-encodes them through FFmpeg's concat filter. Consequently, the
final MP4 is a concatenated presentation artifact; its final stream bitrate is
also influenced by this last CRF-18 encode.

Two implementation details are important when interpreting export behavior:

- `build_decision_stream()` creates all decisions before any segment is
  encoded. Therefore `prev_bitrate` in observations remains at its reset value
  during that decision pass. Although the later encode pass updates
  `env.prev_bitrate`, it does not recompute already-created decisions.
- If the video is longer than the trace, only the traced prefix is exported.
  If the trace is longer than the video, the trace is truncated to the video.

If any decision requests ROI, the exporter requires YOLO metadata. A requested
ROI action on a segment with no detections produces no `addroi` filter and is
reported with `"roi_applied": false`.

## 8. End-to-end run

The following commands assume the shell starts at the repository root. They
use explicit paths to avoid ambiguity from the scripts' current-working-
directory-relative defaults.

```bash
# 1. Extract detections.
python3 extract_yolo_metadata.py \
  --source dataset/videos/input.mp4 \
  --yolo-repo models/yolov5 \
  --weights checkpoints/yolov5s.pt \
  --out outputs/metadata/yolo_metadata.jsonl

# 2. Compute semantic features.
python3 rl_env_offline/yolo_metadata_to_rl.py \
  --input outputs/metadata/yolo_metadata.jsonl \
  --out outputs/metadata/semantic_trace.jsonl

# 3. Generate network conditions.
python3 rl_env_offline/gen_bandwidth_trace.py \
  --num-frames 1500 --fps 25 --seed 42 \
  --out outputs/metadata/bandwidth_trace.jsonl

# 4. Join semantic and network traces.
python3 -c "from rl_env_offline.gen_bandwidth_trace import merge_traces; merge_traces('outputs/metadata/semantic_trace.jsonl', 'outputs/metadata/bandwidth_trace.jsonl', 'outputs/metadata/rl_states.jsonl')"

# 5. Measure the offline rate-distortion grid.
python3 rl_env_offline/encode_grid.py \
  --input dataset/videos/input.mp4 \
  --yolo-metadata outputs/metadata/yolo_metadata.jsonl \
  --out outputs/metadata/vcu_encode_grid.json \
  --bitrate-levels 600,900,1500,2500,4000,6000 \
  --segment-frames 25 --workers 4 --metrics vmaf

# 6. Train from rl_env_offline so its sibling imports resolve.
cd rl_env_offline
python3 train.py \
  --trace-path ../outputs/metadata/rl_states.jsonl \
  --encode-grid-path ../outputs/metadata/vcu_encode_grid.json \
  --episodes 300 --output dqn_policy.pt

# 7. Evaluate the same checkpoint and data contracts.
python3 evaluate.py \
  --trace ../outputs/metadata/rl_states.jsonl \
  --encode-grid ../outputs/metadata/vcu_encode_grid.json \
  --policy dqn_policy.pt

# 8. Produce a real encoded video.
python3 export_encoded_video.py \
  --input ../dataset/videos/input.mp4 \
  --policy dqn_policy.pt \
  --trace ../outputs/metadata/rl_states.jsonl \
  --yolo-metadata ../outputs/metadata/yolo_metadata.jsonl \
  --out ../outputs/exported_policy_video.mp4 \
  --segments-report ../outputs/exported_policy_segments.json
```

The grid step requires FFmpeg/FFprobe and an FFmpeg build with `libvmaf` when
VMAF is requested. YOLO extraction additionally requires the local YOLOv5
repository, model weights, OpenCV, PyTorch, and YOLOv5's Python dependencies.

## 9. Invariants and common failure modes

- The source video, YOLO metadata, semantic trace, bandwidth trace, and encode
  grid must share the same zero-based frame numbering.
- Training and evaluation must use the same `max_bitrate_kbps`, ROI levels,
  action ordering, and grid meaning.
- `segment_len` controls when resolution and ROI may change in the environment;
  it is not the `encode_grid.py --segment-frames` value and is not necessarily
  the physical segment length used by the exporter.
- A grid generated without VMAF cannot provide the training reward used by
  this environment.
- A grid without `roi_vmaf` is accepted only for `roi_idx=0`; ROI actions raise
  a clear `KeyError` asking for a regenerated 27-action grid.
- `VCUSimEnv` validates ROI qoffset values when the grid declares them, but it
  does not validate source filename, FPS, frame count, or resolution metadata.
- The policy checkpoint validates only state and action dimensions. It does
  not store the trace path, encode-grid identity, `max_bitrate_kbps`, reward
  weights, or `segment_len`; those settings must be tracked externally.
