"""
NOTE THIET KE:
Env nay dung state gon cho Semantic-aware Bitrate Control:
  [bandwidth, prev_bitrate, semantic_score].
Reward theo cong thuc R = 0.6 * QoE_norm - 0.4 * bitrate_cost.
Action duoc roi rac hoa de hop voi DQN va duoc "sanitize" truoc khi dua vao
surrogate VCU, tranh de policy xung dot voi rate control cap thap cua encoder.
Resolution va ROI qoffset duoc xem la quyet dinh cap segment/GOP, khong phai
thay tung frame.
"""

import json
import math
import random
from dataclasses import dataclass
from itertools import product

import gymnasium as gym
import numpy as np
from gymnasium import spaces

RESOLUTIONS = [(640, 480), (1280, 720), (1920, 1080)]
BITRATE_RATIOS = [0.50, 0.75, 0.95]
RESOLUTION_ACTIONS = [0, 1, 2]
ROI_QOFFSET_LEVELS = [0.0, -0.3, -0.6]
ROI_ACTIONS = list(range(len(ROI_QOFFSET_LEVELS)))

SEMANTIC_SCORE_MAX = 30.0
MOTION_MAX = 1.0
ROI_AREA_MAX = 1.0
LATENCY_BUDGET_MS = 40.0
DEFAULT_QOE_WEIGHT = 0.60
DEFAULT_BITRATE_COST_WEIGHT = 0.40
DEFAULT_METADATA_DIR = "outputs/metadata"
DEFAULT_TRACE_PATH = "outputs/metadata/rl_states.jsonl"
DEFAULT_YOLO_METADATA_PATH = "outputs/metadata/yolo_metadata.jsonl"


@dataclass(frozen=True)
class EncoderAction:
    bitrate_ratio: float
    resolution_idx: int
    roi_idx: int


ACTIONS = [
    EncoderAction(bitrate_ratio, resolution_idx, roi_idx)
    for bitrate_ratio, resolution_idx, roi_idx in product(
        BITRATE_RATIOS, RESOLUTION_ACTIONS, ROI_ACTIONS
    )
]


def load_trace(path):
    trace = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                trace.append(json.loads(line))
    if not trace:
        raise ValueError(f"Trace rong: {path}")
    return trace


def build_synthetic_trace(num_frames=240, seed=7):
    """Build an explicit synthetic trace for smoke tests."""
    rng = random.Random(seed)
    trace = []
    bandwidth_levels = [900.0, 1800.0, 3200.0, 5200.0, 7600.0]
    level_idx = 2

    for frame_idx in range(num_frames):
        if frame_idx % 40 == 0 and frame_idx > 0:
            level_idx = min(max(level_idx + rng.choice([-1, 0, 1]), 0), len(bandwidth_levels) - 1)

        motion = 0.25 + 0.25 * math.sin(frame_idx / 13.0) + rng.uniform(-0.08, 0.08)
        motion = float(np.clip(motion, 0.0, 1.0))
        num_objects = rng.choice([0, 1, 1, 2, 3, 4])
        roi_area = float(np.clip(0.04 * num_objects + rng.uniform(0.0, 0.12), 0.0, 0.65))
        priority = rng.choice([0.5, 0.8, 1.0, 1.3])
        semantic_score = 1.0 * num_objects + 10.0 * roi_area + 5.0 * motion + 2.0 * priority
        bandwidth = bandwidth_levels[level_idx] * (1.0 + rng.uniform(-0.06, 0.06))

        trace.append(
            {
                "frame_idx": frame_idx,
                "semantic_score": round(semantic_score, 4),
                "bandwidth": round(max(100.0, bandwidth), 2),
                "motion": round(motion, 4),
                "roi_area": round(roi_area, 4),
                "num_objects": num_objects,
                "priority": priority,
                "current_qp": 30.0,
            }
        )
    return trace


def _class_priority(det):
    class_name = str(det.get("class_name", "")).lower()
    class_id = int(det.get("class_id", -1))
    if class_name in {"person", "pedestrian"} or class_id == 0:
        return 1.0
    if class_name in {"car", "bus", "truck"}:
        return 0.8
    if class_name in {"motorcycle", "motorbike", "bicycle"}:
        return 0.6
    if class_name == "train":
        return 0.7
    return 0.3


def load_yolo_metadata_as_trace(
    path,
    max_bitrate_kbps=8000.0,
    seed=11,
    alpha=1.0,
    beta=10.0,
    gamma=5.0,
    delta=2.0,
):
    """Build RL trace from raw YOLO metadata when rl_states.jsonl is not ready yet."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"YOLO metadata rong: {path}")

    synthetic_bw = build_synthetic_trace(len(rows), seed=seed)
    trace = []
    prev_centroids = []

    for i, row in enumerate(rows):
        detections = row.get("detections", [])
        width = max(float(row.get("width", 1.0)), 1.0)
        height = max(float(row.get("height", 1.0)), 1.0)

        roi_area = 0.0
        priority_sum = 0.0
        centroids = []

        for det in detections:
            roi_area += float(det.get("area_ratio", 0.0))
            priority = _class_priority(det)
            priority_sum += priority
            if "centroid" in det:
                centroids.append(det["centroid"])
            elif "bbox" in det:
                x1, y1, x2, y2 = det["bbox"]
                centroids.append([(x1 + x2) * 0.5, (y1 + y2) * 0.5])

        motion = 0.0
        if centroids and prev_centroids:
            pairs = min(len(centroids), len(prev_centroids))
            distances = []
            for j in range(pairs):
                dx = (centroids[j][0] - prev_centroids[j][0]) / width
                dy = (centroids[j][1] - prev_centroids[j][1]) / height
                distances.append(math.sqrt(dx * dx + dy * dy))
            motion = float(np.clip(np.mean(distances) * 8.0, 0.0, 1.0)) if distances else 0.0
        prev_centroids = centroids

        num_objects = len(detections)
        roi_area = float(np.clip(roi_area, 0.0, 1.0))
        semantic_score = (
            alpha * num_objects
            + beta * roi_area
            + gamma * motion
            + delta * priority_sum
        )

        trace.append(
            {
                "frame_idx": int(row.get("frame_idx", i)),
                "semantic_score": round(float(semantic_score), 4),
                "bandwidth": synthetic_bw[i]["bandwidth"] if i < len(synthetic_bw) else max_bitrate_kbps * 0.5,
                "motion": round(motion, 4),
                "roi_area": round(roi_area, 5),
                "num_objects": num_objects,
                "priority": round(priority_sum, 4),
                "current_qp": 30.0,
            }
        )

    return trace


def load_encode_grid(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class VCUSimEnv(gym.Env):
    """
    Offline surrogate cho VCU encoder.

    State normalized:
      [bandwidth, prev_bitrate, semantic_score]

    Action:
      index roi rac trong ACTIONS = bitrate target ratio x resolution x ROI qoffset.
    """

    metadata = {"render_modes": []}
    NUM_ACTIONS = len(ACTIONS)
    STATE_DIM = 3

    def __init__(
        self,
        trace_path=DEFAULT_TRACE_PATH,
        trace=None,
        max_bitrate_kbps=8000.0,
        min_qp=10,
        max_qp=51,
        loop=False,
        encode_grid_path=None,
        segment_len=8,
        qoe_weight=DEFAULT_QOE_WEIGHT,
        bitrate_cost_weight=DEFAULT_BITRATE_COST_WEIGHT,
    ):
        super().__init__()
        if trace is None:
            if not trace_path:
                raise ValueError("Can cung cap trace hoac trace_path")
            trace = load_trace(trace_path)
        if not trace:
            raise ValueError("Trace khong duoc rong")
        required_trace_fields = {
            "frame_idx", "semantic_score", "bandwidth", "motion", "roi_area"
        }
        for row_index, row in enumerate(trace):
            missing = required_trace_fields - set(row)
            if missing:
                raise ValueError(
                    f"Trace row {row_index} thieu field: {sorted(missing)}"
                )

        self.trace = trace
        self.n = len(trace)
        self.max_bitrate = float(max_bitrate_kbps)
        self.min_qp = float(min_qp)
        self.max_qp = float(max_qp)
        self.loop = loop
        self.segment_len = max(1, int(segment_len))
        self.qoe_weight = float(qoe_weight)
        self.bitrate_cost_weight = float(bitrate_cost_weight)

        self.grid = load_encode_grid(encode_grid_path) if encode_grid_path else None
        if self.grid is not None:
            required_grid_fields = {
                "grid_unit", "num_frames", "resolutions",
                "bitrate_levels_kbps", "roi_qoffset_levels", "grid",
            }
            missing = required_grid_fields - set(self.grid)
            if missing:
                raise ValueError(f"Encode grid thieu field: {sorted(missing)}")
            if self.grid["grid_unit"] != "frame":
                raise ValueError("Encode grid phai duoc tao theo tung frame")
            if list(self.grid["resolutions"]) != [list(r) for r in RESOLUTIONS]:
                raise ValueError("Resolution cua encode grid khong khop env")
            grid_roi_levels = self.grid["roi_qoffset_levels"]
            if list(grid_roi_levels) != ROI_QOFFSET_LEVELS:
                raise ValueError(
                    "ROI levels cua encode grid khong khop env: "
                    f"grid={list(grid_roi_levels)}, env={ROI_QOFFSET_LEVELS}."
                )
            grid_num_frames = int(self.grid["num_frames"])
            if grid_num_frames <= 0:
                raise ValueError("Encode grid num_frames phai > 0")
            if not self.grid["bitrate_levels_kbps"]:
                raise ValueError("Encode grid bitrate_levels_kbps khong duoc rong")
            for resolution_idx in range(len(RESOLUTIONS)):
                for bitrate_idx in range(len(self.grid["bitrate_levels_kbps"])):
                    for roi_idx in range(len(ROI_QOFFSET_LEVELS)):
                        key = f"res{resolution_idx}_br{bitrate_idx}_roi{roi_idx}"
                        if key not in self.grid["grid"]:
                            raise ValueError(f"Encode grid thieu curve: {key}")
                        if len(self.grid["grid"][key]) != grid_num_frames:
                            raise ValueError(
                                f"Encode grid curve {key} co do dai "
                                f"{len(self.grid['grid'][key])}, can {grid_num_frames}"
                            )

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.STATE_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.NUM_ACTIONS)

        self.t = 0
        self.current_resolution_idx = 1
        self.current_roi_idx = 0
        self.prev_bitrate = 0.0
        self.prev_vmaf = 0.0
        self.prev_latency = 0.0
        self.prev_action_idx = 0
        self.prev_bandwidth = float(self.trace[0]["bandwidth"])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.current_resolution_idx = 1
        self.current_roi_idx = 0
        self.prev_bandwidth = float(self.trace[0]["bandwidth"])
        self.prev_bitrate = min(self.prev_bandwidth * 0.7, self.max_bitrate)
        self.prev_vmaf = 70.0
        self.prev_latency = 0.0
        self.prev_action_idx = 0
        return self._get_obs(), {}

    def step(self, action_idx):
        action_idx = int(action_idx)
        raw_action = ACTIONS[action_idx]
        row = self.trace[self.t]

        semantic_score = float(row["semantic_score"])
        bandwidth = float(row["bandwidth"])
        motion = float(row["motion"])
        roi_area = float(row["roi_area"])

        safe_action = self._sanitize_action(raw_action, semantic_score, bandwidth)
        width, height = RESOLUTIONS[safe_action.resolution_idx]
        target_bitrate = safe_action.bitrate_ratio * bandwidth

        if self.grid is not None:
            actual_bitrate, vmaf, latency, power = self._encode_from_grid(
                safe_action, target_bitrate, row["frame_idx"],
                semantic_score, roi_area, width, height
            )
        else:
            actual_bitrate, vmaf, latency, power = self._simulate_encode(
                width, height, safe_action, target_bitrate, semantic_score, motion, roi_area
            )

        reward, reward_terms = self._compute_reward(
            vmaf=vmaf,
            actual_bitrate=actual_bitrate,
        )

        self.prev_bandwidth = bandwidth
        self.prev_bitrate = actual_bitrate
        self.prev_vmaf = vmaf
        self.prev_latency = latency
        self.prev_action_idx = action_idx
        self.current_resolution_idx = safe_action.resolution_idx
        self.current_roi_idx = safe_action.roi_idx

        self.t += 1
        if self.t >= self.n:
            if self.loop:
                self.t = 0
                terminated = False
            else:
                terminated = True
        else:
            terminated = False

        info = {
            "raw_action": raw_action,
            "safe_action": safe_action,
            "actual_bitrate": actual_bitrate,
            "target_bitrate": target_bitrate,
            "vmaf": vmaf,
            "latency": latency,
            "power": power,
            "resolution": (width, height),
            "roi_idx": safe_action.roi_idx,
            "roi_qoffset": ROI_QOFFSET_LEVELS[safe_action.roi_idx],
            "semantic_score": semantic_score,
            "roi_area": roi_area,
            **reward_terms,
        }

        obs = self._get_obs() if not terminated else np.zeros(self.STATE_DIM, dtype=np.float32)
        return obs, reward, terminated, False, info

    def decode_action(self, action_idx):
        return ACTIONS[int(action_idx)]

    def _get_obs(self):
        row = self.trace[self.t]
        bandwidth = float(row["bandwidth"])
        bandwidth_norm = np.clip(bandwidth / self.max_bitrate, 0.0, 1.0)
        bitrate_norm = np.clip(self.prev_bitrate / self.max_bitrate, 0.0, 1.0)
        semantic_norm = np.clip(float(row["semantic_score"]) / SEMANTIC_SCORE_MAX, 0.0, 1.0)
        return np.array([bandwidth_norm, bitrate_norm, semantic_norm], dtype=np.float32)

    def _sanitize_action(self, action, semantic_score, bandwidth):
        bitrate_ratio = min(action.bitrate_ratio, 0.95)
        resolution_idx = action.resolution_idx
        roi_idx = int(np.clip(action.roi_idx, 0, len(ROI_QOFFSET_LEVELS) - 1))

        if bandwidth < 300.0:
            resolution_idx = min(resolution_idx, 1)
            bitrate_ratio = min(bitrate_ratio, 0.75)
        elif bandwidth < 900.0:
            resolution_idx = min(resolution_idx, 1)

        if semantic_score > 0.85 * SEMANTIC_SCORE_MAX and bandwidth > 2500.0:
            resolution_idx = max(resolution_idx, 1)

        # Resolution va ROI qoffset la quyet dinh cap segment/GOP.
        if self.t % self.segment_len != 0:
            resolution_idx = self.current_resolution_idx
            roi_idx = self.current_roi_idx

        return EncoderAction(bitrate_ratio, resolution_idx, roi_idx)

    def _encode_from_grid(self, action, target_bitrate, frame_idx, semantic_score,
                          roi_area, width, height):
        actual_bitrate = float(np.clip(target_bitrate, 50.0, self.max_bitrate))
        global_vmaf, roi_vmaf = self._interpolate_grid_vmaf(
            action.resolution_idx,
            frame_idx,
            actual_bitrate,
            roi_idx=action.roi_idx,
        )
        semantic_norm = np.clip(semantic_score / SEMANTIC_SCORE_MAX, 0.0, 1.0)
        roi_weight = semantic_norm * np.clip(roi_area * 3.0, 0.0, 0.75)
        vmaf = float((1.0 - roi_weight) * global_vmaf + roi_weight * roi_vmaf)

        pixels = width * height
        roi_strength = action.roi_idx / max(len(ROI_QOFFSET_LEVELS) - 1, 1)
        complexity = 1.0 + 0.08 * min(semantic_score, SEMANTIC_SCORE_MAX) + 0.08 * roi_strength
        latency = 5.0 + (pixels / (1920 * 1080)) * 25.0 * complexity
        power = 0.5 + 1.2 * (pixels / (1920 * 1080)) ** 0.8 * complexity
        return actual_bitrate, vmaf, latency, power

    def _simulate_encode(self, width, height, action, target_bitrate, semantic_score, motion, roi_area):
        pixels = width * height
        res_factor = pixels / (1920 * 1080)
        semantic_norm = np.clip(semantic_score / SEMANTIC_SCORE_MAX, 0.0, 1.0)
        roi_strength = action.roi_idx / max(len(ROI_QOFFSET_LEVELS) - 1, 1)
        complexity = (
            1.0 + 0.65 * motion + 0.55 * roi_area
            + 0.25 * semantic_norm + 0.08 * roi_strength
        )

        demanded_bitrate = self.max_bitrate * res_factor * complexity * 0.42
        actual_bitrate = float(np.clip(min(demanded_bitrate, target_bitrate), 50.0, self.max_bitrate))

        bitrate_gain = 17.0 * math.log10(max(actual_bitrate, 50.0) / 350.0)
        resolution_gain = 16.0 * math.log10(max(res_factor, 0.12) / 0.12)
        roi_relevance = semantic_norm * np.clip(roi_area * 3.0, 0.0, 1.0)
        roi_quality_gain = 6.0 * roi_strength * roi_relevance
        vmaf = 58.0 + bitrate_gain + resolution_gain - 9.0 * motion + roi_quality_gain
        vmaf = float(np.clip(vmaf, 0.0, 100.0))

        latency = 5.0 + 25.0 * res_factor * complexity
        power = 0.55 + 1.15 * (res_factor ** 0.8) * complexity
        return actual_bitrate, vmaf, latency, power

    @staticmethod
    def _cell_vmaf(cell):
        if "vmaf" in cell:
            return float(cell["vmaf"])
        raise KeyError("Encode grid thieu field 'vmaf'. Hay chay lai encode_grid.py de tao grid VMAF.")

    @staticmethod
    def _cell_roi_vmaf(cell):
        if "roi_vmaf" not in cell:
            raise KeyError("Encode grid thieu field 'roi_vmaf'")
        return float(cell["roi_vmaf"])

    def _interpolate_grid_vmaf(self, resolution_idx, frame_idx, target_bitrate, roi_idx=0):
        points = []
        frame_idx = int(frame_idx)
        num_frames = int(self.grid["num_frames"])
        if not 0 <= frame_idx < num_frames:
            raise IndexError(
                f"frame_idx={frame_idx} nam ngoai encode grid [0,{num_frames})"
            )
        bitrate_levels = self.grid["bitrate_levels_kbps"]
        for br_idx, _ in enumerate(bitrate_levels):
            key = f"res{resolution_idx}_br{br_idx}_roi{roi_idx}"
            frames = self.grid["grid"][key]
            cell = frames[frame_idx]
            points.append((
                float(cell["bitrate_kbps"]),
                self._cell_vmaf(cell),
                self._cell_roi_vmaf(cell),
            ))

        points.sort(key=lambda p: p[0])
        bitrates = np.array([p[0] for p in points], dtype=np.float32)
        vmafs = np.array([p[1] for p in points], dtype=np.float32)
        roi_vmafs = np.array([p[2] for p in points], dtype=np.float32)
        unique_bitrates, unique_indices = np.unique(bitrates, return_index=True)
        unique_vmafs = vmafs[unique_indices]
        unique_roi_vmafs = roi_vmafs[unique_indices]

        if len(unique_bitrates) == 1:
            return (
                float(np.clip(unique_vmafs[0], 0.0, 100.0)),
                float(np.clip(unique_roi_vmafs[0], 0.0, 100.0)),
            )

        interp_vmaf = np.interp(
            float(target_bitrate),
            unique_bitrates,
            unique_vmafs,
            left=unique_vmafs[0],
            right=unique_vmafs[-1],
        )
        interp_roi_vmaf = np.interp(
            float(target_bitrate),
            unique_bitrates,
            unique_roi_vmafs,
            left=unique_roi_vmafs[0],
            right=unique_roi_vmafs[-1],
        )
        return (
            float(np.clip(interp_vmaf, 0.0, 100.0)),
            float(np.clip(interp_roi_vmaf, 0.0, 100.0)),
        )

    def _compute_reward(self, vmaf, actual_bitrate):
        qoe_norm = np.clip(vmaf / 100.0, 0.0, 1.0)
        bitrate_cost = actual_bitrate / self.max_bitrate
        reward = self.qoe_weight * qoe_norm - self.bitrate_cost_weight * bitrate_cost

        return float(reward), {
            "qoe_norm": float(qoe_norm),
            "vmaf_norm": float(qoe_norm),
            "bitrate_cost": float(bitrate_cost),
            "qoe_reward_term": float(self.qoe_weight * qoe_norm),
            "bitrate_penalty_term": float(self.bitrate_cost_weight * bitrate_cost),
        }

if __name__ == "__main__":
    env = VCUSimEnv(trace=build_synthetic_trace(), loop=False)
    obs, _ = env.reset()
    print("obs0:", obs)
    total_reward = 0.0
    done = False
    steps = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        steps += 1
    print(f"Ran {steps} steps, random-policy total reward: {total_reward:.3f}")
