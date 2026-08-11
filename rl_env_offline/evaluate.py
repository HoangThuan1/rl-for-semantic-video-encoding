"""
evaluate.py
-----------
Kiem tra policy DQN SAU KHI train xong (checkpoint tu train.py moi: dict
{"model_state_dict", "state_dim", "num_actions", "actions"}), TRUOC khi tin
dung. Khop dung interface hien tai cua env.py/train.py:
  - Action: index roi rac (int) vao env.action_space (Discrete), KHONG con
    la vector continuous nhu ban thiet ke truoc -- env.step(action_idx) nhan
    thang int.
  - env.decode_action(idx) -> EncoderAction(bitrate_ratio, resolution_idx)
  - checkpoint la 1 dict (torch.save(...)), khong phai state_dict truc tiep.

Tra loi 2 cau hoi:
  1) Co hoi tu khong? (doc lai file *.rewards.json train.py luu ra)
  2) Policy da hoc co tot hon RANDOM va CBR co dinh khong?

CACH DUNG:
  python3 evaluate.py --trace outputs/metadata/rl_states.jsonl \
      --encode-grid outputs/metadata/vcu_encode_grid_roi.json \
      --policy dqn_policy.pt
"""

import argparse
import json
import os

import numpy as np
import torch

from env import ACTIONS, DEFAULT_TRACE_PATH, VCUSimEnv
from train import QNetwork


def load_policy(path):
    """Checkpoint moi la dict (xem train.py: torch.save({"model_state_dict":...,
    "state_dim":..., "num_actions":..., "actions":...}, path)) -- KHONG phai
    state_dict truc tiep nua."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    state_dim = ckpt["state_dim"]
    num_actions = ckpt["num_actions"]
    qnet = QNetwork(state_dim, num_actions)
    qnet.load_state_dict(ckpt["model_state_dict"])
    qnet.eval()
    return qnet, num_actions


def make_greedy_policy(qnet):
    def choose(state):
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).unsqueeze(0)
            q_values = qnet(state_tensor)
            return int(torch.argmax(q_values, dim=1).item())
    return choose


def make_random_policy(rng, num_actions):
    return lambda state: rng.randrange(num_actions)


def make_cbr_policy(num_actions):
    """Tim action_idx gan nhat voi (bitrate_ratio~0.75, resolution_idx=1)
    tuc 720p va GIU CO DINH -- baseline "khong AI"."""
    target = np.array([0.75, 1.0])
    diffs = [
        np.sum((np.array([a.bitrate_ratio, a.resolution_idx]) - target) ** 2)
        for a in ACTIONS[:num_actions]
    ]
    fixed_idx = int(np.argmin(diffs))
    return (lambda state: fixed_idx), fixed_idx


def run_episode(env, choose_action_fn):
    state, _ = env.reset()
    done = False
    total_reward = 0.0
    bitrates, vmafs, roi_vmafs, powers, latencies = [], [], [], [], []
    overflow_steps = 0
    bitrate_switches, resolution_switches = [], 0
    semantic_values = []
    action_counts = np.zeros(env.NUM_ACTIONS, dtype=int)
    steps = 0
    prev_bitrate = None
    prev_resolution = None

    while not done:
        bandwidth_now = env.trace[env.t]["bandwidth"]
        action_idx = choose_action_fn(state)
        action_counts[action_idx] += 1

        next_state, reward, terminated, truncated, info = env.step(action_idx)
        done = terminated or truncated

        total_reward += reward
        bitrates.append(info["actual_bitrate"])
        vmafs.append(info["vmaf"])
        roi_vmafs.append(info.get("roi_vmaf", info["vmaf"]))
        powers.append(info["power"])
        latencies.append(info["latency"])
        semantic_values.append(info["semantic_score"])
        if info["actual_bitrate"] > bandwidth_now:
            overflow_steps += 1
        if prev_bitrate is not None:
            bitrate_switches.append(abs(info["actual_bitrate"] - prev_bitrate))
        if prev_resolution is not None and info["resolution"] != prev_resolution:
            resolution_switches += 1
        prev_bitrate = info["actual_bitrate"]
        prev_resolution = info["resolution"]

        state = next_state
        steps += 1

    semantic_values = np.asarray(semantic_values, dtype=np.float32)
    bitrates_arr = np.asarray(bitrates, dtype=np.float32)
    low_mask = semantic_values <= np.percentile(semantic_values, 33)
    high_mask = semantic_values >= np.percentile(semantic_values, 67)

    return {
        "total_reward": total_reward,
        "avg_reward_per_step": total_reward / max(steps, 1),
        "steps": steps,
        "avg_bitrate_kbps": float(np.mean(bitrates)),
        "avg_vmaf": float(np.mean(vmafs)),
        "avg_roi_vmaf": float(np.mean(roi_vmafs)),
        "avg_power": float(np.mean(powers)),
        "avg_latency_ms": float(np.mean(latencies)),
        "overflow_rate": overflow_steps / max(steps, 1),
        "avg_bitrate_low_semantic": float(np.mean(bitrates_arr[low_mask])) if np.any(low_mask) else 0.0,
        "avg_bitrate_high_semantic": float(np.mean(bitrates_arr[high_mask])) if np.any(high_mask) else 0.0,
        "avg_bitrate_switch_kbps": float(np.mean(bitrate_switches)) if bitrate_switches else 0.0,
        "resolution_switch_rate": resolution_switches / max(steps - 1, 1),
        "action_counts": action_counts,
    }


def print_comparison(results):
    headers = ["Policy", "Tong reward", "Reward/buoc", "Bitrate TB (kbps)",
               "VMAF TB", "ROI VMAF", "Vuot BW", "Doi res"]
    rows = []
    for name, m in results.items():
        rows.append([
            name,
            f"{m['total_reward']:.2f}",
            f"{m['avg_reward_per_step']:.3f}",
            f"{m['avg_bitrate_kbps']:.1f}",
            f"{m['avg_vmaf']:.2f}",
            f"{m['avg_roi_vmaf']:.2f}",
            f"{m['overflow_rate']*100:.1f}%",
            f"{m['resolution_switch_rate']*100:.1f}%",
        ])
    widths = [max(len(h), max((len(r[i]) for r in rows), default=0))
              for i, h in enumerate(headers)]
    line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print(" | ".join(c.ljust(w) for c, w in zip(r, widths)))


def print_learning_checks(env, qnet):
    """Counterfactual checks: giu state history/noi dung co dinh, chi thay
    bandwidth hoac semantic de xem policy co sensitivity dung chieu khong."""
    def choose_for(row, bandwidth, semantic_score, prev_vmaf=82.0, prev_bitrate=1800.0, res_idx=1):
        saved_t = env.t
        saved_row = env.trace[0]
        saved_prev_vmaf = env.prev_vmaf
        saved_prev_bitrate = env.prev_bitrate
        saved_res = env.current_resolution_idx

        env.t = 0
        env.trace[0] = dict(row, bandwidth=bandwidth, semantic_score=semantic_score)
        env.prev_vmaf = prev_vmaf
        env.prev_bitrate = prev_bitrate
        env.current_resolution_idx = res_idx
        obs = env._get_obs()
        action_idx = make_greedy_policy(qnet)(obs)
        action = env.decode_action(action_idx)

        env.t = saved_t
        env.trace[0] = saved_row
        env.prev_vmaf = saved_prev_vmaf
        env.prev_bitrate = saved_prev_bitrate
        env.current_resolution_idx = saved_res
        return action_idx, action

    semantics = [float(r.get("semantic_score", 0.0)) for r in env.trace]
    low_s = float(np.percentile(semantics, 20))
    high_s = float(np.percentile(semantics, 80))
    base_row = dict(env.trace[int(len(env.trace) / 2)])

    print("\n=== Kiem tra policy da hoc 4 quan he ===")
    print("1) Network -> cau hinh, giu semantic trung binh:")
    mid_s = float(np.percentile(semantics, 50))
    for bw in [600.0, 1500.0, 3500.0, 6500.0]:
        _, action = choose_for(base_row, bandwidth=bw, semantic_score=mid_s)
        print(
            f"  bandwidth={bw:6.0f} kbps -> "
            f"target={action.bitrate_ratio * env.max_bitrate:6.0f} kbps, "
            f"resolution_idx={action.resolution_idx}"
        )

    print("2) Semantic -> phan bo tai nguyen, giu bandwidth trung binh:")
    mid_bw = float(np.percentile([r.get("bandwidth", env.max_bitrate) for r in env.trace], 50))
    for s in [low_s, high_s]:
        _, action = choose_for(base_row, bandwidth=mid_bw, semantic_score=s)
        print(
            f"  semantic={s:5.2f} -> "
            f"target={action.bitrate_ratio * env.max_bitrate:6.0f} kbps, "
            f"resolution_idx={action.resolution_idx}"
        )

    print("3) Rate-quality: xem avg VMAF/ROI VMAF trong bang so sanh o tren.")
    print("4) Tuan tu/on dinh: xem cot Vuot BW va Doi res; cang thap cang on dinh.")


def check_convergence(rewards_path, plot_out=None):
    try:
        with open(rewards_path, "r", encoding="utf-8") as f:
            episode_rewards = json.load(f)
    except FileNotFoundError:
        print(f"[canh bao] Khong tim thay {rewards_path} -- bo qua kiem tra hoi tu. "
              f"(train.py ban moi se tu luu file *.rewards.json canh policy sau khi train)")
        return

    n = len(episode_rewards)
    w = min(25, max(1, n // 2))
    first_w = np.mean(episode_rewards[:w])
    last_w = np.mean(episode_rewards[-w:])
    print(f"\n=== Hoi tu ({rewards_path}, {n} episode) ===")
    print(f"  Avg reward {w} episode DAU  : {first_w:.3f}")
    print(f"  Avg reward {w} episode CUOI : {last_w:.3f}")
    if last_w <= first_w:
        print("  [canh bao] Reward cuoi KHONG cao hon dau -- policy co the chua "
              "hoi tu (thu tang --episodes, giam lr, hoac kiem tra lai reward shaping).")
    else:
        print("  -> Co xu huong hoc len theo thoi gian (tot).")

    if plot_out is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            ax.plot(episode_rewards, alpha=0.3, label="reward/episode")
            if n >= 10:
                window = min(25, n // 2)
                smoothed = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
                ax.plot(range(window - 1, n), smoothed, label=f"trung binh truot ({window} ep)")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Total reward")
            ax.set_title("Duong cong hoi tu qua trinh train")
            ax.legend()
            fig.tight_layout()
            fig.savefig(plot_out, dpi=120)
            print(f"  Da luu bieu do hoi tu vao: {plot_out}")
        except ImportError:
            print("  [luu y] Chua cai matplotlib nen bo qua ve bieu do.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", default=DEFAULT_TRACE_PATH)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--rewards", default=None,
                     help="Mac dinh: tu doan <policy khong duoi>.rewards.json")
    ap.add_argument("--encode-grid", default=None)
    ap.add_argument("--plot-out", default="eval_convergence.png")
    ap.add_argument("--random-runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rewards_path = args.rewards or (os.path.splitext(args.policy)[0] + ".rewards.json")
    check_convergence(rewards_path, plot_out=args.plot_out)

    print("\n=== Danh gia tren trace (chay het 1 luot, loop=False) ===")
    env = VCUSimEnv(trace_path=args.trace, loop=False, encode_grid_path=args.encode_grid)

    qnet, num_actions = load_policy(args.policy)
    if num_actions != env.NUM_ACTIONS:
        raise ValueError(
            "Policy/checkpoint khong khop env hien tai: "
            f"policy num_actions={num_actions}, env num_actions={env.NUM_ACTIONS}. "
            "Env hien tai dung action bitrate_ratio x resolution_idx; can train lai policy."
        )
    greedy_result = run_episode(env, make_greedy_policy(qnet))

    cbr_fn, cbr_idx = make_cbr_policy(num_actions)
    print(f"[info] CBR baseline dung co dinh action_idx={cbr_idx} -> {ACTIONS[cbr_idx]}")
    cbr_result = run_episode(env, cbr_fn)

    import random as _random
    rng = _random.Random(args.seed)
    random_runs = [run_episode(env, make_random_policy(rng, num_actions)) for _ in range(args.random_runs)]
    random_result = {
        k: (np.mean([r[k] for r in random_runs]) if k != "action_counts"
            else sum(r["action_counts"] for r in random_runs))
        for k in random_runs[0]
    }

    print()
    print_comparison({
        "Trained (greedy)": greedy_result,
        "CBR co dinh": cbr_result,
        f"Random (tb {args.random_runs} lan)": random_result,
    })
    print_learning_checks(env, qnet)

    if greedy_result["total_reward"] <= max(cbr_result["total_reward"], random_result["total_reward"]):
        print("\n[canh bao] Trained policy KHONG vuot qua baseline -- can xem lai "
              "truoc khi dung ket qua nay (them episode, kiem tra reward shaping, "
              "hoc lai voi seed khac).")
    else:
        print("\n[ok] Trained policy vuot qua ca CBR va Random tren trace nay.")


if __name__ == "__main__":
    main()
