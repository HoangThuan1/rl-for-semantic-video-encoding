"""
evaluate.py
-----------
Kiem tra policy DQN SAU KHI train xong (checkpoint tu train.py moi: dict
{"model_state_dict", "state_dim", "num_actions", "actions"}), TRUOC khi tin
dung. Khop dung interface hien tai cua env.py/train.py:
  - Action: index roi rac (int) vao env.action_space (Discrete), KHONG con
    la vector continuous nhu ban thiet ke truoc -- env.step(action_idx) nhan
    thang int.
  - env.decode_action(idx) -> EncoderAction(bitrate_ratio, resolution_idx, roi_idx)
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

from env import ACTIONS, DEFAULT_TRACE_PATH, ROI_QOFFSET_LEVELS, VCUSimEnv
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
    """Tim action gan (bitrate_ratio~0.75, resolution_idx=1, roi_idx=0),
    tuc 720p, khong ROI va GIU CO DINH -- baseline "khong AI"."""
    target = np.array([0.75, 1.0, 0.0])
    diffs = [
        np.sum((np.array([a.bitrate_ratio, a.resolution_idx, a.roi_idx]) - target) ** 2)
        for a in ACTIONS[:num_actions]
    ]
    fixed_idx = int(np.argmin(diffs))
    return (lambda state: fixed_idx), fixed_idx


def run_episode(env, choose_action_fn):
    state, _ = env.reset()
    done = False
    total_reward = 0.0
    bitrates, vmafs, powers, latencies = [], [], [], []
    overflow_steps = 0
    action_counts = np.zeros(env.NUM_ACTIONS, dtype=int)
    roi_counts = np.zeros(len(ROI_QOFFSET_LEVELS), dtype=int)
    steps = 0

    while not done:
        bandwidth_now = env.trace[env.t]["bandwidth"]
        action_idx = choose_action_fn(state)
        action_counts[action_idx] += 1

        next_state, reward, terminated, truncated, info = env.step(action_idx)
        done = terminated or truncated
        roi_counts[info["roi_idx"]] += 1

        total_reward += reward
        bitrates.append(info["actual_bitrate"])
        vmafs.append(info["vmaf"])
        powers.append(info["power"])
        latencies.append(info["latency"])
        if info["actual_bitrate"] > bandwidth_now:
            overflow_steps += 1

        state = next_state
        steps += 1

    return {
        "total_reward": total_reward,
        "avg_reward_per_step": total_reward / max(steps, 1),
        "steps": steps,
        "avg_bitrate_kbps": float(np.mean(bitrates)),
        "avg_vmaf": float(np.mean(vmafs)),
        "avg_power": float(np.mean(powers)),
        "avg_latency_ms": float(np.mean(latencies)),
        "overflow_rate": overflow_steps / max(steps, 1),
        "action_counts": action_counts,
        "roi_counts": roi_counts,
    }


def print_comparison(results):
    headers = ["Policy", "Tong reward", "Reward/buoc", "Bitrate TB (kbps)",
               "VMAF TB", "Power TB", "Ty le vuot BW"]
    rows = []
    for name, m in results.items():
        rows.append([
            name,
            f"{m['total_reward']:.2f}",
            f"{m['avg_reward_per_step']:.3f}",
            f"{m['avg_bitrate_kbps']:.1f}",
            f"{m['avg_vmaf']:.2f}",
            f"{m['avg_power']:.3f}",
            f"{m['overflow_rate']*100:.1f}%",
        ])
    widths = [max(len(h), max((len(r[i]) for r in rows), default=0))
              for i, h in enumerate(headers)]
    line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print(" | ".join(c.ljust(w) for c, w in zip(r, widths)))
    print("\nTy le ROI level thuc su sau sanitize:")
    for name, metrics in results.items():
        counts = np.asarray(metrics["roi_counts"], dtype=np.float64)
        percentages = 100.0 * counts / max(float(np.sum(counts)), 1.0)
        detail = ", ".join(
            f"roi{idx}({ROI_QOFFSET_LEVELS[idx]}): {pct:.1f}%"
            for idx, pct in enumerate(percentages)
        )
        print(f"  {name}: {detail}")


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
    ap.add_argument("--encode-grid", required=True)
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
            "Env hien tai dung action bitrate_ratio x resolution_idx x roi_idx; "
            "can train lai policy."
        )
    greedy_result = run_episode(env, make_greedy_policy(qnet))

    cbr_fn, cbr_idx = make_cbr_policy(num_actions)
    print(f"[info] CBR baseline dung co dinh action_idx={cbr_idx} -> {ACTIONS[cbr_idx]}")
    cbr_result = run_episode(env, cbr_fn)

    import random as _random
    rng = _random.Random(args.seed)
    random_runs = [run_episode(env, make_random_policy(rng, num_actions)) for _ in range(args.random_runs)]
    random_result = {
        k: (
            sum(r[k] for r in random_runs)
            if k in {"action_counts", "roi_counts"}
            else np.mean([r[k] for r in random_runs])
        )
        for k in random_runs[0]
    }

    print()
    print_comparison({
        "Trained (greedy)": greedy_result,
        "CBR co dinh": cbr_result,
        f"Random (tb {args.random_runs} lan)": random_result,
    })

    if greedy_result["total_reward"] <= max(cbr_result["total_reward"], random_result["total_reward"]):
        print("\n[canh bao] Trained policy KHONG vuot qua baseline -- can xem lai "
              "truoc khi dung ket qua nay (them episode, kiem tra reward shaping, "
              "hoc lai voi seed khac).")
    else:
        print("\n[ok] Trained policy vuot qua ca CBR va Random tren trace nay.")


if __name__ == "__main__":
    main()
