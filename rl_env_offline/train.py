"""
NOTE THIET KE:
    Training nay dung DQN roi rac hoa action cho Semantic-aware Adaptive Encoding.
Moi action la mot to hop {target bitrate ratio, resolution level}.
Env se sanitize action truoc khi mo phong VCU de giam xung dot voi hardware
rate control. Reward khong dung semantic_score truc tiep, ma dung VMAF
normalized cong voi penalty bitrate.
"""

import argparse
import json
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env import VCUSimEnv


class QNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


def _torch_load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _serialize_transition(transition):
    state, action, reward, next_state, done = transition
    return (
        np.asarray(state, dtype=np.float32).tolist(),
        int(action),
        float(reward),
        np.asarray(next_state, dtype=np.float32).tolist(),
        bool(done),
    )


def _deserialize_transition(transition):
    state, action, reward, next_state, done = transition
    return (
        np.asarray(state, dtype=np.float32),
        int(action),
        float(reward),
        np.asarray(next_state, dtype=np.float32),
        bool(done),
    )


def load_training_checkpoint(path, env, qnet, target_net, optimizer, buffer):
    ckpt = _torch_load_checkpoint(path)

    state_dim = ckpt.get("state_dim")
    num_actions = ckpt.get("num_actions")
    if state_dim != env.STATE_DIM or num_actions != env.NUM_ACTIONS:
        raise ValueError(
            "Checkpoint khong khop env hien tai: "
            f"checkpoint state_dim={state_dim}, num_actions={num_actions}; "
            f"env state_dim={env.STATE_DIM}, num_actions={env.NUM_ACTIONS}. "
            "Neu vua doi state/action, can train moi checkpoint."
        )

    qnet.load_state_dict(ckpt["model_state_dict"])
    target_net.load_state_dict(ckpt.get("target_model_state_dict", ckpt["model_state_dict"]))
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    for transition in ckpt.get("replay_buffer", []):
        buffer.buffer.append(_deserialize_transition(transition))

    start_episode = int(ckpt.get("episode", -1)) + 1
    epsilon = float(ckpt.get("epsilon", 1.0))
    episode_rewards = list(ckpt.get("episode_rewards", []))
    return start_episode, epsilon, episode_rewards


def save_training_checkpoint(
    output_path,
    env,
    qnet,
    target_net,
    optimizer,
    buffer,
    episode,
    epsilon,
    episode_rewards,
):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    ckpt = {
        "model_state_dict": qnet.state_dict(),
        "target_model_state_dict": target_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "state_dim": env.STATE_DIM,
        "num_actions": env.NUM_ACTIONS,
        "actions": [env.decode_action(i).__dict__ for i in range(env.NUM_ACTIONS)],
        "episode": int(episode),
        "epsilon": float(epsilon),
        "episode_rewards": list(episode_rewards),
        "replay_buffer": [_serialize_transition(t) for t in buffer.buffer],
    }
    torch.save(ckpt, output_path)

    rewards_path = os.path.splitext(output_path)[0] + ".rewards.json"
    with open(rewards_path, "w", encoding="utf-8") as f:
        json.dump(list(episode_rewards), f, ensure_ascii=False, indent=2)


def train(
    trace_path=None,
    encode_grid_path=None,
    output_path="dqn_policy.pt",
    resume_path=None,
    num_episodes=300,
    batch_size=64,
    gamma=0.97,
    lr=5e-4,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.992,
    target_update_every=10,
    seed=7,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = VCUSimEnv(trace_path=trace_path, encode_grid_path=encode_grid_path, loop=False)
    qnet = QNetwork(env.STATE_DIM, env.NUM_ACTIONS)
    target_net = QNetwork(env.STATE_DIM, env.NUM_ACTIONS)
    target_net.load_state_dict(qnet.state_dict())

    optimizer = optim.Adam(qnet.parameters(), lr=lr)
    buffer = ReplayBuffer()
    epsilon = epsilon_start
    episode_rewards = []
    start_episode = 0

    if resume_path is not None:
        start_episode, epsilon, episode_rewards = load_training_checkpoint(
            resume_path, env, qnet, target_net, optimizer, buffer
        )
        print(
            f"Resume tu {resume_path}: da co {start_episode} episode, "
            f"epsilon={epsilon:.3f}, replay_buffer={len(buffer)}"
        )

    end_episode = start_episode + num_episodes
    for episode in range(start_episode, end_episode):
        state, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = qnet(torch.from_numpy(state).unsqueeze(0))
                    action = int(torch.argmax(q_values, dim=1).item())

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                states_t = torch.from_numpy(states)
                actions_t = torch.from_numpy(actions)
                rewards_t = torch.from_numpy(rewards)
                next_states_t = torch.from_numpy(next_states)
                dones_t = torch.from_numpy(dones)

                q_sa = qnet(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

                # Double DQN target: online net chon action, target net dinh gia action do.
                with torch.no_grad():
                    next_actions = torch.argmax(qnet(next_states_t), dim=1)
                    next_q = target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    q_target = rewards_t + gamma * next_q * (1.0 - dones_t)

                loss = nn.functional.smooth_l1_loss(q_sa, q_target)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(qnet.parameters(), 5.0)
                optimizer.step()

        episode_rewards.append(total_reward)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if episode % target_update_every == 0:
            target_net.load_state_dict(qnet.state_dict())

        if episode % 25 == 0:
            avg_reward = np.mean(episode_rewards[-25:])
            print(
                f"Episode {episode:4d} | epsilon={epsilon:.3f} | "
                f"avg_reward(25 ep)={avg_reward:.3f}"
            )

    save_training_checkpoint(
        output_path,
        env,
        qnet,
        target_net,
        optimizer,
        buffer,
        end_episode - 1,
        epsilon,
        episode_rewards,
    )
    print(f"\nSaved policy/checkpoint to {output_path}")
    return episode_rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-path", default=None)
    parser.add_argument("--encode-grid-path", default=None)
    parser.add_argument("--output", default="dqn_policy.pt")
    parser.add_argument("--resume", default=None, help="Checkpoint cu de tiep tuc train")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    train(
        trace_path=args.trace_path,
        encode_grid_path=args.encode_grid_path,
        output_path=args.output,
        resume_path=args.resume,
        num_episodes=args.episodes,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

