import copy
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .base_algorithm import Algorithm
from RLcodes.environment.distillation_column_sequencing_env import DistillationColumnSequencingEnv
from RLcodes.network.nn_column_sequencing import JointQMLP
from RLcodes.utility.buffer import ReplayBuffer


class DQNColumnSequencing(Algorithm):
    def __init__(self, config):
        self.config = config
        self.device = config["device"]
        self.s_dim = config["s_dim"]
        self.a_dim = config["a_dim"]
        self.nT = config["nT"]

        self.split_action_dim = int(config.get("split_action_dim", 5))
        self.reflux_n_bins = int(config.get("reflux_n_bins", 21))
        self.reflux_low = float(config.get("reflux_low", 1.05))
        self.reflux_high = float(config.get("reflux_high", 2.0))
        self.reflux_mesh = np.linspace(
            self.reflux_low, self.reflux_high, self.reflux_n_bins, dtype=np.float32
        )

        self.num_hidden_nodes = config["num_hidden_nodes"]
        self.num_hidden_layers = config["num_hidden_layers"]
        hidden_dim_lst = [self.num_hidden_nodes for _ in range(self.num_hidden_layers)]

        self.gamma = config["gamma"]
        self.critic_lr = config["critic_lr"]
        self.adam_eps = config["adam_eps"]
        self.l2_reg = config["l2_reg"]
        self.grad_clip_mag = config["grad_clip_mag"]
        self.tau = config["tau"]

        self.eps0 = 0.1
        self.epi_denom = 1.0
        self.step_count = 0
        self.episode = 0
        self.epsilon = self.eps0

        self.joint_action_dim = self.split_action_dim * self.reflux_n_bins
        self.replay_buffer = ReplayBuffer(config)

        self.critic = JointQMLP(
            self.s_dim, self.split_action_dim, self.reflux_n_bins, hidden_dim_lst, F.silu
        ).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.adam_eps,
            weight_decay=self.l2_reg,
        )

        self.loss_lst = ["Critic loss"]

    @staticmethod
    def _split_idx_to_scaled(split_idx):
        return split_idx.astype(np.float32) / 2.0 - 1.0

    @staticmethod
    def _scaled_to_split_idx(split_scaled):
        idx = np.round((np.clip(split_scaled, -1.0, 1.0) + 1.0) * 2.0)
        return int(np.clip(idx, 0, 4))

    def _reflux_to_scaled(self, reflux):
        return (2.0 * reflux - (self.reflux_high + self.reflux_low)) / (
            self.reflux_high - self.reflux_low + 1e-8
        )

    def _scaled_to_reflux(self, reflux_scaled):
        return (self.reflux_high - self.reflux_low) / 2.0 * reflux_scaled + (
            self.reflux_high + self.reflux_low
        ) / 2.0

    def _encode_joint_idx(self, split_idx, reflux_bin):
        return int(split_idx * self.reflux_n_bins + reflux_bin)

    def _decode_joint_idx(self, joint_idx):
        split_idx = int(joint_idx // self.reflux_n_bins)
        reflux_bin = int(joint_idx % self.reflux_n_bins)
        return split_idx, reflux_bin

    def _nearest_reflux_bin(self, reflux_value):
        return int(np.argmin(np.abs(self.reflux_mesh - reflux_value)))

    def _split_mask_from_scaled_state_np(self, state_scaled):
        mask, _, valid = DistillationColumnSequencingEnv.split_mask_from_scaled_state(state_scaled)
        if valid <= 0:
            mask = np.zeros_like(mask)
            mask[0] = 1.0
        return mask.astype(np.float32)

    def _build_joint_mask_from_states(self, states):
        state_np = states.detach().cpu().numpy()
        masks = []
        for s in state_np:
            split_mask = self._split_mask_from_scaled_state_np(s)
            joint_mask = np.repeat(split_mask[:, None], self.reflux_n_bins, axis=1).reshape(-1)
            masks.append(joint_mask)
        return torch.tensor(np.asarray(masks), dtype=torch.float32, device=self.device)

    def _build_joint_mask_single(self, state):
        split_mask = self._split_mask_from_scaled_state_np(state)
        return np.repeat(split_mask[:, None], self.reflux_n_bins, axis=1).reshape(-1)

    def _update_epsilon_schedule(self):
        self.step_count += 1
        if self.step_count >= self.nT:
            self.step_count = 0
            self.episode += 1
            self.epsilon = self.eps0 / (1.0 + (self.episode / self.epi_denom))

    def ctrl(self, state):
        with torch.no_grad():
            state_tensor = torch.tensor(state.T, dtype=torch.float32, device=self.device)
            q_values = self.critic(state_tensor).cpu().numpy().reshape(-1)

        joint_mask = self._build_joint_mask_single(state.reshape(-1))
        masked_q = q_values.copy()
        masked_q[joint_mask <= 0.0] = 1e9
        greedy_idx = int(np.argmin(masked_q))

        valid_indices = np.where(joint_mask > 0.0)[0]
        if len(valid_indices) == 0:
            action_idx = 0
        elif np.random.random() <= self.epsilon:
            action_idx = int(np.random.choice(valid_indices))
        else:
            action_idx = greedy_idx

        split_idx, reflux_bin = self._decode_joint_idx(action_idx)
        reflux = float(self.reflux_mesh[reflux_bin])
        split_scaled = self._split_idx_to_scaled(np.array([split_idx], dtype=np.float32))[0]
        reflux_scaled = float(np.clip(self._reflux_to_scaled(reflux), -1.0, 1.0))

        self._update_epsilon_schedule()

        action = np.array([[split_scaled], [reflux_scaled]], dtype=np.float32)
        return action

    def _action_to_joint_idx(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(self.a_dim, 1)
        split_scaled = float(action[0, 0])
        reflux_scaled = float(action[1, 0])

        split_idx = self._scaled_to_split_idx(split_scaled)
        reflux = float(np.clip(self._scaled_to_reflux(reflux_scaled), self.reflux_low, self.reflux_high))
        reflux_bin = self._nearest_reflux_bin(reflux)
        return self._encode_joint_idx(split_idx, reflux_bin)

    def add_experience(self, experience):
        state, action, reward, next_state, done, deriv = experience
        joint_idx = self._action_to_joint_idx(action)
        self.replay_buffer.add((state, np.array([[joint_idx]]), reward, next_state, done, deriv))

    def warm_up_train(self):
        pass

    def train(self):
        sample = self.replay_buffer.sample()
        states = sample["states"]
        action_indices = sample["actions"]
        rewards = sample["rewards"]
        next_states = sample["next_states"]
        dones = sample["dones"]

        with torch.no_grad():
            next_q_all = self.target_critic(next_states).detach()
            next_masks = self._build_joint_mask_from_states(next_states)
            next_q_all = next_q_all + (1.0 - next_masks) * 1e9
            next_q, _ = torch.min(next_q_all, dim=1, keepdim=True)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        current_q_all = self.critic(states)
        current_q = torch.gather(current_q_all, dim=1, index=action_indices.long())
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_mag)
        self.critic_optimizer.step()

        for to_model, from_model in zip(self.target_critic.parameters(), self.critic.parameters()):
            to_model.data.copy_(self.tau * from_model.data + (1 - self.tau) * to_model.data)

        return np.array([critic_loss.detach().cpu().item()], dtype=np.float32)

    def save(self, path, file_name):
        torch.save(self.critic.state_dict(), os.path.join(path, file_name + "_critic.pt"))
        torch.save(
            self.critic_optimizer.state_dict(),
            os.path.join(path, file_name + "_critic_optimizer.pt"),
        )

    def load(self, path, file_name):
        self.critic.load_state_dict(torch.load(os.path.join(path, file_name + "_critic.pt")))
        self.critic_optimizer.load_state_dict(
            torch.load(os.path.join(path, file_name + "_critic_optimizer.pt"))
        )
        self.target_critic = copy.deepcopy(self.critic)

