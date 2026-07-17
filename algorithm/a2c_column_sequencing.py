import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .base_algorithm import Algorithm
from RLcodes.environment.distillation_column_sequencing_env import DistillationColumnSequencingEnv
from RLcodes.network.nn_column_sequencing import CriticMLP, HybridActorMLP
from RLcodes.utility.buffer import RolloutBuffer


class A2CColumnSequencing(Algorithm):
    def __init__(self, config):
        self.config = config
        self.device = config["device"]
        self.s_dim = config["s_dim"]
        self.a_dim = config["a_dim"]
        self.nT = config["nT"]

        self.split_action_dim = int(config.get("split_action_dim", 5))

        self.num_hidden_nodes = config["num_hidden_nodes"]
        self.num_hidden_layers = config["num_hidden_layers"]
        hidden_dim_lst = [self.num_hidden_nodes for _ in range(self.num_hidden_layers)]

        self.gamma = config["gamma"]
        self.critic_lr = config["critic_lr"]
        self.actor_lr = config["actor_lr"]
        self.adam_eps = config["adam_eps"]
        self.l2_reg = config["l2_reg"]
        self.grad_clip_mag = config["grad_clip_mag"]
        self.use_mc_return = config["use_mc_return"]

        config["buffer_size"] = self.nT
        config["batch_size"] = self.nT
        self.rollout_buffer = RolloutBuffer(config)

        self.critic = CriticMLP(self.s_dim, 1, hidden_dim_lst, F.silu).to(self.device)
        self.actor = HybridActorMLP(
            self.s_dim, self.split_action_dim, hidden_dim_lst, F.silu, cont_dim=1
        ).to(self.device)

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.adam_eps,
            weight_decay=self.l2_reg,
        )
        self.actor_optimizer = optim.RMSprop(
            self.actor.parameters(),
            lr=self.actor_lr,
            eps=self.adam_eps,
            weight_decay=self.l2_reg,
        )

        self.loss_lst = ["Critic loss", "Actor loss"]

    @staticmethod
    def _split_idx_to_scaled(split_idx):
        return split_idx.float() / 2.0 - 1.0

    @staticmethod
    def _scaled_to_split_idx(split_scaled):
        idx = torch.round((torch.clamp(split_scaled, -1.0, 1.0) + 1.0) * 2.0)
        return torch.clamp(idx, 0, 4).long()

    def _build_split_masks_from_states(self, states):
        masks = []
        state_np = states.detach().cpu().numpy()
        for s in state_np:
            mask, _, valid = DistillationColumnSequencingEnv.split_mask_from_scaled_state(s)
            if valid <= 0:
                mask = np.zeros_like(mask)
                mask[0] = 1.0
            masks.append(mask)
        return torch.tensor(np.asarray(masks), dtype=torch.float32, device=self.device)

    def ctrl(self, state):
        with torch.no_grad():
            state_tensor = torch.tensor(state.T, dtype=torch.float32, device=self.device)
            split_mask, _, valid = DistillationColumnSequencingEnv.split_mask_from_scaled_state(state)
            if valid <= 0:
                split_mask = np.zeros_like(split_mask)
                split_mask[0] = 1.0
            split_mask = torch.tensor(split_mask.reshape(1, -1), dtype=torch.float32, device=self.device)

            split_idx, cont_action, _, _, _ = self.actor.sample(
                state_tensor,
                split_mask=split_mask,
                deterministic=False,
                reparam_trick=False,
                return_log_prob=False,
            )

        split_scaled = self._split_idx_to_scaled(split_idx)
        action = torch.cat([split_scaled, cont_action], dim=1)
        action = np.clip(action.T.cpu().numpy(), -1.0, 1.0)
        return action

    def add_experience(self, experience):
        self.rollout_buffer.add(experience)

    def warm_up_train(self):
        pass

    def train(self):
        sample = self.rollout_buffer.sample()
        states = sample["states"]
        actions = sample["actions"]
        rewards = sample["rewards"]
        next_states = sample["next_states"]
        dones = sample["dones"]

        if self.use_mc_return:
            return_values = [rewards[-1]]
            for i in range(self.nT - 1):
                return_values.append(rewards[-i - 2] + self.gamma * return_values[-1])
            return_values.reverse()
            target_values = torch.stack(return_values)
        else:
            with torch.no_grad():
                target_values = rewards + self.gamma * self.critic(next_states) * (1 - dones)

        current_values = self.critic(states)
        critic_loss = F.mse_loss(current_values, target_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_mag)
        self.critic_optimizer.step()

        advantages = target_values - current_values.detach()

        split_scaled = actions[:, 0:1]
        split_indices = self._scaled_to_split_idx(split_scaled).squeeze(-1)
        cont_actions = torch.clamp(actions[:, 1:2], -1.0, 1.0)
        split_masks = self._build_split_masks_from_states(states)

        split_log_prob, cont_log_prob = self.actor.get_log_prob(
            states,
            split_indices=split_indices,
            cont_actions=cont_actions,
            split_mask=split_masks,
        )
        log_prob_traj = split_log_prob + cont_log_prob
        actor_loss = (log_prob_traj * advantages).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_mag)
        self.actor_optimizer.step()

        self.rollout_buffer.reset()
        return np.array(
            [critic_loss.detach().cpu().item(), actor_loss.detach().cpu().item()],
            dtype=np.float32,
        )

    def save(self, path, file_name):
        torch.save(self.critic.state_dict(), os.path.join(path, file_name + "_critic.pt"))
        torch.save(
            self.critic_optimizer.state_dict(),
            os.path.join(path, file_name + "_critic_optimizer.pt"),
        )
        torch.save(self.actor.state_dict(), os.path.join(path, file_name + "_actor.pt"))
        torch.save(
            self.actor_optimizer.state_dict(),
            os.path.join(path, file_name + "_actor_optimizer.pt"),
        )

    def load(self, path, file_name):
        self.critic.load_state_dict(torch.load(os.path.join(path, file_name + "_critic.pt")))
        self.critic_optimizer.load_state_dict(
            torch.load(os.path.join(path, file_name + "_critic_optimizer.pt"))
        )
        self.actor.load_state_dict(torch.load(os.path.join(path, file_name + "_actor.pt")))
        self.actor_optimizer.load_state_dict(
            torch.load(os.path.join(path, file_name + "_actor_optimizer.pt"))
        )

