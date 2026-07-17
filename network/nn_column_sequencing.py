import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


class CriticMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim_lst, activation_function):
        super().__init__()
        self.fc_lst = nn.ModuleList()
        current_dim = input_dim
        for hidden_dim in hidden_dim_lst:
            self.fc_lst.append(nn.Linear(current_dim, hidden_dim))
            current_dim = hidden_dim
        self.last_fc = nn.Linear(current_dim, output_dim)
        self.activation_function = activation_function

    def forward(self, inputs):
        h = inputs
        for fc in self.fc_lst:
            h = self.activation_function(fc(h))
        return self.last_fc(h)


class HybridActorMLP(nn.Module):
    """
    Shared trunk + discrete split head + continuous reflux head.
    """

    def __init__(self, input_dim, n_split_logits, hidden_dim_lst, activation_function, cont_dim=1):
        super().__init__()
        self.n_split_logits = n_split_logits
        self.cont_dim = cont_dim

        self.fc_lst = nn.ModuleList()
        current_dim = input_dim
        for hidden_dim in hidden_dim_lst:
            self.fc_lst.append(nn.Linear(current_dim, hidden_dim))
            current_dim = hidden_dim

        self.split_head = nn.Linear(current_dim, n_split_logits)
        self.cont_mean_head = nn.Linear(current_dim, cont_dim)
        self.cont_logstd_head = nn.Linear(current_dim, cont_dim)
        self.activation_function = activation_function

        self.log_std_min = -10.0
        self.log_std_max = 1.0

    def _forward_shared(self, inputs):
        h = inputs
        for fc in self.fc_lst:
            h = self.activation_function(fc(h))
        return h

    def _mask_logits(self, logits, split_mask):
        if split_mask is None:
            return logits

        mask = split_mask.to(logits.device).float()
        # Ensure at least one valid action per row to avoid invalid Categorical.
        empty_rows = (mask.sum(dim=1, keepdim=True) <= 0.0).float()
        if empty_rows.any():
            fallback = torch.zeros_like(mask)
            fallback[:, 0] = 1.0
            mask = torch.where(empty_rows > 0, fallback, mask)

        invalid = (mask <= 0.0)
        masked_logits = logits.masked_fill(invalid, -1e9)
        return masked_logits

    def _continuous_distribution(self, h):
        mean = self.cont_mean_head(h)
        log_std = self.cont_logstd_head(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return Normal(mean, std)

    def sample(self, inputs, split_mask=None, deterministic=False, reparam_trick=True, return_log_prob=True):
        h = self._forward_shared(inputs)

        logits = self.split_head(h)
        masked_logits = self._mask_logits(logits, split_mask)
        split_dist = Categorical(logits=masked_logits)

        if deterministic:
            split_idx = torch.argmax(masked_logits, dim=1)
        else:
            split_idx = split_dist.sample()

        cont_dist = self._continuous_distribution(h)
        if deterministic:
            z = cont_dist.mean
        elif reparam_trick:
            z = cont_dist.rsample()
        else:
            z = cont_dist.sample()
        cont_action = torch.tanh(z)

        split_log_prob = None
        cont_log_prob = None
        if return_log_prob:
            split_log_prob = split_dist.log_prob(split_idx).unsqueeze(-1)
            cont_log_prob = cont_dist.log_prob(z) - torch.log(1.0 - cont_action.pow(2) + 1e-7)

        return split_idx.unsqueeze(-1), cont_action, split_log_prob, cont_log_prob, masked_logits

    def get_log_prob(self, inputs, split_indices, cont_actions, split_mask=None):
        h = self._forward_shared(inputs)

        logits = self.split_head(h)
        masked_logits = self._mask_logits(logits, split_mask)
        split_dist = Categorical(logits=masked_logits)
        split_log_prob = split_dist.log_prob(split_indices).unsqueeze(-1)

        cont_dist = self._continuous_distribution(h)
        cont_actions = torch.clamp(cont_actions, -0.999999, 0.999999)
        z = 0.5 * torch.log((1.0 + cont_actions) / (1.0 - cont_actions))
        cont_log_prob = cont_dist.log_prob(z) - torch.log(1.0 - cont_actions.pow(2) + 1e-7)

        return split_log_prob, cont_log_prob


class JointQMLP(nn.Module):
    """
    Q(s, joint_action_idx) over split x reflux_bin Cartesian product.
    """

    def __init__(self, input_dim, n_split_logits, n_reflux_bins, hidden_dim_lst, activation_function):
        super().__init__()
        self.n_split_logits = n_split_logits
        self.n_reflux_bins = n_reflux_bins

        output_dim = n_split_logits * n_reflux_bins
        self.q_net = CriticMLP(input_dim, output_dim, hidden_dim_lst, activation_function)

    def forward(self, inputs):
        return self.q_net(inputs)

