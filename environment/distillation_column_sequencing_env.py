import numpy as np

from .column_sequencing_env import HydrocarbonDistillationEnv


class DistillationColumnSequencingEnv:
    """
    Distillation sequencing wrapper for Trainer interface.

    Action (scaled, shape (2, 1)):
      action[0]: split index mapped to [-1, 1] for physical [0, 4]
      action[1]: reflux ratio mapped to [-1, 1] for physical [reflux_low, reflux_high]
    """

    env_name = "DistillationColumnSequencing"
    max_split_actions = 5

    def __init__(self, config):
        self.config = config
        self.reflux_low = float(config.get("reflux_low", 1.05))
        self.reflux_high = float(config.get("reflux_high", 2.0))

        self.s_dim = 6
        self.a_dim = 2
        self.o_dim = 1
        self.p_dim = 1

        self.nT = 5
        self.dt = 1.0
        self.t0 = 0.0
        self.tT = 5.0

        self.xmin = np.zeros((self.s_dim, 1), dtype=np.float32)
        self.xmax = np.ones((self.s_dim, 1), dtype=np.float32)
        self.umin = np.array([[0.0], [self.reflux_low]], dtype=np.float32)
        self.umax = np.array([[4.0], [self.reflux_high]], dtype=np.float32)
        self.ymin = np.array([[0.0]], dtype=np.float32)
        self.ymax = np.array([[1.0]], dtype=np.float32)
        self.zero_center_scale = True

        self.need_derivs = False
        self.need_noise_derivs = False
        self.need_deriv_inverse = False

        seed = int(config.get("seed", 0))
        self.inner_env = HydrocarbonDistillationEnv(seed=seed)
        self._done = False

        self.plot_info = {
            "ref_idx_lst": [],
            "state_plot_shape": (2, 3),
            "state_plot_idx_lst": range(0, 6),
            "action_plot_shape": (1, 2),
            "variable_tag_lst": [
                "Step",
                r"$z_{C2}$",
                r"$z_{C3}$",
                r"$z_{iC4}$",
                r"$z_{nC4}$",
                r"$z_{iC5}$",
                r"$z_{nC5}$",
                r"Split index",
                r"$R/R_{min}$",
            ],
        }

    def reset(self, x0=None, random_init=False):
        state_raw = self.inner_env.reset()
        self._done = False
        state = self.scale(state_raw.reshape(-1, 1), self.xmin, self.xmax)
        u0 = np.zeros((self.a_dim, 1), dtype=np.float32)
        return state.astype(np.float32), u0

    def step(self, state, action):
        if self._done:
            return state, np.zeros((1, 1), dtype=np.float32), True, None

        action = np.asarray(action, dtype=np.float32).reshape(self.a_dim, 1)
        action_phys = self.descale(action, self.umin, self.umax)

        split_idx = int(np.clip(np.round(action_phys[0, 0]), 0, self.max_split_actions - 1))
        reflux_ratio = float(np.clip(action_phys[1, 0], self.reflux_low, self.reflux_high))

        split_mask = self.get_split_mask_for_current_stream()
        valid_split = int(split_mask.sum())
        if valid_split <= 0:
            split_idx = 0
        elif split_idx >= valid_split:
            split_idx = 0

        next_raw, reward, done, info = self.inner_env.step(split_idx, reflux_ratio)
        self._done = bool(done)

        next_state = self.scale(next_raw.reshape(-1, 1), self.xmin, self.xmax).astype(np.float32)
        cost = np.array([[-reward]], dtype=np.float32)

        if info is None:
            info = {}
        info = dict(info)
        info["split_mask"] = split_mask.copy()
        info["valid_split_count"] = valid_split

        return next_state, cost, self._done, info

    def get_episode_summary(self):
        return self.inner_env.get_episode_summary()

    def ref_traj(self):
        return np.array([0.0], dtype=np.float32)

    def get_observ(self, state, action):
        return state[: self.o_dim]

    @classmethod
    def split_mask_from_normalized_state(cls, state_normalized):
        flows = np.asarray(state_normalized, dtype=np.float32).reshape(-1)
        flows = np.clip(flows, 0.0, 1.0)

        stream_total = float(np.sum(flows))
        if stream_total <= 1e-12:
            mask = np.zeros((cls.max_split_actions,), dtype=np.float32)
            return mask, np.array([], dtype=np.int64), 0

        threshold = stream_total * 0.01
        active_components = np.where(flows > threshold)[0]
        n_active = int(len(active_components))
        valid_splits = max(0, n_active - 1)

        mask = np.zeros((cls.max_split_actions,), dtype=np.float32)
        if valid_splits > 0:
            mask[:valid_splits] = 1.0

        return mask, active_components.astype(np.int64), valid_splits

    @classmethod
    def split_mask_from_scaled_state(cls, state_scaled):
        state_scaled = np.asarray(state_scaled, dtype=np.float32).reshape(-1)
        normalized = np.clip(0.5 * (state_scaled + 1.0), 0.0, 1.0)
        return cls.split_mask_from_normalized_state(normalized)

    def get_split_mask_from_state(self, state_scaled):
        mask, _, _ = self.split_mask_from_scaled_state(state_scaled)
        return mask

    def get_split_mask_for_current_stream(self):
        if len(self.inner_env.stream_queue) == 0:
            return np.zeros((self.max_split_actions,), dtype=np.float32)

        current = self.inner_env.stream_queue[0]
        flows = current["flows"] / max(self.inner_env.feed_total, 1e-8)
        mask, _, _ = self.split_mask_from_normalized_state(flows)
        return mask

    def scale(self, var, min_val, max_val, shift=True):
        if self.zero_center_scale:
            shifting = (max_val + min_val) if shift else 0.0
            return (2.0 * var - shifting) / (max_val - min_val + 1e-8)
        shifting = min_val if shift else 0.0
        return (var - shifting) / (max_val - min_val + 1e-8)

    def descale(self, scaled_var, min_val, max_val):
        if self.zero_center_scale:
            return (max_val - min_val) / 2.0 * scaled_var + (max_val + min_val) / 2.0
        return (max_val - min_val) * scaled_var + min_val

