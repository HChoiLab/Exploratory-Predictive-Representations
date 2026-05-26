from typing import Optional, Tuple, Dict
import os
import numpy as np
import math

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch.distributions import Normal, Dirichlet
    from torch.distributions.kl import kl_divergence
except Exception as e:  
    raise ImportError(
        "Perception requires PyTorch. Please install it (e.g. `pip install torch torchvision`)" 
        f". Original error: {e}")

try:
    import pytorch_lightning as pl
except Exception as e:
    raise ImportError(
        "PyTorch Lightning is required for training scripts. Install with `pip install pytorch-lightning`."
        f" Original error: {e}")

import math
from typing import Optional, Tuple, Dict, Callable




class PerceptionModel(pl.LightningModule):
    """Abstract interface for perception models.

    Required methods:
    - forward(obs_seq, loc_seq) -> returns reconstructions, latent samples and posterior / prior dists
    - posterior_entropy(obs_seq, loc_seq) -> per-batch entropy (float tensor)
    - prior_entropy(states) -> entropy of the predictive prior given current state

    """

    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.lr = lr

    def forward(self, obs_seq: torch.Tensor, loc_seq: Optional[torch.Tensor] = None):
        raise NotImplementedError

    def posterior_entropy(self, obs_seq: torch.Tensor, loc_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return posterior entropy scalar per batch (or tensor with shape (batch,))."""
        raise NotImplementedError

    def prior_entropy(self, state: torch.Tensor) -> torch.Tensor:
        """Return entropy of the predictive prior for the given state."""
        raise NotImplementedError

    def predict_obs(self, obs_seq: torch.Tensor, loc_seq: Optional[torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        
        raise NotImplementedError

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)




class SequentialMazePerception(PerceptionModel):
   
    def __init__(
        self,
        num_nodes: int,
        z_dim: int,
        hidden_dim: int,
        num_actions: int = 3,
        lr: float = 1e-3,
        beta: float = 1e-3,
        reward_beta: float = 1.0,
        sparse_beta: float = 1e-3
    ):
        super().__init__(lr=lr)

        self.num_nodes = num_nodes
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.beta = beta

        
        self.rnn = nn.GRUCell(
            input_size= self.num_actions + self.z_dim,
            hidden_size=self.hidden_dim,
        )
        # --- Prior network p(z_t | h_{t-1}, a_{t-1}, x_{t-1}) ---
        
        prior_in_dim = self.hidden_dim + self.num_actions + self.num_nodes # no reward input

        self.prior_net = nn.Sequential(
            nn.Linear(prior_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * z_dim),  # -> [mu_p, logvar_p]
        )

        # --- Posterior network q(z_t | h_{t-1}, a_{t-1}, x_t) ---
        
        post_in_dim  = self.hidden_dim + self.num_actions + self.num_nodes # no reward input
        self.post_net = nn.Sequential(
            nn.Linear(post_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * z_dim),  # -> [mu_q, logvar_q]
        )

        # --- Decoder p(x_t | h_t, z_t) ---
        # Decoder must see transition context + z.
        dec_in_dim = self.hidden_dim + self.z_dim  # [h_t, z_t]
        self.decoder = nn.Sequential(
            nn.Linear(dec_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_nodes),  # categorical logits over next node
        )

        self.reward_head = nn.Sequential(
            nn.Linear(self.hidden_dim + self.z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # running class-balance stats for pos_weight
        self.register_buffer("rew_pos_count", torch.tensor(1.0))
        self.register_buffer("rew_neg_count", torch.tensor(1.0))

        # reward is binary {0,1}
        self.reward_beta = reward_beta
        self.sparse_beta = sparse_beta


    # ------------------------- utilities -------------------------

    def configure_params(self):
        # prior-only params
        prior_params = list(self.prior_net.parameters())
        # posterior-only params
        post_params = list(self.post_net.parameters())
        # shared
        other_params = (
            list(self.rnn.parameters())
            + list(self.decoder.parameters())
            + list(self.reward_head.parameters())
        )
        return [prior_params, post_params, other_params]


    @staticmethod
    def kl_diag_gauss(mu_q, logvar_q, mu_p, logvar_p) -> torch.Tensor:
        """KL(q||p) for diagonal Gaussians. Shapes (..., z_dim) -> (...,)"""
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)
        kl = 0.5 * (
            logvar_p - logvar_q
            + (var_q + (mu_q - mu_p) ** 2) / (var_p + 1e-8)
            - 1.0
        )
        return kl.sum(dim=-1)

    def _prepare_obs(self, obs_seq: torch.Tensor):
        """
        Returns:
        node_ids: (B,T) long
        x_onehot: (B,T,num_nodes) float
        """
        if obs_seq.dim() == 2:
            node_ids = obs_seq.long()
        elif obs_seq.dim() == 3 and obs_seq.size(-1) == self.num_nodes:
            node_ids = obs_seq.argmax(dim=-1)
            return node_ids, obs_seq.float()
        else:
            raise ValueError("obs_seq must be (B,T) ints or (B,T,num_nodes) one-hot")

        x_onehot = torch.nn.functional.one_hot(
            node_ids, num_classes=self.num_nodes
        ).float()

        return node_ids, x_onehot

    
    def _prepare_actions(self, act_seq: Optional[torch.Tensor], B: int, T: int, device):
        """
        act_seq expected shape (B,T) with a_t being between x_{t-1}->x_t.
        If None, we fill zeros.
        """
        if act_seq is None:
            return torch.zeros(B, T, dtype=torch.long, device=device)

        # if already one-hot: (B,T,A)
        if act_seq.dim() == 3 and act_seq.size(-1) == self.num_actions:
            return act_seq.float()

        # else assume indices: (B,T)
        return act_seq.long()


    def _action_vec(self, actions: torch.Tensor, t: int, device) -> torch.Tensor:
        # actions can be (B,T) long indices OR (B,T,A) one-hot floats
        if actions.dim() == 3:
            return actions[:, t, :]  # (B, A)

        a_idx = actions[:, t]  # (B,)
        return torch.nn.functional.one_hot(a_idx, num_classes=self.num_actions).float()

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """x and mask same shape; mask is bool."""
        m = mask.float()
        return (x * m).sum() / (m.sum() + eps)



    @torch.no_grad()
    def calculate_EIG(
        self,
        mu_p: torch.Tensor,
        logvar_p: torch.Tensor,
        a_prev: torch.Tensor,
        h_prev: torch.Tensor,
        k: int = 8,
        deterministic_z: bool = True,
    ):
        """
        Monte Carlo expected information gain:
        - sample z_p only once
        - decode to p(x_{t+1} | z_p, h_prev, a_prev)
        - sample k observations from that categorical distribution
        - compute posterior entropy / KL for sampled observations
        - average over samples
        """
        device = mu_p.device
        B = mu_p.size(0)
        std_p = torch.exp(0.5 * logvar_p)

        # ------------------------------------------------------------
        # 1) Sample ONE prior latent z_p
        # ------------------------------------------------------------
        if deterministic_z:
            z_p = mu_p
        else:
            eps = torch.randn_like(std_p)
            z_p = mu_p + eps * std_p  # (B, Z)

        # Predict next hidden state and observation distribution
        rnn_in = torch.cat([a_prev, z_p], dim=-1)
        h_pred = self.rnn(rnn_in, h_prev)

        dec_in = torch.cat([h_pred, z_p], dim=-1)
        logits_p = self.decoder(dec_in)                  # (B, num_nodes)
        p_x = torch.softmax(logits_p, dim=-1)            # (B, num_nodes)

        # ------------------------------------------------------------
        # 2) Sample k observations from decoder distribution
        # ------------------------------------------------------------
        # Categorical sampling wants probs shape (..., num_classes)
        # sample_shape=(k,) gives indices of shape (k, B)
        sampled_idx = torch.distributions.Categorical(logits=logits_p).sample((k,))

        H_q_list = []
        kl_list = []

        for idx_s in sampled_idx:
            x_i = F.one_hot(idx_s, num_classes=self.num_nodes).float()  # (B, num_nodes)

            post_in = torch.cat([h_prev, a_prev, x_i], dim=-1)
            mu_q, logvar_q = torch.chunk(self.post_net(post_in), 2, dim=-1)

            # Posterior entropy
            H_q_i = 0.5 * torch.sum(
                1.0 + math.log(2.0 * math.pi) + logvar_q,
                dim=-1
            )  # (B,)

            # KL(q || p)
            kl_i = self.kl_diag_gauss(mu_q, logvar_q, mu_p, logvar_p)  # (B,)

            H_q_list.append(H_q_i)
            kl_list.append(kl_i)

        H_q_all = torch.stack(H_q_list, dim=0)   # (k, B)
        kl_all = torch.stack(kl_list, dim=0)     # (k, B)

        # ------------------------------------------------------------
        # 3) Monte Carlo expectation over sampled observations
        # ------------------------------------------------------------
        expected_H_q = H_q_all.mean(dim=0)        # (B,)
        expected_IG_kl = kl_all.mean(dim=0)       # (B,)

        # Prior entropy
        H_p = 0.5 * torch.sum(
            1.0 + math.log(2.0 * math.pi) + logvar_p,
            dim=-1
        )  # (B,)

        EIG = H_p - expected_H_q

        return {
            "sampled_obs_idx": sampled_idx,              # (k, B)
            "p_x": p_x,                                  # (B, num_nodes)
            "expected_H_q": expected_H_q,                # (B,)
            "expected_IG_kl": expected_IG_kl,            # (B,)
            "EIG": EIG,  # (B,)
            "H_p": H_p                                   # (B,)
        }



    # ------------------------- forward -------------------------

    def forward(
        self,
        obs_seq: torch.Tensor,
        act_seq: Optional[torch.Tensor] = None,
        rew_seq: Optional[torch.Tensor] = None,
        *,
        deterministic_z: bool = True,
        h_init: Optional[torch.Tensor] = None,
        calculate_expected_info: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Outputs are for timesteps t=1..T-1, so time length is (T-1):
          logits_q: (B, T-1, num_nodes)  decoded using posterior z_q_t  (for training recon)
          logits_p: (B, T-1, num_nodes)  decoded using prior z_p_t      (for prediction eval)
          mu_q/logvar_q, mu_p/logvar_p: (B, T-1, z_dim)
          z_q/z_p: (B, T-1, z_dim)
        """
        node_ids, x_emb = self._prepare_obs(obs_seq)
        B, T = node_ids.shape
        device = obs_seq.device

        actions = self._prepare_actions(act_seq, B, T, device)

        # deterministic belief state
        if h_init is not None:
            h = h_init
        else:
            h = torch.zeros(B, self.hidden_dim, device=device)

        mu_q_list, logvar_q_list = [], []
        mu_p_list, logvar_p_list = [], []
        zq_list, zp_list = [], []
        logits_q_list, logits_p_list = [], []
        reward_logits_p_list, reward_logits_q_list = [], []
        h_list = []
        expected_H_q_list, expected_IG_kl_list, EIG_list, H_p_list = [], [], [], []

        # Predict x_t for t=1..T-1
        for t in range(1, T):
            h_prev = h
            h_list.append(h)

            x_t = x_emb[:, t, :]       
            x_prev = x_emb[:, t - 1, :] 
            a_prev = self._action_vec(actions, t, device)

            r_t = rew_seq[:, t].float().unsqueeze(-1)
            r_prev = rew_seq[:, t - 1].float().unsqueeze(-1)

            # ----- prior: does NOT use x_t -----
            prior_in = torch.cat([h_prev, a_prev, x_prev], dim=-1) # no reward input
            mu_p_t, logvar_p_t = torch.chunk(self.prior_net(prior_in), 2, dim=-1)



            if calculate_expected_info:
            # EXPECTED IG CALCULATION
                expected_dict = self.calculate_EIG(
                    mu_p=mu_p_t,
                    logvar_p=logvar_p_t,
                    a_prev=a_prev,
                    h_prev=h_prev,
                    k=10,
                    deterministic_z=True,
                )

            # ----- posterior: uses x_t -----
            post_in = torch.cat([h_prev, a_prev, x_t], dim=-1) # no reward input
            mu_q_t, logvar_q_t = torch.chunk(self.post_net(post_in), 2, dim=-1)

            # ----- sample z -----
            if deterministic_z:
                z_p_t, z_q_t = mu_p_t, mu_q_t
            else:
                std_p = torch.exp(0.5 * logvar_p_t)
                std_q = torch.exp(0.5 * logvar_q_t)
                z_p_t = mu_p_t + torch.randn_like(std_p) * std_p
                z_q_t = mu_q_t + torch.randn_like(std_q) * std_q

            # ----- deterministic transition: NO x_t here -----
            rnn_in_p = torch.cat([a_prev, z_p_t], dim=-1)
            rnn_in_q = torch.cat([a_prev, z_q_t], dim=-1)

            h_p = self.rnn(rnn_in_p, h_prev)   # predicted time-t state
            h_q = self.rnn(rnn_in_q, h_prev)   # filtered time-t state

            # ----- decode x_t from time-t states -----
            dec_in_p = torch.cat([h_p, z_p_t], dim=-1)
            dec_in_q = torch.cat([h_q, z_q_t], dim=-1)

            logits_p_t = self.decoder(dec_in_p)
            logits_q_t = self.decoder(dec_in_q)

            reward_logits_p_t = self.reward_head(dec_in_p).squeeze(-1)
            reward_logits_q_t = self.reward_head(dec_in_q).squeeze(-1)

            # carry filtered state forward
            h = h_q

            # collect
            mu_q_list.append(mu_q_t)
            logvar_q_list.append(logvar_q_t)
            mu_p_list.append(mu_p_t)
            logvar_p_list.append(logvar_p_t)
            zq_list.append(z_q_t)
            zp_list.append(z_p_t)
            logits_q_list.append(logits_q_t)
            logits_p_list.append(logits_p_t)
            reward_logits_p_list.append(reward_logits_p_t)
            reward_logits_q_list.append(reward_logits_q_t)

            if calculate_expected_info:
                expected_H_q_list.append(expected_dict["expected_H_q"])
                expected_IG_kl_list.append(expected_dict["expected_IG_kl"])
                EIG_list.append(expected_dict["EIG"])
                H_p_list.append(expected_dict["H_p"])

        # stack over time => (B, T-1, ...)
        mu_q = torch.stack(mu_q_list, dim=1)
        logvar_q = torch.stack(logvar_q_list, dim=1)
        mu_p = torch.stack(mu_p_list, dim=1)
        logvar_p = torch.stack(logvar_p_list, dim=1)
        z_q = torch.stack(zq_list, dim=1)
        z_p = torch.stack(zp_list, dim=1)
        logits_q = torch.stack(logits_q_list, dim=1)
        logits_p = torch.stack(logits_p_list, dim=1)
        reward_logits_p = torch.stack(reward_logits_p_list, dim=1)
        reward_logits_q = torch.stack(reward_logits_q_list, dim=1)

        h_list = torch.stack(h_list, dim=1)  # (B, T-1, H)
        if calculate_expected_info:
            expected_H_q = torch.stack(expected_H_q_list, dim=1)  # (B, T-1)
            expected_IG_kl = torch.stack(expected_IG_kl_list, dim=1)  # (B, T-1)
            EIG = torch.stack(EIG_list, dim=1)  # (B, T-1)
            H_p = torch.stack(H_p_list, dim=1)  # (B, T-1)


        res_dict = {
            "logits_q": logits_q,
            "logits_p": logits_p,
            "mu_q": mu_q,
            "logvar_q": logvar_q,
            "mu_p": mu_p,
            "logvar_p": logvar_p,
            "z_q": z_q,
            "z_p": z_p,
            "h": h,  # final deterministic state (B,H)
            "h_list": h_list,  # all deterministic states (B,T-1,H)
            "reward_logits_p": reward_logits_p,
            "reward_logits_q": reward_logits_q,
        }
        if calculate_expected_info:
            res_dict["expected_H_q"] = expected_H_q
            res_dict["expected_IG_kl"] = expected_IG_kl
            res_dict["EIG"] = EIG
            res_dict["H_p"] = H_p


        return res_dict

    def random_action_sampler(h, step):
        B = h.size(0)
        return torch.randint(0, 3, (B,), device=h.device)

    @torch.no_grad()
    def rollout_replay(
        self,
        h_start: torch.Tensor,
        x_prev_start: torch.Tensor,
        *,
        k: int,
        action_seq: Optional[torch.Tensor] = None,
        action_sampler: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
        deterministic_z: bool = True,
        deterministic_obs: bool = True,
        mu_p_start: Optional[torch.Tensor] = None,
        logvar_p_start: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Prior-only replay / preplay rollout.
        """
        device = h_start.device
        B = h_start.size(0)

        if x_prev_start.dim() != 2 or x_prev_start.size(-1) != self.num_nodes:
            raise ValueError(
                f"x_prev_start must have shape (B, {self.num_nodes})"
            )

        if action_seq is None and action_sampler is None:
            raise ValueError("Provide either action_seq or action_sampler")

        h = h_start
        x_prev = x_prev_start.float()

        logits_list = []
        pred_obs_list = []
        pred_node_list = []
        mu_p_list = []
        logvar_p_list = []
        z_p_list = []
        h_list = []
        reward_logits_list = []
        p_reward_list = []
        entropy_obs_list = []
        action_used_list = []

        for step in range(k):
            # -----------------------------
            # choose / prepare action
            # -----------------------------
            if action_seq is not None:
                if action_seq.dim() == 2:
                    a_idx = action_seq[:, step]  # (B,)
                    a_vec = F.one_hot(a_idx, num_classes=self.num_actions).float()
                    action_used = a_idx
                elif action_seq.dim() == 3:
                    a_vec = action_seq[:, step, :].float()  # (B, A)
                    action_used = a_vec
                else:
                    raise ValueError("action_seq must be (B,k) or (B,k,num_actions)")
            else:
                a = action_sampler(h, step)
                if a.dim() == 1:
                    a_idx = a
                    a_vec = F.one_hot(a_idx, num_classes=self.num_actions).float()
                    action_used = a_idx
                elif a.dim() == 2 and a.size(-1) == self.num_actions:
                    a_vec = a.float()
                    action_used = a_vec
                else:
                    raise ValueError("action_sampler must return (B,) or (B,num_actions)")

            action_used_list.append(action_used)

            # -----------------------------
            # prior
            # -----------------------------
            if step == 0 and mu_p_start is not None and logvar_p_start is not None:
                mu_p_t = mu_p_start
                logvar_p_t = logvar_p_start
            else:
                prior_in = torch.cat([h, a_vec, x_prev], dim=-1)
                mu_p_t, logvar_p_t = torch.chunk(self.prior_net(prior_in), 2, dim=-1)

            if deterministic_z:
                z_p_t = mu_p_t
            else:
                std_p_t = torch.exp(0.5 * logvar_p_t)
                z_p_t = mu_p_t + torch.randn_like(std_p_t) * std_p_t

            # -----------------------------
            # transition
            # -----------------------------
            rnn_in = torch.cat([a_vec, z_p_t], dim=-1)
            h = self.rnn(rnn_in, h)

            # -----------------------------
            # decode imagined observation
            # -----------------------------
            dec_in = torch.cat([h, z_p_t], dim=-1)
            logits_p_t = self.decoder(dec_in)                     # (B, num_nodes)
            reward_logits_p_t = self.reward_head(dec_in).squeeze(-1)  # (B,)

            p_obs_t = torch.softmax(logits_p_t, dim=-1)          # (B, num_nodes)
            entropy_obs_t = -(p_obs_t * torch.log(p_obs_t + 1e-8)).sum(dim=-1)

            if deterministic_obs:
                pred_node_t = torch.argmax(logits_p_t, dim=-1)
                pred_obs_t = F.one_hot(pred_node_t, num_classes=self.num_nodes).float()
            else:
                pred_obs_t = p_obs_t
                pred_node_t = torch.argmax(p_obs_t, dim=-1)

            # store
            logits_list.append(logits_p_t)
            pred_obs_list.append(pred_obs_t)
            pred_node_list.append(pred_node_t)
            mu_p_list.append(mu_p_t)
            logvar_p_list.append(logvar_p_t)
            z_p_list.append(z_p_t)
            h_list.append(h)
            reward_logits_list.append(reward_logits_p_t)
            p_reward_list.append(torch.sigmoid(reward_logits_p_t))
            entropy_obs_list.append(entropy_obs_t)

            # next step uses predicted observation
            x_prev = pred_obs_t

        def _stack(items):
            return torch.stack(items, dim=1)

        # action_used can be ints or one-hot, so stack separately
        if isinstance(action_used_list[0], torch.Tensor):
            action_used = torch.stack(action_used_list, dim=1)
        else:
            action_used = action_used_list

        return {
            "logits_p": _stack(logits_list),              # (B, k, num_nodes)
            "pred_obs": _stack(pred_obs_list),            # (B, k, num_nodes)
            "pred_node": _stack(pred_node_list),          # (B, k)
            "mu_p": _stack(mu_p_list),                    # (B, k, z_dim)
            "logvar_p": _stack(logvar_p_list),            # (B, k, z_dim)
            "z_p": _stack(z_p_list),                      # (B, k, z_dim)
            "h_roll": _stack(h_list),                     # (B, k, hidden_dim)
            "reward_logits_p": _stack(reward_logits_list),# (B, k)
            "p_reward_p": _stack(p_reward_list),          # (B, k)
            "entropy_obs": _stack(entropy_obs_list),      # (B, k)
            "action_used": action_used,
        }





