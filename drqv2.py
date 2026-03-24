# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class DrQV2Agent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb,
                 use_consistency=False, consistency_alpha=0.1,
                 use_contrastive=False, contrastive_alpha=0.1,
                 contrastive_tau=0.1, contrastive_epsilon=5.0):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        # auxiliary losses
        self.use_consistency = use_consistency
        self.consistency_alpha = consistency_alpha
        self.use_contrastive = use_contrastive
        self.contrastive_alpha = contrastive_alpha
        self.contrastive_tau = contrastive_tau
        self.contrastive_epsilon = contrastive_epsilon

        # models
        self.encoder = Encoder(obs_shape).to(device)
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)

        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step,
                      consistency_loss=0.0, contrastive_loss=0.0):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        # add auxiliary losses
        total_loss = critic_loss
        if self.use_consistency and torch.is_tensor(consistency_loss):
            total_loss = total_loss + self.consistency_alpha * consistency_loss
            if self.use_tb:
                metrics['consistency_loss'] = consistency_loss.item()
        if self.use_contrastive and torch.is_tensor(contrastive_loss):
            total_loss = total_loss + self.contrastive_alpha * contrastive_loss
            if self.use_tb:
                metrics['contrastive_loss'] = contrastive_loss.item()

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()
            if total_loss is not critic_loss:
                metrics['total_critic_loss'] = total_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        total_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def compute_contrastive_loss(self, obs_encoded, action):
        """Modification B: Task-conditional contrastive loss.

        Positive pairs are defined by behavioral similarity:
        (i, j) is positive if |Q(si,ai) - Q(sj,aj)| < epsilon.
        """
        B = obs_encoded.shape[0]

        # Mine positive pairs using Q-values (no grad - just for mining)
        with torch.no_grad():
            q1, q2 = self.critic(obs_encoded.detach(), action)
            q_values = torch.min(q1, q2).squeeze(-1)  # (B,)
            q_diff = torch.abs(q_values.unsqueeze(0) - q_values.unsqueeze(1))
            positive_mask = (q_diff < self.contrastive_epsilon).float()
            positive_mask.fill_diagonal_(0)

        # Normalize representations for cosine similarity
        z = F.normalize(obs_encoded, dim=-1)
        sim_matrix = torch.mm(z, z.t()) / self.contrastive_tau  # (B, B)

        # Numerical stability
        sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True).values.detach()

        # Mask out self-similarity
        self_mask = torch.eye(B, device=obs_encoded.device).bool()
        sim_matrix.masked_fill_(self_mask, -1e9)

        # InfoNCE: average log_softmax over positive pairs
        log_prob = F.log_softmax(sim_matrix, dim=1)
        num_positives = positive_mask.sum(dim=1)
        has_positives = num_positives > 0

        if has_positives.sum() == 0:
            return torch.tensor(0.0, device=obs_encoded.device)

        loss = -(positive_mask * log_prob).sum(dim=1)
        loss = loss[has_positives] / num_positives[has_positives]
        return loss.mean()

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # keep raw pixels for auxiliary losses
        obs_pixels = obs.float()
        action = action.float()
        reward = reward.float()
        discount = discount.float()

        # primary augmentation (view 1)
        obs_aug1 = self.aug(obs_pixels.clone())
        next_obs = self.aug(next_obs.float())

        # Modification A: stop-gradient consistency regularizer
        consistency_loss = 0.0
        if self.use_consistency:
            obs_aug2 = self.aug(obs_pixels.clone())
            z1 = self.encoder(obs_aug1)
            z2 = self.encoder(obs_aug2)
            # L = ||z2 - sg[z1]||^2  (gradients flow only through z2)
            consistency_loss = F.mse_loss(z2, z1.detach())
            obs = z1
        else:
            obs = self.encoder(obs_aug1)

        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        # Modification B: task-conditional contrastive loss
        contrastive_loss = 0.0
        if self.use_contrastive:
            contrastive_loss = self.compute_contrastive_loss(obs, action)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic (with auxiliary losses)
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step,
                               consistency_loss=consistency_loss,
                               contrastive_loss=contrastive_loss))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
