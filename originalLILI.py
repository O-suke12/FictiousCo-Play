import os

import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
from torch.distributions import Categorical


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.state_latent = []
        self.tau = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.state_latent[:]
        del self.tau[:]


class Encoder(nn.Module):
    def __init__(self, input_dim, z_dim):
        super().__init__()
        self.hidden_dim = 128
        self.lr = nn.Linear(input_dim, self.hidden_dim)
        self.lr2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lr3 = nn.Linear(self.hidden_dim, z_dim)  # log(sigma^2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.lr(x)
        x = self.relu(x)
        x = self.lr2(x)
        x = self.relu(x)
        x = nn.Dropout(p=0.5)(x)
        z = self.lr3(x)

        return z


class Decoder(nn.Module):
    def __init__(self, state_dim, action_dim, input_dim, output_dim):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = 128
        self.lr = nn.Linear(input_dim, self.hidden_dim)
        self.lr2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.reward_mu = nn.Linear(self.hidden_dim, 1)
        self.reward_log_var = nn.Linear(self.hidden_dim, 1)
        self.state_mu = nn.Linear(self.hidden_dim, state_dim)
        self.state_log_var = nn.Linear(self.hidden_dim, state_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.lr(x)
        x = self.relu(x)
        x = self.lr2(x)
        x = self.relu(x)
        x = nn.Dropout(p=0.5)(x)

        reward_mu = self.reward_mu(x)
        reward_log_var = self.reward_log_var(x)
        state_mu = self.state_mu(x)
        state_log_var = self.state_log_var(x)

        reward_vars = torch.exp(reward_log_var)
        state_vars = torch.exp(state_log_var)
        reward_dist = dist.Normal(reward_mu, reward_vars)
        state_dist = dist.Normal(state_mu, state_vars)

        return reward_dist, state_dist


class ActorCritic(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        z_dim,
    ):
        super(ActorCritic, self).__init__()
        input_dim = state_dim + z_dim
        hidden_dim = 256

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1),
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class LILI:
    def __init__(
        self,
        state_dim,
        action_dim,
        z_dim,
        device,
        cfg,
        **kwargs,
    ):
        self.cfg = cfg
        self.run = kwargs.get("run")
        self.state_dim = state_dim
        self.z_dim = z_dim
        self.gamma = cfg.gamma
        self.K_epochs = cfg.K_epochs
        self.eps_clip = cfg.eps_clip
        self.buffer = RolloutBuffer()
        self.device = device
        self.policy = ActorCritic(state_dim, action_dim, z_dim).to(device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": cfg.lr_actor},
                {"params": self.policy.critic.parameters(), "lr": cfg.lr_critic},
            ]
        )
        encoder_input_dim = state_dim + 1 + 1 + state_dim
        decoder_input_dim = state_dim + 1 + z_dim
        decoder_output_dim = 1 + action_dim
        self.encoder = Encoder(encoder_input_dim, z_dim).to(self.device)
        self.decoder = Decoder(
            state_dim, action_dim, decoder_input_dim, decoder_output_dim
        ).to(self.device)
        self.policy_old = ActorCritic(state_dim, action_dim, z_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.ed_optimizer = torch.optim.Adam(
            [
                {"params": self.encoder.parameters(), "lr": 0.001},
                {"params": self.decoder.parameters(), "lr": 0.001},
            ]
        )
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=0.001)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=0.001)

        self.MseLoss = nn.MSELoss()

    def make_one_tau(self, state):
        state_t_1 = self.buffer.states[-1].clone().detach().float().unsqueeze(0)
        action_t_1 = (
            self.buffer.actions[-1].clone().detach().float().unsqueeze(0).unsqueeze(0)
        )
        reward_t_1 = (
            (torch.tensor(self.buffer.rewards[-1]).float().to(self.device))
            .unsqueeze(0)
            .unsqueeze(0)
        )
        state_t = torch.tensor(state).float().unsqueeze(0).to(self.device)
        tau = torch.cat(
            (
                state_t_1,
                action_t_1,
                reward_t_1,
                state_t,
            ),
            dim=1,
        )
        return tau

    def select_action(self, state):
        with torch.no_grad():
            if len(self.buffer.states) > 1:
                tau = self.make_one_tau(state)
                self.buffer.tau.append(tau)
                z = self.get_latent_strategies(tau)
                z = z.cpu().detach().numpy()
            else:
                z = np.zeros((1, self.z_dim))

            state_latent = np.concatenate((state, z[0]))
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob, state_val = self.policy_old.act(state)

        self.buffer.state_latent.append(state[self.state_dim :])
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.item()

    def just_select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob, state_val = self.policy_old.act(state)
        return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = (
            torch.squeeze(torch.stack(self.buffer.states, dim=0))
            .detach()
            .to(self.device)
        )
        old_actions = (
            torch.squeeze(torch.stack(self.buffer.actions, dim=0))
            .detach()
            .to(self.device)
        )
        old_logprobs = (
            torch.squeeze(torch.stack(self.buffer.logprobs, dim=0))
            .detach()
            .to(self.device)
        )
        old_state_values = (
            torch.squeeze(torch.stack(self.buffer.state_values, dim=0))
            .detach()
            .to(self.device)
        )

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        total_loss = 0
        total_J_rep = 0
        for _ in range(self.K_epochs):
            actions_t = old_actions[1:-1].unsqueeze(1).detach()
            tau = torch.cat(self.buffer.tau, 0)
            tau = torch.cat((tau, actions_t), 1)
            self.predicted_rewards = (
                torch.tensor(self.buffer.rewards[2:], dtype=torch.float32)
                .unsqueeze(1)
                .detach()
                .to(self.device)
            )

            reward_dist, state_dist = self.encoder_decoder(tau)
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            # final loss of clipped objective PPO
            critique_loss = self.MseLoss(state_values, rewards)
            loss = -torch.min(surr1, surr2) + 0.5 * critique_loss - 0.01 * dist_entropy

            state_loss = -state_dist.log_prob(old_states[1:-1]).mean()
            reward_loss = -reward_dist.log_prob(rewards[1:-1]).mean()
            J_rep = state_loss + reward_loss
            # encoder_loss = critique_loss - J_rep

            # self.encoder_optimizer.zero_grad()
            # encoder_loss.backward()
            # self.encoder_optimizer.step()

            # self.coder_optimizer.zero_grad()
            # J_rep.backward()
            # self.ed_optimizer.step()

            self.ed_optimizer.zero_grad()
            J_rep.backward()
            self.ed_optimizer.step()

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            total_J_rep += J_rep
            total_loss += loss.mean()

        if self.cfg is not None and self.cfg.track:
            if self.cfg.track:
                self.run.log(
                    {
                        "loss": total_loss.item(),
                        "J_rep": total_J_rep.item(),
                    }
                )
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def ed_update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = (
            torch.squeeze(torch.stack(self.buffer.states, dim=0))
            .detach()
            .to(self.device)
        )
        old_actions = (
            torch.squeeze(torch.stack(self.buffer.actions, dim=0))
            .detach()
            .to(self.device)
        )
        old_logprobs = (
            torch.squeeze(torch.stack(self.buffer.logprobs, dim=0))
            .detach()
            .to(self.device)
        )
        old_state_values = (
            torch.squeeze(torch.stack(self.buffer.state_values, dim=0))
            .detach()
            .to(self.device)
        )

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs

        total_J_rep = 0
        for _ in range(self.K_epochs):
            actions_t = old_actions[1:-1].unsqueeze(1).detach()
            tau = torch.cat(self.buffer.tau, 0)
            tau = torch.cat((tau, actions_t), 1)
            self.predicted_rewards = (
                torch.tensor(self.buffer.rewards[2:], dtype=torch.float32)
                .unsqueeze(1)
                .detach()
                .to(self.device)
            )

            reward_dist, state_dist = self.encoder_decoder(tau)
            # Evaluating old actions and values
            # with torch.no_grad():
            #     logprobs, state_values, dist_entropy = self.policy.evaluate(
            #         old_states, old_actions
            #     )

            #     # match state_values tensor dimensions with rewards tensor
            #     state_values = torch.squeeze(state_values)

            #     # Finding the ratio (pi_theta / pi_theta__old)
            #     ratios = torch.exp(logprobs - old_logprobs.detach())

            #     critique_loss = self.MseLoss(state_values, rewards)
            state_loss = -state_dist.log_prob(old_states[1:-1]).mean()
            reward_loss = -reward_dist.log_prob(rewards[1:-1]).mean()
            J_rep = state_loss + reward_loss

            self.ed_optimizer.zero_grad()
            J_rep.backward()
            self.ed_optimizer.step()

            total_J_rep += J_rep

        if self.cfg is not None and self.cfg.track:
            if self.cfg.track:
                self.run.log(
                    {
                        "pretrain_J_rep": total_J_rep.item(),
                    }
                )
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
        torch.save(
            self.encoder.state_dict(), checkpoint_path.replace("lili", "encoder")
        )
        torch.save(
            self.decoder.state_dict(), checkpoint_path.replace("lili", "decoder")
        )

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )
        self.policy.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )

    def load_ed(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            self.encoder.load_state_dict(
                torch.load(
                    checkpoint_path.replace("lili", "encoder"),
                    map_location=lambda storage, loc: storage,
                )
            )
            self.decoder.load_state_dict(
                torch.load(
                    checkpoint_path.replace("lili", "decoder"),
                    map_location=lambda storage, loc: storage,
                )
            )
            print("Encoder-Decoder loaded successfully.")
        else:
            print("Checkpoint path does not exist.")
            exit(1)

    def get_latent_strategies(self, tau):
        return self.encoder(tau)

    def encoder_decoder(self, tau):
        z = self.encoder(tau[:, :-1])
        decoder_input = torch.cat((z, tau[:, -(self.state_dim + 1) :]), 1)
        reward_dist, state_dist = self.decoder(decoder_input)
        return reward_dist, state_dist
