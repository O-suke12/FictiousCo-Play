import os

import numpy as np
import torch
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
        self.state_all = []
        self.tau = []
        self.predicted_states = []
        self.predicted_rewards = []
        self.hidden = None

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.state_all[:]
        del self.tau[:]
        del self.predicted_states[:]
        del self.predicted_rewards[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorCritic, self).__init__()
        input_dim = state_dim + hidden_dim

        self.actor = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, action_dim),
            nn.Softmax(dim=-1),
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
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


class Latent_Predictor(nn.Module):
    def __init__(self, lp_input_dim, hidden_dim, state_dim):
        super(Latent_Predictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(lp_input_dim, self.hidden_dim, batch_first=True)
        self.state_layer = nn.Linear(self.hidden_dim, state_dim)

    def forward(self, tau, hidden):
        output, hidden = self.lstm(tau, hidden)
        next_state = self.state_layer(output)
        return next_state, hidden


class LILI_LSTM:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim,
        device,
        cfg,
        **kwargs,
    ):
        self.cfg = cfg
        self.run = kwargs.get("run")
        self.state_dim = state_dim
        self.gamma = cfg.gamma
        self.K_epochs = cfg.K_epochs
        self.eps_clip = cfg.eps_clip
        self.buffer = RolloutBuffer()
        self.device = device
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": cfg.lr_actor},
                {"params": self.policy.critic.parameters(), "lr": cfg.lr_critic},
            ]
        )
        self.policy_old = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

        self.hidden_dim = hidden_dim
        self.lp_input_dim = state_dim + 1 + 1 + state_dim + 1
        self.latent_predictor = Latent_Predictor(
            self.lp_input_dim, hidden_dim, state_dim
        ).to(self.device)
        self.hidden = None
        self.lp_state_loss = nn.MSELoss()
        self.lp_reward_loss = nn.MSELoss()
        self.hidden_optimizer = torch.optim.Adam(
            self.latent_predictor.parameters(), lr=0.001
        )

    def make_one_tau(self, state, end):
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
        end = torch.tensor(end).float().unsqueeze(0).unsqueeze(0).to(self.device)

        tau = torch.cat(
            (state_t_1, action_t_1, reward_t_1, state_t, end),
            dim=1,
        )
        return tau

    def select_action(self, state, t, end):
        with torch.no_grad():
            if t > 1:
                tau = self.make_one_tau(state, end)
            else:
                tau = torch.zeros((self.lp_input_dim)).unsqueeze(0)
                self.hidden = None

            _, self.hidden = self.latent_predictor(tau, self.hidden)
            cell_state = self.hidden[1][0].detach().numpy()
            state_hidden = np.concatenate((state, cell_state))
            state = torch.FloatTensor(state_hidden).to(self.device)
            action, action_logprob, state_val = self.policy_old.act(state)

        self.buffer.tau.append(tau)
        # self.buffer.predicted_states.append(next_state)
        # self.buffer.predicted_rewards.append(next_reward)
        self.buffer.state_all.append(state)
        self.buffer.states.append(state[: self.state_dim])
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.item()

    def just_select_action(self, state, t):
        with torch.no_grad():
            if t > 1:
                tau = self.make_one_tau(state)
            else:
                tau = torch.zeros((1, self.state_dim + 1 + 1 + self.state_dim))
                self.hidden = None

            next_z, self.hidden = self.latent_predictor(tau, self.hidden)
            cell_state = self.hidden[1][0].detach().numpy()
            state_latent = np.concatenate((state, cell_state))
            state = torch.FloatTensor(state_latent).to(self.device)
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
        old_state_all = (
            torch.squeeze(torch.stack(self.buffer.state_all, dim=0))
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
        total_lstm_loss = 0
        actions_t = old_actions.unsqueeze(1).detach()
        taus = torch.cat(self.buffer.tau, 0)

        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_state_all, old_actions
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

            self.hidden = None
            predicted_states = []
            for tau in taus:
                predicted_state, self.hidden = self.latent_predictor(
                    tau.unsqueeze(0), self.hidden
                )
                if tau[-1] == 1:
                    self.hidden = None
                predicted_states.append(predicted_state.squeeze(0))

            # cut first state of each episode
            mask = torch.ones_like(old_states)
            mask[:: self.cfg.max_cycle + 1] = 0
            mask = mask.bool()
            states = torch.masked_select(old_states, mask).view(-1, self.state_dim)

            predicted_states = torch.stack(predicted_states)
            mask = torch.ones_like(predicted_states)
            mask[self.cfg.max_cycle :: self.cfg.max_cycle + 1] = 0
            mask = mask.bool()
            predicted_states = torch.masked_select(predicted_states, mask).view(
                -1, self.state_dim
            )

            state_loss = self.lp_state_loss(predicted_states, states)

            # mask = torch.ones_like(rewards)
            # mask[:: self.cfg.max_cycle + 1] = 0
            # mask = mask.bool()
            # old_rewards = torch.masked_select(rewards, mask).view(-1, 1)

            # predicted_rewards = torch.stack(predicted_rewards)
            # mask = torch.ones_like(predicted_rewards)
            # mask[self.cfg.max_cycle :: self.cfg.max_cycle + 1] = 0
            # mask = mask.bool()
            # predicted_rewards = torch.masked_select(predicted_rewards, mask).view(-1, 1)

            # reward_loss = self.lp_reward_loss(predicted_rewards, old_rewards)

            lstm_loss = state_loss

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            self.hidden_optimizer.zero_grad()
            lstm_loss.backward()
            self.hidden_optimizer.step()

            total_loss += loss.mean()
            total_lstm_loss += lstm_loss

        if self.cfg is not None and self.cfg.track:
            if self.cfg.track:
                self.run.log(
                    {
                        "loss": total_loss.item(),
                        "hidden_loss": total_lstm_loss.item(),
                    }
                )
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

        torch.save(
            self.latent_predictor.state_dict(),
            checkpoint_path.replace("lili_lstm", "lp"),
        )

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )
        self.policy.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )

    def load_lp(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            self.latent_predictor.load_state_dict(
                torch.load(
                    checkpoint_path.replace("lili_lstm", "lp"),
                    map_location=lambda storage, loc: storage,
                )
            )
            print("Encoder-Decoder loaded successfully.")
        else:
            print("Checkpoint path does not exist.")
            exit(1)
