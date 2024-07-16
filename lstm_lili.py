import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.taus = []
        self.predicted_states = []
        self.predicted_rewards = []
        self.hidden = None
        self.latents = []
        self.another_actions = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.state_all[:]
        del self.taus[:]
        del self.predicted_states[:]
        del self.predicted_rewards[:]
        del self.latents[:]
        del self.another_actions[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim):
        super(ActorCritic, self).__init__()
        input_dim = state_dim + latent_dim
        layer_dim1 = 16
        layer_dim2 = 8

        self.actor = nn.Sequential(
            nn.Linear(input_dim, layer_dim1),
            nn.Tanh(),
            nn.Linear(layer_dim1, layer_dim2),
            nn.Tanh(),
            nn.Linear(layer_dim2, action_dim),
            nn.Softmax(dim=-1),
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(input_dim, layer_dim1),
            nn.Tanh(),
            nn.Linear(layer_dim1, layer_dim2),
            nn.Tanh(),
            nn.Linear(layer_dim2, 1),
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
        # self.reward_layer = nn.Linear(self.hidden_dim, 1)

    def forward(self, tau, hidden):
        output, hidden = self.lstm(tau, hidden)
        next_state = self.state_layer(output)
        # next_reward = self.reward_layer(output)
        return next_state, hidden


class LILIEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, latent_size)

    def forward(self, x, hidden):
        _, hidden = self.lstm(x, hidden)
        latent = self.fc(hidden[1])
        return latent, hidden


class LILIDecoder(nn.Module):
    def __init__(self, state_size, latent_size, hidden_size, output_size):
        super().__init__()
        input_dim = state_size + latent_size
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, z):
        x = self.relu(self.fc1(z))
        output = self.softmax(self.fc2(x))
        return output


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
        self.latent_dim = 8
        self.cfg = cfg
        self.run = kwargs.get("run")
        self.state_dim = state_dim
        self.gamma = cfg.gamma
        self.K_epochs = cfg.K_epochs
        self.eps_clip = cfg.eps_clip
        self.buffer = RolloutBuffer()
        self.device = device
        self.policy = ActorCritic(state_dim, action_dim, self.latent_dim).to(device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": cfg.lr_actor},
                {"params": self.policy.critic.parameters(), "lr": cfg.lr_critic},
            ]
        )
        self.policy_old = ActorCritic(state_dim, action_dim, self.latent_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

        # self.lp_input_dim = state_dim + 1 + state_dim + 1
        # self.latent_predictor = Latent_Predictor(
        #     self.lp_input_dim, hidden_dim, state_dim
        # ).to(self.device)
        # self.hidden = None
        # self.lp_state_loss = nn.MSELoss()
        # self.lp_reward_loss = nn.MSELoss()
        # self.hidden_optimizer = torch.optim.Adam(
        #     self.latent_predictor.parameters(), lr=0.001
        # )

        self.hidden_dim = hidden_dim

        # self.encoder_input_dim = state_dim + state_dim + state_dim + 1
        self.encoder_input_dim = 10 + 10 + 10 + 1
        self.encoder = LILIEncoder(
            self.encoder_input_dim, hidden_dim, self.latent_dim
        ).to(device)
        self.decoder = LILIDecoder(state_dim, self.latent_dim, 8, 5).to(device)
        self.ed_optimizer = torch.optim.Adam(
            [
                {"params": self.encoder.parameters(), "lr": 0.01},
                {"params": self.decoder.parameters(), "lr": 0.01},
            ]
        )

    def arrange_state(self, state):
        arranged_state = []
        self_ab_pos = state[0][0:2]
        another_rel_pos = state[0][10:12]
        another_ab_pos = self_ab_pos + another_rel_pos
        arranged_state.append(another_ab_pos)

        self_rel_pos = self_ab_pos - another_ab_pos
        arranged_state.append(self_rel_pos)

        for i in range(3):
            land_rel_pos = state[0][4 + 2 * i : 6 + 2 * i] - another_ab_pos
            arranged_state.append(land_rel_pos)
        return torch.concat(arranged_state).unsqueeze(0)

    def make_one_tau(self, state, end):
        state_t_2 = self.buffer.states[-2].clone().detach().float().unsqueeze(0)
        state_t_1 = self.buffer.states[-1].clone().detach().float().unsqueeze(0)
        # action_t_1 = (
        #     self.buffer.actions[-1].clone().detach().float().unsqueeze(0).unsqueeze(0)
        # )
        # reward_t_1 = (
        #     (torch.tensor(self.buffer.rewards[-1]).float().to(self.device))
        #     .unsqueeze(0)
        #     .unsqueeze(0)
        # )
        state_t = torch.tensor(state).float().unsqueeze(0).to(self.device)

        state_t_2 = self.arrange_state(state_t_2)
        state_t_1 = self.arrange_state(state_t_1)
        state_t = self.arrange_state(state_t)

        end = torch.tensor(end).float().unsqueeze(0).unsqueeze(0).to(self.device)

        # tau = torch.cat(
        #     (state_t_1, action_t_1, reward_t_1, state_t, end),
        #     dim=1,
        # )
        tau = torch.cat(
            (state_t_2, state_t_1, state_t, end),
            dim=1,
        )
        return tau

    def select_action(self, state, t, end):
        with torch.no_grad():
            if t > 2:
                tau = self.make_one_tau(state, end)
            else:
                tau = torch.zeros((self.encoder_input_dim)).unsqueeze(0)
                self.hidden = None

            latent, self.hidden = self.encoder(tau, self.hidden)
            # hidden_state = self.hidden[0][0].detach().numpy()
            # cell_state = self.hidden[1][0].detach().numpy()

            # state_hidden = np.concatenate((state, cell_state))
            self.buffer.latents.append(latent.squeeze().detach().numpy())
            state_latent = np.concatenate((state, latent.squeeze().detach().numpy()))
            state = torch.FloatTensor(state_latent).to(self.device)
            action, action_logprob, state_val = self.policy_old.act(state)

        self.buffer.taus.append(tau)

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

            # next_z, self.hidden = self.latent_predictor(tau, self.hidden)
            next_z, self.hidden = self.encoder(tau)
            cell_state = self.hidden[1][0].detach().numpy()
            state_latent = np.concatenate((state, cell_state))
            state = torch.FloatTensor(state_latent).to(self.device)
            action, action_logprob, state_val = self.policy_old.act(state)
        return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        # rewards = []
        # discounted_reward = 0
        # for reward, is_terminal in zip(
        #     reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)
        # ):
        #     if is_terminal:
        #         discounted_reward = 0
        #     discounted_reward = reward + (self.gamma * discounted_reward)
        #     rewards.insert(0, discounted_reward)

        # # Normalizing the rewards
        # rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # # convert list to tensor
        old_states = (
            torch.squeeze(torch.stack(self.buffer.states, dim=0))
            .detach()
            .to(self.device)
        )
        # old_state_all = (
        #     torch.squeeze(torch.stack(self.buffer.state_all, dim=0))
        #     .detach()
        #     .to(self.device)
        # )
        # old_actions = (
        #     torch.squeeze(torch.stack(self.buffer.actions, dim=0))
        #     .detach()
        #     .to(self.device)
        # )
        old_another_actions = (
            torch.squeeze(torch.stack(self.buffer.another_actions, dim=0))
            .detach()
            .to(self.device)
        )
        # old_logprobs = (
        #     torch.squeeze(torch.stack(self.buffer.logprobs, dim=0))
        #     .detach()
        #     .to(self.device)
        # )
        # old_state_values = (
        #     torch.squeeze(torch.stack(self.buffer.state_values, dim=0))
        #     .detach()
        #     .to(self.device)
        # )

        # calculate advantages
        # advantages = rewards.detach() - old_state_values.detach()
        # actions_t = old_actions.unsqueeze(1).detach()

        # Optimize policy for K epochs
        total_loss = 0
        total_ed_loss = 0

        taus = torch.cat(self.buffer.taus, 0)

        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            # logprobs, state_values, dist_entropy = self.policy.evaluate(
            #     old_state_all, old_actions
            # )

            # # match state_values tensor dimensions with rewards tensor
            # state_values = torch.squeeze(state_values)

            # # Finding the ratio (pi_theta / pi_theta__old)
            # ratios = torch.exp(logprobs - old_logprobs.detach())

            # # Finding Surrogate Loss
            # surr1 = ratios * advantages
            # surr2 = (
            #     torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            # )

            # # final loss of clipped objective PPO
            # critique_loss = self.MseLoss(state_values, rewards)
            # loss = -torch.min(surr1, surr2) + 0.5 * critique_loss - 0.01 * dist_entropy

            # self.optimizer.zero_grad()
            # loss.mean().backward()
            # self.optimizer.step()

            for i in range(5):
                action_dist = self.encoder_decoder(old_states, taus)
                ed_loss = F.cross_entropy(action_dist, old_another_actions)
                self.ed_optimizer.zero_grad()
                ed_loss.backward()
                self.ed_optimizer.step()

            # total_loss += loss.mean()
            total_ed_loss += ed_loss

            # self.hidden = None
            # predicted_states = []
            # # predicted_rewards = []
            # for tau in taus:
            #     predicted_state, self.hidden = self.latent_predictor(
            #         tau.unsqueeze(0), self.hidden
            #     )
            #     if tau[-1] == 1:
            #         self.hidden = None
            #     predicted_states.append(predicted_state.squeeze(0))
            #     # predicted_rewards.append(predicted_reward.squeeze(0))

            # # cut first state of each episode
            # mask = torch.ones_like(old_states)
            # mask[:: self.cfg.max_cycle + 1] = 0
            # mask = mask.bool()
            # states = torch.masked_select(old_states, mask).view(-1, self.state_dim)

            # predicted_states = torch.stack(predicted_states)
            # mask = torch.ones_like(predicted_states)
            # mask[self.cfg.max_cycle :: self.cfg.max_cycle + 1] = 0
            # mask = mask.bool()
            # predicted_states = torch.masked_select(predicted_states, mask).view(
            #     -1, self.state_dim
            # )

            # state_loss = self.lp_state_loss(predicted_states, states)

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
            # lstm_loss = state_loss

            # self.hidden_optimizer.zero_grad()
            # lstm_loss.backward()
            # self.hidden_optimizer.step()

        if self.cfg is not None and self.cfg.track:
            if self.cfg.track:
                self.run.log(
                    {
                        "loss": total_loss.item(),
                        "hidden_loss": total_ed_loss.item(),
                    }
                )
        print(total_ed_loss)
        # Copy new weights into old policy

        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

        # torch.save(
        #     self.latent_predictor.state_dict(),
        #     checkpoint_path.replace("lili_lstm", "lp"),
        # )
        torch.save(
            self.encoder.state_dict(),
            checkpoint_path.replace("lili_lstm", "ll_encoder"),
        )
        torch.save(
            self.decoder.state_dict(),
            checkpoint_path.replace("lili_lstm", "ll_decoder"),
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
            # self.latent_predictor.load_state_dict(
            #     torch.load(
            #         checkpoint_path.replace("lili_lstm", "lp"),
            #         map_location=lambda storage, loc: storage,
            #     )
            # )
            self.encoder.load_state_dict(
                torch.load(
                    checkpoint_path.replace("lili_lstm", "ll_encoder"),
                    map_location=lambda storage, loc: storage,
                )
            )
            self.decoder.load_state_dict(
                torch.load(
                    checkpoint_path.replace("lili_lstm", "ll_decoder"),
                    map_location=lambda storage, loc: storage,
                )
            )
            print("Encoder-Decoder loaded successfully.")
        else:
            print("Checkpoint path does not exist.")
            exit(1)

    def encoder_decoder(self, old_states, taus):
        self.hidden = None
        latents = []
        for tau in taus:
            latent, self.hidden = self.encoder(tau.unsqueeze(0), self.hidden)
            latents.append(latent.squeeze(0))
            if tau[-1] == 1:
                self.hidden = None
        latents = torch.stack(latents)
        state_latents = torch.concat((old_states, latents), dim=1)
        action_dist = self.decoder(state_latents)
        return action_dist
