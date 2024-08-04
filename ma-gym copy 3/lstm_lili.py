import numpy as np
import torch
import torch.nn as nn
from actor_critic import ActorCritic


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
        self.other_positions = []

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
        del self.other_positions[:]


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
    def __init__(self, latent_size, prediction_steps):
        super().__init__()
        self.fc1 = nn.Linear(latent_size + 2, 16)
        self.fc2 = nn.Linear(16, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, latent, state):
        x = torch.cat([latent, state], dim=-1)
        hidden = torch.relu(self.fc1(x))
        output = self.fc2(hidden)
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
        self.ed_pretrain_count = 0
        self.latent_dim = cfg.latent_dim
        self.cfg = cfg
        self.run = kwargs.get("run")
        self.state_dim = state_dim
        self.gamma = cfg.gamma
        self.K_epochs = cfg.K_epochs
        self.eps_clip = cfg.eps_clip
        self.buffer = RolloutBuffer()
        self.device = device
        self.policy = ActorCritic(self.state_dim, action_dim, self.latent_dim).to(
            device
        )
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": cfg.lr_actor},
                {"params": self.policy.critic.parameters(), "lr": cfg.lr_critic},
            ]
        )
        self.policy_old = ActorCritic(self.state_dim, action_dim, self.latent_dim).to(
            device
        )
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

        self.criterion_mse = nn.MSELoss()  # 再構成誤差 (MSE)
        self.criterion_kld = nn.KLDivLoss(
            reduction="batchmean"
        )  # 正則化項 (KLダイバージェンス)

        self.encoder_input_dim = 2 + 2 + 1
        self.encoder = LILIEncoder(
            self.encoder_input_dim, hidden_dim, self.latent_dim
        ).to(device)
        # self.decoder = LILIDecoder(state_dim, self.latent_dim, 8, 5).to(device)
        prediction_steps = 4
        self.decoder = LILIDecoder(self.latent_dim, prediction_steps).to(device)
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

    def make_one_tau(self, self_position, other_position, end):
        # self_position_t_2 = (
        #     torch.tensor(self.buffer.states[-2][1:3]).unsqueeze(0).to(self.device)
        # )
        # self_position_t_1 = (
        #     torch.tensor(self.buffer.states[-1][1:3]).unsqueeze(0).to(self.device)
        # )
        # self_position_t = (
        #     torch.tensor(self_position).float().unsqueeze(0).to(self.device)
        # )
        # other_position_t_2 = (
        #     torch.tensor(self.buffer.other_positions[-2]).unsqueeze(0).to(self.device)
        # )
        # other_position_t_1 = (
        #     torch.tensor(self.buffer.other_positions[-1]).unsqueeze(0).to(self.device)
        # )
        # other_position_t = (
        #     torch.tensor(other_position).float().unsqueeze(0).to(self.device)
        # )

        difference_t = np.array(other_position) - np.array(self_position)
        difference_t_1 = (
            torch.tensor(self.buffer.other_positions[-1]).to(self.device)
            - self.buffer.states[-1][1:3]
        )

        difference_t = torch.tensor(difference_t).float().unsqueeze(0).to(self.device)
        difference_t_1 = difference_t_1.unsqueeze(0)

        end = torch.tensor(end).float().unsqueeze(0).unsqueeze(0).to(self.device)
        tau = torch.cat(
            (difference_t, difference_t_1, end),
            dim=1,
        )
        return tau

    def select_action(self, state, t, end):
        with torch.no_grad():
            self_position = state[1:3]
            other_position = state[-2:]
            state = state[:-2]

            if t > 2:
                tau = self.make_one_tau(self_position, other_position, end)
            else:
                tau = torch.zeros((self.encoder_input_dim)).unsqueeze(0).to(self.device)
                self.hidden = None

            latent, self.hidden = self.encoder(tau, self.hidden)
            # hidden_state = self.hidden[0][0].detach().numpy()
            # cell_state = self.hidden[1][0].detach().numpy()

            # state_hidden = np.concatenate((state, cell_state))
            # self.buffer.latents.append(latent.squeeze().detach().numpy())

            state_latent = np.concatenate(
                (state, latent.squeeze().detach().cpu().numpy())
            )
            state = torch.FloatTensor(state_latent).to(self.device)
            action, action_logprob, state_val = self.policy_old.act(state)

        self.buffer.taus.append(tau)
        self.buffer.latents.append(latent.squeeze().detach().cpu().numpy())

        self.buffer.state_all.append(state)
        self.buffer.states.append(state[: self.state_dim])
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        self.buffer.other_positions.append(other_position)

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

        # # convert list to tensor
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
        old_another_actions = (
            torch.squeeze(torch.stack(self.buffer.another_actions, dim=0))
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

        old_another_positions_tensor = [
            torch.tensor(item) for item in self.buffer.other_positions
        ]
        old_another_positions = (
            torch.stack(old_another_positions_tensor, dim=0).detach().to(self.device)
        )

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()
        actions_t = old_actions.unsqueeze(1).detach()

        # Optimize policy for K epochs
        total_loss = 0
        total_ed_loss = 0

        taus = torch.cat(self.buffer.taus, 0)

        if self.ed_pretrain_count > 400:
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
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                    * advantages
                )

                # final loss of clipped objective PPO
                critique_loss = self.MseLoss(state_values, rewards)
                loss = (
                    -torch.min(surr1, surr2) + 0.5 * critique_loss - 0.01 * dist_entropy
                )

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

                total_loss += loss.mean()

            if self.cfg is not None and self.cfg.track:
                if self.cfg.track:
                    self.run.log(
                        {
                            "loss": total_loss.item(),
                        }
                    )
        # elif (self.ed_pretrain_count > 100) & (self.ed_pretrain_count <= 400):
        #     for _ in range(self.K_epochs):
        #         # Evaluating old actions and values
        #         logprobs, state_values, dist_entropy = self.policy.evaluate(
        #             old_state_all, old_actions
        #         )

        #         # match state_values tensor dimensions with rewards tensor
        #         state_values = torch.squeeze(state_values)

        #         # Finding the ratio (pi_theta / pi_theta__old)
        #         ratios = torch.exp(logprobs - old_logprobs.detach())

        #         # Finding Surrogate Loss
        #         surr1 = ratios * advantages
        #         surr2 = (
        #             torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
        #             * advantages
        #         )

        #         # final loss of clipped objective PPO
        #         critique_loss = self.MseLoss(state_values, rewards)
        #         loss = (
        #             -torch.min(surr1, surr2) + 0.5 * critique_loss - 0.01 * dist_entropy
        #         )

        #         self.optimizer.zero_grad()
        #         loss.mean().backward()
        #         self.optimizer.step()
        #         total_loss += loss.mean()
        #         for i in range(5):
        #             state_dist = self.encoder_decoder(taus)
        #             reconstruction_loss = self.criterion_mse(
        #                 state_dist[:-3], old_another_positions[3:]
        #             )

        #             ed_loss = reconstruction_loss
        #             self.ed_optimizer.zero_grad()
        #             ed_loss.backward()
        #             self.ed_optimizer.step()
        #             total_ed_loss += ed_loss

        #     if self.cfg is not None and self.cfg.track:
        #         if self.cfg.track:
        #             self.run.log(
        #                 {
        #                     "loss": total_loss.item(),
        #                     "hidden_loss": total_ed_loss.item(),
        #                 }
        #             )
        else:
            for _ in range(self.K_epochs):
                for i in range(5):
                    state_dist = self.encoder_decoder(taus)
                    reconstruction_loss = self.criterion_mse(
                        state_dist[:-3], old_another_positions[3:]
                    )

                    ed_loss = reconstruction_loss
                    self.ed_optimizer.zero_grad()
                    ed_loss.backward()
                    self.ed_optimizer.step()

                    total_ed_loss += ed_loss

            if self.cfg is not None and self.cfg.track:
                if self.cfg.track:
                    self.run.log(
                        {
                            "hidden_loss": total_ed_loss.item(),
                        }
                    )

        self.ed_pretrain_count += 1
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(
            self.policy_old.state_dict(),
            checkpoint_path.replace("flex", "lstm"),
        )
        torch.save(
            self.encoder.state_dict(),
            checkpoint_path.replace("flex", "encoder"),
        )
        torch.save(
            self.decoder.state_dict(),
            checkpoint_path.replace("flex", "decoder"),
        )

    def load(self, checkpoint_path):
        mode = "lstm"
        self.policy_old.load_state_dict(
            torch.load(
                checkpoint_path.replace("flex", mode),
                map_location=lambda storage, loc: storage,
            )
        )
        self.policy.load_state_dict(
            torch.load(
                checkpoint_path.replace("flex", mode),
                map_location=lambda storage, loc: storage,
            )
        )

    def load_ed(self, checkpoint_path):
        self.encoder.load_state_dict(
            torch.load(
                checkpoint_path.replace("flex", "encoder"),
                map_location=lambda storage, loc: storage,
            )
        )
        self.decoder.load_state_dict(
            torch.load(
                checkpoint_path.replace("flex", "decoder"),
                map_location=lambda storage, loc: storage,
            )
        )
        print("Encoder-Decoder loaded successfully.")

    def encoder_decoder(self, taus):
        self.hidden = None
        latents = []
        for tau in taus:
            latent, self.hidden = self.encoder(tau.unsqueeze(0), self.hidden)
            latents.append(latent.squeeze(0))
            if tau[-1] == 1:
                self.hidden = None
        latents = torch.stack(latents)
        state_dist = self.decoder(latents, taus[:, -3:-1])
        # kld_loss = self.criterion_kld(latents.log(), torch.randn_like(latents))
        return state_dist
