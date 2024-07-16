import torch
import torch.distributions as dist
import torch.nn as nn
from torch.utils.data import Dataset


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


class mpeDataset(Dataset):
    def __init__(self, agent, device):
        states = torch.stack(agent.buffer.states[:-2])
        next_states = torch.stack(agent.buffer.states[1:-1])
        actions = torch.stack(agent.buffer.actions[1:-1]).unsqueeze(1)
        next_actions = torch.stack(agent.buffer.actions[1:-1]).unsqueeze(1)
        rewards = (
            torch.tensor(agent.buffer.rewards[1:-1], dtype=torch.float32)
            .unsqueeze(1)
            .to(device)
        )
        self.predicted_states = torch.stack(agent.buffer.states[2:])
        self.predicted_rewards = (
            torch.tensor(agent.buffer.rewards[2:], dtype=torch.float32)
            .unsqueeze(1)
            .to(device)
        )
        self.tau = torch.cat((states, actions, rewards, next_states, next_actions), 1)

    def __len__(self):
        return len(self.tau)

    def __getitem__(self, idx):
        return self.tau[idx], self.predicted_rewards[idx], self.predicted_states[idx]


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        encoder_input_dim,
        z_dim,
        decoder_input_dim,
        decoder_output_dim,
        **kwargs,
    ):
        super().__init__()
        self.device = "cuda:0"
        self.state_dim = state_dim
        self.encoder = Encoder(encoder_input_dim, z_dim).to(self.device)
        self.decoder = Decoder(
            state_dim, action_dim, decoder_input_dim, decoder_output_dim
        ).to(self.device)
        self.num_epochs = 2

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[15], gamma=0.1
        )
        self.loss = nn.MSELoss()
        self.run = kwargs.get("run")
        self.cfg = kwargs.get("cfg")

    def get_latent_strategies(self, tau):
        return self.encoder(tau)

    def forward(self, tau):
        z = self.encoder(tau[:, :-1])
        decoder_input = torch.cat((z, tau[:, -(self.state_dim + 1) :]), 1)
        reward, state_mean, state_log_var = self.decoder(decoder_input)
        return reward, state_mean, state_log_var

    def update(self, agent):
        dataset = mpeDataset(agent, self.device)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, 5, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(val_dataset, 5, shuffle=True)

        for epoch in range(self.num_epochs):
            self.train()
            for x, y_r, y_s in train_loader:
                reward_dist, state_dist = self(x)

                state_loss = -state_dist.log_prob(y_s).mean()
                reward_loss = -reward_dist.log_prob(y_r).mean()
                J_rep = state_loss + reward_loss

                self.optimizer.zero_grad()
                J_rep.backward()
                self.optimizer.step()

            self.eval()
            with torch.no_grad():
                J_rep = 0
                for x, y_r, y_s in valid_loader:
                    reward, state_mean, state_log_var = self(x)

                    reward_dist, state_dist = self(x)

                    state_loss = -state_dist.log_prob(y_s).mean()
                    reward_loss = -reward_dist.log_prob(y_r).mean()
                    J_rep = state_loss + reward_loss

            if self.cfg is not None and self.cfg.track:
                if self.cfg.track:
                    self.run.log({"vae_loss": J_rep.item()})
            self.scheduler.step()
