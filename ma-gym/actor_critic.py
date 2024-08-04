import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim=None):
        super(ActorCritic, self).__init__()
        if latent_dim is None:
            latent_dim = 0
        input_dim = state_dim + latent_dim
        layer_dim1 = 32
        layer_dim2 = 16

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
