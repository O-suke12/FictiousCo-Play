import os

import torch.nn as nn

from envs.mpe_fixed_env import simple_spread_v3
from scripted_agent import ANOTHER_AGENT
from utils.recoder import VideoRecorder


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


env = simple_spread_v3.env(
    N=2,
    max_cycles=120,
    continuous_actions=False,
    render_mode="rgb_array",
)

seed = 42
another_type = "nothing"
env.reset(seed, options={"agent_type": another_type, "seed": seed})
scripted_agent = ANOTHER_AGENT(env, 0.1)
scripted_agent.set_agent_type(env.world.another_agent_type)

root_dir = os.getcwd()
recoder = VideoRecorder(root_dir)
recoder.init()

state_buffer = []
action_buffer = []
pred_update_step = 0
for num_step in range(100000):
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            if agent == "another_agent":
                action = scripted_agent.select_action(observation, 0, truncation)
                state_buffer.append(observation)
                action_buffer.append(action)
            else:
                action = env.action_space(agent).sample()
        recoder.record(env)
        env.step(action)

    if num_step - pred_update_step > 1000:
        update_ed()
        pred_update_step = num_step

video_name = "test.mp4"
recoder.save(video_name)
env.close()


def update_ed():
    action_dist = encoder_decoder(old_states, taus)
    ed_loss = F.cross_entropy(action_dist, old_another_actions)
    self.ed_optimizer.zero_grad()
    ed_loss.backward()
    self.ed_optimizer.step()


def encoder_decoder(self, old_states, taus):
    hidden = None
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
