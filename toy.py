import datetime
import os

import hydra
import numpy as np
import pytz
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from pettingzoo.mpe import simple_v3
from torch.distributions.categorical import Categorical
from tqdm import tqdm

import wandb
from utils.recoder import VideoRecorder

SWEEP = False


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def batchify_obs(obs, device):
    """Converts PZ style observations to batch of torch arrays."""
    # convert to list of np arrays
    obs = np.stack([obs[a] for a in obs], axis=0)
    # transpose to be (batch, channel, height, width)
    # obs = obs.transpose(0, -1, 1, 2)
    # convert to torch
    obs = torch.tensor(obs).to(device)

    return obs


def batchify(x, device):
    """Converts PZ style returns to batch of torch arrays."""
    # convert to list of np arrays
    x = np.stack([x[a] for a in x], axis=0)
    # convert to torch
    x = torch.tensor(x).to(device)

    return x


def unbatchify(x, env):
    """Converts np array to PZ style arguments."""
    x = x.cpu().numpy()
    x = {a: x[i] for i, a in enumerate(env.possible_agents)}

    return x


def training(
    agent,
    optimizer,
    env,
    device,
    cfg,
    num_agents,
    observation_size,
    run,
    ent_coef,
    vf_coef,
    clip_coef,
    gamma,
    batch_size,
):
    end_step = 0
    total_episodic_return = 0
    rb_obs = torch.zeros((cfg.max_cycles, num_agents, observation_size)).to(device)
    rb_actions = torch.zeros((cfg.max_cycles, num_agents)).to(device)
    rb_logprobs = torch.zeros((cfg.max_cycles, num_agents)).to(device)
    rb_rewards = torch.zeros((cfg.max_cycles, num_agents)).to(device)
    rb_terms = torch.zeros((cfg.max_cycles, num_agents)).to(device)
    rb_values = torch.zeros((cfg.max_cycles, num_agents)).to(device)
    # train for n number of episodes
    for episode in tqdm(range(cfg.total_episodes)):
        # collect an episode
        with torch.no_grad():
            # collect observations and convert to batch of torch tensors
            next_obs, info = env.reset(seed=None)
            # reset the episodic return
            total_episodic_return = 0

            # each episode has num_steps
            for step in range(0, cfg.max_cycles):
                # rollover the observation
                obs = batchify_obs(next_obs, device)

                # get action from the agent
                actions, logprobs, _, values = agent.get_action_and_value(obs)

                # execute the environment and log data
                next_obs, rewards, terms, truncs, infos = env.step(
                    unbatchify(actions, env)
                )

                # add to episode storage
                rb_obs[step] = obs
                rb_rewards[step] = batchify(rewards, device)
                rb_terms[step] = batchify(terms, device)
                rb_actions[step] = actions
                rb_logprobs[step] = logprobs
                rb_values[step] = values.flatten()

                # compute episodic return
                total_episodic_return += rb_rewards[step].cpu().numpy()

                # if we reach termination or truncation, end
                if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
                    end_step = step
                    break

        # bootstrap value if not done
        with torch.no_grad():
            rb_advantages = torch.zeros_like(rb_rewards).to(device)
            for t in reversed(range(end_step)):
                delta = (
                    rb_rewards[t]
                    + gamma * rb_values[t + 1] * rb_terms[t + 1]
                    - rb_values[t]
                )
                rb_advantages[t] = delta + gamma * gamma * rb_advantages[t + 1]
            rb_returns = rb_advantages + rb_values

        # convert our episodes to batch of individual transitions
        b_obs = torch.flatten(rb_obs[:end_step], start_dim=0, end_dim=1)
        b_logprobs = torch.flatten(rb_logprobs[:end_step], start_dim=0, end_dim=1)
        b_actions = torch.flatten(rb_actions[:end_step], start_dim=0, end_dim=1)
        b_returns = torch.flatten(rb_returns[:end_step], start_dim=0, end_dim=1)
        b_values = torch.flatten(rb_values[:end_step], start_dim=0, end_dim=1)
        b_advantages = torch.flatten(rb_advantages[:end_step], start_dim=0, end_dim=1)

        # Optimizing the policy and value network
        b_index = np.arange(len(b_obs))
        clip_fracs = []
        for repeat in range(3):
            # shuffle the indices we use to access the data
            np.random.shuffle(b_index)
            for start in range(0, len(b_obs), batch_size):
                # select the indices we want to train on
                end = start + batch_size
                batch_index = b_index[start:end]

                _, newlogprob, entropy, value = agent.get_action_and_value(
                    b_obs[batch_index], b_actions.long()[batch_index]
                )
                logratio = newlogprob - b_logprobs[batch_index]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_fracs += [
                        ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                    ]

                # normalize advantaegs
                advantages = b_advantages[batch_index]
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                # Policy loss
                pg_loss1 = -b_advantages[batch_index] * ratio
                pg_loss2 = -b_advantages[batch_index] * torch.clamp(
                    ratio, 1 - clip_coef, 1 + clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                value = value.flatten()
                v_loss_unclipped = (value - b_returns[batch_index]) ** 2
                v_clipped = b_values[batch_index] + torch.clamp(
                    value - b_values[batch_index],
                    -clip_coef,
                    clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[batch_index]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if cfg.track:
            run.log(
                {
                    "value_loss": v_loss.item(),
                    "policy_loss": pg_loss.item(),
                    "loss": loss.item(),
                    "episode_return": np.mean(total_episodic_return),
                    "old_approx_kl": old_approx_kl.item(),
                    "approx_kl": approx_kl.item(),
                    "clip_fraction": np.mean(clip_fracs),
                    "explained_variance": explained_var,
                }
            )

    if cfg.save_model:
        torch.save(agent.state_dict(), "models/model.pth")
    return loss.item()


def evaluate(agent, env, device, cfg, run):
    agent.eval()
    root_dir = os.getcwd()
    if cfg.render:
        recoder = VideoRecorder(root_dir)
    with torch.no_grad():
        # render 5 episodes out
        for episode in range(3):
            if cfg.render:
                recoder.init()
            obs, infos = env.reset(seed=42)
            obs = batchify_obs(obs, device)
            terms = [False]
            truncs = [False]
            episode_reward = 0
            while not any(terms) and not any(truncs):
                actions, logprobs, _, values = agent.get_action_and_value(obs)
                obs, rewards, terms, truncs, infos = env.step(unbatchify(actions, env))
                if cfg.render:
                    recoder.record(env)
                obs = batchify_obs(obs, device)
                terms = [terms[a] for a in terms]
                truncs = [truncs[a] for a in truncs]
                episode_reward += rewards

            if cfg.render:
                video_name = f"eval{episode+1}.mp4"
                recoder.save(video_name)
            if cfg.track:
                run.log(
                    {
                        "video": wandb.Video(
                            f"{recoder.save_dir}/{video_name}", fps=30
                        ),
                        "episode_reward": episode_reward,
                    }
                )


@hydra.main(version_base=None, config_path="config", config_name="toy")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.track:
        japan_tz = pytz.timezone("Japan")
        now = datetime.datetime.now(japan_tz)
        run_name = f"{now.strftime('%m_%d_%H:%M')}"
        run = wandb.init(
            project=cfg.project_name,
            sync_tensorboard=True,
            monitor_gym=True,
            name=run_name,
        )

    else:
        run = None

    env = simple_v3(
        max_cycles=cfg.max_cycles,
        continuous_actions=False,
        render_mode="rgb_array",
    )
    num_agents = len(env.possible_agents)
    num_actions = env.action_space(env.possible_agents[0]).n
    observation_size = env.observation_space(env.possible_agents[0]).shape[0]

    if SWEEP:
        ent_coef = wandb.config.ent_coef
        vf_coef = wandb.config.vf_coef
        lr = wandb.config.lr
        clip_coef = wandb.config.clip_coef
        gamma = wandb.config.gamma
        batch_size = wandb.config.batch_size
    else:
        ent_coef = 0.09
        vf_coef = 0.1
        lr = 0.001
        clip_coef = 0.1
        gamma = 0.98
        batch_size = 16

    agent = Agent(num_actions=num_actions, observation_size=observation_size).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

    if cfg.train:
        training(
            agent,
            optimizer,
            env,
            device,
            cfg,
            num_agents,
            observation_size,
            run,
            ent_coef,
            vf_coef,
            clip_coef,
            gamma,
            batch_size,
        )

    evaluate(agent, env, device, cfg, run)


if __name__ == "__main__":
    if SWEEP:
        sweep_config = {
            "method": "bayes",
            "metric": {"name": "loss", "goal": "minimize"},
            "parameters": {
                "ent_coef": {"min": 0.001, "max": 0.1},
                "vf_coef": {"min": 0.1, "max": 1.0},
                "lr": {"min": 0.0001, "max": 0.01},
                "clip_coef": {"min": 0.1, "max": 0.3},
                "gamma": {"min": 0.9, "max": 0.99},
                "batch_size": {
                    "values": [
                        16,
                        32,
                        64,
                        128,
                    ]
                },
            },
        }
        sweep_id = wandb.sweep(sweep_config, project="mysweep")
        wandb.config = sweep_config
        wandb.agent(sweep_id, function=main)
    else:
        main()


# import datetime
# import os

# import gym
# import hydra
# import numpy as np
# import pytz
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from omegaconf import DictConfig
# from torch.distributions.categorical import Categorical
# from tqdm import tqdm

# import wandb
# from utils.recoder import VideoRecorder

# SWEEP = False


# class Agent(nn.Module):
#     def __init__(self, num_actions, observation_size):
#         super().__init__()
#         self.hidden_size = 128

#         self.network = nn.Sequential(
#             nn.Linear(observation_size, self.hidden_size),
#             nn.ReLU(),
#             nn.Linear(self.hidden_size, self.hidden_size),
#             nn.ReLU(),
#             nn.Linear(self.hidden_size, self.hidden_size),
#             nn.ReLU(),
#         )
#         self.actor = nn.Sequential(
#             self.network, nn.Linear(self.hidden_size, num_actions)
#         )
#         self.critic = nn.Sequential(self.network, nn.Linear(self.hidden_size, 1))

#     def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
#         torch.nn.init.orthogonal_(layer.weight, std)
#         torch.nn.init.constant_(layer.bias, bias_const)
#         return layer

#     def get_action_and_value(self, x, action=None):
#         logits = self.actor(x)
#         probs = Categorical(logits=logits)
#         if action is None:
#             action = probs.sample()
#         return action, probs.log_prob(action), probs.entropy(), self.critic(x)


# def training(
#     agent,
#     optimizer,
#     env,
#     num_envs,
#     device,
#     cfg,
#     num_agents,
#     observation_size,
#     run,
#     ent_coef,
#     vf_coef,
#     clip_coef,
#     gamma,
#     batch_size,
# ):
#     end_step = 0
#     total_episodic_return = 0
#     rb_obs = torch.zeros((cfg.max_cycles, num_envs, observation_size)).to(device)
#     rb_actions = torch.zeros((cfg.max_cycles, num_envs)).to(device)
#     rb_logprobs = torch.zeros((cfg.max_cycles, num_envs)).to(device)
#     rb_rewards = torch.zeros((cfg.max_cycles, num_envs)).to(device)
#     rb_terms = torch.zeros((cfg.max_cycles, num_envs)).to(device)
#     rb_values = torch.zeros((cfg.max_cycles, num_envs)).to(device)
#     # train for n number of episodes
#     for episode in tqdm(range(cfg.total_episodes)):
#         # collect an episode
#         with torch.no_grad():
#             # collect observations and convert to batch of torch tensors
#             next_obs, info = env.reset(seed=None)
#             # reset the episodic return
#             total_episodic_return = 0

#             # each episode has num_steps
#             for step in range(0, cfg.max_cycles):
#                 # rollover the observation
#                 obs = torch.from_numpy(next_obs).to(device)

#                 # get action from the agent
#                 actions, logprobs, _, values = agent.get_action_and_value(obs)

#                 # execute the environment and log data

#                 next_obs, rewards, terms, truncs, infos = env.step(
#                     actions.cpu().detach().numpy()
#                 )

#                 # add to episode storage
#                 rb_obs[step] = obs
#                 rb_rewards[step] = torch.from_numpy(rewards).to(device)
#                 rb_terms[step] = torch.from_numpy(terms).to(device)
#                 rb_actions[step] = actions
#                 rb_logprobs[step] = logprobs
#                 rb_values[step] = values.flatten()

#                 # compute episodic return
#                 total_episodic_return += rb_rewards[step].cpu().numpy()

#                 # if we reach termination or truncation, end
#                 if np.any(terms) or np.any(truncs):
#                     end_step = step
#                     break

#         # bootstrap value if not done
#         with torch.no_grad():
#             rb_advantages = torch.zeros_like(rb_rewards).to(device)
#             for t in reversed(range(end_step)):
#                 delta = (
#                     rb_rewards[t]
#                     + gamma * rb_values[t + 1] * rb_terms[t + 1]
#                     - rb_values[t]
#                 )
#                 rb_advantages[t] = delta + gamma * gamma * rb_advantages[t + 1]
#             rb_returns = rb_advantages + rb_values

#         # convert our episodes to batch of individual transitions
#         b_obs = torch.flatten(rb_obs[:end_step], start_dim=0, end_dim=1)
#         b_logprobs = torch.flatten(rb_logprobs[:end_step], start_dim=0, end_dim=1)
#         b_actions = torch.flatten(rb_actions[:end_step], start_dim=0, end_dim=1)
#         b_returns = torch.flatten(rb_returns[:end_step], start_dim=0, end_dim=1)
#         b_values = torch.flatten(rb_values[:end_step], start_dim=0, end_dim=1)
#         b_advantages = torch.flatten(rb_advantages[:end_step], start_dim=0, end_dim=1)

#         # Optimizing the policy and value network
#         b_index = np.arange(len(b_obs))
#         clip_fracs = []
#         for repeat in range(3):
#             # shuffle the indices we use to access the data
#             np.random.shuffle(b_index)
#             for start in range(0, len(b_obs), batch_size):
#                 # select the indices we want to train on
#                 end = start + batch_size
#                 batch_index = b_index[start:end]

#                 _, newlogprob, entropy, value = agent.get_action_and_value(
#                     b_obs[batch_index], b_actions.long()[batch_index]
#                 )
#                 logratio = newlogprob - b_logprobs[batch_index]
#                 ratio = logratio.exp()

#                 with torch.no_grad():
#                     # calculate approx_kl http://joschu.net/blog/kl-approx.html
#                     old_approx_kl = (-logratio).mean()
#                     approx_kl = ((ratio - 1) - logratio).mean()
#                     clip_fracs += [
#                         ((ratio - 1.0).abs() > clip_coef).float().mean().item()
#                     ]

#                 # normalize advantaegs
#                 advantages = b_advantages[batch_index]
#                 advantages = (advantages - advantages.mean()) / (
#                     advantages.std() + 1e-8
#                 )

#                 # Policy loss
#                 pg_loss1 = -b_advantages[batch_index] * ratio
#                 pg_loss2 = -b_advantages[batch_index] * torch.clamp(
#                     ratio, 1 - clip_coef, 1 + clip_coef
#                 )
#                 pg_loss = torch.max(pg_loss1, pg_loss2).mean()

#                 # Value loss
#                 value = value.flatten()
#                 v_loss_unclipped = (value - b_returns[batch_index]) ** 2
#                 v_clipped = b_values[batch_index] + torch.clamp(
#                     value - b_values[batch_index],
#                     -clip_coef,
#                     clip_coef,
#                 )
#                 v_loss_clipped = (v_clipped - b_returns[batch_index]) ** 2
#                 v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
#                 v_loss = 0.5 * v_loss_max.mean()

#                 entropy_loss = entropy.mean()
#                 loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#         y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
#         var_y = np.var(y_true)
#         explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

#         if cfg.track:
#             run.log(
#                 {
#                     "value_loss": v_loss.item(),
#                     "policy_loss": pg_loss.item(),
#                     "loss": loss.item(),
#                     "episode_return": np.mean(total_episodic_return),
#                     "old_approx_kl": old_approx_kl.item(),
#                     "approx_kl": approx_kl.item(),
#                     "clip_fraction": np.mean(clip_fracs),
#                     "explained_variance": explained_var,
#                     "step": step,
#                 }
#             )

#     if cfg.save_model:
#         torch.save(agent.state_dict(), "models/model.pth")
#     return loss.item()


# def evaluate(agent, env, device, cfg, run):
#     agent.eval()
#     root_dir = os.getcwd()
#     env = gym.make("CartPole-v1", render_mode="rgb_array")
#     if cfg.render:
#         recoder = VideoRecorder(root_dir)
#     with torch.no_grad():
#         # render 5 episodes out
#         for episode in range(3):
#             if cfg.render:
#                 recoder.init()
#             obs, infos = env.reset(seed=None)
#             obs = torch.from_numpy(obs).to(device)
#             terms = False
#             truncs = False
#             while not terms and not truncs:
#                 actions, logprobs, _, values = agent.get_action_and_value(obs)
#                 obs, rewards, terms, truncs, infos = env.step(int(actions))
#                 if cfg.render:
#                     recoder.record(env)
#                 obs = torch.from_numpy(obs).to(device)

#             if cfg.render:
#                 video_name = f"eval{episode+1}.mp4"
#                 recoder.save(video_name)
#             if cfg.track:
#                 run.log(
#                     {"video": wandb.Video(f"{recoder.save_dir}/{video_name}", fps=30)}
#                 )


# @hydra.main(version_base=None, config_path="config", config_name="toy")
# def main(cfg: DictConfig):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     if cfg.track:
#         japan_tz = pytz.timezone("Japan")
#         now = datetime.datetime.now(japan_tz)
#         run_name = f"{now.strftime('%m_%d_%H:%M')}"
#         run = wandb.init(
#             project=cfg.project_name,
#             sync_tensorboard=True,
#             monitor_gym=True,
#             name=run_name,
#         )

#     else:
#         run = None
#     num_envs = 1
#     env = gym.vector.make("CartPole-v1", num_envs=num_envs)
#     num_agents = 1
#     observation_size = 4

#     if SWEEP:
#         ent_coef = wandb.config.ent_coef
#         vf_coef = wandb.config.vf_coef
#         lr = wandb.config.lr
#         clip_coef = wandb.config.clip_coef
#         gamma = wandb.config.gamma
#         batch_size = wandb.config.batch_size
#     else:
#         ent_coef = 0.09
#         vf_coef = 0.1
#         lr = 0.003
#         clip_coef = 0.1
#         gamma = 0.98
#         batch_size = 16

#     agent = Agent(num_actions=2, observation_size=observation_size).to(device)
#     optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

#     if cfg.train:
#         training(
#             agent,
#             optimizer,
#             env,
#             num_envs,
#             device,
#             cfg,
#             num_agents,
#             observation_size,
#             run,
#             ent_coef,
#             vf_coef,
#             clip_coef,
#             gamma,
#             batch_size,
#         )

#     evaluate(agent, env, device, cfg, run)


# if __name__ == "__main__":
#     if SWEEP:
#         sweep_config = {
#             "method": "bayes",
#             "metric": {"name": "loss", "goal": "minimize"},
#             "parameters": {
#                 "ent_coef": {"min": 0.001, "max": 0.1},
#                 "vf_coef": {"min": 0.1, "max": 1.0},
#                 "lr": {"min": 0.0001, "max": 0.01},
#                 "clip_coef": {"min": 0.1, "max": 0.3},
#                 "gamma": {"min": 0.9, "max": 0.99},
#                 "batch_size": {
#                     "values": [
#                         16,
#                         32,
#                         64,
#                         128,
#                     ]
#                 },
#             },
#         }
#         sweep_id = wandb.sweep(sweep_config, project="mysweep")
#         wandb.config = sweep_config
#         wandb.agent(sweep_id, function=main)
#     else:
#         main()
