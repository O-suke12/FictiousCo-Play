import datetime

import hydra
import numpy as np
import pytz
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from pettingzoo.mpe import simple_v3
from torch.distributions.categorical import Categorical

import wandb

SWEEP = False


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, observation_size, num_actions):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_size).prod(), 64)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_size).prod(), 64)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(64, num_actions), std=0.01),
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
    action_size,
    run,
    ent_coef,
    vf_coef,
    clip_coef,
    gamma,
    batch_size,
    lr,
):
    obs = torch.zeros(cfg.num_steps, observation_size).to(device)
    actions = torch.zeros(cfg.num_steps, cfg.num_envs, action_size).to(device)
    logprobs = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    rewards = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    dones = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    values = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    global_step = 0

    next_done = torch.zeros(cfg.num_envs).to(device)

    gae_lambda = 0.95
    max_grad_norm = 0.5

    for update in range(1, cfg.total_episodes + 1):
        env.reset()
        # Annealing the rate if instructed to do so.

        frac = 1.0 - (update - 1.0) / cfg.num_steps
        lrnow = frac * lr
        optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, cfg.num_steps):
            global_step += 1 * cfg.num_envs

            next_obs, reward, done, truncated, info = env.last()

            # ALGO LOGIC: action logic
            with torch.no_grad():
                next_obs = torch.from_numpy(next_obs).to(device)
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            if done or truncated:
                action = None
                env.step(action)
                break
            else:
                env.step(action.detach().cpu().numpy())

            # TRY NOT TO MODIFY: execute the game and log data.

            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = (
                torch.tensor(next_obs).to(device),
                torch.tensor(int(done)).to(device),
            )
            obs[step] = next_obs
            dones[step] = next_done

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)

            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(cfg.num_steps)):
                if t == cfg.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs[:step]
        b_logprobs = logprobs.reshape(-1)[:step]
        b_actions = actions.reshape((-1,), action_size)[:step]
        b_advantages = advantages.reshape(-1)[:step]
        b_returns = returns.reshape(-1)[:step]
        b_values = values.reshape(-1)[:step]

        # Optimizing the policy and value network
        b_inds = np.arange(step)
        batch_size = step // cfg.minibatch_size
        clipfracs = []
        for epoch in range(3):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size):
                start = start * cfg.minibatch_size
                end = start + cfg.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]

                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - clip_coef, 1 + clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)

                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -clip_coef,
                    clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()


# def evaluate(agent, env, device, cfg, run):
#     agent.eval()
#     root_dir = os.getcwd()
#     if cfg.render:
#         recoder = VideoRecorder(root_dir)
#     with torch.no_grad():
#         # render 5 episodes out
#         for episode in range(3):
#             if cfg.render:
#                 recoder.init()
#             env.reset(seed=42)
#             obs = batchify_obs(obs, device)
#             terms = [False]
#             truncs = [False]
#             episode_reward = 0
#             while not any(terms) and not any(truncs):
#                 actions, logprobs, _, values = agent.get_action_and_value(obs)
#                 obs, rewards, terms, truncs, infos = env.step(unbatchify(actions, env))
#                 if cfg.render:
#                     recoder.record(env)
#                 obs = batchify_obs(obs, device)
#                 terms = [terms[a] for a in terms]
#                 truncs = [truncs[a] for a in truncs]
#                 episode_reward += rewards

#             if cfg.render:
#                 video_name = f"eval{episode+1}.mp4"
#                 recoder.save(video_name)
#             if cfg.track:
#                 run.log(
#                     {
#                         "video": wandb.Video(
#                             f"{recoder.save_dir}/{video_name}", fps=30
#                         ),
#                         "episode_reward": episode_reward,
#                     }
#                 )


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

    env = simple_v3.env(
        max_cycles=50,
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

    agent = Agent(observation_size=observation_size, num_actions=num_actions).to(device)
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
            num_actions,
            run,
            ent_coef,
            vf_coef,
            clip_coef,
            gamma,
            batch_size,
            lr,
        )

    # evaluate(agent, env, device, cfg, run)


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
