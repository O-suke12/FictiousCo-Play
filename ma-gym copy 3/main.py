import os
from datetime import datetime

import gym
import hydra
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pytz
import torch
from omegaconf import DictConfig
from sklearn.decomposition import PCA

import wandb
from lstm_lili import LILI_LSTM
from ppo import PPO
from scripted_agent import ANOTHER_AGENT
from utils.recoder import VideoRecorder


def train(cfg, run, name):
    env = gym.make("ma_gym:Lumberjacks-v0")
    japan_tz = pytz.timezone("Japan")
    now = datetime.now(japan_tz)

    state_dim = env.observation_space[0].shape[0]
    action_dim = env.action_space[0].n

    if cfg.model == "ppo":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(
            "============================================================================================"
        )
        print("Currently using PPO")
        print(
            "============================================================================================"
        )
        flex_agent = PPO(state_dim, action_dim, device, cfg, run=run)
    elif cfg.model == "lstm":
        device = "cuda:1" if torch.cuda.is_available() else "cpu"
        print(
            "============================================================================================"
        )
        print("Currently using LSTM")
        print(
            "============================================================================================"
        )
        flex_agent = LILI_LSTM(
            state_dim, action_dim, cfg.hidden_dim, device, cfg, run=run
        )
    else:
        raise ValueError("Invalid model type")
    another_agent = ANOTHER_AGENT(env, 0.0)

    directory = "ma-gym/models"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + "/" + name + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    flex_checkpoint_path = directory + f"{name}_flex_pena{env.invalid_penalty}.pth"

    time_step = 0
    i_episode = 0
    print_running_reward = 0
    print_running_episodes = 0
    last_update_time_step = 0
    start_time = now

    while time_step <= cfg.max_training_timesteps:
        obs_n = env.reset()
        done_n = [False for _ in range(env.n_agents)]
        current_ep_reward = 0
        another_agent.set_agent_type()
        step = 0
        while not all(done_n):
            flex_action = flex_agent.select_action(obs_n[0], step, done_n[0])
            another_action = another_agent.select_action(obs_n[1], step, done_n[1])

            obs_n, reward_n, done_n, info = env.step([flex_action, another_action])
            flex_agent.buffer.rewards.append(reward_n[0])
            flex_agent.buffer.is_terminals.append(done_n[0])
            flex_agent.buffer.another_actions.append(torch.tensor(another_action))

            current_ep_reward += reward_n[0]
            time_step += 1
            step += 1

            # printing average reward
            if time_step % cfg.print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print(
                    "Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(
                        i_episode, time_step, print_avg_reward
                    )
                )

                print_running_reward = 0
                print_running_episodes = 0
                if cfg.track:
                    run.log(
                        {
                            "average_reward": print_avg_reward,
                        }
                    )

            if time_step % cfg.save_model_freq == 0:
                print(
                    "--------------------------------------------------------------------------------------------"
                )
                flex_agent.save(flex_checkpoint_path)

                # another_agent.save(another_checkpoint_path)
                print("model saved")
                print(
                    "Elapsed Time  : ",
                    now - start_time,
                )
                print(
                    "--------------------------------------------------------------------------------------------"
                )

        if time_step - last_update_time_step >= cfg.update_timestep:
            # another_agent.update()
            flex_agent.update()
            last_update_time_step = time_step

        print_running_reward += current_ep_reward
        print_running_episodes += 1
        i_episode += 1
    # recoder.save("ma-gym_test.mp4")
    env.close()


def test(cfg, run, name):
    root_dir = os.getcwd() + "/ma-gym"
    recoder = VideoRecorder(root_dir, fps=3)
    recoder.init()

    env = gym.make("ma_gym:Lumberjacks-v0")
    state_dim = env.observation_space[0].shape[0]
    action_dim = env.action_space[0].n
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    directory = "models"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + "/" + name + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    flex_checkpoint_path = (
        "ma-gym/" + directory + f"{name}_flex_pena{env.invalid_penalty}.pth"
    )

    agents = {}
    agents["ppo"] = PPO(state_dim, action_dim, device, cfg, run=run)

    agents["lstm"] = LILI_LSTM(
        state_dim, action_dim, cfg.hidden_dim, device, cfg, run=run
    )
    agents["ppo"].load(flex_checkpoint_path)
    agents["lstm"].load(flex_checkpoint_path)
    agents["lstm"].load_ed(flex_checkpoint_path)
    another_agent = ANOTHER_AGENT(env, 0.0)

    another_agent_types = [
        "Following without Overlapping",
        "Following with Overlapping",
    ]
    seeds = np.random.randint(
        0, 1001, cfg.test_episode_num * len(another_agent_types)
    ).tolist()
    pca = PCA(n_components=2)

    latents = {}
    all_results = {}
    for agent in agents.keys():
        all_results[agent] = {}
        for another_agent_type in another_agent_types:
            all_results[agent][another_agent_type] = 0
    # cumulative_rewards = {
    #     "Following without Overlapping": [],
    #     "Following with Overlapping": [],
    # }
    env.seed(0)
    env.reset(another_agent_type=another_agent.agent_type)

    for i in range(len(another_agent_types)):
        another_agent_type = another_agent_types[i]
        another_agent.set_agent_type(agent_type=another_agent_type)
        difference = np.array([0 for _ in range(cfg.test_episode_num)])
        total_step = 0

        for agent in agents.keys():
            step = 0
            flex_agent = agents[agent]
            for episode in range(cfg.test_episode_num):
                seed = seeds[episode * (i + 1)]
                env.seed(seed)
                obs_n = env.reset(another_agent_type=another_agent.agent_type)
                done_n = [False for _ in range(env.n_agents)]
                current_ep_reward = 0
                recoder.init()

                while not all(done_n):
                    flex_action = flex_agent.select_action(obs_n[0], step, done_n[0])
                    another_action = another_agent.select_action(obs_n[1], 0, done_n[1])

                    obs_n, reward_n, done_n, info = env.step(
                        [flex_action, another_action]
                    )
                    recoder.record(env, mode="rgb_array")
                    current_ep_reward += sum(reward_n)
                    current_ep_reward += reward_n[0]

                    # cumulative_reward += sum(reward_n)
                    # cumulative_rewards[another_agent_types[i]].append(cumulative_reward)
                    step += 1
                    total_step += 1
                all_results[agent][another_agent.agent_type] += current_ep_reward

                if agent == "ppo":
                    difference[episode] = difference[episode] + current_ep_reward
                else:
                    difference[episode] = -difference[episode] + current_ep_reward

                print(
                    f"{another_agent.agent_type}_{agent}_{episode} reward: {current_ep_reward}"
                )
                recoder.save(
                    f"{another_agent.agent_type}_{agent}_{episode}_pena{env.invalid_penalty}.mp4"
                )
                imageio.mimsave("simulation.gif", recoder.frames[:50], fps=3)
                if cfg.track:
                    run.log(
                        {
                            "video": wandb.Video(
                                f"{recoder.save_dir}/ma-gym_test{another_agent.agent_type}_pena{env.invalid_penalty}.mp4",
                                fps=15,
                            )
                        }
                    )

                if agent == "lstm":
                    if another_agent.agent_type not in list(latents.keys()):
                        latents[another_agent.agent_type] = np.array(
                            flex_agent.buffer.latents[-3:]
                        )
                    else:
                        latents[another_agent_type] = np.concatenate(
                            (
                                latents[another_agent_type],
                                np.array(flex_agent.buffer.latents[-2:]),
                            )
                        )

            all_results[agent][another_agent.agent_type] = (
                all_results[agent][another_agent.agent_type] / cfg.test_episode_num
            )

            # plt.figure(figsize=(10, 6))
            # for agent_type, avg_rewards in cumulative_rewards.items():
            #     plt.plot(avg_rewards, label=f"Average Cumulative Reward - {agent_type}")
            # plt.xlabel("Episode Step")
            # plt.ylabel("Average Cumulative Reward")
            # plt.title(
            #     f"Average Cumulative Reward vs. Episode Step({another_agent_types[i]})"
            # )
            # plt.legend()
            # plt.savefig(
            #     f"ma-gym/{another_agent_types[i]}_Reward_Comparison_{env.invalid_penalty}.png"
            # )

        print(difference)

    all_latents = np.concatenate(list(latents.values()))
    labels = np.repeat(list(latents.keys()), [len(arr) for arr in latents.values()])
    all_latents_2d = pca.fit_transform(all_latents)
    unique_labels = list(latents.keys())
    colors = plt.cm.get_cmap("tab10", len(unique_labels))
    for i, label in enumerate(unique_labels):
        indices = np.where(labels == label)
        plt.scatter(
            all_latents_2d[indices, 0],
            all_latents_2d[indices, 1],
            c=colors(i),
            label=label,
        )
    plt.legend()
    plt.savefig(f"ma-gym/latent_dim_{env.invalid_penalty}.png")
    env.close()

    flex_types = list(agents.keys())

    labels = all_results[flex_types[0]].keys()
    x = range(len(labels))

    fig, ax = plt.subplots()
    ax.bar(
        x,
        all_results[flex_types[0]].values(),
        width=0.3,
        label="PPO",
    )
    ax.bar(
        [i + 0.3 for i in x],
        all_results[flex_types[1]].values(),
        width=0.3,
        label="Proposed Method",
    )

    ax.set_xlabel("The Other Agent Type")
    ax.set_ylabel("Average Reward")
    ax.set_title("Comparison between PPO and Proposed method")
    ax.set_xticks([i + 0.2 for i in x])
    ax.set_xticklabels(labels)
    ax.legend()
    fig.savefig(f"ma-gym/Comparison_{env.invalid_penalty}.png")
    if cfg.track:
        run.log({"test_results": wandb.Image("Comparison.png")})


@hydra.main(version_base=None, config_path="config", config_name="lumberjacks")
def main(cfg: DictConfig):
    japan_tz = pytz.timezone("Japan")
    now = datetime.now(japan_tz)
    name = "lumberjacks"
    if cfg.track:
        run_name = f"{name}_{now.strftime('%m_%d_%H:%M')}"
        run = wandb.init(
            project=cfg.project_name,
            sync_tensorboard=True,
            monitor_gym=True,
            name=run_name,
        )
    else:
        run = None

    if cfg.train:
        train(cfg, run, name)
    else:
        test(cfg, run, name)


if __name__ == "__main__":
    main()
