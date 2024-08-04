import os
from datetime import datetime

import hydra
import numpy as np
import pytz
import torch
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from sklearn.decomposition import PCA

import wandb
from envs.mpe_fixed_env import simple_spread_v3
from lili import LILI
from lstm_lili import LILI_LSTM
from ppo import PPO
from scripted_agent import ANOTHER_AGENT
from utils.recoder import VideoRecorder

val_count = 0
japan_tz = pytz.timezone("Japan")
now = datetime.now(japan_tz)


def train(
    cfg: DictConfig,
    run,
    env,
    validation_env,
    recoder,
    device,
    flex_checkpoint_path,
    agent_type,
):
    state_dim = env.observation_space(env.possible_agents[0]).shape[0]
    action_dim = env.action_space(env.possible_agents[0]).n

    if agent_type == "lili":
        print(
            "============================================================================================"
        )
        print("Currently using LILI")
        print(
            "============================================================================================"
        )
        flex_agent = LILI(
            state_dim,
            action_dim,
            cfg.z_dim,
            device,
            cfg,
            run=run,
        )

    elif agent_type == "PPO":
        print(
            "============================================================================================"
        )
        print("Currently using PPO")
        print(
            "============================================================================================"
        )
        flex_agent = PPO(
            state_dim,
            action_dim,
            device,
            cfg,
            run=run,
        )
    else:
        print(
            "============================================================================================"
        )
        print("Currently using LILI_LSTM")
        print(
            "============================================================================================"
        )
        flex_agent = LILI_LSTM(
            state_dim,
            action_dim,
            cfg.hidden_dim,
            device,
            cfg,
            run=run,
        )

    another_agent = ANOTHER_AGENT(env, 0.1)

    agents = {}

    agents[env.possible_agents[0]] = another_agent
    agents[env.possible_agents[1]] = flex_agent

    # track total training time
    start_time = now

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    last_update_time_step = 0

    time_step = 0
    i_episode = 0

    # if cfg.data_collect:
    #     collect(agents, env, another_checkpoint_path, flex_checkpoint_path, cfg)

    # training loop
    while time_step <= cfg.max_training_timesteps:
        env.reset()
        current_ep_reward = 0
        another_agent.set_agent_type(env.world.another_agent_type)

        for t in range(1, cfg.max_cycle + 2):
            for agent in env.possible_agents:
                state, reward, done, truncated, info = env.last()
                end = done or truncated
                action = agents[agent].select_action(state, t, end)

                if agents[agent] == flex_agent:
                    agents[agent].buffer.rewards.append(reward)
                    agents[agent].buffer.is_terminals.append(done)
                else:
                    agents["flex_agent"].buffer.another_actions.append(
                        torch.tensor(action)
                    )

                time_step += 1
                current_ep_reward += reward

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

                # save model weights
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
                    # validation(cfg, run, agents, validation_env, recoder)
                    agents["another_agent"].change_env(env)
                    agents["another_agent"].set_agent_type(env.world.another_agent_type)

                # break; if the episode is over
                if done or truncated:
                    action = None

                env.step(action)
            if done or truncated:
                break

        # update PPO agent
        if time_step - last_update_time_step >= cfg.update_timestep:
            # another_agent.update()
            flex_agent.update()
            last_update_time_step = time_step

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        i_episode += 1

    env.close()

    # print total training time
    print(
        "============================================================================================"
    )
    end_time = now
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print(
        "============================================================================================"
    )


def validation(cfg, run, agents, validation_env, recoder):
    test_running_reward = 0
    global val_count

    with torch.no_grad():
        ep_reward = 0
        validation_env.reset()
        agents["another_agent"].change_env(validation_env)
        agents["another_agent"].set_agent_type(validation_env.world.another_agent_type)
        recoder.init()

        for t in range(1, cfg.max_cycle + 1):
            for agent in validation_env.possible_agents:
                state, reward, done, truncated, info = validation_env.last()
                action = agents[agent].just_select_action(state, t)

                ep_reward += reward

                recoder.record(validation_env)
                # break; if the episode is over
                if done or truncated:
                    action = None
                    break
                validation_env.step(action)
            if done or truncated:
                break
    val_count += 1
    video_name = f"valid_{val_count}.mp4"
    recoder.save(video_name)
    if cfg.track:
        run.log(
            {
                "valid_average_reward": ep_reward,
            }
        )
        run.log({"video": wandb.Video(f"{recoder.save_dir}/{video_name}", fps=24)})

    print(
        "============================================================================================"
    )

    avg_test_reward = ep_reward
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print(
        "============================================================================================"
    )


def test(cfg: DictConfig, run, test_env, recoder, device, directory):
    print(
        "============================================================================================"
    )
    print("Testing")
    print(
        "============================================================================================"
    )
    state_dim = test_env.observation_space(test_env.possible_agents[0]).shape[0]
    action_dim = test_env.action_space(test_env.possible_agents[0]).n

    lili_lstm_model_name = f"lili_lstm_{cfg.agent_num}agent_{cfg.landmark_num}land"
    lili_lstm_check_point = directory + f"{lili_lstm_model_name}.pth"
    lili_lstm = LILI_LSTM(
        state_dim,
        action_dim,
        cfg.hidden_dim,
        device,
        cfg,
        run=run,
    )
    lili_lstm.load(lili_lstm_check_point)
    lili_lstm.load_ed(lili_lstm_check_point)

    ppo_model_name = f"ppo_{cfg.agent_num}agent_{cfg.landmark_num}land"
    ppo_check_point = directory + f"{ppo_model_name}.pth"
    ppo = PPO(
        state_dim,
        action_dim,
        device,
        cfg,
        run=run,
    )
    ppo.load(ppo_check_point)

    another_agent = ANOTHER_AGENT(test_env, 0.0)

    flex_types = [
        "PPO",
        "Proposed Method",
    ]

    ppo_agents = {}
    lili_lstm_agents = {}
    ppo_agents[test_env.possible_agents[0]] = another_agent
    ppo_agents[test_env.possible_agents[1]] = ppo
    lili_lstm_agents[test_env.possible_agents[0]] = another_agent
    lili_lstm_agents[test_env.possible_agents[1]] = lili_lstm

    test_agents_dict = {}
    test_agents_dict["PPO"] = ppo_agents
    test_agents_dict["Proposed Method"] = lili_lstm_agents

    ppo_results = {}
    lili_lstm_results = {}
    results = {}
    results["PPO"] = ppo_results
    results["Proposed Method"] = lili_lstm_results

    ppo_collision = {}
    lili_lstm_collision = {}
    collision_results = {}
    collision_results["PPO"] = ppo_collision
    collision_results["Proposed Method"] = lili_lstm_collision

    ppo_position = {}
    lili_lstm_position = {}
    position_results = {}
    position_results["PPO"] = ppo_position
    position_results["Proposed Method"] = lili_lstm_position

    latents = {}
    for another_type in test_env.world.another_agent_type_list:
        latents[another_type] = np.empty((0, 8))
    pca = PCA(n_components=3)

    seeds = np.random.randint(0, 1001, cfg.test_episode_num).tolist()
    for flex_type in flex_types:
        test_agents = test_agents_dict[flex_type]
        for another_type in test_env.world.another_agent_type_list:
            each_agent_reward = 0
            each_agent_collision_reward = 0
            each_agent_position_reward = 0
            for i in range(cfg.test_episode_num):
                ep_reward = 0
                test_env.reset(
                    seeds[i], options={"agent_type": another_type, "seed": seeds[i]}
                )
                test_agents["another_agent"].change_env(test_env)
                test_agents["another_agent"].set_agent_type(
                    test_env.world.another_agent_type
                )
                recoder.init()
                ep_reward = 0
                ep_collision_reward = 0
                ep_position_reward = 0
                for t in range(1, cfg.max_cycle + 2):
                    for agent in test_env.possible_agents:
                        state, reward, done, truncated, info = test_env.last()
                        end = done or truncated
                        action = test_agents[agent].select_action(state, t, end)
                        if agent == "flex_agent":
                            test_agents[agent].buffer.rewards.append(reward)
                            test_agents[agent].buffer.is_terminals.append(done)
                            ep_reward += reward
                            if info != {}:
                                ep_position_reward += info["position_reward"]
                                ep_collision_reward += info["collision_reward"]
                            if flex_type == "Proposed Method" and t > 100:
                                latent = test_agents[agent].buffer.latents[-1]
                                latents[another_type] = np.concatenate(
                                    (latents[another_type], latent.reshape(1, -1)), 0
                                )

                        if i < 3:
                            recoder.record(test_env)
                        if done or truncated:
                            action = None
                        test_env.step(action)
                    if done or truncated:
                        break

                video_name = f"test_{flex_type}_{another_type}_{i}.mp4"
                recoder.save(video_name)
                each_agent_reward += ep_reward
                each_agent_collision_reward += ep_collision_reward
                each_agent_position_reward += ep_position_reward
                if i < 3 and cfg.track:
                    run.log(
                        {
                            "test_video": wandb.Video(
                                f"{recoder.save_dir}/{video_name}", fps=24
                            )
                        }
                    )
            results[flex_type][another_type] = each_agent_reward / cfg.test_episode_num
            collision_results[flex_type][another_type] = (
                each_agent_collision_reward / cfg.test_episode_num
            )
            position_results[flex_type][another_type] = (
                each_agent_position_reward / cfg.test_episode_num
            )

    result_types = ["total_results", "collision_results", "position_results"]
    all_results = {
        "total_results": results,
        "collision_results": collision_results,
        "position_results": position_results,
    }

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
    plt.savefig("latent_dim.png")

    for result_type in result_types:
        labels = all_results[result_type][flex_types[0]].keys()
        x = range(len(labels))

        fig, ax = plt.subplots()
        ax.bar(
            x,
            all_results[result_type][flex_types[0]].values(),
            width=0.3,
            label=flex_types[0],
        )
        ax.bar(
            [i + 0.3 for i in x],
            all_results[result_type][flex_types[1]].values(),
            width=0.3,
            label=flex_types[1],
        )

        ax.set_xlabel("Another Agent Type")
        ax.set_ylabel("Average Reward")
        # ax.set_title(f"{result_type}_comparison of {flex_types[0]} and {flex_types[1]}")
        ax.set_xticks([i + 0.2 for i in x])
        ax.set_xticklabels(labels)
        ax.legend()
        fig.savefig(f"{result_type}_comparison.png")
        if cfg.track:
            run.log({"test_results": wandb.Image(f"{result_type}_comparison.png")})

    print(
        "============================================================================================"
    )
    print("All done")
    print(
        "============================================================================================"
    )


@hydra.main(version_base=None, config_path="config", config_name="mpe")
def main(cfg: DictConfig):
    env_name = "simple_spread_v3"

    env = simple_spread_v3.env(
        N=cfg.agent_num,
        LN=cfg.landmark_num,
        local_ratio=0.5,
        max_cycles=cfg.max_cycle,
        continuous_actions=False,
    )
    validation_env = simple_spread_v3.env(
        N=cfg.agent_num,
        LN=cfg.landmark_num,
        local_ratio=0.5,
        max_cycles=cfg.max_cycle,
        continuous_actions=False,
        render_mode="rgb_array",
    )
    test_env = simple_spread_v3.env(
        N=cfg.agent_num,
        LN=cfg.landmark_num,
        local_ratio=0.5,
        max_cycles=cfg.max_cycle,
        continuous_actions=False,
        render_mode="rgb_array",
    )

    agent_type = ["PPO", "lili", "Proposed Method"][cfg.model_number]

    if agent_type == "PPO" or agent_type == "lili":
        cuda_num = 0
    else:
        cuda_num = 0

    if cfg.train:
        name = f"{agent_type}_{cfg.agent_num}agent_{cfg.landmark_num}land"
    else:
        name = f"test_{cfg.agent_num}agent_{cfg.landmark_num}land_test"

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

    root_dir = os.getcwd()
    recoder = VideoRecorder(root_dir)

    device = torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{cuda_num}")
        torch.cuda.empty_cache()

    directory = "models"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + "/" + env_name + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    flex_checkpoint_path = directory + f"{name}.pth"

    if cfg.train:
        train(
            cfg,
            run,
            env,
            validation_env,
            recoder,
            device,
            flex_checkpoint_path,
            agent_type,
        )
    else:
        test(cfg, run, test_env, recoder, device, directory)


if __name__ == "__main__":
    main()
