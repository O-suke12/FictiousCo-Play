import os
from datetime import datetime

import hydra
import pytz
import torch
from omegaconf import DictConfig

import wandb
from envs.mpe_fixed_env import simple_spread_v3
from originalLILI import LILI
from originPPO import PPO
from scripted_agent import ANOTHER_AGENT
from utils.recoder import VideoRecorder

lili = False
env_name = "simple_spread_v3"
agent_num = 2
landmark_num = 3
agent_type = [None, "fixed", "fixed_dynamics", "following"]
another_agent_type = agent_type[2]
encoder_decoder_pretrain = False
max_cycle = 100

env = simple_spread_v3.env(
    N=agent_num,
    LN=landmark_num,
    another_agent_type=another_agent_type,
    local_ratio=0.5,
    max_cycles=max_cycle,
    continuous_actions=False,
)
validation_env = simple_spread_v3.env(
    N=agent_num,
    LN=landmark_num,
    another_agent_type=another_agent_type,
    local_ratio=0.5,
    max_cycles=max_cycle,
    continuous_actions=False,
    render_mode="rgb_array",
)

if lili:
    agent_type = "lili"
    cuda_num = 0
else:
    agent_type = "ppo"
    cuda_num = 1

name = f"{another_agent_type}_{agent_type}_{agent_num}agent_{landmark_num}land"

state_dim = env.observation_space(env.possible_agents[0]).shape[0]
action_dim = env.action_space(env.possible_agents[0]).n
val_count = 0


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

another_checkpoint_path = directory + f"another_{name}.pth"
flex_checkpoint_path = directory + f"flex_{name}.pth"


################################### Training ###################################
@hydra.main(version_base=None, config_path="config", config_name="lili")
def train(cfg: DictConfig):
    japan_tz = pytz.timezone("Japan")
    now = datetime.now(japan_tz)

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

    ################# training procedure ################
    if lili:
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

    else:
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

    another_agent = ANOTHER_AGENT(env, another_agent_type, 0.1)

    agents = {}

    agents[env.possible_agents[0]] = another_agent
    agents[env.possible_agents[1]] = flex_agent

    if lili and cfg.load_ed:
        flex_agent.load_ed(flex_checkpoint_path)

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

        for t in range(1, cfg.max_ep_len + 1):
            for agent in env.possible_agents:
                # select action with policy
                state, reward, done, truncated, info = env.last()
                action = agents[agent].select_action(state)

                # saving reward and is_terminals
                if agents[agent] == flex_agent:
                    agents[agent].buffer.rewards.append(reward)
                    agents[agent].buffer.is_terminals.append(done)

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
                    validation(agents, cfg, run)
                    agents["another_agent"].change_env(env)
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


def validation(agents, cfg, run):
    test_running_reward = 0
    global val_count

    agents["another_agent"].change_env(env)

    with torch.no_grad():
        ep_reward = 0
        validation_env.reset()
        recoder.init()

        for t in range(1, cfg.max_ep_len + 1):
            for agent in validation_env.possible_agents:
                state, reward, done, truncated, info = validation_env.last()
                action = agents[agent].just_select_action(state)

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


if __name__ == "__main__":
    train()
