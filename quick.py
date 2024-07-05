import os

from envs.mpe_fixed_env import simple_spread_v3
from utils.recoder import VideoRecorder

env = simple_spread_v3.env(
    N=1,
    max_cycles=120,
    continuous_actions=False,
    render_mode="rgb_array",
)
env.reset(seed=42)
root_dir = os.getcwd()
recoder = VideoRecorder(root_dir)
recoder.init()
# scripted_agent = ANOTHER_AGENT(env, 0.1)
# scripted_agent.set_agent_type(env.world.another_agent_type)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        # if agent == "another_agent":
        #     action = scripted_agent.select_action(observation)
        # else:
        action = env.action_space(agent).sample()
    recoder.record(env)

    env.step(action)
video_name = "test.mp4"
recoder.save(video_name)
env.close()
