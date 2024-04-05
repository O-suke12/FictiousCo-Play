from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_env import Overcooked
from overcooked_ai_py.agents.agent import AgentPair


def main():
    mdp = OvercookedGridworld.from_layout_name("cramped_room")
    base_env = OvercookedEnv.from_mdp(mdp, horizon=500)
    env = Overcooked(base_env, base_env.featurize_state_mdp)
    obs = env.reset()['overcooked_state']
    agent_path = "overcooked_demo/server/static/assets/agents/RllibCrampedRoomSP/agent"

    from human_aware_rl.rllib.rllib import load_agent_pair
    ap_sp = load_agent_pair(agent_path,"ppo","ppo")

    for _ in range(10):
        action = agents.joint_action(obs)
        obs, reward, done, info = env.step()
        print(obs)
        print(reward)
        print(done)
        print(info)
        print()
        if done:
            env.reset()



if __name__ == "__main__":
    main()