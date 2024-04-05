# #Training and save
# from human_aware_rl.ppo.ppo_rllib_client import ex
# # For all the tunable paramters, check out ppo_rllib_client.py file
# # Note this is not what the configuration should look like for a real experiment
# config_updates = {
#     "results_dir": "path/to/results", #change this to your local directory
#     "layout_name": "cramped_room",
#     "clip_param": 0.2,
#     'gamma': 0.9,
#     'num_training_iters': 10, #this should usually be a lot higher
#     'num_workers': 1,
#     'num_gpus': 0,
#     "verbose": False,
#     'train_batch_size': 800,
#     'sgd_minibatch_size': 800,
#     'num_sgd_iter': 1,
#     "evaluation_interval": 2
# }
# run = ex.run(config_updates=config_updates, options={"--loglevel": "ERROR"})


#Loading agent pair
from human_aware_rl.rllib.rllib import load_agent_pair
agent_path = "src/overcooked_demo/server/static/assets/agents/RllibCrampedRoomSP/agent"
ap_sp = load_agent_pair(agent_path,"ppo","ppo")


#evaluation
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
layout = "cramped_room"
ae = AgentEvaluator.from_layout_name(mdp_params={"layout_name": layout, "old_dynamics": True}, 
                                     env_params={"horizon": 400})
trajs = ae.evaluate_agent_pair(ap_sp,10,400)


#visualise
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
StateVisualizer().display_rendered_trajectory(trajs, ipython_display=True)