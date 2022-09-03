from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.agents.agent import AgentPair, FixedPlanAgent, GreedyHumanModel, RandomAgent, SampleAgent
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from human_aware_rl.rllib.rllib import load_agent, load_agent_pair


ae = AgentEvaluator.from_layout_name({"layout_name": "cramped_room"}, {"horizon": 400})
trajs = ae.evaluate_human_model_pair(num_games=10)

print(trajs)




