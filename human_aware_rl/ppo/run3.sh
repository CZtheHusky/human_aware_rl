CUDA_VISIBLE_DEVICES=4 python ppo_rllib_client.py with lr=5e-4 layout_name=multiplayer_schelling num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=4 python ppo_rllib_client.py with lr=5e-4 layout_name=pipeline num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=4 python ppo_rllib_client.py with lr=5e-4 layout_name=schelling num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=4 python ppo_rllib_client.py with lr=5e-4 layout_name=schelling_s num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=4 python ppo_rllib_client.py with lr=5e-4 layout_name=small_corridor num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=4 python ppo_rllib_client.py with lr=5e-4 layout_name=soup_coordination num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=4 python ppo_rllib_client.py with lr=5e-4 layout_name=unident num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=4 python ppo_rllib_client.py with lr=5e-4 layout_name=you_shall_not_pass num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100
