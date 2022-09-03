CUDA_VISIBLE_DEVICES=5 python ppo_rllib_client.py with lr=5e-4 layout_name=cramped_room_tomato num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=5 python ppo_rllib_client.py with lr=5e-4 layout_name=five_by_five num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=5 python ppo_rllib_client.py with lr=5e-4 layout_name=forced_coordination_tomato num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=5 python ppo_rllib_client.py with lr=5e-4 layout_name=inverse_marshmallow_experiment num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=5 python ppo_rllib_client.py with lr=5e-4 layout_name=large_room num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=5 python ppo_rllib_client.py with lr=5e-4 layout_name=long_cook_time num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=5 python ppo_rllib_client.py with lr=5e-4 layout_name=marshmallow_experiment_coordination num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=5 python ppo_rllib_client.py with lr=5e-4 layout_name=marshmallow_experiment num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100
