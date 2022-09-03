CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=asymmetric_advantages_tomato num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=bottleneck num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=centre_objects num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=centre_pots num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=corridor num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=counter_circuit_o_1order num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=cramped_room_o_3orders num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=cramped_room_single num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100
