#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=asymmetric_advantages_tomato num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=bottleneck num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=centre_objects num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=centre_pots num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=corridor num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=counter_circuit_o_1order num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=cramped_room_o_3orders num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=cramped_room_single num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=5 python ppo_rllib_client.py with lr=5e-4 layout_name=cramped_room_tomato num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=5 python ppo_rllib_client.py with lr=5e-4 layout_name=five_by_five num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=5 python ppo_rllib_client.py with lr=5e-4 layout_name=forced_coordination_tomato num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=5 python ppo_rllib_client.py with lr=5e-4 layout_name=inverse_marshmallow_experiment num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=5 python ppo_rllib_client.py with lr=5e-4 layout_name=large_room num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=5 python ppo_rllib_client.py with lr=5e-4 layout_name=long_cook_time num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=5 python ppo_rllib_client.py with lr=5e-4 layout_name=marshmallow_experiment_coordination num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=5 python ppo_rllib_client.py with lr=5e-4 layout_name=marshmallow_experiment num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=4 python ppo_rllib_client.py with lr=5e-4 layout_name=multiplayer_schelling num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=4 python ppo_rllib_client.py with lr=5e-4 layout_name=pipeline num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=4 python ppo_rllib_client.py with lr=5e-4 layout_name=schelling num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=4 python ppo_rllib_client.py with lr=5e-4 layout_name=schelling_s num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=4 python ppo_rllib_client.py with lr=5e-4 layout_name=small_corridor num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=4 python ppo_rllib_client.py with lr=5e-4 layout_name=soup_coordination num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=4 python ppo_rllib_client.py with lr=5e-4 layout_name=unident num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

CUDA_VISIBLE_DEVICES=4 python ppo_rllib_client.py with lr=5e-4 layout_name=you_shall_not_pass num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100

# CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=counter_circuit num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100
# CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=counter_circuit num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100
# CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=counter_circuit num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100
# CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=counter_circuit num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100
# CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=counter_circuit num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100
# CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=counter_circuit num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100
# CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=counter_circuit num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100
# CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=counter_circuit num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100
# CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=counter_circuit num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100
# CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=counter_circuit num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100
# CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=counter_circuit num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100
# CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=counter_circuit num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100
# CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=counter_circuit num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100
# CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=counter_circuit num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100
# CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=counter_circuit num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100
# CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=counter_circuit num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100
# CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=counter_circuit num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100
# CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=counter_circuit num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100
# CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=counter_circuit num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100
# CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=counter_circuit num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100
# CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=counter_circuit num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100
# CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=counter_circuit num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100
# CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=counter_circuit num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100
# CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=counter_circuit num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100
# CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=counter_circuit num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100
# CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=counter_circuit num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100
# CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=counter_circuit num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100
# CUDA_VISIBLE_DEVICES=6 python ppo_rllib_client.py with lr=5e-4 layout_name=counter_circuit num_gpus=1  reward_shaping_horizon=5000000 num_training_iters=900 save_freq=100