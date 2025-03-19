"""This tutorial shows how to train an MADDPG agent on the space invaders atari environment.

Authors: Michael (https://github.com/mikepratt1), Nick (https://github.com/nicku-a)
"""
import numpy as np
import torch

from pettingzoo.mpe import simple_speaker_listener_v4, simple_tag_v3, simple_spread_v3, simple_push_v3, simple_adversary_v3, simple_crypto_v3

from get_args import get_args
from train import train_algorithm

if __name__ == "__main__":
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #   Network architecture based on paper.
    #   'Unless otherwise specified, our policies are parameterized by a two-layer ReLU MLP with 64 units per layer.'
    NET_CONFIG = {
        "arch": "mlp",  # Network architecture
        "hidden_size": [32, 32],  # Actor hidden size
    }

    # Define the initial hyperparameters
    INIT_HP = {
        "POPULATION_SIZE": 4,
        "ALGO": "MADDPG",  # Algorithm
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        "CHANNELS_LAST": False,
        "BATCH_SIZE": 1024,  # Batch size
        "O_U_NOISE": True,  # Ornstein Uhlenbeck action noise
        "EXPL_NOISE": 0.1,  # Action noise scale
        "MEAN_NOISE": 0.0,  # Mean action noise
        "THETA": 0.15,  # Rate of mean reversion in OU noise
        "DT": 0.01,  # Timestep for OU noise
        "LR_ACTOR": 0.01,  # Actor learning rate
        "LR_CRITIC": 0.01,  # Critic learning rate
        "GAMMA": 0.95,  # Discount factor
        "MEMORY_SIZE": 100000,  # Max memory buffer size
        "LEARN_STEP": 100,  # Learning frequency
        "TAU": 0.01,  # For soft update of target parameters
        "POLICY_FREQ": 2,  # Policy frequnecy
    }

    num_envs = 8
    # Define the simple speaker listener environment as a parallel environment

    env = None
    if args.env == "simple_tag":
        env = simple_tag_v3
    elif args.env == "simple_speaker_listener":
        env = simple_speaker_listener_v4
    elif args.env == "simple_spread":
        env = simple_spread_v3
    elif args.env == "simple_push":
        env = simple_push_v3
    elif args.env == "simple_adversary":
        env = simple_adversary_v3
    elif args.env == "simple_crypto":
        env = simple_crypto_v3
    
    torch.manual_seed(0)
    train_algorithm(env, args.env, NET_CONFIG, INIT_HP, num_envs, args.max_steps, args.use_ernie, device)