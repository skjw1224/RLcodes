import argparse
import os
import torch
import numpy as np
import random

from RLcodes import algorithm
from RLcodes import environment


def get_config():
    parser = argparse.ArgumentParser(description='EPEL RL')

    # Basic settings
    parser.add_argument('--algo', type=str, default='DDPG', help='RL algorithm')
    parser.add_argument('--env', type=str, default='CSTR', help='Environment')
    parser.add_argument('--seed', type=int, default=0, help='Seed number')
    parser.add_argument('--device', type=str, default='cpu', help='Device - cuda or cpu')
    parser.add_argument('--save_freq', type=int, default=20, help='Save frequency')
    parser.add_argument('--save_model', action='store_true', help='Whether to save model or not')
    parser.add_argument('--load_model', action='store_true', help='Whether to load saved model or not')

    # Training settings
    parser.add_argument('--max_episode', type=int, default=20, help='Maximum training episodes')
    parser.add_argument('--init_ctrl_idx', type=int, default=0, help='Episodes for training with initial controller')
    parser.add_argument('--buffer_size', type=int, default=1000000, help='Replay buffer size')
    parser.add_argument('--batch_size', type=int, default=1024, help='Mini-batch size')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount')
    parser.add_argument('--warm_up_episode', type=int, default=10, help='Number of warm up episode')
    parser.add_argument('--num_evaluate', type=int, default=3, help='Number of evaluation per episode')

    # Neural network parameters
    parser.add_argument('--num_hidden_nodes', type=int, default=128, help='Number of hidden nodes in MLP')
    parser.add_argument('--num_hidden_layers', type=int, default=2, help='Number of hidden layers in MLP')
    parser.add_argument('--tau', type=float, default=0.005, help='Parameter for soft target update')
    parser.add_argument('--adam_eps', type=float, default=1e-6, help='Epsilon for numerical stability')
    parser.add_argument('--l2_reg', type=float, default=1e-3, help='Weight decay (L2 penalty)')
    parser.add_argument('--grad_clip_mag', type=float, default=5.0, help='Gradient clipping magnitude')
    parser.add_argument('--critic_lr', type=float, default=1e-4, help='Critic network learning rate')
    parser.add_argument('--actor_lr', type=float, default=1e-4, help='Actor network learning rate')

    # RBF parameters
    parser.add_argument('--rbf_dim', type=int, default=100, help='Dimension of RBF basis function')
    parser.add_argument('--rbf_type', type=str, default='gaussian', help='Type of RBF basis function')

    args = parser.parse_args()

    # Algorithm specific settings
    if args.algo == 'A2C':
        args.use_mc_return = False
    elif args.algo == 'DDPG':
        pass
    elif args.algo == 'DQN':
        args.max_n_action_grid = 200

    # Derivative setting
    args.need_derivs = False
    args.need_noise_derivs = False
    args.need_deriv_inverse = False


    
    # Discrete action space setting
    args.is_discrete_action = False
    if args.algo in ['DQN']:
        args.is_discrete_action = True
       
    return vars(args)


def get_env(config):
    env_name = config['env']

    if env_name == 'CSTR':
        if environment.CSTR is None:
            raise ImportError("CSTR environment requires 'casadi'. Please install casadi to use env='CSTR'.")
        env = environment.CSTR(config)
    else:
        raise NameError('Wrong environment name')

    return env


def get_algo(config, env):
    algo_name = config['algo']
    config['s_dim'] = env.s_dim
    config['a_dim'] = env.a_dim
    config['p_dim'] = env.p_dim
    config['nT'] = env.nT
    config['dt'] = env.dt

    if algo_name == 'A2C':
        algo = algorithm.A2C(config)
    elif algo_name == 'DDPG':
        algo = algorithm.DDPG(config)
    elif algo_name == 'DQN':
        algo = algorithm.DQN(config)
    else:
        raise NameError('Wrong algorithm name')

    return algo


def set_seed(config):
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
