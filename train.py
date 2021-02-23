import ray
from ray import tune
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.models import ModelCatalog
from ray.tune import run_experiments
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer
from QCGym.environments.generic_env import GenericEnv
import argparse
import os

parser = argparse.ArgumentParser(
    description='Reinforcement Learning for Quantum control')
parser.add_argument('--algo', type=str, default="PPO",
                    help='Algorithm to be used for training. Default: PPO')
parser.add_argument('--iter', type=int, default=100000,
                    help='Number of training iterations')


if __name__ == '__main__':
    args = parser.parse_args()

    def env_creator():
        return GenericEnv()
    env_name = "cross_resonance"
    register_env(env_name, env_creator)

    exp_dict = {
        'name': 'cross_res',
        'run_or_experiment': args.algo,
        "stop": {
            "training_iteration": args.iter
        },
        'checkpoint_freq': 20,
        "config": {
            "log_level": "WARN",
            "num_workers": os.cpu_count()-1,
            "num_cpus_for_driver": 1,
            "num_cpus_per_worker": 1,
            # "num_gpus_per_worker": 0.33,
            "num_sgd_iter": 10,
            "train_batch_size": 128,
            "lr": 5e-3,
            "model": {"fcnet_hiddens": [300, 300, 300]},
            # "multiagent": {
            #     "policies": policy_graphs,
            #     "policy_mapping_fn": policy_mapping_fn,
            # },
            "env": env_name},
    }

    # Initialize ray and run
    ray.init()
    tune.run(**exp_dict)
