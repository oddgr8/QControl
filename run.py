import ray
from ray import tune
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.tune import run_experiments
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer
from QCGym.environments.generic_env import GenericEnv


class Model(object):
    def __init__(self, ckpt_path):
        def env_creator(_):
            return GenericEnv()
        env_name = "cross_resonance"
        register_env(env_name, env_creator)
        ray.init()
        config = {
            "log_level": "WARN",
            "num_workers": 0,
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
            "env": "cross_resonance"}

        agent = ppo.PPOTrainer(config, env=GenericEnv)
        agent.restore(ckpt_path)
        exp_dict = {
            'name': "run",
            'run_or_experiment': "PPO",
            "stop": {
                "training_iteration": 1
            },
            'checkpoint_freq': 20,
            "config": config,
            "restore": "/home/onkar/ray_results/cross_res/PPO_cross_resonance_a9d78_00000_0_2021-02-23_22-16-23/checkpoint_720/checkpoint-720"
        }
        tune.run(**exp_dict)
        self.policy = agent.workers.local_worker().get_policy()

    def act(self, state):
        action = self.policy.compute_single_action([i for i in range(1000)])
        return action


A = Model("/home/onkar/ray_results/cross_res/PPO_cross_resonance_bb5f2_00000_0_2021-02-23_23-42-47/checkpoint_500/checkpoint-500")
print("_____________________________", A.act(0))
