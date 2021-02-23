import ray
from ray import tune
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.models import ModelCatalog
from ray.tune import run_experiments
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer
from QCGym.environments.generic_env import GenericEnv


# Driver code for training
def setup_and_train():

    # Create a single environment and register it
    def env_creator(_):
        return GenericEnv()
    single_env = GenericEnv(max_timesteps=100)
    env_name = "cross_resonance"
    register_env(env_name, env_creator)

    # # Get environment obs, action spaces and number of agents
    # obs_space = single_env.observation_space
    # act_space = single_env.action_space
    # num_agents = single_env.num_agents

    # # Create a policy mapping
    # def gen_policy():
    #     return (None, obs_space, act_space, {})

    # policy_graphs = {}
    # for i in range(num_agents):
    #     policy_graphs['agent-' + str(i)] = gen_policy()

    # def policy_mapping_fn(agent_id):
    #     return 'agent-' + str(agent_id)

    # Define configuration with hyperparam and training details
    config = {
        "log_level": "WARN",
        "num_workers": 3,
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

    # Define experiment details
    exp_name = 'cross_res'
    exp_dict = {
        'name': exp_name,
        'run_or_experiment': 'PPO',
        "stop": {
            "training_iteration": 1000
        },
        'checkpoint_freq': 20,
        "config": config,
    }

    # Initialize ray and run
    ray.init()
    tune.run(**exp_dict)


if __name__ == '__main__':
    setup_and_train()
