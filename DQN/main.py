


import matplotlib.pyplot as plt

import gym



from collections import deque

import os.path

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import PopulationBasedTraining

import torch.optim
import torch.nn as nn

import torch
from torch.optim import Adam
import torch.nn.functional as F
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from memory import replayMemory


class DQNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed, layer_1=128, layer_2=64):
        '''
        :param state_size:
        :param action_size:
        :param seed:
        :param layer_1:
        :param layer_2:
        '''

        super(DQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        '''specify the network'''
        self.model = nn.Sequential(
            nn.Linear(state_size, layer_1),
            nn.ReLU(True),
            nn.Linear(layer_1, layer_1),
            nn.ReLU(True),
            nn.Linear(layer_1, layer_2),
            nn.ReLU(True),
            nn.Linear(layer_2, action_size),
        )

    def forward(self, state):
        '''
        overrides the function forward
        :param state:
        :return:
        '''

        x = self.model(state)

        return x

class Agent(nn.Module):
    def __init__(self, state_space, action_space, seed, opts):
        super(Agent, self).__init__()

        self.state_space = state_space
        self.action_space = action_space
        self.seed = random.seed(seed)
        self.opts = opts
        self.batch_size = opts.batch

        '''DQNetwork'''

        self.local_model = DQNetwork(state_space, action_space, seed)
        self.target_model = DQNetwork(state_space, action_space, seed)
        self.optimizer = Adam(self.local_model.parameters(), lr=opts.lr)

        '''Replay Memory'''

        self.memory = replayMemory(action_space, opts.memory_size, self.batch_size, seed)

        '''How often to update the model'''

        self.update_every = opts.update_freq

    def step(self, state, action, reward, next_state, done):
        '''
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        '''

        '''save experience to memory'''
        self.memory.add(state, action, reward, next_state, done)

        self.update_every += 1
        if(self.update_every % self.update_every == 0):
            if(len(self.memory) > self.batch_size):
                experience = self.memory.sample()
                self.learn(experience, self.opts.discount_rate)

    def learn(self, experience, gamma):
        '''
        :param experience:
        :param gamma:
        :return:
        '''

        sampled_state, sampled_action, sampled_reward, sampled_next_state, sampled_done = experience

        next_value = self.target_model(sampled_next_state).detach().max(1)[0].unsqueeze(1)

        DQN_target = sampled_reward + (gamma * next_value * (1 - sampled_done))

        DQN_estimation = self.local_model(sampled_state).gather(1, sampled_action)

        loss = F.mse_loss(DQN_estimation, DQN_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.local_model, self.target_model, self.opts.transfer_rate)

    def soft_update(self, local_model, target_model, transfer_rate):
        '''
        :param local_model:
        :param target_model:
        :param transfer_rate:
        :return:
        '''

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(transfer_rate * local_param.data + (1.0 - transfer_rate) * target_param.data)

    def act(self, state, epsilon=0.):
        '''
        :param state:
        :param epsilon:
        :return:
        '''

        if type(state) is np.ndarray:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.local_model.eval()
        with torch.no_grad():
            action_value = self.local_model(state)
        self.local_model.train()

        if(np.random.uniform(0,1,1) < epsilon):
            action = np.random.choice(np.arange(self.action_space))
        else:
            action = np.argmax(action_value.cpu().data.numpy())

        return action

def train(
        model,
        iters,
        last_100_scores,
        epsilon,
        opts,
          ):

    # initialize the environments
    env = gym.make(opts.env_name)

    # empty list to keep track of scores
    score_lst = []

    for eps in range(opts.epochs):

        # reset the score
        rewards = 0.0
        # reset the env
        state = env.reset()

        for k in range(opts.max_iteration):

            action = model.act(state, epsilon)

            next_state, reward, done, info = env.step(action)

            model.step(state, action, reward, next_state, done)
            state = next_state
            rewards += reward

            if done:
                score_lst.append(rewards)
                last_100_scores.append(rewards)
                break

        epsilon = max(opts.min_epsilon, epsilon * opts.decay)

        # print every opts.print_every to keep track of how well the model is being trained
        if iters % opts.print_every == 0 and eps != 0:
            print("# of episode :{}, avg score : {:.2f}".format(iters, np.mean(last_100_scores)))
            score_lst = []

        # if the average score of the last 100 episode is great than or equal to opts.end_condition, then the env is solved
        if np.mean(last_100_scores) >= opts.win_condition:
            print("Environment Solved\n")
            print("# of episode :{}, avg score : {:.2f}".format(iters, np.mean(last_100_scores)))

        if eps == opts.train_iterations_per_step:
            break

    env.close()

    return last_100_scores, epsilon

def main(config, checkpoint_dir=None):

    step = 0
    opts = config['opts']

    opts.seed = config['seed']
    opts.min_epsilon = config['min_epsilon']
    opts.decay = config['decay']
    running_score = deque(maxlen=100)

    # initialize the environments
    env = gym.make(opts.env_name)

    # instantiate the model
    model = Agent(env.observation_space.shape[0], env.action_space.n, opts=opts, seed=0).to(device)


    if checkpoint_dir is not None:

        path = os.path.join(checkpoint_dir, "checkpoint")
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model"])
        step = checkpoint["step"]
        running_score = checkpoint['running_score']

        if "lr" in config:
            for param_group in model.optimizer.param_groups:
                param_group["lr"] = config["lr"]

        if "lambdas" in config:
            opts.lambdas = config['lambdas']

    running_score.append(0)  # appending 0 here so np.mean() works.

    epsilon = 1.0

    while np.mean(running_score) < opts.win_condition:

        running_score, epsilon = train(
            model,
            step,
            running_score,
            epsilon,
            opts,
            )

        step += 1
        with tune.checkpoint_dir(step=step) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(
                {
                    "model": model.state_dict(),
                    "step": step,
                    "running_score": running_score,
                },
                path,
            )

        if step % 5 == 0:
            print(np.mean(running_score))

        tune.report(iters=step, scores=np.mean(running_score))

def pbt(opts):

    ray.init()

    # PBT scheduler
    scheduler = PopulationBasedTraining(
        perturbation_interval=opts.perturb_iter,
        hyperparam_mutations={
            # distribution for resampling
            "lr": lambda: np.random.uniform(1e-4, 1e-5),
        },
    )

    config = {
        "opts": opts,
        "use_gpu": True,
        "lr": tune.choice([0.00005, 0.00001, 0.000025]),
        "min_epsilon": tune.choice([0.1, 0.05, 0.01]),
        "decay": tune.choice([0.99, 0.995, 0.999]),
        "seed":tune.uniform(0, 1000000),
        }

    reporter = CLIReporter(
        metric_columns=["iters", "scores"])

    analysis = tune.run(
        main,
        name="RL",
        scheduler=scheduler,
        resources_per_trial={"cpu": opts.cpu_use, "gpu": opts.gpu_use},
        verbose=1,
        stop={
            "training_iteration": opts.tune_iter,
        },
        metric="scores",
        mode="max",
        num_samples=opts.num_samples,
        progress_reporter=reporter,
        config=config
    )

    all_trials = analysis.trials
    checkpoint_paths = [
        os.path.join(analysis.get_best_checkpoint(t), "checkpoint")
        for t in all_trials
    ]

    best_trial = analysis.get_best_trial("scores", "max", "last-5-avg")
    best_checkpoint = analysis.get_best_checkpoint(best_trial, metric="scores")

    dfs = analysis.trial_dataframes

    fig = plt.figure()
    ax = None  # This plots everything on the same plot
    for d in dfs.values():
        ax = d.scores.plot(ax=ax, legend=False)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Score")
    ax.plot()
    fig.savefig('basicPPO_Training.png')

if __name__ == "__main__":

    #load the options for testing out different configs
    from options import options
    options = options()
    opts = options.parse()

    pbt(opts)


















