import gym
import pickle
import blosc
import gym_minigrid
import torch
import numpy as np
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output



#device
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")


#same function as in BabyAI
def transform_demos(demos):
    '''
    takes as input a list of demonstrations in the format generated with `make_agent_demos` or `make_human_demos`
    i.e. each demo is a tuple (mission, blosc.pack_array(np.array(images)), directions, actions)
    returns demos as a list of lists. Each demo is a list of (obs, action, done) tuples
    '''
    new_demos = []
    for demo in demos:
        new_demo = []

        mission = demo[0]
        all_images = demo[1]
        directions = demo[2]
        actions = demo[3]

        all_images = blosc.unpack_array(all_images)
        n_observations = all_images.shape[0]
        assert len(directions) == len(actions) == n_observations, "error transforming demos"
        for i in range(n_observations):
            obs = {'image': all_images[i],
                   'direction': directions[i],
                   'mission': mission}
            action = actions[i]
            done = i == n_observations - 1
            new_demo.append((obs, action, done))
        new_demos.append(new_demo)
    return new_demos


#f = open("/Users/leobix/demos/BabyAI-GoToRedBall-v0_agent.pkl","rb")

def add_absorbing(states, actions):
    state = [np.zeros((7,7,3))]
    state = torch.tensor(state, device=device, dtype=torch.float)
    action = torch.tensor([6], device=device, dtype=torch.long)
    states.append(state)
    actions.append(action)

def get_demos(filepath = "/Users/leobix/demos/BabyAI-GoToRedBall-v0_agent.pkl"):
    f = open(filepath,"rb")
    demos = pickle.load(f)
    dem = transform_demos(demos)
    expert_traj = [[],[]]
    for i in range(len(dem)):
        for j in range(len(dem[i])):
            state = np.array([dem[i][j][0]["image"]])
            state = torch.tensor(state, device=device, dtype=torch.float)
            action = np.array([dem[i][j][1].value])
        
            if action != np.array([6]):
                action = torch.tensor(action, device=device, dtype=torch.long)
                expert_traj[0].append(state)
                expert_traj[1].append(action)
        
            else:
                add_absorbing(expert_traj[0], expert_traj[1])
                add_absorbing(expert_traj[0], expert_traj[1])
             
    return torch.cat(expert_traj[0]), torch.cat(expert_traj[1])



def plot(update, successes):
    clear_output(True)
    plt.figure(figsize=(15,15))
    #plt.subplot(331)
    plt.title('update %s. success: %s' % (update, successes[-1]))
    #plt.suptitle('disc_update %s , gen_update %s , lr %s , num_states_actions_gen %s, num_states_actions_exp %s , mini_batch_size %s , ppo_epochs %s' % (disc_update, gen_update, lr, num_states_actions_gen, num_states_actions_exp, mini_batch_size, ppo_epochs))
    plt.plot(successes)
    #plt.savefig('/Users/leobix/Desktop/GAIL/fig2602_random_num_exp%s.png' %num_states_actions_gen)
    plt.show()
    
def test_env(vis=False):
    state = env.reset()
    obs = state["image"]
    if vis: env.render()
    done = False
    total_reward = 0
    i = 0
    success = False
    while not done :
        i +=1
        obs         = np.array([obs])
        obs         = torch.tensor(obs, device=device, dtype=torch.float)
        dist, _ = model(obs)
        action      = dist.sample()
        action_done = (action ==6)
        play        = env.step(action.cpu().numpy())
        next_obs, reward, done = play[0]['image'], play[1], play[2]
        obs = next_obs
        if vis: 
            env.render()
            time.sleep(0.1)
        total_reward += reward
        success = (total_reward > 0)
    if vis: env.render()
    return [total_reward, success, i]


def eval_env(env, obs, model, vis=False):
    done = False
    total_reward = 0
    i = 0
    success = False
    while not done :
        i +=1
        obs         = np.array([obs])
        obs         = torch.tensor(obs, device=device, dtype=torch.float)
        dist, _     = model(obs)
        action      = dist.sample()
        play        = env.step(action.cpu().numpy())
        next_obs, reward, done = play[0]['image'], play[1], play[2]
        obs = next_obs
        if vis: env.render()
        total_reward += reward
        success = (total_reward > 0)
    if vis: env.render()
    return [total_reward, success, i]   



def eval_envs(n=50):
    envs = []
    origin_states = []
    for i in range(n):
        e = gym.make('BabyAI-GoToRedBall-v0')
        envs.append(e)
        origin_states.append(e.step(5)[0]['image'])
    return envs, origin_states

def deep_copy_envs(envs):
    envs2 = []
    for i in range(len(envs)):
        envs2.append(deepcopy(envs[i]))
    return envs2

def eval_gail(envs, origin_states, model):
    envs2 = deep_copy_envs(envs)
    test_success = np.mean([eval_env(envs2[i], origin_states[i], model)[1] for i in range(len(envs2))])
    return test_success