import gym
import gym_minigrid
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from copy import deepcopy
import pandas as pd


# we use ppo_iter to create sub_batches
def ppo_iter(mini_batch_size, states, actions, log_probs, returns, values, advantages):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids], actions[rand_ids], log_probs[rand_ids], returns[rand_ids], values[rand_ids], advantages[rand_ids]
        
        
#I use nearly the same code as in BabyAI but I simplified a lot
def ppo_update(model, optimizer_generator,
    mini_batch_size, states, actions, log_probs, returns, values, advantages, 
               ppo_epochs = 4, 
               clip_param=0.2, 
               value_loss_coef=0.5, 
               entropy_coef = 0.01):
    
    for _ in range(ppo_epochs):
        
        for state, action, old_log_probs, return_, value_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, values, advantages):
            
            dist, value = model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = torch.mul(ratio, advantage)
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
            actor_loss  = - torch.min(surr1, surr2).mean()
            
            value_clipped = value_ + torch.clamp(value - value_, -clip_param, clip_param)
            surr1 = (value - return_).pow(2)
            surr2 = (value_clipped - return_).pow(2)
            value_loss = torch.max(surr1, surr2).mean()
            
            loss = actor_loss - entropy_coef * entropy + value_loss_coef * value_loss
            optimizer_generator.zero_grad()
            loss.backward()
            
            grad_norm = sum(p.grad.data.norm(2) ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer_generator.step()



def compute_gae(next_value, rewards, masks, values, gae_discount = 0.9, gae_lambda =0.85):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gae_discount * values[step + 1] * masks[step] - values[step]
        gae = delta + gae_discount * gae_lambda * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns