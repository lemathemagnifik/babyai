import torch
import gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from gail.utils import *
#from gail.utils import eval_envs
#from gail.utils import add_absorbing
#from gail.algos import compute_gae
from gail.algos import *

#device
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)
         
class ActorCriticGenerator(nn.Module):
    def __init__(self, embedding_size = 64):
        super(ActorCriticGenerator, self).__init__()
        
        # We use the cnn1 architecture for image_conv
        self.image_conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(2, 2)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 2)),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=embedding_size, kernel_size=(2, 2)),
                nn.ReLU()
            )
        
        self.actor = nn.Sequential(
            nn.Linear(embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 7) #7 for 7 possible actions
        )

        self.critic = nn.Sequential(
            nn.Linear(embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1) #1 for 1 value
        )
                
        self.apply(init_weights)

        
    #state has to be a tensor of shape [-1,7,7,3]
    def forward(self, x):
        
        #we transpose to get a tensor of shape [-1,3,7,7]
        x     = torch.transpose(torch.transpose(x, 1, 3), 2, 3)
        x     = self.image_conv(x)
            
        #we now have a tensor of shape [-1,64,1,1] that we reshape in [-1,64]
        x     = x.reshape(x.shape[0], -1)
        embedding = x
        x     = self.actor(embedding)
        dist  = Categorical(logits=F.log_softmax(x,dim=1))
        
        x     = self.critic(embedding)
        value = x.squeeze(1)
        
        return dist, value







class Discriminator(nn.Module):
    def __init__(self, embedding_size = 64, hidden_size = 128):
        super(Discriminator, self).__init__()
        
        self.image_conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(2, 2)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 2)),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=embedding_size, kernel_size=(2, 2)),
                nn.ReLU()
            )
        
        #embedding_size + 7 because we feed an (embedded state, action) pair
        self.linear1   = nn.Linear(embedding_size+7, hidden_size)
        self.linear2   = nn.Linear(hidden_size, hidden_size)
        self.linear3   = nn.Linear(hidden_size, 1)
        self.linear3.weight.data.mul_(0.1)
        self.linear3.bias.data.mul_(0.0)
    
    #remark that the discriminator takes as an input an observation and an action
    #the state will go through the conv_layer
    #then we stack the state and the action together 
    #to create a state_action that we will discriminate
    #state has to be a tensor of shape [-1,7,7,3]
    #action has to be a tensor of shape [-1,1]
    
    def forward(self, state, action):
        
        #we transpose to get a tensor of shape [-1,3,7,7]
        state     = torch.transpose(torch.transpose(state, 1, 3), 2, 3)
        state     = self.image_conv(state) 
    
        #we now have a tensor of shape [-1,64,1,1] that we reshape in [-1,64]
        state     = state.reshape(state.shape[0], -1)
        
        #we create a one-hot action vector 
        one_hot_action = torch.zeros(action.shape[0],7)
        for i in range(action.shape[0]):
            one_hot_action[i][action[i].data] = 1
        
        action  = one_hot_action
        
        #We reshape the action as [-1,7]
        action    = action.view(-1,7).type(torch.float)
        
        #we now create a state_action pair that we wish to discriminate
        x = torch.cat((state, action), 1)
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        prob = torch.sigmoid(self.linear3(x))
        
        return prob
    
    #computes the reward given by the discriminator
    def reward(self, state, action):
        prob = self.forward(state, action)
        return -np.log(prob.cpu().data.numpy())






#this variable is just used for plotting losses
def imitationgail(env                = "BabyAI-GoToRedBall-v0",
         demos_filepath     = "/Users/leobix/demos/BabyAI-GoToRedBall-v0_agent.pkl",
         max_updates        = 4000,
         lr_gen             = 1e-4, 
         lr_disc            = 3e-4,
         beta1              = 0.9,
         beta2              = 0.999,
         eps                = 1e-8,
         gae_discount       = 0.95,
         gae_lambda         = 0.95,
         batch_size         = 512, 
         mini_batch_size    = 64, 
         ppo_epochs         = 4, 
         eval_set_size      = 100,
         eval_num           = 10,
         absorbing          = True):    

    '''
    demos_filepath : filepath of the demos, stored in a pickle file (generated by make_agent_demos.py for example)
    eval_set_size : size of the evaluation set, starting states are generated randomly 
    eval_num : evaluation will be performed every eval_num discriminator updates
    batch_size_gen : number of (state, action) pairs from the current policy we'll use for every
    policy and discriminator update
    batch_size_exp : number of (state, action) pairs from the expert policy we'll use for every
    discriminator update
    mini_batch_size : size of the sub-batch we'll use for every PPO epoch
    ppo_epochs : number of ppo epochs for every policy update
    disc_update and gen_update : ratio of discriminator updates over generator updates
    gae_discount : gamma factor in generalized advantage estimation
    gae_lambda : lambda factor in gae, lambda = 0 -> temporal difference 
    lambda = 1 -> Monte-Carlo 
    max_updates : maximum number of updates before stopping the training
    absorbing : if True, we add absorbing states after trajectory ended
    '''
  
    torch.manual_seed(0)
    np.random.seed(0)
    #this variable will count the number of iterations in the GAIL loop.
    i_update = 0

    env = gym.make('BabyAI-GoToRedBall-v0')


    ###define models
    model         = ActorCriticGenerator().to(device)
    discriminator = Discriminator().to(device)
    discrim_criterion = nn.BCELoss()
    optimizer_generator  = torch.optim.Adam(model.parameters(), lr_gen, (beta1, beta2), eps)
    optimizer_discrim    = torch.optim.Adam(discriminator.parameters(), lr_disc, (beta1, beta2), eps)
    
    #variables for plotting results
    discrim_loss_lst = []
    test_successes = []
    
    #get demos and evaluation set
    expert_states, expert_actions = get_demos(demos_filepath)
    envs, origin_states = eval_envs(eval_set_size)

    while i_update < max_updates :
        
        i_update += 1
        log_probs = []
        values    = []
        obss      = []
        actions   = []
        rewards   = []
        true_rewards = []
        masks     = []
        entropy   = 0
        done      = False
        frames    = 0
        
        #we begin a new trajectory
        obs = env.reset()["image"]
        #a frame corresponds to a step in the environment
        length_traj = 0
        while frames < batch_size :
            frames     += 1
            length_traj +=1
            obs         = np.array([obs])
            obs         = torch.tensor(obs, device=device, dtype=torch.float)

            dist, value = model(obs)
            
            

            #We sample an action from the distribution (stochastic policy)
            action      = dist.sample()
            
            #we compute the reward associated to this (state, action) pair with the discriminator
            reward      = discriminator.reward(obs, action)
            
            #We go one step ahead
            play        = env.step(action.cpu().numpy())
            next_obs, true_reward, done = play[0]['image'], play[1], play[2]

            log_prob    = dist.log_prob(action)

            entropy    += dist.entropy().mean()
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).to(device))
            true_rewards.append(torch.FloatTensor(np.array([true_reward])).to(device))
            obss.append(obs)
            actions.append(action)
            if not done:
                masks.append(torch.tensor(1, device=device, dtype=torch.float))
             
            obs         = next_obs

    
            #we handle the case we achieved the goal or reached the number of max steps of the env
            if done :
                obs = env.reset()["image"]
                if absorbing and true_reward > 0 and frames < batch_size :
                    for i in range(64 - length_traj):
                        masks.append(torch.tensor(1, device=device, dtype=torch.float))
                        add_absorbing(obss, actions)                    
                        log_prob    = dist.log_prob(action)
                        entropy    += dist.entropy().mean()
                        log_probs.append(log_prob)
                        values.append(value)
                        rewards.append(torch.FloatTensor(reward).to(device))
                        true_rewards.append(torch.FloatTensor(np.array([true_reward])).to(device))
                        frames +=1
                masks.append(torch.tensor(0, device=device, dtype=torch.float))
                length_traj = 0

        #we test and plot
        if i_update % 10 == 0:
            #test_success = np.mean([test_env()[1] for _ in range(30)])
            test_success = [eval_gail(envs, origin_states, model)]
            print('test_success %s' %test_success)
            test_successes.append(test_success)
            
            #plot(i_update, test_successes)                
            #plt.plot([i for i in range(len(discrim_loss_lst))], discrim_loss_lst)


        #to handle the last observation of the trajectory
        next_obs         = np.array([next_obs])
        next_obs         = torch.tensor(next_obs, device=device, dtype=torch.float)
        _, next_value    = model(next_obs)

        ####We can choose ground-truth reward by setting true_rewards instead of rewards
        returns          = compute_gae(next_value, rewards, masks, values, gae_discount, gae_lambda)

        returns          = torch.cat(returns).detach().view(-1)
        log_probs        = torch.cat(log_probs).detach()
        values           = torch.cat(values).detach()
        obss             = torch.cat(obss)
        actions          = torch.cat(actions)
        advantages       = returns - values    

        #discriminator_update
        #if i_update % disc_update == 0 or i_update==1:
        if True:

            #to select randomly batch_size_exp expert state-action pairs
            lst = np.random.randint(0, expert_states.shape[0], batch_size)
            expert_states_batch   = expert_states[lst]
            expert_actions_batch  = expert_actions[lst]

            fake = discriminator(obss,actions)
            real = discriminator(expert_states_batch, expert_actions_batch)
            
            optimizer_discrim.zero_grad()
            #We consider prob = 1 if fake and 0 if expert
            discrim_loss = discrim_criterion(fake, torch.ones((obss.shape[0], 1)).to(device)) + \
                discrim_criterion(real, torch.zeros((expert_states_batch.shape[0], 1)).to(device))
            discrim_loss.backward()
            discrim_loss_lst.append(discrim_loss)
            optimizer_discrim.step()
        
        #generator_update
        #if i_update % gen_update == 0:
        if True:
            ppo_update(model            = model,
                        optimizer_generator = optimizer_generator,
                        mini_batch_size = mini_batch_size, 
                        states          = obss,
                        actions         = actions, 
                        log_probs       = log_probs, 
                        returns         = returns, 
                        values          = values, 
                        advantages      = advantages, 
                        ppo_epochs      = ppo_epochs)
            
    return test_successes, discrim_loss_lst