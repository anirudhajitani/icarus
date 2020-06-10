"""Network Model-View-Controller (MVC)

This module contains classes providing an abstraction of the network shown to
the strategy implementation. The network is modelled using an MVC design
pattern.

A strategy performs actions on the network by calling methods of the
`NetworkController`, that in turns updates  the `NetworkModel` instance that
updates the `NetworkView` instance. The strategy can get updated information
about the network status by calling methods of the `NetworkView` instance.

The `NetworkController` is also responsible to notify a `DataCollectorProxy`
of all relevant events.
"""
import logging
import sys
import traceback
import networkx as nx
import fnss
import sys
from itertools import count
from itertools import combinations
import collections
from collections import namedtuple
import numpy as np
from pprint import pprint
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from matplotlib import pyplot as plt
import matplotlib
from icarus.registry import CACHE_POLICY
from icarus.util import iround, path_links

__all__ = [
    'NetworkModel',
    'NetworkView',
    'Agent',
    'Policy',
    'NetworkController'
          ]

logger = logging.getLogger('orchestration')

#Uncomment and provide manual seed if needed
#torch.manual_seed()
torch.autograd.set_detect_anomaly(True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
SavedAction2 = namedtuple('SavedAction2', ['log_prob', 'value'])

def symmetrify_paths(shortest_paths):
    """Make paths symmetric

    Given a dictionary of all-pair shortest paths, it edits shortest paths to
    ensure that all path are symmetric, e.g., path(u,v) = path(v,u)

    Parameters
    ----------
    shortest_paths : dict of dict
        All pairs shortest paths

    Returns
    -------
    shortest_paths : dict of dict
        All pairs shortest paths, with all paths symmetric

    Notes
    -----
    This function modifies the shortest paths dictionary provided
    """
    for u in shortest_paths:
        for v in shortest_paths[u]:
            shortest_paths[u][v] = list(reversed(shortest_paths[v][u]))
    return shortest_paths

def symmetrify_paths_len(shortest_paths_len):
    """Make paths symmetric

    Given a dictionary of all-pair shortest paths, it edits shortest paths to
    ensure that all path are symmetric, e.g., path(u,v) = path(v,u)

    Parameters
    ----------
    shortest_paths : dict of dict
        All pairs shortest paths

    Returns
    -------
    shortest_paths : dict of dict
        All pairs shortest paths, with all paths symmetric

    Notes
    -----
    This function modifies the shortest paths dictionary provided
    """
    for u in shortest_paths_len:
        for v in shortest_paths_len[u]:
            shortest_paths_len[u][v] = shortest_paths_len[v][u]
    return shortest_paths_len



class Policy(nn.Module):
    """

    implements both Actor and Critic in one model
    """

    def __init__(self, s_len, a_len):
        super(Policy, self).__init__()
        #print ("INIT")
        #TODO - change the layers based on the total number actions and values we want
        self.affine1 = nn.Linear(s_len, 256)
        #Don't want to use Dropout with RL (not very stable)
        #self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(256, 128)
        # actor's layer
        self.action_head = nn.Linear(128, a_len)

        # critic's layer
        self.value_head = nn.Linear(128, 1)

        # action and reward buffer
        self.saved_actions = []
        self.rewards = []
        self.to(device)

    def forward(self, x):
        """

        forward pass of both actor and critic
        """
        #print ("POLICY FORWARD")
        x = F.relu(self.affine1(x))
        #x = self.dropout(x)
        x = F.relu(self.affine2(x))

        # actor: choses action to taken based on state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic evaluates being in state s_t
        state_values = self.value_head(x)

        # return value of both actor and critic as a 2 tuple
        # 1. a list of probability of each action over state space
        # 2. the value from state s_t
        return action_prob, state_values

class RNNPolicy(nn.Module):
    """
    A2C Actor and Critic Policy Network with RNN
    """
    def __init__(self, s_len, hidden_dim, a_len):
        super(RNNPolicy, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(s_len, hidden_dim, batch_first = True)
        self.affine1 = nn.Linear(hidden_dim, 256)
        #self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(256, 128)
        # actor's layer
        self.action_head = nn.Linear(128, a_len)

        # critic's layer
        self.value_head = nn.Linear(128, 1)

        # action and reward buffer
        self.saved_actions = []
        self.rewards = []
        self.to(device)

    def forward(self, x, hidden):
        batch_size = x.size(0)
        x, hidden = self.lstm(x, hidden)
        x = x.contiguous().view(-1, self.hidden_dim)
        x = F.relu(self.affine1(x))
        #x = self.dropout(x)
        x = F.relu(self.affine2(x))

        # actor: choses action to taken based on state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)
        action_prob = action_prob.view(batch_size, -1)
        action_prob = action_prob[:,-1]

        # critic evaluates being in state s_t
        state_values = self.value_head(x)
        state_values = state_values.view(batch_size, -1)
        state_values = state_values[:,-1]
        # return value of both actor and critic as a 2 tuple
        # 1. a list of probability of each action over state space
        # 2. the value from state s_t
        return action_prob, state_values, hidden


class Agent(object):
    """

    Each router is considered as an agent. The agent needs to view its state space
    and then select an action (which files to cache based on the current policy it
    is executing.
    """

    def __init__(self, view, router, ver, policy_type, gamma=0.9, lr=3e-2, window=250, comb=0):
        """Constructor
        
        Parameters
        ----------
        model : NetworkView
            The network view instance
        ver : Version
            If ver == 0, select top k softmax actions and append experiences for all during reward calculation
            If ver == 1, action space is combinatorial f^C_k
        policy_type : Policy type
            if policy_type == 0, it is simple Feed-Forward Network
            if policy_type == 1, it is RNN
        """
        if not isinstance(view, NetworkView):
            raise ValueError('The model argument must be an instance of '
                             'NetworkView')
        self.view = view
        self.cache = router
        self.cache_size = self.view.model.cache_size[self.cache]
        #If all cache are equal size, can be moved to model
        self.count = 0
        self.count2 = 1
        self.hidden_state = 256
        self.gamma = gamma
        self.rewards = 0
        self.valid_action = [0, 1]
        print ("VER, POL, COMB", ver, policy_type, comb) 
        #self.indexes = np.zeros((self.view.model.cache_size[self.cache]), dtype=float)
        if self.view.strategy_name in ['INDEX', 'INDEX_DIST']:
            self.indexes = dict()
        if self.view.strategy_name in ['INDEX', 'INDEX_DIST', 'RL_DEC_1', 'RL_DEC_2F', 'RL_DEC_2D']:
            self.requests = collections.deque(maxlen=window)
        #All possible combinations of files (assuming minimum size of file is 1)
        print ("Cache size : ", self.cache_size)
        # if ver == 1 and comb is set to 1 then, combinatorial action, else if 0 find more optimized way of encoding (not yet implemented)
        if self.view.strategy_name in ['RL_DEC_1'] and ver == 1:
            self.action_choice = []
            self.valid_action = []
            for i in range(1, self.view.model.cache_size[self.cache] + 1):
                self.action_choice.extend(list(combinations(self.view.lib, i))) 
            #print ("Action choices : ", self.action_choice)
            #Filter out choice which don't add up to cache size
            for value in self.action_choice:
                val = list(value)
                sum_list = 0
                for v in val:
                    sum_list += self.view.model.workload.contents_len[v]
                #print ("Val : ", val, " Sum : ", sum_list)
                if sum_list <= self.view.model.cache_size[self.cache]:
                    self.valid_action.append(value)
            print ("Content Lens : " ,self.view.model.workload.contents_len)
            print ("Action choices after : ", self.valid_action)
            del self.action_choice
        
        #Try to change this and see the behavior
        """
        if Version == 0
        The state comprises of all the elements currently cached in the router.
        It returns a binary vector of size F (library size) with the entry being
        0 if the file_id is not cached and 1 if it is cached.
        if Version == 1
        The state returns the identifier of the files currently being cached along
        with its binomial/multinomial distribution.
        """
        self.state_ver = ver
        self.policy_type = policy_type
        if self.policy_type == 1:
            self.a_hx = torch.zeros(self.hidden_state).unsqueeze(0).unsqueeze(0).to(device);
            self.a_cx = torch.zeros(self.hidden_state).unsqueeze(0).unsqueeze(0).to(device);
        #self.action_space = combinations(
        if self.state_ver == 0 or self.state_ver == 1:
            if self.view.strategy_name in ['INDEX_DIST', 'RL_DEC_1', 'RL_DEC_2F', 'RL_DEC_2D']:
                self.state_counts = np.full((len(self.view.model.library)), 0, dtype=int) 
            self.state = np.full((len(self.view.model.library)), 0, dtype=int) 
            if self.view.strategy_name in ['INDEX_DIST', 'RL_DEC_2D']:
                self.prob = np.full((len(self.view.model.library)), 1.0, dtype=float) 
                self.prob = self.prob/np.sum(self.prob)
                # Prior distribution (dirchlet)
                self.alpha = np.array(range(1, len(self.view.model.library)+1))
                self.alpha = self.alpha[::-1]
        else:
            #TODO - if needed, we update the statistics
            self.state = np.full((self.view.model.cache_size[self.cache]), 0, dtype=int)
         
        # Initialize the policy and other neural network optimizers
        #state space is lib size + 1 (for the input)

        if self.view.strategy_name in ['RL_DEC_1']:
            try:
                if self.state_ver == 1:
                    if self.policy_type == 0:
                        self.policy = Policy(len(self.view.model.library), len(self.valid_action))
                    else:
                        self.policy = RNNPolicy(len(self.view.model.library), self.hidden_state, len(self.valid_action))
                else:
                    if self.policy_type == 0:
                        self.policy = Policy(len(self.view.model.library), len(self.view.model.library))
                    else:
                        self.policy = RNNPolicy(len(self.view.model.library), self.hidden_state, len(self.view.model.library))

                #print ("POLICY", self.policy)
                self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
                #print ("OPTIMIZER", self.optimizer)
                self.eps = np.finfo(np.float32).eps.item()
                #print ("EPS", self.eps)
            except:
                print(traceback.format_exc())
                print(sys.exc_info()[2])

        if self.view.strategy_name in ['RL_DEC_2F', 'RL_DEC_2D']:
            try:
                if self.policy_type == 0:
                    self.policy = Policy(len(self.view.model.library)*2, len(self.valid_action))
                    #print ("POLICY", self.policy)
                    self.policy2 = Policy(self.cache_size, self.cache_size)
                else:
                    self.policy = RNNPolicy(len(self.view.model.library)*2, self.hidden_state, len(self.valid_action))
                    #print ("POLICY", self.policy)
                    self.policy2 = RNNPolicy(self.cache_size, self.hidden_state, self.cache_size)
                self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
                self.optimizer2 = optim.Adam(self.policy2.parameters(), lr=lr)
                #print ("OPTIMIZER", self.optimizer)
                self.eps = np.finfo(np.float32).eps.item()
                self.eps2 = np.finfo(np.float32).eps.item()
                #print ("EPS", self.eps)
            except:
                print(traceback.format_exc())
                print(sys.exc_info()[2])


    def get_state(self):
        """
        Returns the current state of the cache.
        """
        if self.view.strategy_name in ['INDEX', 'INDEX_DIST', 'RL_DEC_1', 'RL_DEC_2F']:
            return self.state_counts
        if self.view.strategy_name in ['RL_DEC_2D']:
            return self.prob
    
    def get_cache_dump(self):
        """ 
            Return cache dump of node
        """
        contents = self.view.cache_dump(self.cache)
        if self.view.strategy_name in ['RL_DEC_2F']:
            state_con = np.full((self.cache_size), 0, dtype=int)
        if self.view.strategy_name in ['RL_DEC_2D']:
            state_con = np.full((self.cache_size), 0, dtype=float)
        i = 0
        for c in contents :
            if self.view.strategy_name in ['RL_DEC_2F']:
                state_con[i] = self.state_counts[c-1]
            elif self.view.strategy_name in ['RL_DEC_2D']:
                state_con[i] = self.prob[c-1]
            i += 1
        return state_con


    def decode_action(self, action):
        """
        Decode the action and return a vector of binary values signifying which caches
        should cache what content
        """
        action_decoded = np.full((len(self.view.model.library)), 0, dtype=int)
        if self.state_ver == 1:
            files_to_cache = self.valid_action[action[0]]
            files_to_cache = list(files_to_cache)
            #print ("Files to cache", files_to_cache)
            for f in files_to_cache:
                action_decoded[f] = 1
        else:
            for a in action:
                action_decoded[a] = 1
        #print ("Action decoded", action_decoded)
        return action_decoded

    def select_actions(self, state):
        #print ("STATE", state)
        if self.policy_type == 0:
            state = torch.from_numpy(state).float().to(device)
            probs, state_value = self.policy.forward(state)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0).to(device)
            probs, state_value, (self.a_hx, self.a_cx) = self.policy.forward(state, (self.a_hx, self.a_cx))
        #print ("PROBS, STATE_VAL", probs.shape, state_value.shape)
        # create categorical distribution over list of probabilities of actions
        m = Categorical(probs)
        #print ("Output of NN ")
        #print (m)
        # sample an action from the categorical distribution
        # Select top k values from probability softmax output
        if self.state_ver == 0 and self.view.strategy_name in ['RL_DEC_1']:
            top_k_val, top_k_inx = probs.topk(self.cache_size)
            for action in top_k_inx:
                self.policy.saved_actions.append(SavedAction(m.log_prob(action), state_value))
            top_k_inx = top_k_inx.cpu().detach().numpy()
            return top_k_inx
        #print ("TOP K INDEX", top_k_inx, type(top_k_inx))
        #print ("TOP K VAL", top_k_val)
        action = m.sample()
        #print ("Sampled Action", action)
        # save to action buffer
        self.policy.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        cache_cont = self.view.cache_dump(self.cache)
        #print ("CACHE DUMP", cache_cont, len(cache_cont))
        # Only if cache dump is equal to max cache size we need to find what obj to evict
        if action.item() == 1 and len(cache_cont) == self.cache_size and self.view.strategy_name in ['RL_DEC_2D', 'RL_DEC_2F']:
            state2 = self.get_cache_dump()
            action2 = self.select_actions_2(state2)
            #print ("Return evicted content", cache_cont[action2])
            #Return the content that needs to be removed rather than action choice in 2nd argument
            return [action.item(), cache_cont[action2]]
        # return the action
        return [action.item()]

    def select_actions_2(self, state):
        #print ("STATE", state)
        if self.policy_type == 0:
            state = torch.from_numpy(state).float().to(device)
            probs, state_value = self.policy2.forward(state)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0).to(device)
            probs, state_value, (self.a_hx, self.a_cx) = self.policy2.forward(state, (self.a_hx, self.a_cx))
        #print ("PROBS2, STATE_VAL2", probs, state_value)
        # create categorical distribution over list of probabilities of actions
        m = Categorical(probs)
        #print ("Output of NN 2")
        #print (m)
        # sample an action from the categorical distribution
        action = m.sample()
        #print ("Sampled Action 2", action)
        # save to action buffer
        self.policy2.saved_actions.append(SavedAction2(m.log_prob(action), state_value))
        self.count2 += 1
        # return the action
        return action.item()

    def update(self):
        """

        Training code. Calculates actor and critic loss and performn backpropogation.
        """
        #print ("UPDATE FN")
        R = 0
        saved_actions = self.policy.saved_actions
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (policy) loss
        returns = [] # list to save true values

        # calculate the true value using rewards returned from the environment
        # this reward is appended by the environment 
        for r in self.policy.rewards[::-1]:
            # calculate the discounted value
            #print ("Reward : ", r)
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, device=device)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        #print ("Returns ", returns)
        #print ("RETURNS LEN", len(returns))

        #print ("Saved Actions", saved_actions)
        #print ("SAVED ACTIONS LEN", len(saved_actions))
        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)

            # calulate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R], device=device)))
        
        #print ("Policy Loss ", policy_losses)
        #print ("Value Loss ", value_losses)
        # reset gradients
        self.optimizer.zero_grad()
        #print ("Optimizer gradient")
        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        #print ("Total Loss ", loss)
        # perform backprop
        #loss.backward(retain_graph=True)
        loss.backward()
        #print ("Backprop Loss")
        #torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 5)
        #print ("Gradients Clip")
        self.optimizer.step()
        #print ("Optimizer Step")
        # reset rewards and action buffers
        # We dont want to reset it as we want the hidden layers to be preserved
        """
        if self.policy_type == 1:
            self.a_hx = torch.zeros(self.hidden_state).unsqueeze(0).unsqueeze(0).to(device);
            self.a_cx = torch.zeros(self.hidden_state).unsqueeze(0).unsqueeze(0).to(device);
        """
        del self.policy.rewards[:]
        del self.policy.saved_actions[:]
        #print ("Deleted policy and saved actions")

    def update2(self):
        """

        Training code. Calculates actor and critic loss and performn backpropogation.
        """
        #print ("UPDATE FN")
        R = 0
        saved_actions = self.policy2.saved_actions
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (policy) loss
        returns = [] # list to save true values
        # calculate the true value using rewards returned from the environment
        # this reward is appended by the environment 
        for r in self.policy2.rewards[::-1]:
            # calculate the discounted value
            #print ("Reward : ", r)
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, device=device)
        returns = (returns - returns.mean()) / (returns.std() + self.eps2)
        #print ("Returns ", returns)

        #print ("Saved Actions", saved_actions)
        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)

            # calulate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R], device=device)))
        
        #print ("Policy Loss ", policy_losses)
        #print ("Value Loss ", value_losses)
        # reset gradients
        self.optimizer2.zero_grad()
        #print ("Optimizer gradient")
        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        #print ("Total Loss ", loss)
        # perform backprop
        loss.backward(retain_graph=True)
        #print ("Backprop Loss")
        #torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 5)
        #print ("Gradients Clip")
        self.optimizer2.step()
        #print ("Optimizer Step")
        # reset rewards and action buffers
        del self.policy2.rewards[:]
        del self.policy2.saved_actions[:]
        #print ("Deleted policy and saved actions")


class NetworkView(object):
    """Network view

    This class provides an interface that strategies and data collectors can
    use to know updated information about the status of the network.
    For example the network view provides information about shortest paths,
    characteristics of links and currently cached objects in nodes.
    """

    def __init__(self, model, cpus, nnp, strategy_name):
        """Constructor

        Parameters
        ----------
        model : NetworkModel
            The network model instance
        """
        if not isinstance(model, NetworkModel):
            raise ValueError('The model argument must be an instance of '
                             'NetworkModel')
        #TODO - objects allocate only based on strategy (no need to assign data if not used
        self.model = model
        self.count = 0
        self.common_rewards = 0
        self.tot_delay = 0
        self.fetch_delay = 0
        self.strategy_name = strategy_name
        #Different because other library is a set, here we want to preserve ordering
        self.lib = [item for item in range(0, len(self.model.library))]
        #Contains the agents as objects of Class Agent
        self.agents = []
        self.cpus = cpus
        if 'lr' not in nnp:
            nnp['lr'] = 3e-2
        if 'gamma' not in nnp:
            nnp['gamma'] = 0.9
        if 'window' not in nnp:
            nnp['window'] = 250
        if 'update_freq' not in nnp:
            nnp['update_freq'] = 100
        if 'state_ver' not in nnp:
            nnp['state_ver'] = 0
        if 'policy_type' not in nnp:
            nnp['policy_type'] = 0
        if 'comb' not in nnp:
            nnp['comb'] = 0
        #self.status = [False] * cpus
        #self.ind_count = [0] * cpus
        self.update_freq = nnp['update_freq']
        self.window = nnp['window']
        self.state_ver = nnp['state_ver']
        self.policy_type = nnp['policy_type']
        self.comb = nnp['comb']
        #Creating agents depending on the total number of routers
        for r in self.model.routers:
            self.agents.append(Agent(self, r, self.state_ver, self.policy_type, nnp['gamma'], nnp['lr'], nnp['window'], nnp['comb']))
        if strategy_name in ['RL_DEC_1']:
            self.agents_per_thread = int(len(self.agents)/cpus)
            self.extra_agents = len(self.agents) % cpus
            print ("CPUS = ", self.cpus, " Agents per thread = ", self.agents_per_thread, " Extra Agents = ", self.extra_agents)
        if strategy_name in ['INDEX']:
            if 'index_threshold_f' not in nnp:
                nnp['index_threshold_f'] = 10
            self.threshold = nnp['index_threshold_f']
        if strategy_name in ['INDEX_DIST']:
            if 'index_threshold_d' not in nnp:
                nnp['index_threshold_d'] = 0.5
            self.threshold = nnp['index_threshold_d']

    """
    def env_step():

        A step of the environment includes the following for all agents:
        1. get the current state of each agent
        2. select actions
        3. perform the actions
        
        for a in self.agents:
            curr_state = a.get_state()
            action = a.select_actions(curr_state)
            action = a.decode_action(action)
            a.perform_action(action, a.cache)

    def update_gradients():

        Get the rewards from the environment
        Append the rewards to the rewards data structure accessed by the policy class
        Perform actor-critic update
        
        for a in self.agents:
            a.policy.append(self.rewards)
            a.update()
    """

    def content_locations(self, k):
        """Return a set of all current locations of a specific content.

        This include both persistent content sources and temporary caches.

        Parameters
        ----------
        k : any hashable type
            The content identifier

        Returns
        -------
        nodes : set
            A set of all nodes currently storing the given content
        """
        loc = set(v for v in self.model.cache if self.model.cache[v].has(k))
        source = self.content_source(k)
        if source:
            loc.add(source)
        return loc

    def content_source(self, k):
        """Return the node identifier where the content is persistently stored.

        Parameters
        ----------
        k : any hashable type
            The content identifier

        Returns
        -------
        node : any hashable type
            The node persistently storing the given content or None if the
            source is unavailable
        """
        return self.model.content_source.get(k, None)

    def shortest_path(self, s, t):
        """Return the shortest path from *s* to *t*

        Parameters
        ----------
        s : any hashable type
            Origin node
        t : any hashable type
            Destination node

        Returns
        -------
        shortest_path : list
            List of nodes of the shortest path (origin and destination
            included)
        """
        return self.model.shortest_path[s][t]

    def all_pairs_shortest_paths(self):
        """Return all pairs shortest paths

        Return
        ------
        all_pairs_shortest_paths : dict of lists
            Shortest paths between all pairs
        """
        return self.model.shortest_path

    def shortest_path_len(self, s, t):
        """Return the shortest path length from *s* to *t*

        Parameters
        ----------
        s : any hashable type
            Origin node
        t : any hashable type
            Destination node

        Returns
        -------
        shortest_path_len : int
            The shortest path (origin and destination included)
        """
        return self.model.shortest_path_len[s][t]

    def all_pairs_shortest_paths_len(self):
        """Return all pairs shortest paths

        Return
        ------
        all_pairs_shortest_path_len : dict of
            Shortest paths between all pairs
        """
        return self.model.shortest_path_len

    def cluster(self, v):
        """Return cluster to which a node belongs, if any

        Parameters
        ----------
        v : any hashable type
            Node

        Returns
        -------
        cluster : int
            Cluster to which the node belongs, None if the topology is not
            clustered or the node does not belong to any cluster
        """
        if 'cluster' in self.model.topology.node[v]:
            return self.model.topology.node[v]['cluster']
        else:
            return None

    def link_type(self, u, v):
        """Return the type of link *(u, v)*.

        Type can be either *internal* or *external*

        Parameters
        ----------
        u : any hashable type
            Origin node
        v : any hashable type
            Destination node

        Returns
        -------
        link_type : str
            The link type
        """
        return self.model.link_type[(u, v)]

    def link_delay(self, u, v):
        """Return the delay of link *(u, v)*.

        Parameters
        ----------
        u : any hashable type
            Origin node
        v : any hashable type
            Destination node

        Returns
        -------
        delay : float
            The link delay
        """
        return self.model.link_delay[(u, v)]

    def topology(self):
        """Return the network topology

        Returns
        -------
        topology : fnss.Topology
            The topology object

        Notes
        -----
        The topology object returned by this method must not be modified by the
        caller. This object can only be modified through the NetworkController.
        Changes to this object will lead to inconsistent network state.
        """
        return self.model.topology

    def cache_nodes(self, size=False):
        """Returns a list of nodes with caching capability

        Parameters
        ----------
        size: bool, opt
            If *True* return dict mapping nodes with size

        Returns
        -------
        cache_nodes : list or dict
            If size parameter is False or not specified, it is a list of nodes
            with caches. Otherwise it is a dict mapping nodes with a cache
            and their size.
        """
        return {v: c.maxlen for v, c in self.model.cache.items()} if size \
                else list(self.model.cache.keys())

    def has_cache(self, node):
        """Check if a node has a content cache.

        Parameters
        ----------
        node : any hashable type
            The node identifier

        Returns
        -------
        has_cache : bool,
            *True* if the node has a cache, *False* otherwise
        """
        return node in self.model.cache

    def cache_lookup(self, node, content):
        """Check if the cache of a node has a content object, without changing
        the internal state of the cache.

        This method is meant to be used by data collectors to calculate
        metrics. It should not be used by strategies to look up for contents
        during the simulation. Instead they should use
        `NetworkController.get_content`

        Parameters
        ----------
        node : any hashable type
            The node identifier
        content : any hashable type
            The content identifier

        Returns
        -------
        has_content : bool
            *True* if the cache of the node has the content, *False* otherwise.
            If the node does not have a cache, return *None*
        """
        if node in self.model.cache:
            return self.model.cache[node].has(content)

    def local_cache_lookup(self, node, content):
        """Check if the local cache of a node has a content object, without
        changing the internal state of the cache.

        The local cache is an area of the cache of a node reserved for
        uncoordinated caching. This is currently used only by hybrid
        hash-routing strategies.

        This method is meant to be used by data collectors to calculate
        metrics. It should not be used by strategies to look up for contents
        during the simulation. Instead they should use
        `NetworkController.get_content_local_cache`.

        Parameters
        ----------
        node : any hashable type
            The node identifier
        content : any hashable type
            The content identifier

        Returns
        -------
        has_content : bool
            *True* if the cache of the node has the content, *False* otherwise.
            If the node does not have a cache, return *None*
        """
        if node in self.model.local_cache:
            return self.model.local_cache[node].has(content)
        else:
            return False

    def cache_dump(self, node):
        """Returns the dump of the content of a cache in a specific node

        Parameters
        ----------
        node : any hashable type
            The node identifier

        Returns
        -------
        dump : list
            List of contents currently in the cache
        """
        if node in self.model.cache:
            return self.model.cache[node].dump()


class NetworkModel(object):
    """Models the internal state of the network.

    This object should never be edited by strategies directly, but only through
    calls to the network controller.
    """

    def __init__(self, topology, workload, cache_policy, shortest_path=None, shortest_path_len=None):
        """Constructor

        Parameters
        ----------
        topology : fnss.Topology
            The topology object
        cache_policy : dict or Tree
            cache policy descriptor. It has the name attribute which identify
            the cache policy name and keyworded arguments specific to the
            policy
        shortest_path : dict of dict, optional
            The all-pair shortest paths of the network
        """
        # Filter inputs
        if not isinstance(topology, fnss.Topology):
            raise ValueError('The topology argument must be an instance of '
                             'fnss.Topology or any of its subclasses.')
        
        # Shortest paths of the network
        self.shortest_path = dict(shortest_path) if shortest_path is not None \
                             else symmetrify_paths(dict(nx.all_pairs_dijkstra_path(topology)))
       
        # Shortest paths length of the network
        self.shortest_path_len = dict(shortest_path_len) if shortest_path_len is not None \
                                else symmetrify_paths_len(dict(nx.all_pairs_dijkstra_path_length(topology, weight='delay')))
        #print ("Shortest Path Lengths: ", self.shortest_path_len)
        # Network topology
        self.topology = topology
        self.workload = workload
        # Dictionary mapping each content object to its source
        # dict of location of contents keyed by content ID
        self.content_source = {}
        # Dictionary mapping the reverse, i.e. nodes to set of contents stored
        self.source_node = {}
        #List of all routers in the topology
        self.routers = []
        #List of all files in the library
        self.library = set()
        # Color dictionary
        self.color_dict = dict()
        # Dictionary of link types (internal/external)
        self.link_type = nx.get_edge_attributes(topology, 'type')
        self.link_delay = fnss.get_delays(topology)
        # Instead of this manual assignment, I could have converted the
        # topology to directed before extracting type and link delay but that
        # requires a deep copy of the topology that can take long time if
        # many content source mappings are included in the topology
        if not topology.is_directed():
            for (u, v), link_type in list(self.link_type.items()):
                self.link_type[(v, u)] = link_type
            for (u, v), delay in list(self.link_delay.items()):
                self.link_delay[(v, u)] = delay

        self.cache_size = {}
        for node in topology.nodes():
            stack_name, stack_props = fnss.get_stack(topology, node)
            #print ("_____________", type(stack_props))
            #print (node, list(topology.neighbors(node)))
            if stack_name == 'router':
                if 'cache_size' in stack_props:
                    self.cache_size[node] = stack_props['cache_size']
                    self.routers.append(node)
                    self.color_dict[node] = 'r'
            elif stack_name == 'source':
                contents = stack_props['contents']
                self.color_dict[node] = 'b'
                self.source_node[node] = contents
                for content in contents:
                    self.library.add(content)
                    self.content_source[content] = node
            else:
                    self.color_dict[node] = 'g'
        if any(c < 1 for c in self.cache_size.values()):
            logger.warn('Some content caches have size equal to 0. '
                        'I am setting them to 1 and run the experiment anyway')
            for node in self.cache_size:
                if self.cache_size[node] < 1:
                    self.cache_size[node] = 1
        print ("ROUTERS", self.routers)
        policy_name = cache_policy['name']
        policy_args = {k: v for k, v in cache_policy.items() if k != 'name'}
        # The actual cache objects storing the content
        self.cache = {node: CACHE_POLICY[policy_name](self.cache_size[node], **policy_args)
                          for node in self.cache_size}

        # This is for a local un-coordinated cache (currently used only by
        # Hashrouting with edge cache)
        self.local_cache = {}

        # Keep track of nodes and links removed to simulate failures
        self.removed_nodes = {}
        # This keeps track of neighbors of a removed node at the time of removal.
        # It is needed to ensure that when the node is restored only links that
        # were removed as part of the node removal are restored and to prevent
        # restoring nodes that were removed manually before removing the node.
        self.disconnected_neighbors = {}
        self.removed_links = {}
        self.removed_sources = {}
        self.removed_caches = {}
        self.removed_local_caches = {}
        #self.plot_graph(topology, self.color_dict)

    def plot_graph(self, topology, color_dict):
        nx.draw(topology,
            nodelist=color_dict,
            node_size=1000,
            node_color=[i
                    for i in color_dict.values()],
            with_labels=True)
        #nx.draw(G, cmap=plt.get_cmap('viridis'), node_color=values, with_labels=True, font_color='white')
        #nx.draw_networkx(topology, pos=nx.drawing.layout.spring_layout(topology), cmap=color_dict)
        #labels = nx.get_edge_attributes(topology,'delay')
        #nx.draw_networkx_edge_labels(topology,pos=nx.drawing.layout.spring_layout(topology),edge_labels=labels)
        plt.savefig("topo_.png")
        print ("DIAGRAM SAVED")


class NetworkController(object):
    """Network controller

    This class is in charge of executing operations on the network model on
    behalf of a strategy implementation. It is also in charge of notifying
    data collectors of relevant events.
    """

    def __init__(self, model, cpus):
        """Constructor

        Parameters
        ----------
        model : NetworkModel
            Instance of the network model
        """
        self.session = dict()
        self.model = model
        self.collector = None

    def attach_collector(self, collector):
        """Attach a data collector to which all events will be reported.

        Parameters
        ----------
        collector : DataCollector
            The data collector
        """
        self.collector = collector

    def detach_collector(self):
        """Detach the data collector."""
        self.collector = None

    def start_session(self, timestamp, receiver, content, inx, log, count):
        """Instruct the controller to start a new session (i.e. the retrieval
        of a content).

        Parameters
        ----------
        timestamp : int
            The timestamp of the event
        receiver : any hashable type
            The receiver node requesting a content
        content : any hashable type
            The content identifier requested by the receiver
        log : bool
            *True* if this session needs to be reported to the collector,
            *False* otherwise
        """
        #print ("Start Session ", inx, count)
        self.session[inx] = dict(timestamp=timestamp,
                            receiver=receiver,
                            content=content,
                            inx=inx,
                            log=log,
                            count=count)
        if self.collector is not None and self.session[inx]['log']:
            self.collector.start_session(timestamp, receiver, content, inx)

    def forward_request_path(self, s, t, inx, path=None, main_path=True):
        """Forward a request from node *s* to node *t* over the provided path.

        Parameters
        ----------
        s : any hashable type
            Origin node
        t : any hashable type
            Destination node
        path : list, optional
            The path to use. If not provided, shortest path is used
        main_path : bool, optional
            If *True*, indicates that link path is on the main path that will
            lead to hit a content. It is normally used to calculate latency
            correctly in multicast cases. Default value is *True*
        """
        if path is None:
            path = self.model.shortest_path[s][t]
        for u, v in path_links(path):
            self.forward_request_hop(u, v, inx, main_path)

    def forward_content_path(self, u, v, size, inx, path=None, main_path=True):
        """Forward a content from node *s* to node *t* over the provided path.

        Parameters
        ----------
        s : any hashable type
            Origin node
        t : any hashable type
            Destination node
        size : length of the file
        path : list, optional
            The path to use. If not provided, shortest path is used
        main_path : bool, optional
            If *True*, indicates that this path is being traversed by content
            that will be delivered to the receiver. This is needed to
            calculate latency correctly in multicast cases. Default value is
            *True*
        """
        if path is None:
            path = self.model.shortest_path[u][v]
        for u, v in path_links(path):
            self.forward_content_hop(u, v, size, inx, main_path)

    def forward_request_hop(self, u, v, inx, main_path=True):
        """Forward a request over link  u -> v.

        Parameters
        ----------
        u : any hashable type
            Origin node
        v : any hashable type
            Destination node
        main_path : bool, optional
            If *True*, indicates that link link is on the main path that will
            lead to hit a content. It is normally used to calculate latency
            correctly in multicast cases. Default value is *True*
        """
        if self.collector is not None and self.session[inx]['log']:
            self.collector.request_hop(u, v, inx, main_path)

    def forward_content_hop(self, u, v, size, inx, main_path=True):
        """Forward a content over link  u -> v.

        Parameters
        ----------
        u : any hashable type
            Origin node
        v : any hashable type
            Destination node
        size : length of the file
        main_path : bool, optional
            If *True*, indicates that this link is being traversed by content
            that will be delivered to the receiver. This is needed to
            calculate latency correctly in multicast cases. Default value is
            *True*
        """
        if self.collector is not None and self.session[inx]['log']:
            self.collector.content_hop(u, v, size, inx, main_path)

    def put_content(self, node, inx, content=None):
        """Store content in the specified node.

        The node must have a cache stack and the actual insertion of the
        content is executed according to the caching policy. If the caching
        policy has a selective insertion policy, then content may not be
        inserted.

        Parameters
        ----------
        node : any hashable type
            The node where the content is inserted
        content : the content to be searched
            If it is not in the same session 

        Returns
        -------
        evicted : any hashable type
            The evicted object or *None* if no contents were evicted.
        """
        if node in self.model.cache and content is None:
            return self.model.cache[node].put(self.session[inx]['content'])
        if node in self.model.cache and content is not None:
            return self.model.cache[node].put(content)

    def get_content(self, node, inx, content=None):
        """Get a content from a server or a cache.

        Parameters
        ----------
        node : any hashable type
            The node where the content is retrieved
        content : the content to be searched
            If it is not in the same session 

        Returns
        -------
        content : bool
            True if the content is available, False otherwise
        """
        if node in self.model.cache and content is not None:
            return self.model.cache[node].get(content)
        if node in self.model.cache:
            cache_hit = self.model.cache[node].get(self.session[inx]['content'], inx)
            if cache_hit:
                if self.session[inx]['log']:
                    self.collector.cache_hit(node, inx)
            else:
                if self.session[inx]['log']:
                    self.collector.cache_miss(node, inx)
            return cache_hit
        name, props = fnss.get_stack(self.model.topology, node)
        if name == 'source' and self.session[inx]['content'] in props['contents']:
            if self.collector is not None and self.session[inx]['log']:
                self.collector.server_hit(node, inx)
            return True
        else:
            return False

    def remove_content(self, node, inx, content=None):
        """Remove the content being handled from the cache

        Parameters
        ----------
        node : any hashable type
            The node where the cached content is removed
        content : the content to be searched
            If it is not in the same session 

        Returns
        -------
        removed : bool
            *True* if the entry was in the cache, *False* if it was not.
        """
        if node in self.model.cache and content is None:
            return self.model.cache[node].remove(self.session[inx]['content'])
        if node in self.model.cache and content is not None:
            return self.model.cache[node].remove(content)

    def end_session(self, inx, success=True):
        """Close a session

        Parameters
        ----------
        success : bool, optional
            *True* if the session was completed successfully, *False* otherwise
        """
        if self.collector is not None and self.session[inx]['log']:
            self.collector.end_session(inx, success)
        self.session[inx] = None

    def rewire_link(self, u, v, up, vp, recompute_paths=True):
        """Rewire an existing link to new endpoints

        This method can be used to model mobility patters, e.g., changing
        attachment points of sources and/or receivers.

        Note well. With great power comes great responsibility. Be careful when
        using this method. In fact as a result of link rewiring, network
        partitions and other corner cases might occur. Ensure that the
        implementation of strategies using this method deal with all potential
        corner cases appropriately.

        Parameters
        ----------
        u, v : any hashable type
            Endpoints of link before rewiring
        up, vp : any hashable type
            Endpoints of link after rewiring
        """
        link = self.model.topology.adj[u][v]
        self.model.topology.remove_edge(u, v)
        self.model.topology.add_edge(up, vp, **link)
        if recompute_paths:
            shortest_path = dict(nx.all_pairs_dijkstra_path(self.model.topology))
            self.model.shortest_path = symmetrify_paths(shortest_path)

    def remove_link(self, u, v, recompute_paths=True):
        """Remove a link from the topology and update the network model.

        Note well. With great power comes great responsibility. Be careful when
        using this method. In fact as a result of link removal, network
        partitions and other corner cases might occur. Ensure that the
        implementation of strategies using this method deal with all potential
        corner cases appropriately.

        Also, note that, for these changes to be effective, the strategy must
        use fresh data provided by the network view and not storing local copies
        of network state because they won't be updated by this method.

        Parameters
        ----------
        u : any hashable type
            Origin node
        v : any hashable type
            Destination node
        recompute_paths: bool, optional
            If True, recompute all shortest paths
        """
        self.model.removed_links[(u, v)] = self.model.topology.adj[u][v]
        self.model.topology.remove_edge(u, v)
        if recompute_paths:
            shortest_path = dict(nx.all_pairs_dijkstra_path(self.model.topology))
            self.model.shortest_path = symmetrify_paths(shortest_path)

    def restore_link(self, u, v, recompute_paths=True):
        """Restore a previously-removed link and update the network model

        Parameters
        ----------
        u : any hashable type
            Origin node
        v : any hashable type
            Destination node
        recompute_paths: bool, optional
            If True, recompute all shortest paths
        """
        self.model.topology.add_edge(u, v, **self.model.removed_links.pop((u, v)))
        if recompute_paths:
            shortest_path = dict(nx.all_pairs_dijkstra_path(self.model.topology))
            self.model.shortest_path = symmetrify_paths(shortest_path)

    def remove_node(self, v, recompute_paths=True):
        """Remove a node from the topology and update the network model.

        Note well. With great power comes great responsibility. Be careful when
        using this method. In fact, as a result of node removal, network
        partitions and other corner cases might occur. Ensure that the
        implementation of strategies using this method deal with all potential
        corner cases appropriately.

        It should be noted that when this method is called, all links connected
        to the node to be removed are removed as well. These links are however
        restored when the node is restored. However, if a link attached to this
        node was previously removed using the remove_link method, restoring the
        node won't restore that link as well. It will need to be restored with a
        call to restore_link.

        This method is normally quite safe when applied to remove cache nodes or
        routers if this does not cause partitions. If used to remove content
        sources or receiver, special attention is required. In particular, if
        a source is removed, the content items stored by that source will no
        longer be available if not cached elsewhere.

        Also, note that, for these changes to be effective, the strategy must
        use fresh data provided by the network view and not storing local copies
        of network state because they won't be updated by this method.

        Parameters
        ----------
        v : any hashable type
            Node to remove
        recompute_paths: bool, optional
            If True, recompute all shortest paths
        """
        self.model.removed_nodes[v] = self.model.topology.node[v]
        # First need to remove all links the removed node as endpoint
        neighbors = self.model.topology.adj[v]
        self.model.disconnected_neighbors[v] = set(neighbors.keys())
        for u in self.model.disconnected_neighbors[v]:
            self.remove_link(v, u, recompute_paths=False)
        self.model.topology.remove_node(v)
        if v in self.model.cache:
            self.model.removed_caches[v] = self.model.cache.pop(v)
        if v in self.model.local_cache:
            self.model.removed_local_caches[v] = self.model.local_cache.pop(v)
        if v in self.model.source_node:
            self.model.removed_sources[v] = self.model.source_node.pop(v)
            for content in self.model.removed_sources[v]:
                self.model.countent_source.pop(content)
        if recompute_paths:
            shortest_path = dict(nx.all_pairs_dijkstra_path(self.model.topology))
            self.model.shortest_path = symmetrify_paths(shortest_path)

    def restore_node(self, v, recompute_paths=True):
        """Restore a previously-removed node and update the network model.

        Parameters
        ----------
        v : any hashable type
            Node to restore
        recompute_paths: bool, optional
            If True, recompute all shortest paths
        """
        self.model.topology.add_node(v, **self.model.removed_nodes.pop(v))
        for u in self.model.disconnected_neighbors[v]:
            if (v, u) in self.model.removed_links:
                self.restore_link(v, u, recompute_paths=False)
        self.model.disconnected_neighbors.pop(v)
        if v in self.model.removed_caches:
            self.model.cache[v] = self.model.removed_caches.pop(v)
        if v in self.model.removed_local_caches:
            self.model.local_cache[v] = self.model.removed_local_caches.pop(v)
        if v in self.model.removed_sources:
            self.model.source_node[v] = self.model.removed_sources.pop(v)
            for content in self.model.source_node[v]:
                self.model.countent_source[content] = v
        if recompute_paths:
            shortest_path = dict(nx.all_pairs_dijkstra_path(self.model.topology))
            self.model.shortest_path = symmetrify_paths(shortest_path)

    def reserve_local_cache(self, ratio=0.1):
        """Reserve a fraction of cache as local.

        This method reserves a fixed fraction of the cache of each caching node
        to act as local uncoodinated cache. Methods `get_content` and
        `put_content` will only operated to the coordinated cache. The reserved
        local cache can be accessed with methods `get_content_local_cache` and
        `put_content_local_cache`.

        This function is currently used only by hybrid hash-routing strategies.

        Parameters
        ----------
        ratio : float
            The ratio of cache space to be reserved as local cache.
        """
        if ratio < 0 or ratio > 1:
            raise ValueError("ratio must be between 0 and 1")
        for v, c in list(self.model.cache.items()):
            maxlen = iround(c.maxlen * (1 - ratio))
            if maxlen > 0:
                self.model.cache[v] = type(c)(maxlen)
            else:
                # If the coordinated cache size is zero, then remove cache
                # from that location
                if v in self.model.cache:
                    self.model.cache.pop(v)
            local_maxlen = iround(c.maxlen * (ratio))
            if local_maxlen > 0:
                self.model.local_cache[v] = type(c)(local_maxlen)

    def get_content_local_cache(self, node, inx):
        """Get content from local cache of node (if any)

        Get content from a local cache of a node. Local cache must be
        initialized with the `reserve_local_cache` method.

        Parameters
        ----------
        node : any hashable type
            The node to query
        """
        if node not in self.model.local_cache:
            return False
        cache_hit = self.model.local_cache[node].get(self.session['content'])
        if cache_hit:
            if self.session['log']:
                self.collector.cache_hit(node, inx)
        else:
            if self.session['log']:
                self.collector.cache_miss(node, inx)
        return cache_hit

    def put_content_local_cache(self, node):
        """Put content into local cache of node (if any)

        Put content into a local cache of a node. Local cache must be
        initialized with the `reserve_local_cache` method.

        Parameters
        ----------
        node : any hashable type
            The node to query
        """
        if node in self.model.local_cache:
            return self.model.local_cache[node].put(self.session['content'])
