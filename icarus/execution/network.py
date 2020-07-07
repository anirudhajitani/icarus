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
if device == 'cuda':
    torch.set_gpu_as_default_device()
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


def batch_to_seq(x):
    print ("x shape", x.shape, type(x))
    n_step = x.shape[0]
    print ("Batch to seq", n_step)
    if len(x.shape) == 1:
        x = torch.unsqueeze(x, -1)
    return torch.chunk(x, n_step)

def run_rnn_cen(layer, xs, dones, s):
    xs = batch_to_seq(xs)
    # need dones to reset states
    dones = batch_to_seq(dones)
    n_in = int(xs[0].shape[1])
    n_out = int(s.shape[0]) // 2
    s = torch.unsqueeze(s, 0)
    h, c = torch.chunk(s, 2, dim=1)
    outputs = []
    for ind, (x, done) in enumerate(zip(xs, dones)):
        c = c * (1-done)
        h = h * (1-done)
        h, c = layer(x, (h, c))
        outputs.append(h)
    s = torch.cat([h, c], dim=1)
    return torch.cat(outputs), torch.squeeze(s)

def run_rnn(layer, xs, dones, s):
    #xs = batch_to_seq(xs)
    # need dones to reset states
    #dones = batch_to_seq(dones)
    print ("XS = ", type(xs), "dones = ", dones)
    n_in = int(xs.shape[1])
    n_out = int(s.shape[0]) // 2
    print ("n_in, n_out", n_in, n_out)
    s = torch.unsqueeze(s, 0)
    print ("s = ", s.shape)
    h, c = torch.chunk(s, 2, dim=1)
    print ("h = , c =", h.shape, c.shape)
    outputs = []
    for ind, (x, done) in enumerate(zip(xs, dones)):
        #print ("ind, x, done", ind, x, done)
        c = c * (1-done)
        h = h * (1-done)
        print ("x, h, c", x.shape, h.shape, c.shape, x.is_cuda, h.is_cuda, c.is_cuda)
        h, c = layer(torch.unsqueeze(x,0), (h, c))
        #outputs.append(h)
        #print ("outputs ", outputs)
    s = torch.cat([h, c], dim=1)
    print ("s later = ", s.shape)
    #return torch.cat(outputs), torch.squeeze(s)
    return torch.squeeze(h), torch.squeeze(s)

def one_hot(x, oh_dim, dim=-1):
    oh_shape = list(x.shape)
    if dim == -1:
        oh_shape.append(oh_dim)
    else:
        oh_shape = oh_shape[:dim+1] + [oh_dim] + oh_shape[dim+1:]
    x_oh = torch.zeros(oh_shape)
    x = torch.unsqueeze(x, -1)
    if dim == -1:
        x_oh = x_oh.scatter(dim, x, 1)
    else:
        x_oh = x_oh.scatter(dim+1, x, 1)
    return x_oh

def init_layer(layer, layer_type):
    if layer_type == 'fc':
        nn.init.orthogonal_(layer.weight.data)
        nn.init.constant_(layer.bias.data, 0)
    elif layer_type == 'lstm':
        nn.init.orthogonal_(layer.weight_ih.data)
        nn.init.orthogonal_(layer.weight_hh.data)
        nn.init.constant_(layer.bias_ih.data, 0)
        nn.init.constant_(layer.bias_hh.data, 0)

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, centralized, alpha, dropout=0.6, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = 0.6
        self.alpha = alpha
        self.concat = concat
        self.centralized = centralized
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj=None):
        h = torch.mm(inp, self.W)
        N = h.size()[0]
        print ("N = ", N)
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        print ("A_input shape = ", a_input.shape)
        #print ("A_input = ", a_input)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        print ("e = ", e.shape)
        zero_vec = -9e15*torch.ones_like(e)
        one_vec = torch.ones_like(e)
        if self.centralized == 0:
            attention = torch.where(one_vec > 0, e, zero_vec)
        else:
            attention = torch.where(adj > 0, e, zero_vec)
        print ("Attention before softmax shape = ", attention.shape)
        #print ("Attention before softmax = ", attention)
        attention = F.softmax(attention, dim=1)
        print ("Attention after softmax shape = ", attention.shape)
        if self.centralized:
            attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        print ("h_prime shape = ", h_prime.shape)
        #print ("h_prime = ", h_prime)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphATPolicy(nn.Module):
    def __init__(self, nfeat, nhid, a_len, alpha, nheads):
        """Dense version of GAT."""
        super(GraphATPolicy, self).__init__()
        self.centralized = 0
        self.attentions = [GraphAttentionLayer(nfeat, nhid, self.centralized, alpha=alpha, concat=True) for _ in range(nheads)]
        print ("Model attention shape : ", self.attentions)
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.affine1 = nn.Linear(nhid * nheads, 256)
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

    
    def forward(self, x, adj=None):
        x = torch.cat([att(x) for att in self.attentions], dim=1)
        #print ("POLICY FORWARD")
        x = F.relu(self.affine1(x[:1, :]))
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

class GraphATPolicyCentralized(nn.Module):
    def __init__(self, nfeat, nhid, a_len, alpha, nheads, dropout=0.6):
        """Dense version of GAT."""
        super(GraphATPolicyCentralized, self).__init__()
        self.centralized = 1
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, self.centralized, alpha=alpha, dropout=self.dropout, concat=True) for _ in range(nheads)]
        print ("Model attention shape : ", self.attentions)
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.affine1 = nn.Linear(nhid * nheads, 256)
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

    
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        #print ("POLICY FORWARD")
        x = F.dropout(x, self.dropout, training=self.training)
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
        #action_prob = action_prob[:,-1]

        # critic evaluates being in state s_t
        state_values = self.value_head(x)
        state_values = state_values.view(batch_size, -1)
        state_values = state_values[:,-1]
        # return value of both actor and critic as a 2 tuple
        # 1. a list of probability of each action over state space
        # 2. the value from state s_t
        return action_prob, state_values, hidden



class BasePolicy(nn.Module):
    def __init__(self, n_s, n_a, policy_name, n_n=0, n_fc=64, n_lstm=64, n_h=64):
        super(BasePolicy, self).__init__()
        print ("Base n_s, n_a, name, n_n, n_fc, n_lstm, n_h ", n_s, n_a, policy_name, n_n, n_fc, n_lstm, n_h)
        self.name = policy_name
        self.n_a = n_a
        self.n_s = n_s
        self.n_fc = n_fc
        self.n_n = n_n
        self.n_h = n_h
        self.n_lstm = n_lstm
        self._init_net()
        self.saved_actions = []
        self.rewards = []
        self.to(device)

    def forward(self, ob, done, action=None, out_type='p'):
        #ob = torch.from_numpy(np.expand_dims(ob, axis=0)).float()
        xs = self._encode_ob(ob)
        if out_type.startswith('p'):
            #print ("O/p of softmax", F.softmax(self.actor_head(xs)).squeeze().detach().cpu().numpy())
            return F.softmax(self.actor_head(xs), dim=-1)
        else:
            #print ("o/p is value fun")
            return self._run_critic_head(xs)

    def backward(self, ob, acts, dones, Rs, Advs, e_coef, v_coef):
        #ob = torch.from_numpy(ob).float()
        dones = torch.from_numpy(dones).float()
        xs = self._encode_ob(ob)
        actor_dist = torch.distributions.categorical.Categorical(logits=F.log_softmax(self.actor_head(xs), dim=1))
        vs = self._run_critic_head(xs)
        self.policy_loss, self.value_loss, self.entropy_loss = \
            self._run_loss(actor_dist, e_coef, v_coef, vs,
                           torch.from_numpy(acts).long(),
                           torch.from_numpy(Rs).float(),
                           torch.from_numpy(Advs).float())
        self.loss = self.policy_loss + self.value_loss + self.entropy_loss
        self.loss.backward()

    def _encode_ob(self, ob):
        return F.relu(self.fc_layer(ob))

    def _init_net(self):
        self.fc_layer = nn.Linear(self.n_s, self.n_fc)
        self._init_actor_head(self.n_fc)
        self._init_critic_head(self.n_fc)

    def _init_actor_head(self, n_h, n_a=None):
        if n_a is None:
            n_a = self.n_a
        # only discrete control is supported for now
        self.actor_head = nn.Linear(n_h, n_a)
        init_layer(self.actor_head, 'fc') 

    def _init_critic_head(self, n_h, n_n=None):
        self.critic_head = nn.Linear(n_h, 1)
        init_layer(self.critic_head, 'fc') 

    def _run_critic_head(self, h, n_n=None):
        print ("Run Critic Head")
        return self.critic_head(h)

    def _run_loss(self, actor_dist, e_coef, v_coef, vs, As, Rs, Advs):
        log_probs = actor_dist.log_prob(As)
        policy_loss = -(log_probs * Advs).mean()
        entropy_loss = -(actor_dist.entropy()).mean() * e_coef
        value_loss = (Rs - vs).pow(2).mean() * v_coef
        return policy_loss, value_loss, entropy_loss

class GATPolicy(BasePolicy):
    def __init__(self, n_s, n_a, n_hid, alpha=0.2, dropout=0.6, n_heads=4, n_fc=64, n_h=64, name=None, na_dim_ls=None):
        super(GATPolicy, self).__init__(n_s, n_a, 'gat', 0, n_fc, n_h)
        self.centralized = 0
        self.n_fc = n_fc
        self.n_heads = n_heads
        self.attentions = [GraphAttentionLayer(n_s, n_hid, self.centralized, alpha, dropout, concat=True) for _ in range(self.n_heads)]
        print ("Model attention shape : ", self.attentions)
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self._init_net()
        self.fc_layer = nn.Linear(self.n_heads * n_hid, self.n_fc)
        self.to(device)

    def forward(self, ob, done, action=None, out_type='p'):
        #ob = torch.from_numpy(ob).float()
        ob = torch.cat([att(ob) for att in self.attentions], dim=1)
        print ("Ob shape : ",ob[:1,:].shape)
        xs = self._encode_ob(ob[:1 , :])
        if out_type == 'p':
            return F.softmax(self.actor_head(xs), dim=-1)
        else:
            return self._run_critic_head(xs)

    def backward(self, obs, nactions, acts, dones, Rs, Advs, e_coef, v_coef):
        #obs = torch.from_numpy(obs).float()
        dones = torch.from_numpy(dones).float()
        obs = torch.cat([att(obs) for att in self.attentions], dim=1)
        xs = self._encode_ob(obs)
        actor_dist = torch.distributions.categorical.Categorical(logits=F.log_softmax(self.actor_head(xs), dim=1))
        vs = self._run_critic_head(xs)
        self.policy_loss, self.value_loss, self.entropy_loss = \
            self._run_loss(actor_dist, e_coef, v_coef, vs,
                           torch.from_numpy(acts).long(),
                           torch.from_numpy(Rs).float(),
                           torch.from_numpy(Advs).float())
        self.loss = self.policy_loss + self.value_loss + self.entropy_loss
        self.loss.backward()


class BaseLstmPolicy(BasePolicy):
    def __init__(self, n_s, n_a, n_n, n_fc=64, n_lstm=64, n_h=64, name='lstm', na_dim_ls=None):
        super(BaseLstmPolicy, self).__init__(n_s, n_a, name, n_n, n_fc, n_lstm, n_h)
        print ("LSTM n_s, n_a, n_n, n_fc, n_lstm, name ", n_s, n_a, n_n, n_fc, n_lstm, name)
        self.n_lstm = n_lstm
        self.n_fc = n_fc
        self.n_n = n_n
        self._init_net()
        self._reset()
        self.to(device)

    def backward(self, obs, nactions, acts, dones, Rs, Advs,
                    e_coef, v_coef):
        #obs = torch.from_numpy(obs).float()
        dones = torch.from_numpy(dones).float().to(device)
        xs = self._encode_ob(obs)
        hs, new_states = run_rnn(self.lstm_layer, torch.unsqueeze(xs, 0), dones, self.states_bw)
        # backward grad is limited to the minibatch
        self.states_bw = new_states.detach()
        actor_dist = torch.distributions.categorical.Categorical(logits=F.log_softmax(self.actor_head(hs), dim=1))
        vs = self._run_critic_head(hs)
        self.policy_loss, self.value_loss, self.entropy_loss = \
            self._run_loss(actor_dist, e_coef, v_coef, vs,
                           torch.from_numpy(acts).long(), 
                           torch.from_numpy(Rs).float(), 
                           torch.from_numpy(Advs).float())
        self.loss = self.policy_loss + self.value_loss + self.entropy_loss
        self.loss.backward()
    
    #TODO - find out shape of done and append with all 0s while calling forward and backward
    def forward(self, ob, done, naction=None, out_type='p'):
        #ob = torch.from_numpy(np.expand_dims(ob, axis=0)).float()
        print ("Out_type=", out_type)
        done = torch.from_numpy(np.expand_dims(done, axis=0)).float().to(device)
        x = self._encode_ob(ob)
        h, new_states = run_rnn(self.lstm_layer, torch.unsqueeze(x, 0), done, self.states_fw)
        print ("Shape after run_rnn", h.shape, new_states.shape)
        if out_type.startswith('p'):
            print ("Actor")
            self.states_fw = new_states.detach()
            return F.softmax(self.actor_head(h), dim=-1)
        else:
            print ("Critic ", self._run_critic_head(h))
            return self._run_critic_head(h)

    def _encode_ob(self, ob):
        return F.relu(self.fc_layer(ob))

    def _init_net(self):
        print ("LSTM init")
        print (self.n_s, self.n_fc, type(self.n_s), type(self.n_fc))
        self.fc_layer = nn.Linear(self.n_s, self.n_fc)
        init_layer(self.fc_layer, 'fc')
        self.lstm_layer = nn.LSTMCell(self.n_fc, self.n_lstm)
        init_layer(self.lstm_layer, 'lstm')
        self._init_actor_head(self.n_lstm)
        self._init_critic_head(self.n_lstm)

    def _reset(self):
        # forget the cumulative states every cum_step
        self.states_fw = torch.zeros(self.n_lstm * 2).to(device)
        self.states_bw = torch.zeros(self.n_lstm * 2).to(device)


class FPPolicy(BaseLstmPolicy):
    def __init__(self, n_s, n_a, n_n, n_lstm=64,n_fc=256, n_h=64, name=None, na_dim_ls=None):
        super(FPPolicy, self).__init__(n_s, n_a, n_n, n_fc, n_lstm, n_h, 'fp')
        self._init_net()

    def _init_net(self):
        print ("INIT FFP")
        self.n_x = self.n_s - self.n_n * self.n_a
        self.fc_x_layer = nn.Linear(self.n_x, self.n_fc)
        init_layer(self.fc_x_layer, 'fc')
        n_h = self.n_fc
        if self.n_n:
            self.fc_p_layer = nn.Linear(self.n_s-self.n_x, self.n_fc)
            init_layer(self.fc_p_layer, 'fc')
            n_h += self.n_fc
        self.lstm_layer = nn.LSTMCell(n_h, self.n_lstm)
        init_layer(self.lstm_layer, 'lstm')
        self._init_actor_head(self.n_lstm)
        self._init_critic_head(self.n_lstm)
        self.to(device)

    def _encode_ob(self, ob):
        #print ("encode fc_x ", ob[:,:self.n_x].shape)
        #print ("encode fc_p ", ob[:,self.n_x:].shape)
        x = F.relu(self.fc_x_layer(ob[:, :self.n_x]))
        if self.n_n:
            p = F.relu(self.fc_p_layer(ob[:, self.n_x:]))
            x = torch.cat([x, p], dim=1)
        return x
    
    def _init_critic_head(self, n_h, n_n=None):
        if n_n is None:
            n_n = int(self.n_n)
        if n_n:
            n_na_sparse = self.n_a*n_n
            n_h += n_na_sparse
        self.critic_head = nn.Linear(n_h, 1)
        init_layer(self.critic_head, 'fc')

    def _run_critic_head(self, h, na, n_n=None):
        print ("Critic head action dims : ", na.shape)
        if n_n is None:
            n_n = int(self.n_n)
        if n_n:
            #TODO - no need for one hot here, we will send action vectors
            na_sparse = na.view(-1, self.n_a*n_n).squeeze()
            print ("DIms na_sparse, h ", na_sparse.shape, h.shape)
            h = torch.cat([h, na_sparse])
        return self.critic_head(h).squeeze()

    def backward(self, obs, nactions, acts, dones, Rs, Advs, e_coef, v_coef):
        #obs = torch.from_numpy(obs).float()
        obs = torch.unsqueeze(obs, 0)
        dones = torch.from_numpy(dones).float().to(device)
        xs = self._encode_ob(obs)
        hs, new_states = run_rnn(self.lstm_layer, xs, dones, self.states_bw)
        # backward grad is limited to the minibatch
        self.states_bw = new_states.detach()
        actor_dist = torch.distributions.categorical.Categorical(logits=F.log_softmax(self.actor_head(hs), dim=1))
        vs = self._run_critic_head(hs, nactions)
        self.policy_loss, self.value_loss, self.entropy_loss = \
            self._run_loss(actor_dist, e_coef, v_coef, vs,
                           torch.from_numpy(acts).long(), 
                           torch.from_numpy(Rs).float(), 
                           torch.from_numpy(Advs).float())
        self.loss = self.policy_loss + self.value_loss + self.entropy_loss
        self.loss.backward()

    def forward(self, ob, done, naction=None, out_type='p'):
        #ob = torch.from_numpy(np.expand_dims(ob, axis=0)).float()
        print ("Ob", ob.shape)
        ob = torch.unsqueeze(ob, 0)
        done = torch.from_numpy(np.expand_dims(done, axis=0)).float().to(device)
        x = self._encode_ob(ob)
        h, new_states = run_rnn(self.lstm_layer, x, done, self.states_fw)
        if out_type.startswith('p'):
            self.states_fw = new_states.detach()
            return F.softmax(self.actor_head(h), dim=-1)
        else:
            return self._run_critic_head(h, naction)


class NCMultiAgentPolicy(BasePolicy):
    """ Inplemented as a centralized meta-DNN. To simplify the implementation, all input
    and output dimensions are identical among all agents, and invalid values are casted as
    zeros during runtime."""
    def __init__(self, n_s, n_a, n_n, n_fc=64, n_h=64, n_lstm=64, name='nc',
                 n_s_ls=None, n_a_ls=None):
        super(NCMultiAgentPolicy, self).__init__(n_s, n_a, name, n_n, n_fc, n_lstm, n_h)
        self.n_n = n_n
        self.n_fc = n_fc
        self.n_h = n_h
        self._init_net()
        self._reset()
        self.to(device)

    def backward(self, obs, fps, acts, dones, Rs, Advs, e_coef, v_coef):
        #obs = torch.from_numpy(obs).float().transpose(0, 1)
        dones = torch.from_numpy(dones).float().to(device)
        #fps = torch.from_numpy(fps).float().transpose(0, 1).to(device)
        acts = torch.from_numpy(acts).long().to(device)
        hs, new_states = self._run_comm_layers(obs, dones, fps, self.states_bw)
        # backward grad is limited to the minibatch
        self.states_bw = new_states.detach()
        ps = self._run_actor_heads(hs)
        vs = self._run_critic_heads(hs, acts)
        self.policy_loss = 0
        self.value_loss = 0
        self.entropy_loss = 0
        Rs = torch.from_numpy(Rs).float()
        Advs = torch.from_numpy(Advs).float()
        actor_dist = torch.distributions.categorical.Categorical(logits=ps)
        policy_loss, value_loss, entropy_loss = \
            self._run_loss(actor_dist, e_coef, v_coef, vs,
                    acts, Rs, Advs)
        self.loss = self.policy_loss + self.value_loss + self.entropy_loss
        self.loss.backward()
        
    def forward(self, ob, done, fp, action=None, out_type='p'):
        # TxNxm
        #ob = torch.from_numpy(np.expand_dims(ob, axis=0)).float()
        done = torch.from_numpy(np.expand_dims(done, axis=0)).float().to(device)
        #fp = torch.from_numpy(np.expand_dims(fp, axis=0)).float().to(device)
        # h dim: NxTxm
        print ("forward shapes ob, fp, states_fw", ob.shape, fp.shape, self.states_fw.shape)
        h, new_states = self._run_comm_layers(ob, done, fp, self.states_fw)
        print ("After run_comm_layers h, new_states ", h.shape, new_states.shape)
        if out_type.startswith('p'):
            self.states_fw = new_states.detach()
            #self.states_fw = new_states
            return self._run_actor_heads(h)
        else:
            #action = torch.from_numpy(np.expand_dims(action, axis=1)).long()
            return self._run_critic_heads(h, action)

    def _get_comm_s(self, n_n, x, h, p):
        # Set the neighbors to 1 and itself as 0
        print ("Input shapes n_n, x, h, p, n_h", n_n, x.shape, h.shape, p.shape, self.n_h) 
        #js = np.ones(n_n + 1)
        #js[0] = 0
        #js = torch.from_numpy(js).long().to(device)
        #print ("JS ", js)
        #m_i = torch.index_select(h, 0, js).view(1, self.n_h * n_n)
        #p_i = torch.index_select(p, 0, js)
        #nx_i = torch.index_select(x, 0, js)
        p_i = p.view(1, self.n_a * n_n)
        nx_i = x.view(1, self.n_s * (n_n+1))
        m_i = h.contiguous().view(1, self.n_h * n_n)
        print ("DIms comm S m_i, p_i, x", m_i.shape, p_i.shape, nx_i.shape)
        s_i = [F.relu(self.fc_x_layer(nx_i)),
               F.relu(self.fc_p_layer(p_i)),
               F.relu(self.fc_m_layer(m_i))]
        return torch.cat(s_i, dim=1)

    def _get_neighbor_dim(self):
        return self.n_n, self.n_s * (self.n_n+1), self.n_a * self.n_n, [self.n_s] * self.n_n, [self.n_a] * self.n_n

    def _init_actor_head(self, n_a):
        # only discrete control is supported for now
        self.actor_head = nn.Linear(self.n_h * self.n_n, n_a)
        init_layer(self.actor_head, 'fc')

    def _init_comm_layer(self, n_n, n_ns, n_na):
        n_lstm_in = 3 * self.n_fc
        self.fc_x_layer = nn.Linear(n_ns, self.n_fc)
        init_layer(self.fc_x_layer, 'fc')
        if n_n:
            self.fc_p_layer = nn.Linear(n_na, self.n_fc)
            init_layer(self.fc_p_layer, 'fc')
            self.fc_m_layer = nn.Linear(self.n_h * n_n, self.n_fc)
            init_layer(self.fc_m_layer, 'fc')
            self.lstm_layer = nn.LSTMCell(n_lstm_in, self.n_h * self.n_n)
        else:
            self.fc_m_layer = None
            self.fc_p_layer = None
            self.lstm_layer = nn.LSTMCell(self.n_fc, self.n_h * self.n_n)
        init_layer(self.lstm_layer, 'lstm')

    def _init_critic_head(self, n_na):
        self.critic_head = nn.Linear(self.n_h * self.n_n + n_na, 1)
        init_layer(self.critic_head, 'fc')

    def _init_net(self):
        print ("INIT NCM")
        n_n, n_ns, n_na, ns_ls, na_ls = self._get_neighbor_dim()
        self._init_comm_layer(n_n, n_ns, n_na)
        n_a = self.n_a 
        self._init_actor_head(n_a)
        self._init_critic_head(n_na)

    def _reset(self):
        self.states_fw = torch.zeros(1, self.n_n * self.n_h * 2).to(device)
        self.states_bw = torch.zeros(1, self.n_n * self.n_h * 2).to(device)

    def _run_actor_heads(self, hs, detach=False):
        print ("RUN ACTOR HEAD ", self)
        if detach:
            ps = F.softmax(self.actor_head(hs), dim=-1).squeeze().detach()
        else:
            ps = F.log_softmax(self.actor_head(hs), dim=-1)
        print ("SOftmax op dim ", ps.shape)
        return ps

    def _run_comm_layers(self, x, done, p, state):
        #x = batch_to_seq(x)
        #dones = batch_to_seq(dones)
        #fps = batch_to_seq(fps)
        h, c = torch.chunk(state, 2, dim=1)
        print ("shapes state, h, c", state.shape, h.shape, c.shape)
        #x = x.squeeze(0)
        #p = p.squeeze(0)
        if self.n_n:
            s_i = self._get_comm_s(self.n_n, x, h, p)
        else:
            s_i = F.relu(self.fc_x_layer(x[0].unsqueeze(0)))
        print("After get_comm_s ", s_i.shape)
        h_i, c_i = h * (1-done), c * (1-done)
        print ("h_i dim , c_ dim :", h_i.shape, c_i.shape)
        h, c = self.lstm_layer(s_i, (h_i, c_i))
        #h, c = torch.cat(next_h), torch.cat(next_c)
        print ("h , c after loop", h.shape, c.shape)
        return h, torch.cat([h, c], dim=1)

    def _run_critic_heads(self, hs, actions, detach=False):
        #TODO this might be problematic (need to see if behaves properly)
        n_n = self.n_n
        if self.n_n:
            js = np.ones(self.n_n + 1)
            js[0] = 0
            js = torch.from_numpy(js).long().to(device)
            print ("JS = ", js)
            print ("critic head action dim before", actions.shape, " neigh + 1 ", self.n_n + 1)
            #TODO - why index_select is not working
            #actions = torch.index_select(actions, 0, js)
            actions = actions[1:,:]
            print ("critic head action dim after", actions.shape)
            actions = torch.flatten(actions) 
            print ("DIms hs, actions ", hs.shape, actions.shape)
            h_i = torch.cat([hs.squeeze(), actions])
            print ("h_i to be feed vs actual", h_i.shape, self.n_h * self.n_n + self.n_a * self.n_n)
        else:
            h_i = hs
        v_i = self.critic_head(h_i).squeeze()
        if detach:
            v_i = v_i.detach()
        print ("Critic head op shape", v_i.shape, type(v_i))
        return v_i

class CommNetMultiAgentPolicy(NCMultiAgentPolicy):
    """Reference code: https://github.com/IC3Net/IC3Net/blob/master/comm.py.
       Note in CommNet, the message is generated from hidden state only, so current state
       and neigbor policies are not included in the inputs."""
    def __init__(self, n_s, n_a, n_n, n_fc=64, n_h=64, n_lstm=64, 
                 n_s_ls=None, n_a_ls=None):
        BasePolicy.__init__(self, n_s, n_a, 'cnet', n_n, n_fc, n_h, n_lstm)
        self.n_n = n_n
        self.n_fc = n_fc
        self.n_h = n_h
        self._init_net()
        self._reset()
        self.to(device)

    def _init_comm_layer(self, n_n, n_ns, n_na):
        self.fc_x_layer = nn.Linear(n_ns, self.n_fc)
        init_layer(self.fc_x_layer, 'fc')
        if n_n:
            self.fc_m_layer = nn.Linear(self.n_h, self.n_fc)
            init_layer(self.fc_m_layer, 'fc')
        else:
            self.fc_m_layer = None
        self.lstm_layer = nn.LSTMCell(self.n_fc, self.n_h * self.n_n)
        init_layer(self.lstm_layer, 'lstm')

    def _get_comm_s(self, n_n, x, h, p):
        print ("GET COMM S dims n_n, x, h, p", n_n, x.shape, h.shape, p.shape)
        js = np.ones(n_n + 1)
        js[0] = 0
        js = torch.from_numpy(js).long().to(device)
        #m_i = torch.index_select(h, 0, js).mean(dim=0, keepdim=True)
        #m_i = torch(h).mean(dim=0, keepdim=True)
        m_i = h.contiguous().view(n_n, self.n_h)
        m_i = torch.mean(m_i, 0, True)
        print ("m_i shape", m_i.shape)
        #nx_i = torch.index_select(x, 0, js)
        nx_i = x.view(1, self.n_s * (n_n+1))
        print ("n_x_i shape", nx_i.shape)
        print ("x shape : ", x.squeeze().shape)
        x = torch.flatten(x)
        return F.relu(self.fc_x_layer(x)) + \
               self.fc_m_layer(m_i)

class DIALMultiAgentPolicy(NCMultiAgentPolicy):
    def __init__(self, n_s, n_a, n_n, n_fc=64, n_h=64, n_lstm=64,
                 n_s_ls=None, n_a_ls=None):
        BasePolicy.__init__(self, n_s, n_a, 'dial', n_n, n_fc, n_h, n_lstm)
        self.n_n = n_n
        self.n_fc = n_fc
        self.n_h = n_h
        self._init_net()
        self._reset()
        self.to(device)

    def _init_comm_layer(self, n_n, n_ns, n_na):
        self.fc_x_layer = nn.Linear(n_ns, n_na)
        init_layer(self.fc_x_layer, 'fc')
        if n_n:
            self.fc_m_layer = nn.Linear(self.n_h*n_n, n_na)
            init_layer(self.fc_m_layer, 'fc')
        else:
            self.fc_m_layer = None
        self.lstm_layer = nn.LSTMCell(n_na, self.n_h * self.n_n)
        init_layer(self.lstm_layer, 'lstm')

    def _get_comm_s(self, n_n, x, h, p):
        js = np.ones(n_n + 1)
        js[0] = 0
        js = torch.from_numpy(js).long().to(device)
        #m_i = torch.index_select(h, 0, js).view(1, self.n_h * n_n)
        #nx_i = torch.index_select(x, 0, js)
        nx_i = x.view(1, self.n_s * (n_n+1))
        print ("x shape : ", x.squeeze().shape)
        #a_i = one_hot(p[i].argmax().unsqueeze(0), self.n_fc)
        x = torch.flatten(x)
        a_i = p.view(1, self.n_a * n_n)
        print ("DIms in comm_s nx_i, h, a_i ", nx_i.shape, h.shape, a_i.shape)
        return F.relu(self.fc_x_layer(nx_i)) + \
               F.relu(self.fc_m_layer(h)) + a_i


class BasePolicyCentralized(nn.Module):
    def __init__(self, n_s, n_a, n_agent, policy_name, n_fc=64, n_lstm=64, n_h=64, neighbor_mask=None):
        super(BasePolicyCentralized, self).__init__()
        print ("Base n_s, n_a, Nagent, name, n_n, n_fc, n_lstm, n_h, N_mask ", n_s, n_a, n_agent, policy_name, n_fc, n_lstm, n_h, neighbor_mask)
        self.name = policy_name
        self.n_a = n_a
        self.n_s = n_s
        self.n_fc = n_fc
        self.n_h = n_h
        self.n_agent = n_agent
        self.n_lstm = n_lstm
        self.neighbor_mask = neighbor_mask
        self._init_net()
        self.saved_actions = []
        self.rewards = []
        self.to(device)

    def forward(self, ob, done, action=None, out_type='p'):
        #ob = torch.from_numpy(np.expand_dims(ob, axis=0)).float()
        xs = self._encode_ob(ob)
        if out_type.startswith('p'):
            #print ("O/p of softmax", F.softmax(self.actor_head(xs)).squeeze().detach().cpu().numpy())
            return F.softmax(self.actor_head(xs), dim=1).squeeze()
        else:
            #print ("o/p is value fun")
            return self._run_critic_head(xs)

    def backward(self, ob, acts, dones, Rs, Advs, e_coef, v_coef):
        #ob = torch.from_numpy(ob).float()
        dones = torch.from_numpy(dones).float()
        xs = self._encode_ob(ob)
        actor_dist = torch.distributions.categorical.Categorical(logits=F.log_softmax(self.actor_head(xs), dim=1))
        vs = self._run_critic_head(xs)
        self.policy_loss, self.value_loss, self.entropy_loss = \
            self._run_loss(actor_dist, e_coef, v_coef, vs,
                           torch.from_numpy(acts).long(),
                           torch.from_numpy(Rs).float(),
                           torch.from_numpy(Advs).float())
        self.loss = self.policy_loss + self.value_loss + self.entropy_loss
        self.loss.backward()

    def _encode_ob(self, ob):
        return F.relu(self.fc_layer(ob))

    def _init_net(self):
        self.fc_layer = nn.Linear(self.n_s, self.n_fc)
        self._init_actor_head(self.n_fc)
        self._init_critic_head(self.n_fc)

    def _init_actor_head(self, n_h, n_a=None):
        if n_a is None:
            n_a = self.n_a
        # only discrete control is supported for now
        self.actor_head = nn.Linear(n_h, n_a)
        init_layer(self.actor_head, 'fc') 

    def _init_critic_head(self, n_h, n_n=None):
        self.critic_head = nn.Linear(n_h, 1)
        init_layer(self.critic_head, 'fc') 

    def _run_critic_head(self, h, n_n=None):
        print ("Run Critic Head")
        return self.critic_head(h)

    def _run_loss(self, actor_dist, e_coef, v_coef, vs, As, Rs, Advs):
        log_probs = actor_dist.log_prob(As)
        policy_loss = -(log_probs * Advs).mean()
        entropy_loss = -(actor_dist.entropy()).mean() * e_coef
        value_loss = (Rs - vs).pow(2).mean() * v_coef
        return policy_loss, value_loss, entropy_loss

class GATPolicyCentralized(BasePolicyCentralized):
    def __init__(self, n_s, n_a, n_hid, n_agent, alpha=0.2, dropout=0.6, n_heads=4, n_fc=64, n_h=64, name=None, na_dim_ls=None, neighbor_mask=None):
        super(GATPolicyCentralized, self).__init__(n_s, n_a, n_agent, 'gat_cen', n_fc, n_h, neighbor_mask)
        self.centralized = 1
        self.n_heads = n_heads
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(n_s, n_hid, self.centralized, alpha, dropout, concat=True) for _ in range(n_heads)]
        print ("Model attention shape : ", self.attentions)
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self._init_net()
        self.fc_layer = nn.Linear(self.n_heads * n_hid, self.n_fc)
        self.to(device)

    def forward(self, ob, done, adj, action=None, out_type='p'):
        #ob = torch.from_numpy(ob).float()
        adj = torch.from_numpy(adj).float().to(device)
        ob = F.dropout(ob, self.dropout, training=self.training)
        ob = torch.cat([att(ob, adj) for att in self.attentions], dim=1)
        ob = F.dropout(ob, self.dropout, training=self.training)
        xs = self._encode_ob(ob)
        if out_type == 'p':
            return F.softmax(self.actor_head(xs), dim=1)
        else:
            return self._run_critic_head(xs)

    def backward(self, obs, nactions, acts, dones, Rs, Advs, e_coef, v_coef, adj=None):
        #obs = torch.from_numpy(obs).float()
        dones = torch.from_numpy(dones).float()
        obs = F.dropout(obs, self.dropout, training=self.training)
        obs = torch.cat([att(obs, adj) for att in self.attentions], dim=1)
        obs = F.dropout(obs, self.dropout, training=self.training)
        xs = self._encode_ob(obs)
        actor_dist = torch.distributions.categorical.Categorical(logits=F.log_softmax(self.actor_head(xs), dim=1))
        vs = self._run_critic_head(xs)
        self.policy_loss, self.value_loss, self.entropy_loss = \
            self._run_loss(actor_dist, e_coef, v_coef, vs,
                           torch.from_numpy(acts).long(),
                           torch.from_numpy(Rs).float(),
                           torch.from_numpy(Advs).float())
        self.loss = self.policy_loss + self.value_loss + self.entropy_loss
        self.loss.backward()

class BaseLstmPolicyCentralized(BasePolicyCentralized):
    def __init__(self, n_s, n_a, n_agent, n_fc=64, n_lstm=64, n_h=64, name='lstm', na_dim_ls=None, neighbor_mask=None):
        super(BaseLstmPolicyCentralized, self).__init__(n_s, n_a, n_agent, name, n_fc, n_lstm, n_h, neighbor_mask)
        print ("LSTM n_s, n_a, n_n, n_fc, n_lstm, name ", n_s, n_a, n_fc, n_lstm, name)
        self.n_lstm = n_lstm
        self.n_fc = n_fc
        self._init_net()
        self._reset()
        self.to(device)

    def backward(self, obs, nactions, acts, dones, Rs, Advs,
                    e_coef, v_coef):
        #obs = torch.from_numpy(obs).float()
        dones = torch.from_numpy(dones).float().to(device)
        xs = self._encode_ob(obs)
        hs, new_states = run_rnn_cen(self.lstm_layer, xs, dones, self.states_bw)
        # backward grad is limited to the minibatch
        self.states_bw = new_states.detach()
        actor_dist = torch.distributions.categorical.Categorical(logits=F.log_softmax(self.actor_head(hs), dim=1))
        vs = self._run_critic_head(hs)
        self.policy_loss, self.value_loss, self.entropy_loss = \
            self._run_loss(actor_dist, e_coef, v_coef, vs,
                           torch.from_numpy(acts).long(), 
                           torch.from_numpy(Rs).float(), 
                           torch.from_numpy(Advs).float())
        self.loss = self.policy_loss + self.value_loss + self.entropy_loss
        self.loss.backward()
    
    #TODO - find out shape of done and append with all 0s while calling forward and backward
    def forward(self, ob, done, naction=None, out_type='p'):
        #ob = torch.from_numpy(np.expand_dims(ob, axis=0)).float()
        print ("Out_type=", out_type)
        done = torch.from_numpy(np.expand_dims(done, axis=0)).float().to(device)
        x = self._encode_ob(ob)
        h, new_states = run_rnn_cen(self.lstm_layer, x, done, self.states_fw)
        print ("Shape after run_rnn", h.shape, new_states.shape)
        if out_type.startswith('p'):
            print ("Actor")
            self.states_fw = new_states.detach()
            return F.softmax(self.actor_head(h), dim=1).squeeze()
        else:
            print ("Critic ", self._run_critic_head(h))
            return self._run_critic_head(h)

    def _encode_ob(self, ob):
        return F.relu(self.fc_layer(ob))

    def _init_net(self):
        print ("LSTM init")
        print (self.n_s, self.n_fc, type(self.n_s), type(self.n_fc))
        self.fc_layer = nn.Linear(self.n_s, self.n_fc)
        init_layer(self.fc_layer, 'fc')
        self.lstm_layer = nn.LSTMCell(self.n_fc, self.n_lstm)
        init_layer(self.lstm_layer, 'lstm')
        self._init_actor_head(self.n_lstm)
        self._init_critic_head(self.n_lstm)

    def _reset(self):
        # forget the cumulative states every cum_step
        self.states_fw = torch.zeros(self.n_agent, self.n_lstm * 2).to(device)
        self.states_bw = torch.zeros(self.n_agent, self.n_lstm * 2).to(device)

class NCMultiAgentPolicyCentralized(BasePolicyCentralized):
    """ Inplemented as a centralized meta-DNN. To simplify the implementation, all input
    and output dimensions are identical among all agents, and invalid values are casted as
    zeros during runtime."""
    def __init__(self, n_s, n_a, n_agent, neighbor_mask, n_fc=64, n_h=64, n_lstm=64, name='nc_cen',
                 n_s_ls=None, n_a_ls=None):
        super(NCMultiAgentPolicyCentralized, self).__init__(n_s, n_a, n_agent, name, n_fc, n_lstm, n_h, neighbor_mask)
        self.n_agent = n_agent
        self.neighbor_mask = neighbor_mask
        self.n_fc = n_fc
        self.n_h = n_h
        self._init_net()
        self._reset()
        self.to(device)

    def backward(self, obs, fps, acts, dones, Rs, Advs, e_coef, v_coef):
        obs = torch.from_numpy(obs).float().transpose(0, 1)
        dones = torch.from_numpy(dones).float()
        fps = torch.from_numpy(fps).float().transpose(0, 1)
        acts = torch.from_numpy(acts).long()
        hs, new_states = self._run_comm_layers(obs, dones, fps, self.states_bw)
        # backward grad is limited to the minibatch
        self.states_bw = new_states.detach()
        ps = self._run_actor_heads(hs)
        vs = self._run_critic_heads(hs, acts)
        self.policy_loss = 0
        self.value_loss = 0
        self.entropy_loss = 0
        Rs = torch.from_numpy(Rs).float()
        Advs = torch.from_numpy(Advs).float()
        for i in range(self.n_agent):
            actor_dist_i = torch.distributions.categorical.Categorical(logits=ps[i])
            policy_loss_i, value_loss_i, entropy_loss_i = \
                self._run_loss(actor_dist_i, e_coef, v_coef, vs[i],
                    acts[i], Rs[i], Advs[i])
            self.policy_loss += policy_loss_i
            self.value_loss += value_loss_i
            self.entropy_loss += entropy_loss_i
        self.loss = self.policy_loss + self.value_loss + self.entropy_loss
        self.loss.backward()
        
    def forward(self, ob, done, fp, action=None, out_type='p'):
        # TxNxm
        #ob = torch.from_numpy(np.expand_dims(ob, axis=0)).float()
        ob = ob.unsqueeze(0)
        fp = fp.unsqueeze(0)
        done = torch.from_numpy(np.expand_dims(done, axis=0)).float().to(device)
        #fp = torch.from_numpy(np.expand_dims(fp, axis=0)).float().to(device)
        # h dim: NxTxm
        print ("forward shapes ob, fp, states_fw", ob.shape, fp.shape, self.states_fw.shape)
        h, new_states = self._run_comm_layers(ob, done, fp, self.states_fw)
        print ("After run_comm_layers h, new_states ", h.shape, new_states.shape)
        if out_type.startswith('p'):
            self.states_fw = new_states.detach()
            return self._run_actor_heads(h)
        else:
            #action = torch.from_numpy(np.expand_dims(action, axis=1)).long()
            return self._run_critic_heads(h, action)

    def _get_comm_s(self, i, n_n, x, h, p):
        print ("Insid get_comm_s : n_n, x, h, p ", n_n, x.shape, h.shape, p.shape) 
        js = torch.from_numpy(np.where(self.neighbor_mask[i])[0]).long().to(device)
        m_i = torch.index_select(h, 0, js).view(1, self.n_h * n_n)
        p_i = torch.index_select(p, 0, js)
        nx_i = torch.index_select(x, 0, js)
        print ("DIms comm S m_i, p_i, nx_i", m_i.shape, p_i.shape, nx_i.shape)
        p_i = p_i.view(1, self.n_a * n_n)
        nx_i = nx_i.view(1, self.n_s * n_n)
        print ("After DIms comm p_i, nx_i", p_i.shape, nx_i.shape)
        s_i = [F.relu(self.fc_x_layers[i](torch.cat([x[i].unsqueeze(0), nx_i], dim=1))),
               F.relu(self.fc_p_layers[i](p_i)),
               F.relu(self.fc_m_layers[i](m_i))]
        return torch.cat(s_i, dim=1)

    def _get_neighbor_dim(self, i_agent):
        n_n = int(np.sum(self.neighbor_mask[i_agent]))
        return n_n, self.n_s * (n_n+1), self.n_a * n_n, [self.n_s] * n_n, [self.n_a] * n_n

    def _init_actor_head(self, n_a):
        # only discrete control is supported for now
        actor_head = nn.Linear(self.n_h, n_a)
        init_layer(actor_head, 'fc')
        self.actor_heads.append(actor_head)

    def _init_comm_layer(self, n_n, n_ns, n_na):
        n_lstm_in = 3 * self.n_fc
        fc_x_layer = nn.Linear(n_ns, self.n_fc)
        init_layer(fc_x_layer, 'fc')
        self.fc_x_layers.append(fc_x_layer)
        if n_n:
            fc_p_layer = nn.Linear(n_na, self.n_fc)
            init_layer(fc_p_layer, 'fc')
            fc_m_layer = nn.Linear(self.n_h * n_n, self.n_fc)
            init_layer(fc_m_layer, 'fc')
            self.fc_m_layers.append(fc_m_layer)
            self.fc_p_layers.append(fc_p_layer)
            lstm_layer = nn.LSTMCell(n_lstm_in, self.n_h)
        else:
            self.fc_m_layers.append(None)
            self.fc_p_layers.append(None)
            lstm_layer = nn.LSTMCell(self.n_fc, self.n_h)
        init_layer(lstm_layer, 'lstm')
        self.lstm_layers.append(lstm_layer)

    def _init_critic_head(self, n_na):
        critic_head = nn.Linear(self.n_h + n_na, 1)
        init_layer(critic_head, 'fc')
        self.critic_heads.append(critic_head)

    def _init_net(self):
        self.fc_x_layers = nn.ModuleList()
        self.fc_p_layers = nn.ModuleList()
        self.fc_m_layers = nn.ModuleList()
        self.lstm_layers = nn.ModuleList()
        self.actor_heads = nn.ModuleList()
        self.critic_heads = nn.ModuleList()
        self.ns_ls_ls = []
        self.na_ls_ls = []
        self.n_n_ls = []
        for i in range(self.n_agent):
            n_n, n_ns, n_na, ns_ls, na_ls = self._get_neighbor_dim(i)
            self.ns_ls_ls.append(ns_ls)
            self.na_ls_ls.append(na_ls)
            self.n_n_ls.append(n_n)
            self._init_comm_layer(n_n, n_ns, n_na)
            n_a = self.n_a 
            self._init_actor_head(n_a)
            self._init_critic_head(n_na)

    def _reset(self):
        self.states_fw = torch.zeros(self.n_agent, self.n_h * 2).to(device)
        self.states_bw = torch.zeros(self.n_agent, self.n_h * 2).to(device)

    def _run_actor_heads(self, hs, detach=False):
        ps = []
        print ("Actor heads ", hs.shape)
        for i in range(self.n_agent):
            if detach:
                p_i = F.softmax(self.actor_heads[i](hs[i])).squeeze().detach()
            else:
                p_i = F.log_softmax(self.actor_heads[i](hs[i]))
            ps.append(p_i)
        return ps

    def _run_comm_layers(self, obs, dones, fps, states):
        obs = batch_to_seq(obs)
        dones = batch_to_seq(dones)
        fps = batch_to_seq(fps)
        h, c = torch.chunk(states, 2, dim=1)
        outputs = []
        for t, (x, p, done) in enumerate(zip(obs, fps, dones)):
            next_h = []
            next_c = []
            x = x.squeeze(0)
            p = p.squeeze(0)
            for i in range(self.n_agent):
                n_n = self.n_n_ls[i]
                if n_n:
                    s_i = self._get_comm_s(i, n_n, x, h, p)
                else:
                    s_i = F.relu(self.fc_x_layers[i](x[i].unsqueeze(0)))
                print ("S-i dim ", s_i.shape)
                h_i, c_i = h[i].unsqueeze(0) * (1-done), c[i].unsqueeze(0) * (1-done)
                print ("h_i, c_i ", h_i.shape, c_i.shape)
                next_h_i, next_c_i = self.lstm_layers[i](s_i, (h_i, c_i))
                print ("next_h_i, next_c_i ", next_h_i.shape, next_c_i.shape)
                next_h.append(next_h_i)
                next_c.append(next_c_i)
            h, c = torch.cat(next_h), torch.cat(next_c)
            print ("AFter append h, c ", h.shape, c.shape)
            outputs.append(h)
        outputs = torch.cat(outputs)
        print ("Outputs ", outputs.shape)
        #return outputs.transpose(0, 1), torch.cat([h, c], dim=1)
        return outputs, torch.cat([h, c], dim=1)

    def _run_critic_heads(self, hs, actions, detach=False):
        vs = []
        print ("RUn critic heads hs, actions ", hs.shape, actions.shape)
        for i in range(self.n_agent):
            n_n = self.n_n_ls[i]
            if n_n:
                js = torch.from_numpy(np.where(self.neighbor_mask[i])[0]).long().to(device)
                na_i = torch.index_select(actions, 0, js)
                print ("na_i shape for i ", i, na_i.shape)
                na_i_ls = []
                for j in range(n_n):
                    na_i_ls.append(na_i[j])
                print ("Before h_i cat hs[i] na_i_ls ", hs[i].shape, na_i_ls[0].shape) 
                h_i = torch.cat([hs[i]] + na_i_ls)
            else:
                h_i = hs[i]
            v_i = self.critic_heads[i](h_i).squeeze()
            if detach:
                vs.append(v_i.detach())
            else:
                vs.append(v_i)
        return vs


class ConsensusPolicyCentralized(NCMultiAgentPolicyCentralized):
    def __init__(self, n_s, n_a, n_agent, neighbor_mask, n_fc=64, n_h=64, n_lstm=64, name='con_cen',
                 n_s_ls=None, n_a_ls=None):
        BasePolicyCentralized.__init__(self, n_s, n_a, n_agent, name, n_fc, n_lstm, n_h, neighbor_mask)
        self.n_agent = n_agent
        self.neighbor_mask = neighbor_mask
        self.n_fc = n_fc
        self.n_h = n_h
        self._init_net()
        self._reset()
        self.to(device)

    def consensus_update(self):
        consensus_update = []
        with torch.no_grad():
            for i in range(self.n_agent):
                mean_wts = self._get_critic_wts(i)
                for param, wt in zip(self.lstm_layers[i].parameters(), mean_wts):
                    param.copy_(wt)

    def _init_net(self):
        self.fc_x_layers = nn.ModuleList()
        self.lstm_layers = nn.ModuleList()
        self.actor_heads = nn.ModuleList()
        self.critic_heads = nn.ModuleList()
        self.na_ls_ls = []
        self.n_n_ls = []
        for i in range(self.n_agent):
            n_n, _, n_na, _, na_ls = self._get_neighbor_dim(i)
            n_s = self.n_s 
            self.na_ls_ls.append(na_ls)
            self.n_n_ls.append(n_n)
            fc_x_layer = nn.Linear(n_s, self.n_fc)
            init_layer(fc_x_layer, 'fc')
            self.fc_x_layers.append(fc_x_layer)
            lstm_layer = nn.LSTMCell(self.n_fc, self.n_h)
            init_layer(lstm_layer, 'lstm')
            self.lstm_layers.append(lstm_layer)
            n_a = self.n_a 
            self._init_actor_head(n_a)
            self._init_critic_head(n_na)

    def _get_critic_wts(self, i_agent):
        wts = []
        for wt in self.lstm_layers[i_agent].parameters():
            wts.append(wt.detach())
        neighbors = list(np.where(self.neighbor_mask[i_agent] == 1)[0])
        for j in neighbors:
            for k, wt in enumerate(self.lstm_layers[j].parameters()):
                wts[k] += wt.detach()
        n = 1 + len(neighbors)
        for k in range(len(wts)):
            wts[k] /= n
        return wts

    def _run_comm_layers(self, obs, dones, fps, states):
        # NxTxm
        obs = obs.transpose(0, 1)
        hs = []
        new_states = []
        for i in range(self.n_agent):
            xs_i = F.relu(self.fc_x_layers[i](obs[i]))
            hs_i, new_states_i = run_rnn(self.lstm_layers[i], xs_i, dones, states[i])
            hs.append(hs_i.unsqueeze(0))
            new_states.append(new_states_i.unsqueeze(0))
        return torch.cat(hs), torch.cat(new_states)


class CommNetMultiAgentPolicyCentralized(NCMultiAgentPolicyCentralized):
    """Reference code: https://github.com/IC3Net/IC3Net/blob/master/comm.py.
       Note in CommNet, the message is generated from hidden state only, so current state
       and neigbor policies are not included in the inputs."""
    def __init__(self, n_s, n_a, n_agent, neighbor_mask, n_fc=64, n_h=64, n_lstm=64, name='cnet_cen',
                 n_s_ls=None, n_a_ls=None):
        BasePolicyCentralized.__init__(self, n_s, n_a, n_agent, name, n_fc, n_lstm, n_h, neighbor_mask)
        self.n_agent = n_agent
        self.neighbor_mask = neighbor_mask
        self.n_fc = n_fc
        self.n_h = n_h
        self._init_net()
        self._reset()
        self.to(device)

    def _init_comm_layer(self, n_n, n_ns, n_na):
        fc_x_layer = nn.Linear(n_ns, self.n_fc)
        init_layer(fc_x_layer, 'fc')
        self.fc_x_layers.append(fc_x_layer)
        if n_n:
            fc_m_layer = nn.Linear(self.n_h, self.n_fc)
            init_layer(fc_m_layer, 'fc')
            self.fc_m_layers.append(fc_m_layer)
        else:
            self.fc_m_layers.append(None)
        lstm_layer = nn.LSTMCell(self.n_fc, self.n_h)
        init_layer(lstm_layer, 'lstm')
        self.lstm_layers.append(lstm_layer)

    def _get_comm_s(self, i, n_n, x, h, p):
        js = torch.from_numpy(np.where(self.neighbor_mask[i])[0]).long().to(device)
        m_i = torch.index_select(h, 0, js).mean(dim=0, keepdim=True)
        nx_i = torch.index_select(x, 0, js)
        nx_i = nx_i.view(1, self.n_s * n_n)
        return F.relu(self.fc_x_layers[i](torch.cat([x[i].unsqueeze(0), nx_i], dim=1))) + \
               self.fc_m_layers[i](m_i)


class DIALMultiAgentPolicyCentralized(NCMultiAgentPolicyCentralized):
    def __init__(self, n_s, n_a, n_agent, neighbor_mask, n_fc=64, n_h=64, n_lstm=64, name='dial_cen',
                 n_s_ls=None, n_a_ls=None):
        BasePolicyCentralized.__init__(self, n_s, n_a, n_agent, name, n_fc, n_lstm, n_h, neighbor_mask)
        self.n_agent = n_agent
        self.neighbor_mask = neighbor_mask
        self.n_fc = n_fc
        self.n_h = n_h
        self._init_net()
        self._reset()
        self.to(device)

    def _init_comm_layer(self, n_n, n_ns, n_na):
        fc_x_layer = nn.Linear(n_ns, self.n_a)
        init_layer(fc_x_layer, 'fc')
        self.fc_x_layers.append(fc_x_layer)
        if n_n:
            fc_m_layer = nn.Linear(self.n_h*n_n, self.n_a)
            init_layer(fc_m_layer, 'fc')
            self.fc_m_layers.append(fc_m_layer)
        else:
            self.fc_m_layers.append(None)
        lstm_layer = nn.LSTMCell(self.n_a, self.n_h)
        init_layer(lstm_layer, 'lstm')
        self.lstm_layers.append(lstm_layer)

    def _get_comm_s(self, i, n_n, x, h, p):
        js = torch.from_numpy(np.where(self.neighbor_mask[i])[0]).long().to(device)
        m_i = torch.index_select(h, 0, js).view(1, self.n_h * n_n)
        nx_i = torch.index_select(x, 0, js)
        nx_i = nx_i.view(1, self.n_s * n_n)
        a_i = p[i]
        return F.relu(self.fc_x_layers[i](torch.cat([x[i].unsqueeze(0), nx_i], dim=1))) + \
               F.relu(self.fc_m_layers[i](m_i)) + a_i

class Agent(object):
    """

    Each router is considered as an agent. The agent needs to view its state space
    and then select an action (which files to cache based on the current policy it
    is executing.
    """

    def __init__(self, view, router, ver, policy_type, gamma=0.9, lr=3e-2, window=250, comb=0, tau=0.95, beta=0.001, use_gae=True, avg_reward_case=True):
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
        self.hidden_state = 64
        self.gamma = gamma
        self.use_gae = use_gae
        self.beta = beta
        self.tau = tau
        self.avg_reward = 0.0
        self.avg_reward_case = avg_reward_case
        self.rewards = 0
        self.action = []
        self.ps = None
        self.valid_action = [0, 1]
        print ("Ver, avg_reward_case, use_gae", ver, self.avg_reward_case, self.use_gae) 
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

        if self.view.strategy_name in ['RL_DEC_1'] and self.view.centralized == False:
            try:
                if self.state_ver == 1:
                    if self.policy_type == 0:
                        self.policy = BasePolicy(len(self.view.model.library), len(self.valid_action), 'fnn')
                    elif self.policy_type == 1:
                        self.policy = BaseLstmPolicy(len(self.view.model.library), len(self.valid_action), 
                                len(self.view.model.neighbor[self.cache]), self.hidden_state)
                else:
                    self.ps = np.full((len(self.view.model.library)), 1.0, dtype=float)
                    self.ps = torch.from_numpy(self.ps/np.sum(self.ps)).float().to(device)
                    act = np.full((len(self.view.model.library)), 0.0, dtype=float)
                    self.action = torch.from_numpy(act).float().to(device)
                    if self.policy_type == 0:
                        self.policy = BasePolicy(len(self.view.model.library), len(self.view.model.library), 'fnn')
                    elif self.policy_type == 1:
                        self.policy = BaseLstmPolicy(len(self.view.model.library), len(self.view.model.library), 
                                len(self.view.model.neighbor[self.cache]), self.hidden_state, self.hidden_state)
                    elif self.policy_type == 2:
                        self.policy = GATPolicy(len(self.view.model.library), len(self.view.model.library), self.hidden_state, alpha=0.2, n_heads=4)
                    #TODO - pass correct parameters
                    elif self.policy_type == 3:
                        self.policy = FPPolicy(int(len(self.view.model.library) * (len(self.view.model.neighbor[self.cache]) + 1)+ 
                                    len(self.view.model.library) * len(self.view.model.neighbor[self.cache])), 
                                    len(self.view.model.library), len(self.view.model.neighbor[self.cache]))
                    elif self.policy_type == 4:
                        self.policy = NCMultiAgentPolicy(len(self.view.model.library), len(self.view.model.library), len(self.view.model.neighbor[self.cache]))
                    elif self.policy_type == 6:
                        self.policy = CommNetMultiAgentPolicy(len(self.view.model.library), len(self.view.model.library), len(self.view.model.neighbor[self.cache]))
                    elif self.policy_type == 7:
                        self.policy = DIALMultiAgentPolicy(len(self.view.model.library), len(self.view.model.library), len(self.view.model.neighbor[self.cache]))

                #print ("POLICY", self.policy)
                self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
                #print ("OPTIMIZER", self.optimizer)
                self.eps = np.finfo(np.float32).eps.item()
                #print ("EPS", self.eps)
            except:
                print(traceback.format_exc())
                print(sys.exc_info()[2])
        elif self.view.strategy_name in ['RL_DEC_1']:
            self.ps = np.full((len(self.view.model.library)), 1.0, dtype=float)
            self.ps = torch.from_numpy(self.ps/np.sum(self.ps)).float().to(device)

        if self.view.resume == True:
            self.load_model()

        if self.view.strategy_name in ['RL_DEC_2F', 'RL_DEC_2D']:
            self.avg_reward2 = 0.0
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
        
            if self.view.resume == True:
                self.load_model()

    def get_state(self):
        """
        Returns the current state of the cache.
        """
        if self.view.strategy_name in ['INDEX', 'INDEX_DIST', 'RL_DEC_1', 'RL_DEC_2F']:
            return self.state_counts
        if self.view.strategy_name in ['RL_DEC_2D']:
            return self.prob
   
    def save_model(self, count):
        """
        Save NN model parameters 
        """
        PATH = self.view.model_path
        if PATH is not None:
            PATH = PATH + "/" + str(self.view.strategy_name) + "agent_" + str(self.cache) + ".pt"
        else:
            PATH = str(self.view.strategy_name) + "agent_" + str(self.cache) + ".pt"
        if self.view.strategy_name in ['RL_DEC_2F', 'RL_DEC_2D']:
            torch.save({
                'epoch': count,
                'model_state_dict': self.policy.state_dict(),
                'model2_state_dict': self.policy2.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'optimizer2_state_dict': self.optimizer2.state_dict(),
                }, PATH)
        else:
            torch.save({
                'epoch': count,
                'model_state_dict': self.policy.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                }, PATH)
        print ("Saved Model to ", PATH)

    def load_model(self):
        """
        Load NN model and initialise them
        """
        PATH = self.view.model_path
        if PATH is not None:
            PATH = PATH + "/" + str(self.view.strategy_name) + "agent_" + str(self.cache) + ".pt"
        else:
            PATH = str(self.view.strategy_name) + "agent_" + str(self.cache) + ".pt"
        try:
            checkpoint = torch.load(PATH)
        except:
            print ("CHECKPOINT not present.. Cannot LOAD Model")
            return
        print ("LOADING MODELS previously stored")
        #print ("MODEL : ", checkpoint['model_state_dict'])
        #print ("OPTIM : ", checkpoint['optimizer_state_dict'])
        try:
            self.policy.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.view.strategy_name in ['RL_DEC_2F', 'RL_DEC_2D']:
                self.policy2.load_state_dict(checkpoint['model2_state_dict'])
                self.optimizer2.load_state_dict(checkpoint['optimizer2_state_dict'])
            # Restore count from where it left off
            self.view.count = checkpoint['epoch']
        except:
            print ("Saved Model does not match the Policy")

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

    def get_policy(self, ob, ps=None):
        """
        ob - set of states of itself as well as neighbor in case of ma2c policies
        ps - fingerprints of neighbors
        """
        
        done = np.array([0])
        if self.policy_type in [4,6,7]:
            policy = self.policy.forward(ob, done, ps)
        else:
            policy = self.policy.forward(ob, done)
        return policy

    def get_value(self, ob, action=None, ps=None):
        """
        ps - neighbor fingerprint
        action - array of actions in case of ma2c
        ob - set of states of itself as well as neighbors in case of ma2c policies
        """
        
        done = np.array([0])
        if self.policy_type in [4,6,7]:
            value = self.policy.forward(ob, done, ps, action, 'v')
        elif self.policy_type in [3]:
            value = self.policy.forward(ob, done, action, 'v')
        else:
            value = self.policy.forward(ob, done, None, 'v')
        return value

    def select_actions(self, state):
        #print ("STATE", state)
        #ob = torch.from_numpy(np.expand_dims(ob, axis=0)).float()
        if self.view.strategy_name in ['RL_DEC_2D', 'RL_DEC_2F']:
            if self.policy_type == 0:
                state = torch.from_numpy(state).float().to(device)
                probs, state_value = self.policy.forward(state)
            elif self.policy_type == 2:
                state = torch.stack([x for x in state]).to(device)
                #print ("State = ", state, state.shape)
                probs, state_value = self.policy.forward(state)
            else:
                state = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0).to(device)
                probs, state_value, (self.a_hx, self.a_cx) = self.policy.forward(state, (self.a_hx, self.a_cx))
        elif self.view.strategy_name in ['RL_DEC_1']:
            if self.policy_type in [0, 1]:
                state = torch.from_numpy(state).float().to(device)
                probs = self.get_policy(state)
            elif self.policy_type in [2]:
                state = torch.stack([x for x in state]).to(device)
                #print ("State = ", state.shape)
                probs = self.get_policy(state)
            elif self.policy_type in [3]:
                neigh_ps = self.view.get_neighbor_policy(self.cache)
                neigh_ps = torch.flatten(neigh_ps)
                state = torch.stack([x for x in state]).to(device)
                state = torch.flatten(state)
                state = torch.cat([state, neigh_ps])
                state = torch.flatten(state)
                #print ("State = ", state.shape)
                probs = self.get_policy(state)
            elif self.policy_type in [4,6,7]:
               # States should consider neighboring states + policies of neighbors
                state = torch.stack([x for x in state]).to(device)
                # Maybe one step late (no sync)
                neigh_ps = self.view.get_neighbor_policy(self.cache)
                probs = self.get_policy(state, neigh_ps)
        # Adding the fingerprint
        self.ps = probs
        #print ("PROBS", probs.shape)
        # create categorical distribution over list of probabilities of actions
        m = Categorical(probs)
        #print ("Output of NN ")
        #print (m)
        # sample an action from the categorical distribution
        # Select top k values from probability softmax output
        if self.state_ver == 0 or self.state_ver == 2 and self.view.strategy_name in ['RL_DEC_1']:
            top_k_val, top_k_inx = probs.topk(self.cache_size)
            action = self.decode_action(top_k_inx.cpu().detach().numpy())
            #print ("Action ", action)
            self.action = torch.from_numpy(np.array(action)).float().to(device)
            # call value function, FP send only neighboring action, in MA2C send both self + neighboring action
            if self.policy_type in [0, 1]:
                #state = torch.from_numpy(state).float().to(device)
                state_value = self.get_value(state)
            elif self.policy_type == 2:
                #state = torch.stack([x for x in state]).to(device)
                #print ("State = ", state, state.shape)
                state_value = self.get_value(state, action)
            elif self.policy_type == 3:
                # here we need to pass neighboring actions only
                # write function in view which does that
                actions = self.view.get_neighbor_action(self.cache, self_=False)
                state_value = self.get_value(state, actions)
            elif self.policy_type in [4,6,7]:
               # States should consider neighboring states + policies of neighbors
                #state = torch.stack([x for x in state]).to(device)
                actions = self.view.get_neighbor_action(self.cache, self_=True) 
                state_value = self.get_value(state, actions, neigh_ps)
            for ac in top_k_inx:
                self.policy.saved_actions.append(SavedAction(m.log_prob(ac), state_value))
            #print ("Value ", state_value, state_value.shape)
            return action
        #print ("TOP K INDEX", top_k_inx, type(top_k_inx))
        #print ("TOP K VAL", top_k_val)
        action = m.sample()
        #self.action = action
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
        self.action = torch.from_numpy(np.array(self.decode_action(action.item()))).float().to(device)
        return [self.decode_action(action.item())]

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
        print ("UPDATE FN ", self.cache)
        R = 0
        saved_actions = self.policy.saved_actions
        #print ("Saved Actions", saved_actions)
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (policy) loss
        returns = [] # list to save true values

        # calculate the true value using rewards returned from the environment
        # this reward is appended by the environment 
        c = 0
        for r in self.policy.rewards[::-1]:
            # calculate the discounted value
            # Rewards are same for multiple actions, hence compute R only once & use it for all actions taken simultaneously
            if self.view.strategy_name in ['RL_DEC_1']:
                if c % self.cache_size == 0:
                    if self.avg_reward_case:
                        R = R + r - self.avg_reward
                        if c != 0:
                            # We dont have the next state here again, hence we do this update for c = 0
                            self.avg_reward = self.beta * (saved_actions[len(self.policy.rewards) - c][1] - saved_actions[len(self.policy.rewards) - c - 1][1])
                    else:
                        R = r + self.gamma * R
            else:
                if self.avg_reward_case:
                    if c != 0:
                        R = R + r - self.avg_reward2
                        self.avg_reward = self.beta * (saved_actions[len(self.policy.rewards) - c][1] - saved_actions[len(self.policy.rewards) - c - 1][1])
                else:
                    R = r + self.gamma * R
            c += 1
            returns.insert(0, R)
        print ("Returns ", self.cache, returns, len(saved_actions))
        returns = torch.tensor(returns, device=device)
        # Already normalised, don't do anything here
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        #print ("Returns ", self.cache, returns)
        #print ("RETURNS LEN", self.cache, len(returns))

        #print ("SAVED ACTIONS LEN", len(saved_actions))
        advantage = torch.tensor(np.zeros((1)), device=device)
        if self.use_gae:
            for i in reversed(range(len(saved_actions))):
                # This is the last set of values saved, we dont have i + cache_size for this
                # We don't have the next state available immidiately, hence we cannot estimate the value function of next state
                if i >= len(saved_actions) - self.cache_size:
                    td_error = self.policy.rewards[i] - saved_actions[i][1]
                    advantage = td_error
                else:
                    # Do it only one once
                    if i % self.cache_size == 0:
                        td_error = self.policy.rewards[i] + self.gamma * saved_actions[i + self.cache_size][1] - saved_actions[i][1]
                        advantage = advantage * self.tau * self.gamma + td_error
                # calculate actor (policy) loss
                policy_losses.append(-saved_actions[i][0] * advantage)
                # calulate critic (value) loss using L1 smooth loss
                print ("Value, R", saved_actions[i][0], torch.tensor([returns[i]], device=device))
                value_losses.append(F.smooth_l1_loss(saved_actions[i][0], torch.tensor([returns[i]], device=device)))
        else:
            for (log_prob, value), R in zip(saved_actions, returns):
                advantage = R - value.item()
                # calculate actor (policy) loss
                policy_losses.append(-log_prob * advantage)
                # calulate critic (value) loss using L1 smooth loss
                print ("Value, R", value, torch.tensor([R], device=device))
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
        loss.backward(retain_graph=True)
        #loss.backward()
        #print ("Backprop Loss")
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
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
        c = 0
        for r in self.policy2.rewards[::-1]:
            # calculate the discounted value
            #print ("Reward : ", r)
            if self.avg_reward_case:
                if c != 0:
                    self.avg_reward2 = self.beta * (saved_actions[len(self.policy2.rewards) - c][1] - saved_actions[len(self.policy2.rewards) - c - 1][1])
                else:
                    R = r + self.gamma * R
            c += 1
            returns.insert(0, R)
        returns = torch.tensor(returns, device=device)
        returns = (returns - returns.mean()) / (returns.std() + self.eps2)
        #print ("Returns ", returns)

        #print ("Saved Actions", saved_actions)
        advantage = torch.tensor(np.zeros((1)), device=device)
        if self.use_gae:
            for i in reversed(range(len(saved_actions))):
                # This is the last set of values saved, we dont have i + cache_size for this
                # We don't have the next state available immidiately, hence we cannot estimate the value function of next state
                if i == len(saved_actions) - 1:
                    td_error = self.policy2.rewards[i] - saved_actions[i][1]
                    advantage = td_error
                else:
                    td_error = self.policy2.rewards[i] + self.gamma * saved_actions[i + self.cache_size][1] - saved_actions[i][1]
                    advantage = advantage * self.tau * self.gamma + td_error
                # calculate actor (policy) loss
                policy_losses.append(-saved_actions[i][0] * advantage)
                # calulate critic (value) loss using L1 smooth loss
                value_losses.append(F.smooth_l1_loss(saved_actions[i][0], torch.tensor([returns[i]], device=device)))
        else:
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

    def __init__(self, model, cpus, nnp, strategy_name, model_path, resume):
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
        self.model_path = model_path
        self.resume = resume
        self.count = 0
        self.centralized = False
        self.step = False
        self.common_rewards = 0
        # 0 - all agents delay , 1 - cache hit , 2 - only agents delay
        self.reward_type = 1
        self.spatio_rewards = True
        self.tot_delay = 0
        self.fetch_delay = 0
        self.alpha = 0.8
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
            nnp['window'] = 500
        if 'update_freq' not in nnp:
            nnp['update_freq'] = 50
        if 'state_ver' not in nnp:
            nnp['state_ver'] = 0
        if 'policy_type' not in nnp:
            nnp['policy_type'] = 0
        if 'comb' not in nnp:
            nnp['comb'] = 0
        if 'tau' not in nnp:
            nnp['tau'] = 0.95
        if 'beta' not in nnp:
            nnp['beta'] = 0.001
        if 'use_gae' not in nnp:
            nnp['use_gae'] = True
        if 'avg_reward_case' not in nnp:
            nnp['avg_reward_case'] = True
        #self.status = [False] * cpus
        #self.ind_count = [0] * cpus
        self.update_freq = nnp['update_freq']
        self.window = nnp['window']
        self.state_ver = nnp['state_ver']
        self.policy_type = nnp['policy_type']
        self.comb = nnp['comb']
        self.nnp = nnp
        #Creating agents depending on the total number of routers
        for r in self.model.routers:
            self.agents.append(Agent(self, r, self.state_ver, self.policy_type, nnp['gamma'], nnp['lr'], nnp['window'], nnp['comb'],
                                nnp['tau'], nnp['beta'], nnp['use_gae'], nnp['avg_reward_case']))
        
        if self.centralized == True and strategy_name in ['RL_DEC_1']:
            self.avg_reward = 0.0
            if self.policy_type == 2:
                self.policy = GATPolicyCentralized(len(self.model.library), len(self.model.library), 64, len(self.model.routers))
            if self.policy_type == 4:
                self.policy = NCMultiAgentPolicyCentralized(len(self.model.library), len(self.model.library), len(self.model.routers), self.model.neighbor_mask)
            if self.policy_type == 5:
                self.policy = ConsensusPolicyCentralized(len(self.model.library), len(self.model.library), len(self.model.routers), self.model.neighbor_mask)
            if self.policy_type == 6:
                self.policy = CommNetMultiAgentPolicyCentralized(len(self.model.library), len(self.model.library), len(self.model.routers), self.model.neighbor_mask)
            if self.policy_type == 7:
                self.policy = DIALMultiAgentPolicyCentralized(len(self.model.library), len(self.model.library), len(self.model.routers), self.model.neighbor_mask)
            self.eps = np.finfo(np.float32).eps.item()
            self.optimizer = optim.Adam(self.policy.parameters(), lr=nnp['lr'])
            
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

    def get_neighbor_policy(self, cache, self_=False):
        #Already a tensor, no need to convert
        policies = []
        if self_:
            policies = [self.agents[self.model.routers.index(cache)].ps]
        for nei in self.model.neighbor[cache]:
            policies.append(self.agents[self.model.routers.index(nei)].ps)
        policies = torch.stack([p.squeeze() for p in policies]).to(device)
        #print("Neighbor policies ", policies) 
        return policies
            

    def get_neighbor_action(self, cache, self_=False):
        actions = []
        if self_:
            actions.append(self.agents[self.model.routers.index(cache)].action)
            #print ("action self", self.cache, type(self.agents[self.model.routers.index(self.cache)].action), self.agents[self.model.routers.index(self.cache)].action)
        for nei in self.model.neighbor[cache]:
            #print ("Nei action", nei, type(self.agents[self.model.routers.index(nei)].action), self.agents[self.model.routers.index(nei)].action)
            actions.append(self.agents[self.model.routers.index(nei)].action)
        actions = torch.stack([a for a in actions]).to(device)
        #print("Neighbor actions ", actions) 
        return actions

    def decode_action(self, action):
        """
        Decode the action and return a vector of binary values signifying which caches
        should cache what content
        """
        action_decoded = np.full((len(self.model.library)), 0, dtype=int)
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

    def get_policy(self, ob, ps=None):
        """
        ob - set of states of itself as well as neighbor in case of ma2c policies
        ps - fingerprints of neighbors
        """
        
        done = np.array([0])
        if self.policy_type in [4,5,6,7]:
            policy = self.policy.forward(ob, done, ps)
        elif self.policy_type in [2]:
            policy = self.policy.forward(ob, done, self.model.neighbor_mask)
        else:
            policy = self.policy.forward(ob, done)
        return policy

    def get_value(self, ob, action=None, ps=None):
        """
        ps - neighbor fingerprint
        action - array of actions in case of ma2c
        ob - set of states of itself as well as neighbors in case of ma2c policies
        """
        
        done = np.array([0])
        if self.policy_type in [4,5,6,7]:
            value = self.policy.forward(ob, done, ps, action, 'v')
        elif self.policy_type in [2]:
            value = self.policy.forward(ob, done, self.model.neighbor_mask, action, 'v')
        else:
            value = self.policy.forward(ob, done, None, 'v')
        return value

    def save_model(self, count):
        """
        Save NN model parameters 
        """
        PATH = self.model_path
        if PATH is not None:
            PATH = PATH + "/" + str(self.strategy_name) + ".pt"
        else:
            PATH = str(self.strategy_name) + ".pt"
            torch.save({
                'epoch': count,
                'model_state_dict': self.policy.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                }, PATH)
        print ("Saved Model to ", PATH)

    def load_model(self):
        """
        Load NN model and initialise them
        """
        PATH = self.model_path
        if PATH is not None:
            PATH = PATH + "/" + str(self.strategy_name) + ".pt"
        else:
            PATH = str(self.strategy_name) + ".pt"
        try:
            checkpoint = torch.load(PATH)
        except:
            print ("CHECKPOINT not present.. Cannot LOAD Model")
            return
        print ("LOADING MODELS previously stored")
        #print ("MODEL : ", checkpoint['model_state_dict'])
        #print ("OPTIM : ", checkpoint['optimizer_state_dict'])
        try:
            self.policy.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.count = checkpoint['epoch']
        except:
            print ("Saved Model does not match the Policy")
    
    def select_actions(self, state, ps):
        #print ("STATE", state)
        #ob = torch.from_numpy(np.expand_dims(ob, axis=0)).float()
        # States should consider neighboring states + policies of neighbors
        actions = []
        state = torch.stack([x for x in state]).to(device)
        ps = torch.stack([p for p in ps]).to(device)
        top_k_val = [0] * len(self.agents)
        top_k_inx = [0] * len(self.agents)
        # Maybe one step late (no sync)
        probs = self.get_policy(state, ps)
        print ("PROBS ", probs, probs[0].shape)
        # Adding the fingerprint
        for i in range(len(self.agents)):
            self.agents[i].ps = probs[i]
            #print ("PROBS", probs.shape)
            # create categorical distribution over list of probabilities of actions
            m = Categorical(probs[i])
            #print ("Output of NN ")
            print (m)
            # sample an action from the categorical distribution
            # Select top k values from probability softmax output
            top_k_val[i], top_k_inx[i] = probs[i].topk(self.model.cache_size[self.model.routers[i]])
            action = self.decode_action(top_k_inx[i].cpu().detach().numpy())
            self.action = torch.from_numpy(np.array(action)).float().to(device)
            actions.append(self.action)
        actions = torch.stack([a for a in actions]).to(device)
        state_value = self.get_value(state, actions, ps)
        for i in range(len(self.agents)):
            for ac in top_k_inx[i]:
                self.policy.saved_actions.append(SavedAction(m.log_prob(ac), state_value[i]))
        print ("Value ", state_value)
        return actions

    def update(self):
        """

        Training code. Calculates actor and critic loss and performn backpropogation.
        """
        print ("UPDATE FN ")
        R = 0
        saved_actions = self.policy.saved_actions
        #print ("Saved Actions", saved_actions)
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (policy) loss
        returns = [] # list to save true values

        # calculate the true value using rewards returned from the environment
        # this reward is appended by the environment 
        c = 0
        for r in self.policy.rewards[::-1]:
            # calculate the discounted value
            # Rewards are same for multiple actions, hence compute R only once & use it for all actions taken simultaneously
            if self.strategy_name in ['RL_DEC_1']:
                if c % self.model.cache_size[self.model.routers[0]] == 0:
                    if self.nnp['avg_reward_case']:
                        R = R + r - self.avg_reward
                        if c != 0:
                            # We dont have the next state here again, hence we do this update for c = 0
                            self.avg_reward = self.nnp['beta'] * (saved_actions[len(self.policy.rewards) - c][1] - saved_actions[len(self.policy.rewards) - c - 1][1])
                    else:
                        R = r + self.nnp['gamma'] * R
            c += 1
            returns.insert(0, R)
        print ("Rewards ", type(self.policy.rewards[0]), self.policy.rewards[0])
        print ("Saved actions ", type(saved_actions), saved_actions[0][0], saved_actions[0][1])
        returns = torch.tensor(returns, device=device)
        # Already normalised, don't do anything here
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        #print ("Returns ", self.cache, returns)
        #print ("RETURNS LEN", self.cache, len(returns))
        print ("Returns ",  type(returns[0]), returns[0])

        #print ("SAVED ACTIONS LEN", len(saved_actions))
        advantage = torch.tensor(np.zeros(1, dtype=float), device=device)
        if self.nnp['use_gae']:
            for i in reversed(range(len(saved_actions))):
                # This is the last set of values saved, we dont have i + cache_size for this
                # We don't have the next state available immidiately, hence we cannot estimate the value function of next state
                if i >= len(saved_actions) - self.model.cache_size[self.model.routers[0]]:
                    td_error = self.policy.rewards[i] - saved_actions[i][1]
                    advantage = td_error
                else:
                    # Do it only one once
                    if i % self.model.cache_size[self.model.routers[0]] == 0:
                        td_error = self.policy.rewards[i] + self.nnp['gamma'] * saved_actions[i + self.model.cache_size[self.model.routers[0]]][1] - saved_actions[i][1]
                        advantage = advantage * self.nnp['tau'] * self.nnp['gamma'] + td_error
                # calculate actor (policy) loss
                print ("Value, R", saved_actions[i][0], returns[i])
                policy_losses.append(-saved_actions[i][0] * advantage)
                # calulate critic (value) loss using L1 smooth loss
                value_losses.append(F.smooth_l1_loss(saved_actions[i][0], torch.tensor([returns[i]], device=device)))
        else:
            for (log_prob, value), R in zip(saved_actions, returns):
                advantage = R - value.item()
                # calculate actor (policy) loss
                policy_losses.append(-log_prob * advantage)
                # calulate critic (value) loss using L1 smooth loss
                print ("Value, R", value, R)
                value_losses.append(F.smooth_l1_loss(value, torch.tensor([R], device=device)))
        
        #print ("Policy Loss ", policy_losses)
        #print ("Value Loss ", value_losses)
        # reset gradients
        self.optimizer.zero_grad()
        #print ("Optimizer gradient")
        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss = loss.type(torch.cuda.DoubleTensor)
        print ("Total Loss ", loss)
        # perform backprop
        loss.backward(retain_graph=True)
        #loss.backward()
        #print ("Backprop Loss")
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
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
        self.neighbor_distance = 2
        self.neighbor = dict()
        self.edges = nx.edges(topology)
        print ("EDGES", self.edges)
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
                    #print (nx.ego_graph(topology, node, radius=3))
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
        self.neighbor_mask = np.zeros((len(self.routers), len(self.routers)), dtype=int)
        self.distance_mask = np.zeros((len(self.routers), len(self.routers)), dtype=float)
        for r in self.routers:
            path_lengths = nx.single_source_dijkstra_path_length(topology, r)
            self.neighbor[r] = [v for v, length in path_lengths.items() \
                if length <= self.neighbor_distance and v in self.routers and v !=r]
            print ("Neighbor for node ", r, " is ", self.neighbor[r])
        for i in range(len(self.routers)):
            v = self.neighbor[self.routers[i]]
            dist_len = nx.single_source_dijkstra_path_length(topology, self.routers[i])
            print ("DIst len ", i, dist_len)
            for j in range(len(self.routers)):
                self.distance_mask[i, j] = dist_len[self.routers[j]] 
                if self.routers[j] in v:
                    self.neighbor_mask[i, j] = 1
        print ("Neighbor mask = ", self.neighbor_mask)
        print ("Distance mask = ", self.distance_mask)
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
        #print ("GET CONTENT", node, inx)
        if node in self.model.cache and content is not None:
            return self.model.cache[node].get(content)
        if node in self.model.cache:
            cache_hit = self.model.cache[node].get(self.session[inx]['content'], inx)
            #print ("NODE IS ROUTER", node, inx, cache_hit)
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
                #print ("NODE IS SERVER", node, inx)
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
