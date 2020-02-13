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

import networkx as nx
import fnss

from itertools import count
from itertools import combinations
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

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

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


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
        self.affine1 = nn.Linear(s_len, 128)

        # actor's layer
        self.action_head = nn.Linear(128, a_len)

        # critic's layer
        self.value_head = nn.Linear(128, 1)

        # action and reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """

        forward pass of both actor and critic
        """
        #print ("POLICY FORWARD")
        x = F.relu(self.affine1(x))

        # actor: choses action to taken based on state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic evaluates being in state s_t
        state_values = self.value_head(x)

        # return value of both actor and critic as a 2 tuple
        # 1. a list of probability of each action over state space
        # 2. the value from state s_t
        return action_prob, state_values


class Agent(object):
    """

    Each router is considered as an agent. The agent needs to view its state space
    and then select an action (which files to cache based on the current policy it
    is executing.
    """

    def __init__(self, view, router, ver):
        """Constructor
        
        Parameters
        ----------
        model : NetworkView
            The network view instance
        """
        if not isinstance(view, NetworkView):
            raise ValueError('The model argument must be an instance of '
                             'NetworkView')
        self.view = view
        self.cache = router
        #If all cache are equal size, can be moved to model
        self.action_choice = list(combinations(self.view.lib, self.view.model.cache_size[self.cache])) 
        #print ("Action choices : ", self.action_choice)
        self.gamma = 0.9
        self.rewards = 0
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
        #self.action_space = combinations(
        if self.state_ver == 0:
            self.state = np.full((len(self.view.model.library)), 0, dtype=int) 
        else:
            #TODO - if needed, we update the statistics
            self.state = np.full((self.view.model.cache_size[self.cache]), 0, dtype=int)
         
        # Initialize the policy and other neural network optimizers
        self.policy = Policy(len(self.view.model.library), len(self.action_choice))
        #print ("POLICY", self.policy)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-2)
        #print ("OPTIMIZER", self.optimizer)
        self.eps = np.finfo(np.float32).eps.item()
        #print ("EPS", self.eps)

    def get_state(self):
        """
        Returns the current state of the cache.
        """
        contents = self.view.cache_dump(self.cache)
        if self.state_ver == 0:
            for c in contents :
                self.state[c-1] = 1
        else:
            for i in range(len(contents)) :
                self.state[i] = contents[i]
        #print ("Agent State ", self.cache, " : ")
        #print (self.state)
        return self.state

    def decode_action(self, action):
        """
        Decode the action and return a vector of binary values signifying which caches
        should cache what content
        """
        files_to_cache = self.action_choice[action]
        files_to_cache = list(files_to_cache)
        #print ("Files to cache", files_to_cache)
        action_decoded = np.full((len(self.view.model.library)), 0, dtype=int)
        for f in files_to_cache:
            action_decoded[f] = 1
        print ("Action decoded", action_decoded)
        return action_decoded

    """
    def perform_action(action):
        Decode the actions provided by the policy network
        Cache files according to the action selected
        
        #TODO - One of the scenarios that can happen that can add delay:
        Files needs to be cached but removed based on LRU, so one file can 
        be deleted and then fetched again in the same iteration.
        SO we create a list of files to be fetched and list of files
        to be deleted, first delete the files and then get the rest to cache.
        
        add_contents = []
        remove_contents = []
        existing_contents = self.view.cache_dump(self.cache)
        for a in range(action.shape(0)):
            if action[a] == 1:
                #here get_content called without content so cache hit/miss can be computed,
                if self.controller.get_content(self.cache, a+1) is False:
                    add_contents.append(a+1)
            else:
                if a+1 in existing_contents:
                    remove_contents.append(a+1)
        
        print ("To be added ", add_contents)
        print ("To be removed ", remove_contents)
        
        for rc in remove_contents:
            self.controller.remove_content(self.cache, rc)             
        for ac in add_contents:
            # Get location of all nodes that has the content stored
            content_loc = self.view.content_locations(content)
            min_delay = sys.maxint
            # Finding the path with the minimum delay in the network
            for c in content_loc :
                delay = self.view.shortest_path_len(self.cache, c)
                if delay < min_delay:
                    min_delay = delay
                    serving_node = c

            # fetching the data
            min_path = self.view.shortest_path(self.cache, c)
            for u, v in path_links(min_path):
                self.controller.forward_request_hop(u, v)
            
            # update the rewards for the episode
            self.view.rewards -= min_delay
            path = list(reversed(self.view.shortest_path(self.cache, serving_node)))
            self.controller.forward_content_path(serving_node, self.cache, path)
            self.controller.put_content(self.cache, ac)
    """

    def select_actions(self, state):
        state = torch.from_numpy(state).float()
        #print ("STATE", state)
        probs, state_value = self.policy.forward(state)
        #print ("PROBS, STATE_VAL", probs, state_value)
        # create categorical distribution over list of probabilities of actions
        m = Categorical(probs)
        #print ("Output of NN ")
        #print (m)
        # sample an action from the categorical distribution
        action = m.sample()
        #print ("Sampled Action", action)
        # save to action buffer
        self.policy.saved_actions.append(SavedAction(m.log_prob(action), state_value))

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
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        #print ("Returns ", returns)

        #print ("Saved Actions", saved_actions)
        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)

            # calulate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))
        
        #print ("Policy Loss ", policy_losses)
        #print ("Value Loss ", value_losses)
        # reset gradients
        self.optimizer.zero_grad()
        #print ("Optimizer gradient")
        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        print ("Total Loss ", loss)
        # perform backprop
        loss.backward()
        #print ("Backprop Loss")
        self.optimizer.step()
        #print ("Optimizer Step")
        # reset rewards and action buffers
        del self.policy.rewards[:]
        del self.policy.saved_actions[:]
        #print ("Deleted policy and saved actions")

class NetworkView(object):
    """Network view

    This class provides an interface that strategies and data collectors can
    use to know updated information about the status of the network.
    For example the network view provides information about shortest paths,
    characteristics of links and currently cached objects in nodes.
    """

    def __init__(self, model):
        """Constructor

        Parameters
        ----------
        model : NetworkModel
            The network model instance
        """
        if not isinstance(model, NetworkModel):
            raise ValueError('The model argument must be an instance of '
                             'NetworkModel')
        self.model = model
        self.count = 0
        self.common_rewards = 0
        #Different because other library is a set, here we want to preserve ordering
        self.lib = [item for item in range(0, len(self.model.library))]
        #Contains the agents as objects of Class Agent
        self.agents = []
        #Creating agents depending on the total number of routers
        for r in self.model.routers:
            self.agents.append(Agent(self, r, 0))

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

    def __init__(self, topology, cache_policy, shortest_path=None, shortest_path_len=None):
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
       
        print ("Shortest Path Lengths: ", dict(nx.all_pairs_dijkstra_path_length(topology)))
        # Shortest paths length of the network
        self.shortest_path_len = dict(shortest_path_len) if shortest_path_len is not None \
                                else symmetrify_paths_len(dict(nx.all_pairs_dijkstra_path_length(topology)))
        # Network topology
        self.topology = topology

        # Dictionary mapping each content object to its source
        # dict of location of contents keyed by content ID
        self.content_source = {}
        # Dictionary mapping the reverse, i.e. nodes to set of contents stored
        self.source_node = {}
        #List of all routers in the topology
        self.routers = []
        #List of all files in the library
        self.library = set()
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
            if stack_name == 'router':
                if 'cache_size' in stack_props:
                    self.cache_size[node] = stack_props['cache_size']
                    self.routers.append(node)
            elif stack_name == 'source':
                contents = stack_props['contents']
                self.source_node[node] = contents
                for content in contents:
                    self.library.add(content)
                    self.content_source[content] = node
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


class NetworkController(object):
    """Network controller

    This class is in charge of executing operations on the network model on
    behalf of a strategy implementation. It is also in charge of notifying
    data collectors of relevant events.
    """

    def __init__(self, model):
        """Constructor

        Parameters
        ----------
        model : NetworkModel
            Instance of the network model
        """
        self.session = None
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

    def start_session(self, timestamp, receiver, content, log):
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
        self.session = dict(timestamp=timestamp,
                            receiver=receiver,
                            content=content,
                            log=log)
        if self.collector is not None and self.session['log']:
            self.collector.start_session(timestamp, receiver, content)

    def forward_request_path(self, s, t, path=None, main_path=True):
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
            self.forward_request_hop(u, v, main_path)

    def forward_content_path(self, u, v, path=None, main_path=True):
        """Forward a content from node *s* to node *t* over the provided path.

        Parameters
        ----------
        s : any hashable type
            Origin node
        t : any hashable type
            Destination node
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
            self.forward_content_hop(u, v, main_path)

    def forward_request_hop(self, u, v, main_path=True):
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
        if self.collector is not None and self.session['log']:
            self.collector.request_hop(u, v, main_path)

    def forward_content_hop(self, u, v, main_path=True):
        """Forward a content over link  u -> v.

        Parameters
        ----------
        u : any hashable type
            Origin node
        v : any hashable type
            Destination node
        main_path : bool, optional
            If *True*, indicates that this link is being traversed by content
            that will be delivered to the receiver. This is needed to
            calculate latency correctly in multicast cases. Default value is
            *True*
        """
        if self.collector is not None and self.session['log']:
            self.collector.content_hop(u, v, main_path)

    def put_content(self, node, content=None):
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
            return self.model.cache[node].put(self.session['content'])
        if node in self.model.cache and content is not None:
            return self.model.cache[node].put(content)

    def get_content(self, node, content=None):
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
            cache_hit = self.model.cache[node].get(self.session['content'])
            if cache_hit:
                if self.session['log']:
                    self.collector.cache_hit(node)
            else:
                if self.session['log']:
                    self.collector.cache_miss(node)
            return cache_hit
        name, props = fnss.get_stack(self.model.topology, node)
        if name == 'source' and self.session['content'] in props['contents']:
            if self.collector is not None and self.session['log']:
                self.collector.server_hit(node)
            return True
        else:
            return False

    def remove_content(self, node, content=None):
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
            return self.model.cache[node].remove(self.session['content'])
        if node in self.model.cache and content is not None:
            return self.model.cache[node].remove(content)

    def end_session(self, success=True):
        """Close a session

        Parameters
        ----------
        success : bool, optional
            *True* if the session was completed successfully, *False* otherwise
        """
        if self.collector is not None and self.session['log']:
            self.collector.end_session(success)
        self.session = None

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

    def get_content_local_cache(self, node):
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
                self.collector.cache_hit(node)
        else:
            if self.session['log']:
                self.collector.cache_miss(node)
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
