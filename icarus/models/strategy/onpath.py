"""Implementations of all on-path strategies"""
from __future__ import division
import random

import networkx as nx
import sys

from icarus.registry import register_strategy
from icarus.util import inheritdoc, path_links

from .base import Strategy

__all__ = [
       'Partition',
       'Edge',
       'RlDec',
       'LeaveCopyEverywhere',
       'LeaveCopyDown',
       'ProbCache',
       'CacheLessForMore',
       'RandomBernoulli',
       'RandomChoice',
           ]


@register_strategy('PARTITION')
class Partition(Strategy):
    """Partition caching strategy.

    In this strategy the network is divided into as many partitions as the number
    of caching nodes and each receiver is statically mapped to one and only one
    caching node. When a request is issued it is forwarded to the cache mapped
    to the receiver. In case of a miss the request is routed to the source and
    then returned to cache, which will store it and forward it back to the
    receiver.

    This requires median cache placement, which optimizes the placement of
    caches for this strategy.

    This strategy is normally used with a small number of caching nodes. This
    is the the behaviour normally adopted by Network CDN (NCDN). Google Global
    Cache (GGC) operates this way.
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller):
        super(Partition, self).__init__(view, controller)
        if 'cache_assignment' not in self.view.topology().graph:
            raise ValueError('The topology does not have cache assignment '
                             'information. Have you used the optimal median '
                             'cache assignment?')
        self.cache_assignment = self.view.topology().graph['cache_assignment']

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, size, log):
        source = self.view.content_source(content)
        self.controller.start_session(time, receiver, content, log)
        cache = self.cache_assignment[receiver]
        self.controller.forward_request_path(receiver, cache)
        if not self.controller.get_content(cache):
            self.controller.forward_request_path(cache, source)
            self.controller.get_content(source)
            self.controller.forward_content_path(source, cache, size)
            self.controller.put_content(cache)
        self.controller.forward_content_path(cache, receiver, size)
        self.controller.end_session()


@register_strategy('EDGE')
class Edge(Strategy):
    """Edge caching strategy.

    In this strategy only a cache at the edge is looked up before forwarding
    a content request to the original source.

    In practice, this is like an LCE but it only queries the first cache it
    finds in the path. It is assumed to be used with a topology where each
    PoP has a cache but it simulates a case where the cache is actually further
    down the access network and it is not looked up for transit traffic passing
    through the PoP but only for PoP-originated requests.
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller):
        super(Edge, self).__init__(view, controller)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, size, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        edge_cache = None
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                edge_cache = v
                if self.controller.get_content(v):
                    serving_node = v
                else:
                    # Cache miss, get content from source
                    self.controller.forward_request_path(v, source)
                    self.controller.get_content(source)
                    serving_node = source
                break
        else:
            # No caches on the path at all, get it from source
            self.controller.get_content(v)
            serving_node = v

        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        self.controller.forward_content_path(serving_node, receiver, size, path)
        if serving_node == source:
            self.controller.put_content(edge_cache)
        self.controller.end_session()

@register_strategy('RL_DEC')
class RlDec(Strategy):
    """Reinforcement Learning Decentralized caching strategy.

    In this strategy, we aim to find an optimal policy where the caches
    look at the current state of the network and try decide if it needs
    to cache certain files or not
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller):
        super(RlDec, self).__init__(view, controller)

    def env_step(self, size):
        """

        A step of the environment includes the following for all agents:
        1. get the current state of each agent
        2. select actions
        3. perform the actions
        """
        #print ("ENV STEP")
        for a in self.view.agents:
            curr_state = a.get_state()
            action = a.select_actions(curr_state)
            action = a.decode_action(action)
            a.rewards -= self.perform_action(action, a.cache, size)

    def update_gradients(self):
        """

        Get the rewards from the environment
        Append the rewards to the rewards data structure accessed by the policy class
        Perform actor-critic update
        """
        #print ("UPGARDE GRADIENTS")
        for a in self.view.agents:
            a.update()


    def perform_action(self,action, cache, size):
        """
        Decode the actions provided by the policy network
        Cache files according to the action selected

        #TODO - One of the scenarios that can happen that can add delay:
        Files needs to be cached but removed based on LRU, so one file can
        be deleted and then fetched again in the same iteration.
        SO we create a list of files to be fetched and list of files
        to be deleted, first delete the files and then get the rest to cache.
        """
        add_contents = []
        remove_contents = []
        existing_contents = self.view.cache_dump(cache)
        for a in range(action.size):
            if action[a] == 1:
                #here get_content called without content so cache hit/miss can be computed,
                if self.controller.get_content(cache, a+1) is False:
                    add_contents.append(a+1)
            else:
                if a+1 in existing_contents:
                    remove_contents.append(a+1)

        #print ("To be added ", add_contents)
        #print ("To be removed ", remove_contents)
    
        min_delay = sys.maxsize
        for rc in remove_contents:
            self.controller.remove_content(cache, rc)
        for ac in add_contents:
            # Get location of all nodes that has the content stored
            content_loc = self.view.content_locations(ac)
            # Finding the path with the minimum delay in the network
            for c in content_loc :
                delay = self.view.shortest_path_len(cache, c)
                if delay < min_delay:
                    min_delay = delay
                    serving_node = c

            # fetching the data
            min_path = self.view.shortest_path(cache, serving_node)
            for u, v in path_links(min_path):
                self.controller.forward_request_hop(u, v)

            # update the rewards for the episode
            #print ("DELAY IN FETCHING", min_delay)
            path = list(reversed(self.view.shortest_path(cache, serving_node)))
            self.controller.forward_content_path(serving_node, cache, size, path)
            self.controller.put_content(cache, ac)
        print ("CACHE DUMP ", cache, " = ", self.view.cache_dump(cache))
        return min_delay

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, size, log):
        # get all required data
        print ("PROCESS EVENT", time, receiver, content)
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        self.view.count += 1
        print ("View Count : ", self.view.count)
        if self.view.count % 50 == 0:
            self.env_step(size) 
        # Get location of all nodes that has the content stored
        content_loc = self.view.content_locations(content)
        min_delay = sys.maxsize
        #print ("Min Delay ", min_delay) 
        # Finding the path with the minimum delay in the network
        for c in content_loc :
            delay = self.view.shortest_path_len(receiver, c)
            #print ("Delay : ", receiver, " , ", c, " : ", delay) 
            if delay < min_delay:
                min_delay = delay
                serving_node = c

        # fetching the data 
        min_path = self.view.shortest_path(receiver, serving_node)
        for u, v in path_links(min_path):
            self.controller.forward_request_hop(u, v)
            self.controller.get_content(v)

        # update the rewards for the episode
        self.view.common_rewards -= min_delay
        if self.view.count % 50 == 0:
            for a in self.view.agents:
                a.rewards -= self.view.common_rewards
                a.policy.rewards.append(a.rewards)
                a.rewards = 0
            self.view.common_rewards = 0
            #print ("AGENT ", a , " REWARDS ", a.policy.rewards.append)
        if self.view.count % 500 == 0:
            self.update_gradients()
        # Return content
        #print ("Serving Node", serving_node)
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        self.controller.forward_content_path(serving_node, receiver, size, path)
        self.controller.end_session()
        #print ("Session End")
        
@register_strategy('LCE')
class LeaveCopyEverywhere(Strategy):
    """Leave Copy Everywhere (LCE) strategy.

    In this strategy a copy of a content is replicated at any cache on the
    path between serving node and receiver.
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(LeaveCopyEverywhere, self).__init__(view, controller)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, size, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v, size)
            if self.view.has_cache(v):
                # insert content
                self.controller.put_content(v)
        self.controller.end_session()


@register_strategy('LCD')
class LeaveCopyDown(Strategy):
    """Leave Copy Down (LCD) strategy.

    According to this strategy, one copy of a content is replicated only in
    the caching node you hop away from the serving node in the direction of
    the receiver. This strategy is described in [2]_.

    Rereferences
    ------------
    ..[1] N. Laoutaris, H. Che, i. Stavrakakis, The LCD interconnection of LRU
          caches and its analysis.
          Available: http://cs-people.bu.edu/nlaout/analysis_PEVA.pdf
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(LeaveCopyDown, self).__init__(view, controller)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, size, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
        else:
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        # Leave a copy of the content only in the cache one level down the hit
        # caching node
        copied = False
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v, size)
            if not copied and v != receiver and self.view.has_cache(v):
                self.controller.put_content(v)
                copied = True
        self.controller.end_session()


@register_strategy('PROB_CACHE')
class ProbCache(Strategy):
    """ProbCache strategy [3]_

    This strategy caches content objects probabilistically on a path with a
    probability depending on various factors, including distance from source
    and destination and caching space available on the path.

    This strategy was originally proposed in [2]_ and extended in [3]_. This
    class implements the extended version described in [3]_. In the extended
    version of ProbCache the :math`x/c` factor of the ProbCache equation is
    raised to the power of :math`c`.

    References
    ----------
    ..[2] I. Psaras, W. Chai, G. Pavlou, Probabilistic In-Network Caching for
          Information-Centric Networks, in Proc. of ACM SIGCOMM ICN '12
          Available: http://www.ee.ucl.ac.uk/~uceeips/prob-cache-icn-sigcomm12.pdf
    ..[3] I. Psaras, W. Chai, G. Pavlou, In-Network Cache Management and
          Resource Allocation for Information-Centric Networks, IEEE
          Transactions on Parallel and Distributed Systems, 22 May 2014
          Available: http://doi.ieeecomputersociety.org/10.1109/TPDS.2013.304
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, t_tw=10):
        super(ProbCache, self).__init__(view, controller)
        self.t_tw = t_tw
        self.cache_size = view.cache_nodes(size=True)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, size, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        for hop in range(1, len(path)):
            u = path[hop - 1]
            v = path[hop]
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
        else:
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        c = len([node for node in path if self.view.has_cache(node)])
        x = 0.0
        for hop in range(1, len(path)):
            u = path[hop - 1]
            v = path[hop]
            N = sum([self.cache_size[n] for n in path[hop - 1:]
                     if n in self.cache_size])
            if v in self.cache_size:
                x += 1
            self.controller.forward_content_hop(u, v, size)
            if v != receiver and v in self.cache_size:
                # The (x/c) factor raised to the power of "c" according to the
                # extended version of ProbCache published in IEEE TPDS
                prob_cache = float(N) / (self.t_tw * self.cache_size[v]) * (x / c) ** c
                if random.random() < prob_cache:
                    self.controller.put_content(v)
        self.controller.end_session()


@register_strategy('CL4M')
class CacheLessForMore(Strategy):
    """Cache less for more strategy [4]_.

    This strategy caches items only once in the delivery path, precisely in the
    node with the greatest betweenness centrality (i.e., that is traversed by
    the greatest number of shortest paths). If the argument *use_ego_betw* is
    set to *True* then the betweenness centrality of the ego-network is used
    instead.

    References
    ----------
    ..[4] W. Chai, D. He, I. Psaras, G. Pavlou, Cache Less for More in
          Information-centric Networks, in IFIP NETWORKING '12
          Available: http://www.ee.ucl.ac.uk/~uceeips/centrality-networking12.pdf
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, use_ego_betw=False, **kwargs):
        super(CacheLessForMore, self).__init__(view, controller)
        topology = view.topology()
        if use_ego_betw:
            self.betw = dict((v, nx.betweenness_centrality(nx.ego_graph(topology, v))[v])
                             for v in topology.nodes())
        else:
            self.betw = nx.betweenness_centrality(topology)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, size, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
        # No cache hits, get content from source
        else:
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        # get the cache with maximum betweenness centrality
        # if there are more than one cache with max betw then pick the one
        # closer to the receiver
        max_betw = -1
        designated_cache = None
        for v in path[1:]:
            if self.view.has_cache(v):
                if self.betw[v] >= max_betw:
                    max_betw = self.betw[v]
                    designated_cache = v
        # Forward content
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v, size)
            if v == designated_cache:
                self.controller.put_content(v)
        self.controller.end_session()


@register_strategy('RAND_BERNOULLI')
class RandomBernoulli(Strategy):
    """Bernoulli random cache insertion.

    In this strategy, a content is randomly inserted in a cache on the path
    from serving node to receiver with probability *p*.
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, p=0.2, **kwargs):
        super(RandomBernoulli, self).__init__(view, controller)
        self.p = p

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, size, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
        else:
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v, size)
            if v != receiver and self.view.has_cache(v):
                if random.random() < self.p:
                    self.controller.put_content(v)
        self.controller.end_session()


@register_strategy('RAND_CHOICE')
class RandomChoice(Strategy):
    """Random choice strategy

    This strategy stores the served content exactly in one single cache on the
    path from serving node to receiver selected randomly.
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(RandomChoice, self).__init__(view, controller)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, size, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
        else:
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        caches = [v for v in path[1:-1] if self.view.has_cache(v)]
        designated_cache = random.choice(caches) if len(caches) > 0 else None
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v, size)
            if v == designated_cache:
                self.controller.put_content(v)
        self.controller.end_session()
