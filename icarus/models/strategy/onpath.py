"""Implementations of all on-path strategies"""
from __future__ import division
import random

import networkx as nx
import numpy as np
import itertools

from icarus.registry import register_strategy
from icarus.util import inheritdoc, path_links

from .base import Strategy

__all__ = [
       'Partition',
       'CentralizedRL',
       'DecRL',
       'Edge',
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
    def process_event(self, time, receiver, content, log):
        source = self.view.content_source(content)
        self.controller.start_session(time, receiver, content, log)
        cache = self.cache_assignment[receiver]
        self.controller.forward_request_path(receiver, cache)
        if not self.controller.get_content(cache):
            self.controller.forward_request_path(cache, source)
            self.controller.get_content(source)
            self.controller.forward_content_path(source, cache)
            self.controller.put_content(cache)
        self.controller.forward_content_path(cache, receiver)
        self.controller.end_session()


@register_strategy('CENTRALIZED_RL')
class CentralizedRL(Strategy):
    """Edge caching strategy.
    MAKE SURE CACHING POLICY IS LRU
    In this strategy only a cache at the edge is looked up before forwarding
    a content request to the original source.

    In practice, this is like an LCE but it only queries the cache it
    finds in the path. It is assumed to be used with a topology where each
    PoP has a cache but it simulates a case where the cache is actually further
    down the access network and it is not looked up for transit traffic passing
    through the PoP but only for PoP-originated requests.
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller):
        super(CentralizedRL, self).__init__(view, controller)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        print ("SESSION BEGIN")
        print ("Process Event ", time, receiver, content)
        source = self.view.content_source(content)
        self.controller.start_session(time, receiver, content, log)
        if self.view.count == 0 :
            try:
                #if not self.view.first :
                self.view.update_q_table(self.view.model.rewards, 0.1, 0.9)
                #else :
                #    self.view.first = False
                self.view.model.rewards = 0
                #get state and action
                print ("Before get action")
                action = self.view.get_action()
                print ("Action", action)
                #cache the relevant contents (for now assume LRU eviction policy)
                #In case of more contents to be cached then allowed, it will use LRU to evict the first contents
                for (x,y), value in np.ndenumerate(action):
                    if value == 1:
                        if self.controller.get_content(self.view.model.routers[x],y+1) is False:
                            self.controller.put_content(self.view.model.routers[x],y+1)
                            #negative reward for cache eviction and swapping
                            self.view.model.rewards -= 1
                print ("AFTER ACTION TAKEN")
                #perform the caching
                #reset the popularity of the contents 
                self.view.model.popularity *= 0
                #self.view.get_state()
            except:
                print ("EROOR HEREEEEEEE")
        self.view.count = (self.view.count + 1) % 100
        content_loc = self.view.content_locations(content)
        print ("Locations of content", content_loc)
        min_delay_path = 1000000
        min_path = []
        for c in content_loc :
            path = self.view.shortest_path(receiver, c)
            print ("Path from rx to content location", path)
            #print ("State : ", self.view.get_state())
            # Route requests to original source and queries caches on the path
            current_delay = 0
            print ("Path Links", path_links(path))
            for u, v in path_links(path):
                print ("Link Delay", u, v, self.view.link_delay(u, v))
                current_delay = current_delay + self.view.link_delay(u,v)
                self.controller.forward_request_hop(u, v)
                #print ("Cache dump of ", v , "is: ", self.view.cache_dump(v))
            if current_delay < min_delay_path:
                min_delay_path = current_delay
                min_path = path
                serving_node = c
        print ("Min Path : ", min_path)
        for u, v in path_links(min_path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v) and v != source:
                print (v, " has cache")
                print ("Routers :", self.view.model.routers) 
                try:
                    print ("Inx", self.view.model.routers.index(v))
                    print ("val", self.view.model.popularity[self.view.model.routers.index(v), content-1])
                    self.view.model.popularity[self.view.model.routers.index(v), content-1] += 1.0
                    print ("val after", self.view.model.popularity[self.view.model.routers.index(v), content-1])
                except:
                    print ("ERROR HERE", v)
        # No caches on the path at all, get it from source
        print ("Serving Node : ", serving_node)
        print ("Total Delay : ", min_delay_path)
        self.view.model.rewards -= min_delay_path
        self.controller.get_content(serving_node)
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        self.controller.forward_content_path(serving_node, receiver, path)
        #if serving_node == source:
        #    self.controller.put_content(edge_cache)
        self.controller.end_session()
        print ("SESSION END")


@register_strategy('DEC_RL')
class DecRL(Strategy):
    """Edge caching strategy.
    MAKE SURE CACHING POLICY IS LRU
    In this strategy only a cache at the edge is looked up before forwarding
    a content request to the original source.

    In practice, this is like an LCE but it only queries the cache it
    finds in the path. It is assumed to be used with a topology where each
    PoP has a cache but it simulates a case where the cache is actually further
    down the access network and it is not looked up for transit traffic passing
    through the PoP but only for PoP-originated requests.
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller):
        super(DecRL, self).__init__(view, controller)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        print ("SESSION BEGIN")
        print ("Process Event ", time, receiver, content)
        alpha = 0.1
        gamma = 0.9
        source = self.view.content_source(content)
        self.controller.start_session(time, receiver, content, log)
        old_state = []
        actions = []
        common_reward = 0.0
        rewards = np.full((len(self.view.model.routers)), 0.0)
        for r in self.view.model.routers :
            contents, state = self.view.get_state(r)
            state = self.view.encode_state(r)
            old_state.append(state)
            #get valid actions (by caching content and if needed removing other contents)
            #content not present in cache 
            valid_actions = []
            cache_len = self.view.model.cache[r].maxlen
            print ("MAX CACHE LEN FOR ", r, " is ", cache_len)
            print ("CONTENTS", contents, contents.shape[0])
            #append current state too (i.e. take no action)
            valid_actions.append(self.view.encode_action(contents))
            if contents[content - 1] == 0 :
                all_comb = [list(i) for i in itertools.product([0, 1], repeat=contents.shape[0])]
                #print ("ALL COMB", all_comb)
                for m in all_comb:
                    #print ("COMB ", m)
                    if m[content - 1] == 1 and np.sum(m) == cache_len :
                        valid_actions.append(self.view.encode_action(np.array(m)))
                        print ("VALID ACTION", self.view.encode_action(np.array(m)))
            print ("Max action of ", self.view.model.routers.index(r),state,valid_actions)
            max_action = np.argmax(self.view.model.q_table[self.view.model.routers.index(r),state,valid_actions])
            print ("MAX ACTION", valid_actions[max_action]) 
            actions.append(valid_actions[max_action])
            max_action_matrix = self.view.decode_action(max_action)
            #put contents in the cache
            for x in range(max_action_matrix.shape[0]):
                if max_action_matrix[x] == 1:
                    if self.controller.get_content(r, x+1) is False:
                        self.controller.put_content(r, x+1)
                        rewards[self.view.model.routers.index(r)] -= 1
                else:
                    if self.controller.get_content(r, x+1) is True:
                        self.controller.remove_content(r, x+1)
                        rewards[self.view.model.routers.index(r)] -= 1
            self.view.model.popularity *= 0
        
        content_loc = self.view.content_locations(content)
        print ("Locations of content", content_loc)
        min_delay_path = 1000000
        min_path = []
        for c in content_loc :
            path = self.view.shortest_path(receiver, c)
            print ("Path from rx to content location", path)
            #print ("State : ", self.view.get_state())
            # Route requests to original source and queries caches on the path
            current_delay = 0
            print ("Path Links", path_links(path))
            for u, v in path_links(path):
                print ("Link Delay", u, v, self.view.link_delay(u, v))
                current_delay = current_delay + self.view.link_delay(u,v)
                #self.controller.forward_request_hop(u, v)
                #print ("Cache dump of ", v , "is: ", self.view.cache_dump(v))
            if current_delay < min_delay_path:
                min_delay_path = current_delay
                min_path = path
                serving_node = c
        print ("Min Path : ", min_path)
        for u, v in path_links(min_path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v) and v != source:
                print (v, " has cache")
                print ("Routers :", self.view.model.routers) 
                try:
                    print ("Inx", self.view.model.routers.index(v))
                    print ("val", self.view.model.popularity[self.view.model.routers.index(v), content-1])
                    self.view.model.popularity[self.view.model.routers.index(v), content-1] += 1.0
                    print ("val after", self.view.model.popularity[self.view.model.routers.index(v), content-1])
                except:
                    print ("ERROR HERE", v)
        # No caches on the path at all, get it from source
        print ("Serving Node : ", serving_node)
        print ("Total Delay : ", min_delay_path)
        common_reward -= min_delay_path
        self.controller.get_content(serving_node)
        #get maximum of those actions
        #update q_table accordingly
        rewards += common_reward
        #print ("REWARDS MATRIX", rewards)
        for r in range(len(self.view.model.routers)):
            #contents, state = self.view.get_state(self.view.model.routers[r])
            next_state = self.view.encode_state(self.view.model.routers[r])
            self.view.model.q_table[r, old_state[r], actions[r]] = (1.0 - alpha) * self.view.model.q_table[r, old_state[r], actions[r]] + alpha * ((rewards[r] + gamma * np.max(self.view.model.q_table[r, next_state,:]) - self.view.model.q_table[r, old_state[r], actions[r]]))

        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        self.controller.forward_content_path(serving_node, receiver, path)
        #if serving_node == source:
        #    self.controller.put_content(edge_cache)
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
    def process_event(self, time, receiver, content, log):
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
        self.controller.forward_content_path(serving_node, receiver, path)
        if serving_node == source:
            self.controller.put_content(edge_cache)
        self.controller.end_session()


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
    def process_event(self, time, receiver, content, log):
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
            self.controller.forward_content_hop(u, v)
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
    def process_event(self, time, receiver, content, log):
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
            self.controller.forward_content_hop(u, v)
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
    def process_event(self, time, receiver, content, log):
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
            self.controller.forward_content_hop(u, v)
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
    def process_event(self, time, receiver, content, log):
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
            self.controller.forward_content_hop(u, v)
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
    def process_event(self, time, receiver, content, log):
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
            self.controller.forward_content_hop(u, v)
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
    def process_event(self, time, receiver, content, log):
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
            self.controller.forward_content_hop(u, v)
            if v == designated_cache:
                self.controller.put_content(v)
        self.controller.end_session()
