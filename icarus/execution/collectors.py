"""Performance metrics loggers

This module contains all data collectors that record events while simulations
are being executed and compute performance metrics.

Currently implemented data collectors allow users to measure cache hit ratio,
latency, path stretch and link load.

To create a new data collector, it is sufficient to create a new class
inheriting from the `DataCollector` class and override all required methods.
"""
from __future__ import division
import collections

from icarus.registry import register_data_collector
from icarus.tools import cdf
from icarus.util import Tree, inheritdoc


__all__ = [
    'DataCollector',
    'CollectorProxy',
    'CacheHitRatioCollector',
    'LinkLoadCollector',
    'LatencyCollector',
    'PathStretchCollector',
    'DummyCollector'
           ]


class DataCollector(object):
    """Object collecting notifications about simulation events and measuring
    relevant metrics.
    """

    def __init__(self, view, **params):
        """Constructor

        Parameters
        ----------
        view : NetworkView
            An instance of the network view
        params : keyworded parameters
            Collector parameters
        """
        self.view = view

    def start_session(self, timestamp, receiver, content, inx):
        """Notifies the collector that a new network session started.

        A session refers to the retrieval of a content from a receiver, from
        the issuing of a content request to the delivery of the content.

        Parameters
        ----------
        timestamp : int
            The timestamp of the event
        receiver : any hashable type
            The receiver node requesting a content
        content : any hashable type
            The content identifier requested by the receiver
        """
        pass

    def cache_hit(self, node, inx):
        """Reports that the requested content has been served by the cache at
        node *node*.

        Parameters
        ----------
        node : any hashable type
            The node whose cache served the content
        """
        pass

    def cache_miss(self, node, inx):
        """Reports that the cache at node *node* has been looked up for
        requested content but there was a cache miss.

        Parameters
        ----------
        node : any hashable type
            The node whose cache served the content
        """
        pass

    def server_hit(self, node, inx):
        """Reports that the requested content has been served by the server at
        node *node*.

        Parameters
        ----------
        node : any hashable type
            The server node which served the content
        """
        pass

    def request_hop(self, u, v, inx, main_path=True):
        """Reports that a request has traversed the link *(u, v)*

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
        pass

    def content_hop(self, u, v, size, inx, main_path=True):
        """Reports that a content has traversed the link *(u, v)*

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
        pass

    def end_session(self, inx, success=True):
        """Reports that the session is closed, i.e. the content has been
        successfully delivered to the receiver or a failure blocked the
        execution of the request

        Parameters
        ----------
        success : bool, optional
            *True* if the session was completed successfully, *False* otherwise
        """
        pass

    def results(self):
        """Returns the aggregated results measured by the collector.

        Returns
        -------
        results : dict
            Dictionary mapping metric with results.
        """
        pass


# Note: The implementation of CollectorProxy could be improved to avoid having
# to rewrite almost identical methods, for example by playing with __dict__
# attribute. However, it was implemented this way to make it more readable and
# easier to understand.
class CollectorProxy(DataCollector):
    """This class acts as a proxy for all concrete collectors towards the
    network controller.

    An instance of this class registers itself with the network controller and
    it receives notifications for all events. This class is responsible for
    dispatching events of interests to concrete collectors.
    """

    EVENTS = ('start_session', 'end_session', 'cache_hit', 'cache_miss', 'server_hit',
              'request_hop', 'content_hop', 'results')

    def __init__(self, view, collectors):
        """Constructor

        Parameters
        ----------
        view : NetworkView
            An instance of the network view
        collector : list of DataCollector
            List of instances of DataCollector that will be notified of events
        """
        self.view = view
        self.collectors = {e: [c for c in collectors if e in type(c).__dict__]
                           for e in self.EVENTS}

    @inheritdoc(DataCollector)
    def start_session(self, timestamp, receiver, content, inx):
        for c in self.collectors['start_session']:
            c.start_session(timestamp, receiver, content, inx)

    @inheritdoc(DataCollector)
    def cache_hit(self, node, inx):
        for c in self.collectors['cache_hit']:
            c.cache_hit(node, inx)

    @inheritdoc(DataCollector)
    def cache_miss(self, node, inx):
        for c in self.collectors['cache_miss']:
            c.cache_miss(node, inx)

    @inheritdoc(DataCollector)
    def server_hit(self, node, inx):
        for c in self.collectors['server_hit']:
            c.server_hit(node, inx)

    @inheritdoc(DataCollector)
    def request_hop(self, u, v, inx, main_path=True):
        for c in self.collectors['request_hop']:
            c.request_hop(u, v, inx, main_path)

    @inheritdoc(DataCollector)
    def content_hop(self, u, v, size, inx, main_path=True):
        for c in self.collectors['content_hop']:
            c.content_hop(u, v, size, inx, main_path)

    @inheritdoc(DataCollector)
    def end_session(self, inx, success=True):
        for c in self.collectors['end_session']:
            c.end_session(inx, success)

    @inheritdoc(DataCollector)
    def results(self):
        return Tree(**{c.name: c.results() for c in self.collectors['results']})


@register_data_collector('LINK_LOAD')
class LinkLoadCollector(DataCollector):
    """Data collector measuring the link load
    """

    def __init__(self, view, threads, results_dict, n_, routers, edges, req_size=1, content_size=10):
        """Constructor

        Parameters
        ----------
        view : NetworkView
            The network view instance
        req_size : int
            Average size (in bytes) of a request
        content_size : int
            Average size (in byte) of a content
        """
        print (results_dict, n_)
        self.view = view
        self.results_dict = results_dict
        self.n_ = n_
        self.req_count = collections.defaultdict(int)
        self.cont_count = collections.defaultdict(int)
        for i in range(threads):
            self.req_count[i] = collections.defaultdict(int)
            self.cont_count[i] = collections.defaultdict(int)
            for e in edges:
                self.req_count[i][e] = 0
                self.cont_count[i][e] = 0
        if req_size <= 0 or content_size <= 0:
            raise ValueError('req_size and content_size must be positive')
        self.req_size = req_size
        self.content_size = content_size
        self.t_start = -1
        self.t_end = 1

    @inheritdoc(DataCollector)
    def start_session(self, timestamp, receiver, content, inx):
        if self.t_start < 0:
            self.t_start = timestamp
        self.t_end = timestamp

    @inheritdoc(DataCollector)
    def request_hop(self, u, v, inx, main_path=True):
        self.req_count[inx][(u, v)] += 1

    @inheritdoc(DataCollector)
    def content_hop(self, u, v, size, inx, main_path=True):
        #size here instead of count to reflect the actual size used
        self.cont_count[inx][(u, v)] += size

    @inheritdoc(DataCollector)
    def results(self):
        duration = self.t_end - self.t_start
        req_count_keys = set()
        for i in range(len(self.req_count.keys())):
            k = [key.keys() for key in self.req_count.values()][i]
            for link in k:
                req_count_keys.add(link)
        cont_count_keys = set()
        for i in range(len(self.cont_count.keys())):
            k = [key.keys() for key in self.cont_count.values()][i]
            for link in k:
                cont_count_keys.add(link)
        used_links = req_count_keys.union(cont_count_keys)
        #used_links = set(self.req_count.keys()).union(set(self.cont_count.keys()))
        #print ("USED LINKS ", used_links)
        link_loads = {link: (self.req_size * sum(d[link] for d in self.req_count.values() if d) +
                             self.content_size * sum(d[link] for d in self.cont_count.values() if d)) / duration
                      for link in used_links}
        #print ("LINK LOADS ", link_loads)
        link_loads_int = {link: load
                          for link, load in link_loads.items()
                          if self.view.link_type(*link) == 'internal'}
        #print ("LINK LOADS_INT ", link_loads_int)
        link_loads_ext = {link: load
                          for link, load in link_loads.items()
                          if self.view.link_type(*link) == 'external'}
        #print ("LINK LOADS_EXT ", link_loads_int)
        mean_load_int = sum(link_loads_int.values()) / len(link_loads_int) \
                        if len(link_loads_int) > 0 else 0
        mean_load_ext = sum(link_loads_ext.values()) / len(link_loads_ext) \
                        if len(link_loads_ext) > 0 else 0
        
        # Update stats based on average of pervious runs and this one
        if self.results_dict is not None:
            for k in link_loads_int.keys():
                link_loads_int[k] = ((self.results_dict['PER_LINK_INTERNAL'][k] * self.n_) + link_loads_int[k]) / (self.n_ + 1)
            for k in link_loads_ext.keys():
                link_loads_ext[k] = ((self.results_dict['PER_LINK_EXTERNAL'][k] * self.n_) + link_loads_ext[k]) / (self.n_ + 1)
            mean_load_int = ((self.results_dict['MEAN_INTERNAL'] * self.n_) + mean_load_int) / (self.n_ + 1)
            mean_load_ext = ((self.results_dict['MEAN_EXTERNAL'] * self.n_) + mean_load_ext) / (self.n_ + 1)
        
        return Tree({'MEAN_INTERNAL':     mean_load_int,
                     'MEAN_EXTERNAL':     mean_load_ext,
                     'PER_LINK_INTERNAL': link_loads_int,
                     'PER_LINK_EXTERNAL': link_loads_ext})


@register_data_collector('LATENCY')
class LatencyCollector(DataCollector):
    """Data collector measuring latency, i.e. the delay taken to delivery a
    content.
    """

    def __init__(self, view, threads, results_dict, n_, routers, edges, cdf=False):
        """Constructor

        Parameters
        ----------
        view : NetworkView
            The network view instance
        cdf : bool, optional
            If *True*, also collects a cdf of the latency
        """
        self.cdf = cdf
        self.view = view
        self.results_dict = results_dict
        self.n_ = n_
        self.sess_latency = dict()
        self.sess_count = dict()
        for i in range(threads): 
            self.sess_latency[i] = 0.0
            self.sess_count[i] = 0
        self.latency = 0.0
        if cdf:
            self.latency_data = collections.deque()

    @inheritdoc(DataCollector)
    def start_session(self, timestamp, receiver, content, inx):
        self.sess_count[inx] += 1
        self.sess_latency[inx] = 0.0

    @inheritdoc(DataCollector)
    def request_hop(self, u, v, inx, main_path=True):
        if main_path:
            self.sess_latency[inx] += self.view.link_delay(u, v)
            #print ("REQUEST HOP ", u, v, inx, self.sess_latency[inx])

    @inheritdoc(DataCollector)
    def content_hop(self, u, v, size, inx, main_path=True):
        if main_path:
            #Multiply by size of file 
            self.sess_latency[inx] += (self.view.link_delay(u, v))
            #print ("CONTENT HOP ", u, v, inx, self.sess_latency[inx])

    @inheritdoc(DataCollector)
    def end_session(self, inx, success=True):
        if not success:
            return
        if self.cdf:
            self.latency_data.append(self.sess_latency[inx])
        self.latency += self.sess_latency[inx]
        #print ("LATENCY AFTER SESSION ", inx, self.sess_count, self.latency)

    @inheritdoc(DataCollector)
    def results(self):
        print ("Latency Sessions ", sum(self.sess_count.values()))
        latency = self.latency / sum(self.sess_count.values())
        if self.results_dict is not None:
            latency = (latency + (self.results_dict['MEAN'] * self.n_)) / (self.n_ + 1)
        print ("Latency ", latency)
        results = Tree({'MEAN': latency})
        if self.cdf:
            results['CDF'] = cdf(self.latency_data)
        return results


@register_data_collector('CACHE_HIT_RATIO')
class CacheHitRatioCollector(DataCollector):
    """Collector measuring the cache hit ratio, i.e. the portion of content
    requests served by a cache.
    """

    def __init__(self, view, threads, results_dict, n_, routers, edges, off_path_hits=False, per_node=True, content_hits=False):
        """Constructor

        Parameters
        ----------
        view : NetworkView
            The NetworkView instance
        off_path_hits : bool, optional
            If *True* also records cache hits from caches not on located on the
            shortest path. This metric may be relevant only for some strategies
        content_hits : bool, optional
            If *True* also records cache hits per content instead of just
            globally
        """
        self.view = view
        self.n_ = n_
        self.results_dict = results_dict
        self.off_path_hits = off_path_hits
        self.per_node = per_node
        self.cont_hits = content_hits
        self.sess_count = dict()
        self.cache_hits = dict()
        self.serv_hits = dict()
        self.off_path_hit_count = dict()
        if per_node:
            self.per_node_cache_hits = collections.defaultdict(int)
            self.per_node_server_hits = collections.defaultdict(int)
        if content_hits:
            self.curr_cont = dict()
            self.cont_cache_hits = collections.defaultdict(int)
            self.cont_serv_hits = collections.defaultdict(int)
        for i in range(threads):
            self.sess_count[i] = 0
            self.cache_hits[i] = 0
            self.serv_hits[i] = 0
            if off_path_hits:
                self.off_path_hit_count[i] = 0
            if per_node:
                self.per_node_cache_hits[i] = collections.defaultdict(int)
                self.per_node_server_hits[i] = collections.defaultdict(int)
                for r in routers:
                    self.per_node_cache_hits[i][r] = 0 
            if content_hits:
                self.curr_cont[i] = None
                self.cont_cache_hits[i] = collections.defaultdict(int)
                self.cont_serv_hits[i] = collections.defaultdict(int) 

    @inheritdoc(DataCollector)
    def start_session(self, timestamp, receiver, content, inx):
        self.sess_count[inx] += 1
        #print ("SESSION COUNT", self.sess_count)
        if self.off_path_hits:
            source = self.view.content_source(content)
            self.curr_path = self.view.shortest_path(receiver, source)
        if self.cont_hits:
            self.curr_cont[inx] = content

    @inheritdoc(DataCollector)
    def cache_hit(self, node, inx):
        self.cache_hits[inx] += 1
        #print ("CACHE HIT", node, inx)
        #print ("SESS COUNT ", sum(self.sess_count.values()), sum(self.serv_hits.values()), sum(self.cache_hits.values()), inx)
        if self.off_path_hits and node not in self.curr_path:
            self.off_path_hit_count[inx] += 1
        if self.cont_hits:
            self.cont_cache_hits[inx][self.curr_cont[inx]] += 1
        if self.per_node:
            self.per_node_cache_hits[inx][node] += 1
            #print ("Cache Hit !!!! ")

    @inheritdoc(DataCollector)
    def server_hit(self, node, inx):
        #print ("SERVER HIT", node, inx)
        #print ("SESS COUNT ", sum(self.sess_count.values()), sum(self.serv_hits.values()), sum(self.cache_hits.values()), inx)
        self.serv_hits[inx] += 1
        if self.cont_hits:
            self.cont_serv_hits[inx][self.curr_cont[inx]] += 1
        if self.per_node:
            self.per_node_server_hits[inx][node] += 1

    @inheritdoc(DataCollector)
    def results(self):
        #print ("RESULTS BEGIN")
        n_sess = sum(self.cache_hits.values()) + sum(self.serv_hits.values())
        print ("Cache Hit Sessions", n_sess)
        hit_ratio = sum(self.cache_hits.values()) / n_sess
        if self.results_dict is not None:
            hit_ratio = (hit_ratio + (self.results_dict['MEAN'] * self.n_)) / (self.n_ + 1)
        results = Tree(**{'MEAN': hit_ratio})
        if self.off_path_hits:
            mean_off_path = sum(self.off_path_hit_count.values()) / n_sess
            if self.results_dict is not None:
                mean_off_path = (mean_off_path + (self.results_dict['MEAN_OFF_PATH'] * self.n_)) / (self.n_ + 1)
            results['MEAN_OFF_PATH'] = mean_off_path
            results['MEAN_ON_PATH'] = results['MEAN'] - results['MEAN_OFF_PATH']
        if self.cont_hits:
            cont_set_cache = set()
            cont_set_server = set()
            for i in range(len(self.cont_cache_hits.keys())):
                k = [key.keys() for key in self.cont_cache_hits.values()][i]
                for node in k:
                    cont_set_cache.add(node)
            for i in range(len(self.cont_serv_hits.keys())):
                k = [key.keys() for key in self.cont_serv_hits.values()][i]
                for node in k:
                    cont_set_server.add(node)
            #cont_set = set(list(self.cont_cache_hits.keys()) + list(self.cont_serv_hits.keys()))
            cont_set = cont_set_cache.union(cont_set_server)
            cont_hits = {i: (
                                sum(d[i] for d in self.cont_cache_hits.values() if d) /
                                (sum(d[i] for d in self.cont_cache_hits.values() if d) + sum(d[i] for d in self.cont_serv_hits.values() if d))
                            )
                         for i in cont_set}
            if self.results_dict is not None:
                for k in cont_hits.keys():
                    cont_hits[k] = ((self.results_dict['PER_CONTENT'][k] * self.n_) + cont_hits[k]) / (self.n_ + 1)
            results['PER_CONTENT'] = cont_hits
        #print ("Per node cache hit all", self.per_node_cache_hits)
        #print ("Per node server hit all", self.per_node_server_hits)
        if self.per_node:
            per_node_cache = set()
            per_node_server = set()
            for i in range(len(self.per_node_cache_hits.keys())):
                k = [key.keys() for key in self.per_node_cache_hits.values()][i]
                for node in k:
                    per_node_cache.add(node)
            for i in range(len(self.per_node_server_hits.keys())):
                k = [key.keys() for key in self.per_node_server_hits.values()][i]
                for node in k:
                    per_node_server.add(node)
            #print ("Node set", per_node_cache)
            #print ("Server set", per_node_server)
            #for v in self.per_node_cache_hits.values():
            per_node_cache_hit = dict()
            per_node_server_hit = dict()
            for v in per_node_cache:
                per_node_cache_hit[v] = sum(d[v] for d in self.per_node_cache_hits.values() if d) / n_sess
                #self.per_node_cache_hits[v] /= n_sess
            #for v in self.per_node_server_hits.values():
            for v in per_node_server:
                per_node_server_hit[v] = sum(d[v] for d in self.per_node_server_hits.values() if d) / n_sess
                #self.per_node_server_hits[v] /= n_sess
            #print ("Per Node Cache Hit", per_node_cache_hit)
            #print ("Per Node Server Hit", per_node_server_hit)
            if self.results_dict is not None:
                #print ("Inside stats")
                #print (self.results_dict)
                #print (per_node_cache_hit)
                for k in per_node_cache_hit.keys():
                    per_node_cache_hit[k] = ((self.results_dict['PER_NODE_CACHE_HIT_RATIO'][k] * self.n_) + per_node_cache_hit[k]) / (self.n_ + 1)
                for k in per_node_server_hit.keys():
                    per_node_server_hit[k] = ((self.results_dict['PER_NODE_SERVER_HIT_RATIO'][k] * self.n_) + per_node_server_hit[k]) / (self.n_ + 1)
            results['PER_NODE_CACHE_HIT_RATIO'] = per_node_cache_hit
            results['PER_NODE_SERVER_HIT_RATIO'] = per_node_server_hit
        #print ("RESULTS END")
        return results


@register_data_collector('PATH_STRETCH')
class PathStretchCollector(DataCollector):
    """Collector measuring the path stretch, i.e. the ratio between the actual
    path length and the shortest path length.
    """

    def __init__(self, view, threads, results_dict, n_, routers, edges, cdf=False):
        """Constructor

        Parameters
        ----------
        view : NetworkView
            The network view instance
        cdf : bool, optional
            If *True*, also collects a cdf of the path stretch
        """
        self.view = view
        self.cdf = cdf
        self.n_ = n_
        self.results_dict = results_dict
        self.req_path_len = dict()
        self.cont_path_len = dict()
        self.sess_count = dict()
        self.receiver = dict()
        self.source = dict()
        for i in range(threads):
            self.req_path_len[i] = 0
            self.cont_path_len[i] = 0
            self.sess_count[i] = 0
            self.receiver[i] = None
            self.source[i] = None
        self.mean_req_stretch = 0.0
        self.mean_cont_stretch = 0.0
        self.mean_stretch = 0.0
        if self.cdf:
            self.req_stretch_data = collections.deque()
            self.cont_stretch_data = collections.deque()
            self.stretch_data = collections.deque()

    @inheritdoc(DataCollector)
    def start_session(self, timestamp, receiver, content, inx):
        self.receiver[inx] = receiver
        self.source[inx] = self.view.content_source(content)
        self.req_path_len[inx] = 0
        self.cont_path_len[inx] = 0
        self.sess_count[inx] += 1

    @inheritdoc(DataCollector)
    def request_hop(self, u, v, inx, main_path=True):
        self.req_path_len[inx] += 1

    @inheritdoc(DataCollector)
    def content_hop(self, u, v, size, inx,  main_path=True):
        self.cont_path_len[inx] += 1

    @inheritdoc(DataCollector)
    def end_session(self, inx, success=True):
        if not success:
            return
        req_sp_len = len(self.view.shortest_path(self.receiver[inx], self.source[inx]))
        cont_sp_len = len(self.view.shortest_path(self.source[inx], self.receiver[inx]))
        req_stretch = self.req_path_len[inx] / req_sp_len
        cont_stretch = self.cont_path_len[inx] / cont_sp_len
        stretch = (self.req_path_len[inx] + self.cont_path_len[inx]) / (req_sp_len + cont_sp_len)
        self.mean_req_stretch += req_stretch
        self.mean_cont_stretch += cont_stretch
        self.mean_stretch += stretch
        if self.cdf:
            self.req_stretch_data.append(req_stretch)
            self.cont_stretch_data.append(cont_stretch)
            self.stretch_data.append(stretch)

    @inheritdoc(DataCollector)
    def results(self):
        mean_stretch = self.mean_stretch / sum(self.sess_count.values())
        mean_req_stretch = self.mean_req_stretch / sum(self.sess_count.values())
        mean_cont_stretch = self.mean_cont_stretch / sum(self.sess_count.values())
        if self.results_dict is not None:
            mean_stretch = (mean_stretch + (self.results_dict['MEAN'] * self.n_)) / (self.n_ + 1)
            mean_req_stretch = (mean_req_stretch + (self.results_dict['MEAN_REQUEST'] * self.n_)) / (self.n_ + 1)
            mean_cont_stretch = (mean_cont_stretch + (self.results_dict['MEAN_CONTENT'] * self.n_)) / (self.n_ + 1)
        
        results = Tree({'MEAN': mean_stretch, 
                        'MEAN_REQUEST': mean_req_stretch, 
                        'MEAN_CONTENT': mean_cont_stretch}) 
        if self.cdf:
            results['CDF'] = cdf(self.stretch_data)
            results['CDF_REQUEST'] = cdf(self.req_stretch_data)
            results['CDF_CONTENT'] = cdf(self.cont_stretch_data)
        print ("Path Stretch Sessions ", sum(self.sess_count.values()))
        return results


@register_data_collector('DUMMY')
class DummyCollector(DataCollector):
    """Dummy collector to be used for test cases only."""

    def __init__(self, view):
        """Constructor

        Parameters
        ----------
        view : NetworkView
            The network view instance
        output : stream
            Stream on which debug collector writes
        """
        self.view = view

    @inheritdoc(DataCollector)
    def start_session(self, timestamp, receiver, content, inx):
        self.session = dict(timestamp=timestamp, receiver=receiver,
                            content=content, cache_misses=[],
                            request_hops=[], content_hops=[])

    @inheritdoc(DataCollector)
    def cache_hit(self, node, inx):
        self.session['serving_node'] = node

    @inheritdoc(DataCollector)
    def cache_miss(self, node, inx):
        self.session['cache_misses'].append(node)

    @inheritdoc(DataCollector)
    def server_hit(self, node, inx):
        self.session['serving_node'] = node

    @inheritdoc(DataCollector)
    def request_hop(self, u, v, inx, main_path=True):
        self.session['request_hops'].append((u, v))

    @inheritdoc(DataCollector)
    def content_hop(self, u, v, size, inx, main_path=True):
        self.session['content_hops'].append((u, v))

    @inheritdoc(DataCollector)
    def end_session(self, inx, success=True):
        self.session['success'] = success

    def session_summary(self):
        """Return a summary of latest session

        Returns
        -------
        session : dict
            Summary of session
        """
        return self.session
