"""This module implements the simulation engine.

The simulation engine, given the parameters according to which a single
experiments needs to be run, instantiates all the required classes and executes
the experiment by iterating through the event provided by an event generator
and providing them to a strategy instance.
"""
from icarus.execution import NetworkModel, NetworkView, NetworkController, CollectorProxy
from icarus.registry import DATA_COLLECTOR, STRATEGY, WORKLOAD
from pprint import pprint
from itertools import islice, takewhile, repeat
from icarus.util import Tree
import more_itertools as mit
import torch.multiprocessing as mp
import threading as th
import sys

__all__ = ['exec_experiment']


def experiment_callback(args):
    print ("Run Done!!!!!!!")

def error_callback(args):
    print ("ERROR!!!!!")

def process_event(lock, barrier, requests, strategy, inx):
    #print ("REQUEST PROCESS", requests)
    for i, req in enumerate(requests):
        #print ("Time, Event ", req[0], req[1])
        #print ("IDDDD", id(strategy))
        strategy.process_event(req[0], lock, barrier, inx, i+1, **req[1])

def generate_workload(topology, workload_name, workload_spec):
    if workload_name not in WORKLOAD:
        logger.error('No workload implementation named %s was found.'
                        % workload_name)
        return None
    workload_obj = WORKLOAD[workload_name](topology, **workload_spec)
    return workload_obj
    

def exec_experiment(topology, workload, workload_name, workload_spec, workload_iterations, requests, 
        netconf, strategy, cache_policy, collectors, nnp, n_rep, model_path, model_resume, collectors_result_dict):
    """Execute the simulation of a specific scenario.

    Parameters
    ----------
    topology : Topology
        The FNSS Topology object modelling the network topology on which
        experiments are run.
    workload : iterable
        An iterable object whose elements are (time, event) tuples, where time
        is a float type indicating the timestamp of the event to be executed
        and event is a dictionary storing all the attributes of the event to
        execute
    netconf : dict
        Dictionary of attributes to inizialize the network model
    strategy : tree
        Strategy definition. It is tree describing the name of the strategy
        to use and a list of initialization attributes
    cache_policy : tree
        Cache policy definition. It is tree describing the name of the cache
        policy to use and a list of initialization attributes
    collectors: dict
        The collectors to be used. It is a dictionary in which keys are the
        names of collectors to use and values are dictionaries of attributes
        for the collector they refer to.

    Returns
    -------
    results : Tree
        A tree with the aggregated simulation results from all collectors
    """
    cpus = mp.cpu_count()
    print ("CPUS = ", cpus, " N_rep = ", n_rep)
    cpus = int(cpus/n_rep)
    #pprint(vars(topology))
    strategy_name = strategy['name']
    model = NetworkModel(topology, workload, cache_policy, **netconf)
    agents = len(model.routers)
    if cpus > agents:
        cpus = agents
    view = NetworkView(model, cpus, nnp, strategy_name, model_path, model_resume)
    print ("Network View Done")
    controller = NetworkController(model, cpus)
    print ("Network Controller Done")
    view_counts = view.count
    total_req_iter = workload.n_warmup + workload.n_measured
    total_req = (workload.n_warmup + workload.n_measured) * workload_iterations
    resume_valid = 0
    print ("View Counts ", view.count)
    print ("Total Requests ", total_req)
    if collectors_result_dict is not None:
        if 'CACHE_HIT_RATIO' in collectors_result_dict:
            cache_hit_ratio = collectors_result_dict['CACHE_HIT_RATIO']
            print (cache_hit_ratio)
        if 'PATH_STRETCH' in collectors_result_dict:
            path_stretch = collectors_result_dict['PATH_STRETCH']
            print (path_stretch)
        if 'LATENCY' in collectors_result_dict:
            latency = collectors_result_dict['LATENCY']
            print (latency)
        if 'LINK_LOAD' in collectors_result_dict:
            link_load = collectors_result_dict['LINK_LOAD']
            print (link_load)
        collectors_inst = [DATA_COLLECTOR[name](view, cpus, collectors_result_dict[name], int(view_counts/total_req), model.routers, model.edges, **params)
                        for name, params in collectors.items()]
        if int(view_counts/total_req) > 0:
            # Check this condition for for setting up warm up requests
            resume_valid = 1
            print ("VALID RESUME")
    else:     
        collectors_inst = [DATA_COLLECTOR[name](view, cpus, collectors_result_dict, 0, model.routers, model.edges, **params)
                        for name, params in collectors.items()]
    collector = CollectorProxy(view, collectors_inst)
    controller.attach_collector(collector)
    print ("Collector done")
    strategy_args = {k: v for k, v in strategy.items() if k != 'name'}
    strategy_inst = STRATEGY[strategy_name](view, controller, **strategy_args)
    print ("Strategy done")
    #for request in requests:
    workload_spec = workload_spec.dict()
    workload_spec['n_warmup'] = 0
    workload_spec['n_measured'] = total_req_iter
    workload_spec = Tree(**{k: v for k,v in workload_spec.items()})
    #pool = mp.Pool(cpus)
    #manager = mp.Manager()
    #strategy = manager.dict()
    #strategy['key'] =  strategy_inst
    """
    Problem with this approach is that it just splits it, we need to split ar distance of n
    split_every = (lambda n, workload:
                #takewhile(bool, (list(islice(workload, n)) for _ in repeat(None))))
    workload_len = sum(1 for _ in iter(workload))
    requests = list(split_every(int(workload_len/cpus), iter(workload)))
    """
    print ("WORKLOAD_SPEC", workload_spec, type(workload_spec))
    for it in range(workload_iterations):
        # in case of first request from resumed experiment, we dont want training workloads, all should be evaluated
        if it != 0 or resume_valid == 1:
            print ("Generate workload with no training, all test")
            # From next time we dont want to train the model, diretly use for testing
            workload = generate_workload(topology, workload_name, workload_spec)
        requests = list(mit.distribute(cpus, iter(workload)))
        list_req = []
        for r in iter(requests):
            list_req.append(list(r))
        del requests
        callbacks = {"callback": experiment_callback}
        lock = th.Lock()
        barrier = th.Barrier(cpus)
        jobs = []
        print ("BEFORE THREAD CALL")
        #if sys.version_info > (3, 2):
        #    callbacks["error_callback"] = error_callback
        for inx, req in enumerate(list_req):
            #pool.apply_async(process_event, args=(req, strategy), callback=experiment_callback)
            #print (it, inx, req)
            t = th.Thread(target=process_event, args=(lock, barrier, req, strategy_inst, inx)) 
            jobs.append(t)
        for j in jobs:
            j.start()
        for j in jobs:
            j.join()
    #pool.close()
    #pool.join()
    #time.sleep(60)
    print (collector.results()) 
    return collector.results()
