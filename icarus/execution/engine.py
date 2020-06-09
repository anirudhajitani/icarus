"""This module implements the simulation engine.

The simulation engine, given the parameters according to which a single
experiments needs to be run, instantiates all the required classes and executes
the experiment by iterating through the event provided by an event generator
and providing them to a strategy instance.
"""
from icarus.execution import NetworkModel, NetworkView, NetworkController, CollectorProxy
from icarus.registry import DATA_COLLECTOR, STRATEGY
from pprint import pprint
from itertools import islice, takewhile, repeat
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


def exec_experiment(topology, workload, requests, netconf, strategy, cache_policy, collectors, nnp, n_rep):
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
    view = NetworkView(model, cpus, nnp, strategy_name)
    print ("Network View Done")
    controller = NetworkController(model, cpus)
    print ("Network Controller Done")
    collectors_inst = [DATA_COLLECTOR[name](view, cpus, **params)
                       for name, params in collectors.items()]
    collector = CollectorProxy(view, collectors_inst)
    controller.attach_collector(collector)
    print ("Collector done")
    strategy_args = {k: v for k, v in strategy.items() if k != 'name'}
    strategy_inst = STRATEGY[strategy_name](view, controller, **strategy_args)
    print ("Strategy done")
    #for request in requests:
    #pool = mp.Pool(cpus)
    #manager = mp.Manager()
    #strategy = manager.dict()
    #strategy['key'] =  strategy_inst

    split_every = (lambda n, workload:
                takewhile(bool, (list(islice(workload, n)) for _ in repeat(None))))
    workload_len = sum(1 for _ in iter(workload))
    requests = list(split_every(int(workload_len/cpus), iter(workload)))
    #requests = list(mit.distribute(cpus, iter(workload)))
    callbacks = {"callback": experiment_callback}
    lock = th.Lock()
    barrier = th.Barrier(cpus)
    jobs = []
    print ("BEFORE THREAD CALL")
    #if sys.version_info > (3, 2):
    #    callbacks["error_callback"] = error_callback
    for inx, req in enumerate(requests):
        #pool.apply_async(process_event, args=(req, strategy), callback=experiment_callback)
        print (inx, req)
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
