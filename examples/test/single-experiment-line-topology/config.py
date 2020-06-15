"""Configuration file for running a single simple simulation."""
from multiprocessing import cpu_count
from collections import deque
import copy
from icarus.util import Tree

# GENERAL SETTINGS

# Level of logging output
# Available options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL = 'INFO'

# If True, executes simulations in parallel using multiple processes
# to take advantage of multicore CPUs
PARALLEL_EXECUTION = False
PARALLEL_EXECUTION_RUNS = True

# Number of processes used to run simulations in parallel.
# This option is ignored if PARALLEL_EXECUTION = False
N_PROCESSES = cpu_count()

# Setting to overrride experiments if already performed and stored in results
OVERRIDE = True

# Number of times each experiment is replicated
N_REPLICATIONS = 1
WORKLOAD_ITERATIONS = 100

# Granularity of caching.
# Currently, only OBJECT is supported
CACHING_GRANULARITY = 'OBJECT'
MODEL_PATH = 'models'
MODEL_RESUME = True

# Format in which results are saved.
# Result readers and writers are located in module ./icarus/results/readwrite.py
# Currently only PICKLE is supported
RESULTS_FORMAT = 'PICKLE'
# Resume training from already configured file
RESULTS_RESUME = 1
# List of metrics to be measured in the experiments
# The implementation of data collectors are located in ./icarus/execution/collectors.py
DATA_COLLECTORS = ['CACHE_HIT_RATIO', 'LATENCY', 'LINK_LOAD', 'PATH_STRETCH']
#DATA_COLLECTORS = ['CACHE_HIT_RATIO']

# Queue of experiments
EXPERIMENT_QUEUE = deque()

# Create experiment
experiment = Tree()

# Set topology
"""
experiment['topology']['name'] = 'PATH'
experiment['topology']['n'] = 10
experiment['topology']['delay'] = 40
"""

#"""
experiment['topology']['name'] = 'TREE'
experiment['topology']['k'] = 2
experiment['topology']['h'] = 4
experiment['topology']['delay'] = 30
"""
experiment['topology']['name'] = 'ROCKET_FUEL'
experiment['topology']['asn'] = 1221
"""
# Set workload
experiment['workload'] = {
         'name':       'STATIONARY',
         'n_contents': 200,
         'n_warmup':   1 * 10 ** 5,
         'n_measured': 2 * 10 ** 5,
         'alpha':      1.0,
         'rate':       12.0
                       }

# NN Parameters
experiment['nnp']['window'] = 200
experiment['nnp']['lr'] = 0.0015
experiment['nnp']['gamma'] = 0.9
experiment['nnp']['index_threshold_f'] = experiment['topology']['delay']
experiment['nnp']['policy_type'] = 1
experiment['nnp']['state_ver'] = 0
experiment['nnp']['index_threshold_d'] = (experiment['topology']['delay'] / experiment['workload']['n_contents']) * 2
experiment['nnp']['tau'] = 0.95
experiment['nnp']['update_freq'] = 100
experiment['nnp']['beta'] = 0.001
experiment['nnp']['use_gae'] = True
experiment['nnp']['avg_reward_case'] = True

# Set cache placement
experiment['cache_placement']['name'] = 'UNIFORM'
experiment['cache_placement']['network_cache'] = 0.1

# Set content placement
experiment['content_placement']['name'] = 'UNIFORM'

# Set cache replacement policy
experiment['cache_policy']['name'] = 'IN_CACHE_LFU'

# Set caching meta-policy
experiment['strategy']['name'] = 'RL_DEC_1'
#experiment['strategy']['name'] = 'LCE'

# Description of the experiment
experiment['desc'] = "Line topology with 10 nodes"

# Append experiment to queue
EXPERIMENT_QUEUE.append(experiment)