#!/usr/bin/env python
"""Plot results read from a result set
"""
from __future__ import division
import os
import argparse
import logging

import matplotlib.pyplot as plt

from icarus.util import Settings, config_logging
from icarus.results import plot_lines, plot_bar_chart
from icarus.registry import RESULTS_READER


# Logger object
logger = logging.getLogger('plot')

# These lines prevent insertion of Type 3 fonts in figures
# Publishers don't want them
plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True

# If True text is interpreted as LaTeX, e.g. underscore are interpreted as
# subscript. If False, text is interpreted literally
plt.rcParams['text.usetex'] = False

# Aspect ratio of the output figures
plt.rcParams['figure.figsize'] = 8, 5

# Size of font in legends
LEGEND_SIZE = 14

# Line width in pixels
LINE_WIDTH = 1.5

# Plot
PLOT_EMPTY_GRAPHS = True

# This dict maps strategy names to the style of the line to be used in the plots
# Off-path strategies: solid lines
# On-path strategies: dashed lines
# No-cache: dotted line
STRATEGY_STYLE = {
         'INDEX':           'b-o',
         'INDEX_DIST':      'r-o',
         'RL_DEC_2F':       'y-o',
         'RL_DEC_2D':       'g-o',
         'RL_DEC_1':        'm-o',
         'LCE':             'b-->',
         'LCD':             'g-->',
         'CL4M':            'r-->',
         'PROB_CACHE':      'y-->',
         'RAND_CHOICE':     'k-->',
         'RAND_BERNOULLI':  'c-->',
         'NO_CACHE':        'm-->',
         'OPTIMAL':         'k-o'
                }

# This dict maps name of strategies to names to be displayed in the legend
STRATEGY_LEGEND = {
         'INDEX':           'INDEX',
         'INDEX_DIST':      'INDEX_DIST',
         'RL_DEC_2F':       'RL_2F',
         'RL_DEC_2D':       'RL_2D',
         'RL_DEC_1':        'RL_1',
         'LCE':             'LCE',
         'LCD':             'LCD',
         'CL4M':            'CL4M',
         'PROB_CACHE':      'ProbCache',
         'RAND_CHOICE':     'Random (choice)',
         'RAND_BERNOULLI':  'Random (Bernoulli)',
         'NO_CACHE':        'No caching',
         'OPTIMAL':         'Optimal'
                    }

# Color and hatch styles for bar charts of cache hit ratio and link load vs topology
STRATEGY_BAR_COLOR = {
    'LCE':          'k',
    'LCD':          '0.4',
    'RL_DEC_1':     '0.5',
    'INDEX':        '0.6',
    'INDEX_DIST':   '0.7',
    'RL_DEC_2F':    '0.8',
    'RL_DEC_2D':    '0.9'
    }

STRATEGY_BAR_HATCH = {
    'LCE':          None,
    'LCD':          '//',
    'NO_CACHE':     'x',
    'INDEX':        '+',
    'INDEX_DIST':   '=',
    'RL_DEC_2F':    '*',
    'RL_DEC_2D':    '-',
    'RL_DEC_1':     '#'
    }

INDEX_THRESHOLD_D_LEGEND = {}
INDEX_THRESHOLD_D_STYLE = {}
INDEX_THRESHOLD_D_BAR_COLOR = {}
INDEX_THRESHOLD_D_BAR_HATCH = {}
INDEX_THRESHOLD_F_LEGEND = {}
INDEX_THRESHOLD_F_STYLE = {}
INDEX_THRESHOLD_F_BAR_COLOR = {}
INDEX_THRESHOLD_F_BAR_HATCH = {}
WINDOW_LEGEND = {}
WINDOW_STYLE = {}
WINDOW_BAR_COLOR = {}
WINDOW_BAR_HATCH = {}
LR_LEGEND = {}
LR_STYLE = {}
LR_BAR_COLOR = {}
LR_BAR_HATCH = {}
GAMMA_LEGEND = {}
GAMMA_STYLE = {}
GAMMA_BAR_COLOR = {}
GAMMA_BAR_HATCH = {}
UPDATE_FREQ_LEGEND = {}
UPDATE_FREQ_STYLE = {}
UPDATE_FREQ_BAR_COLOR = {}
UPDATE_FREQ_BAR_HATCH = {}


def plot_cache_hits_vs_alpha(resultset, topology, cache_size, alpha_range, strategies, plotdir, line_style=STRATEGY_STYLE, legend=STRATEGY_LEGEND, hyper_strategy=None, plot_type=None): 
    if 'NO_CACHE' in strategies:
        strategies.remove('NO_CACHE')
    desc = {}
    desc['title'] = 'Cache hit ratio: T=%s C=%s H=%s P=%s' % (topology, cache_size, hyper_strategy, plot_type)
    desc['ylabel'] = 'Cache hit ratio'
    desc['xlabel'] = u'Content distribution \u03b1'
    desc['xparam'] = ('workload', 'alpha')
    desc['xvals'] = alpha_range
    if hyper_strategy is None:
        desc['filter'] = {'topology': {'name': topology},
                        'cache_placement': {'network_cache': cache_size}}
        desc['ycondnames'] = [('strategy', 'name')] * len(strategies)
    else:
        desc['filter'] = {'topology': {'name': topology},
                        'cache_placement': {'network_cache': cache_size},
                        'strategy': {'name': hyper_strategy}}
        desc['ycondnames'] = [('nnp', plot_type)] * len(strategies)
    desc['ymetrics'] = [('CACHE_HIT_RATIO', 'MEAN')] * len(strategies)
    desc['line_style'] = line_style
    desc['legend'] = legend
    desc['ycondvals'] = strategies
    desc['errorbar'] = True
    desc['legend_loc'] = 'upper left'
    desc['plotempty'] = PLOT_EMPTY_GRAPHS
    plot_lines(resultset, desc, 'CACHE_HIT_RATIO_T=%s@C=%s@H=%s@P=%s.pdf'
               % (topology, cache_size, hyper_strategy, plot_type), plotdir)


def plot_cache_hits_vs_cache_size(resultset, topology, alpha, cache_size_range, strategies, plotdir, line_style=STRATEGY_STYLE, legend=STRATEGY_LEGEND, hyper_strategy=None, plot_type=None):
    desc = {}
    if 'NO_CACHE' in strategies:
        strategies.remove('NO_CACHE')
    desc['title'] = 'Cache hit ratio: T=%s A=%s H=%s P=%s' % (topology, alpha, hyper_strategy, plot_type)
    desc['xlabel'] = u'Cache to population ratio'
    desc['ylabel'] = 'Cache hit ratio'
    desc['xscale'] = 'log'
    desc['xparam'] = ('cache_placement', 'network_cache')
    desc['xvals'] = cache_size_range
    if hyper_strategy is None:
        desc['filter'] = {'topology': {'name': topology},
                        'workload': {'name': 'STATIONARY', 'alpha': alpha}}
        desc['ycondnames'] = [('strategy', 'name')] * len(strategies)
    else:
        desc['filter'] = {'topology': {'name': topology},
                         'workload': {'name': 'STATIONARY', 'alpha': alpha},
                        'strategy': {'name': hyper_strategy}}
        desc['ycondnames'] = [('nnp', plot_type)] * len(strategies)
    desc['ymetrics'] = [('CACHE_HIT_RATIO', 'MEAN')] * len(strategies)
    desc['line_style'] = line_style
    desc['legend'] = legend
    desc['ycondvals'] = strategies
    desc['errorbar'] = True
    desc['legend_loc'] = 'upper left'
    desc['plotempty'] = PLOT_EMPTY_GRAPHS
    plot_lines(resultset, desc, 'CACHE_HIT_RATIO_T=%s@A=%s@H=%s@P=%s.pdf'
               % (topology, alpha, hyper_strategy, plot_type), plotdir)


def plot_link_load_vs_alpha(resultset, topology, cache_size, alpha_range, strategies, plotdir, line_style=STRATEGY_STYLE, legend=STRATEGY_LEGEND, hyper_strategy=None, plot_type=None):
    desc = {}
    desc['title'] = 'Internal link load: T=%s C=%s H=%s P=%s' % (topology, cache_size, hyper_strategy, plot_type)
    desc['xlabel'] = u'Content distribution \u03b1'
    desc['ylabel'] = 'Internal link load'
    desc['xparam'] = ('workload', 'alpha')
    desc['xvals'] = alpha_range
    if hyper_strategy is None:
        desc['filter'] = {'topology': {'name': topology},
                        'cache_placement': {'network_cache': cache_size}}
        desc['ycondnames'] = [('strategy', 'name')] * len(strategies)
    else:
        desc['filter'] = {'topology': {'name': topology},
                        'cache_placement': {'network_cache': cache_size},
                        'strategy': {'name': hyper_strategy}}
        desc['ycondnames'] = [('nnp', plot_type)] * len(strategies)
    desc['ymetrics'] = [('LINK_LOAD', 'MEAN_INTERNAL')] * len(strategies)
    desc['line_style'] = line_style
    desc['legend'] = legend
    desc['ycondvals'] = strategies
    desc['errorbar'] = True
    desc['legend_loc'] = 'upper right'
    desc['plotempty'] = PLOT_EMPTY_GRAPHS
    plot_lines(resultset, desc, 'LINK_LOAD_INTERNAL_T=%s@C=%s@H=%s@P=%s.pdf'
               % (topology, cache_size, hyper_strategy, plot_type), plotdir)


def plot_link_load_vs_cache_size(resultset, topology, alpha, cache_size_range, strategies, plotdir, line_style=STRATEGY_STYLE, legend=STRATEGY_LEGEND, hyper_strategy=None, plot_type=None):
    desc = {}
    desc['title'] = 'Internal link load: T=%s A=%s H=%s P=%s' % (topology, alpha, hyper_strategy, plot_type)
    desc['xlabel'] = 'Cache to population ratio'
    desc['ylabel'] = 'Internal link load'
    desc['xscale'] = 'log'
    desc['xparam'] = ('cache_placement', 'network_cache')
    desc['xvals'] = cache_size_range
    if hyper_strategy is None:
        desc['filter'] = {'topology': {'name': topology},
                          'workload': {'name': 'STATIONARY', 'alpha': alpha}}
        desc['ycondnames'] = [('strategy', 'name')] * len(strategies)
    else:
        desc['filter'] = {'topology': {'name': topology},
                        'workload': {'name': 'STATIONARY', 'alpha': alpha},
                        'strategy': {'name': hyper_strategy}}
        desc['ycondnames'] = [('nnp', plot_type)] * len(strategies)
    desc['ymetrics'] = [('LINK_LOAD', 'MEAN_INTERNAL')] * len(strategies)
    desc['line_style'] = line_style
    desc['legend'] = legend
    desc['ycondvals'] = strategies
    desc['errorbar'] = True
    desc['legend_loc'] = 'upper right'
    desc['plotempty'] = PLOT_EMPTY_GRAPHS
    plot_lines(resultset, desc, 'LINK_LOAD_INTERNAL_T=%s@A=%s@H=%s@P=%s.pdf'
               % (topology, alpha, hyper_strategy, plot_type), plotdir)


def plot_latency_vs_alpha(resultset, topology, cache_size, alpha_range, strategies, plotdir, line_style=STRATEGY_STYLE, legend=STRATEGY_LEGEND, hyper_strategy=None, plot_type=None):
    desc = {}
    desc['title'] = 'Latency: T=%s C=%s H=%s P=%s' % (topology, cache_size, hyper_strategy, plot_type)
    desc['xlabel'] = u'Content distribution \u03b1'
    desc['ylabel'] = 'Latency (ms)'
    desc['xparam'] = ('workload', 'alpha')
    desc['xvals'] = alpha_range
    if hyper_strategy is None:
        desc['filter'] = {'topology': {'name': topology},
                        'cache_placement': {'network_cache': cache_size}}
        desc['ycondnames'] = [('strategy', 'name')] * len(strategies)
    else:
        desc['filter'] = {'topology': {'name': topology},
                        'cache_placement': {'network_cache': cache_size},
                        'strategy': {'name': hyper_strategy}}
        desc['ycondnames'] = [('nnp', plot_type)] * len(strategies)
    desc['ymetrics'] = [('LATENCY', 'MEAN')] * len(strategies)
    desc['line_style'] = line_style
    desc['legend'] = legend
    desc['ycondvals'] = strategies
    desc['errorbar'] = True
    desc['legend_loc'] = 'upper right'
    desc['plotempty'] = PLOT_EMPTY_GRAPHS
    plot_lines(resultset, desc, 'LATENCY_T=%s@C=%s@H=%s@P=%s.pdf'
               % (topology, cache_size, hyper_strategy, plot_type), plotdir)


def plot_latency_vs_cache_size(resultset, topology, alpha, cache_size_range, strategies, plotdir, line_style=STRATEGY_STYLE, legend=STRATEGY_LEGEND, hyper_strategy=None, plot_type=None):
    desc = {}
    desc['title'] = 'Latency: T=%s A=%s H=%s P=%s' % (topology, alpha, hyper_strategy, plot_type)
    desc['xlabel'] = 'Cache to population ratio'
    desc['ylabel'] = 'Latency'
    desc['xscale'] = 'log'
    desc['xparam'] = ('cache_placement', 'network_cache')
    desc['xvals'] = cache_size_range
    if hyper_strategy is None:
        desc['filter'] = {'topology': {'name': topology},
                         'workload': {'name': 'STATIONARY', 'alpha': alpha}}
        desc['ycondnames'] = [('strategy', 'name')] * len(strategies)
    else:
        desc['filter'] = {'topology': {'name': topology},
                        'workload': {'name': 'STATIONARY', 'alpha': alpha},
                        'strategy': {'name': hyper_strategy}}
        desc['ycondnames'] = [('nnp', plot_type)] * len(strategies)
    desc['ymetrics'] = [('LATENCY', 'MEAN')] * len(strategies)
    desc['line_style'] = line_style
    desc['legend'] = legend
    desc['ycondvals'] = strategies
    desc['metric'] = ('LATENCY', 'MEAN')
    desc['errorbar'] = True
    desc['legend_loc'] = 'upper right'
    desc['plotempty'] = PLOT_EMPTY_GRAPHS
    plot_lines(resultset, desc, 'LATENCY_T=%s@A=%s@H=%s@P=%s.pdf'
               % (topology, alpha, hyper_strategy, plot_type), plotdir)


def plot_path_stretch_vs_alpha(resultset, topology, cache_size, alpha_range, strategies, plotdir, line_style=STRATEGY_STYLE, legend=STRATEGY_LEGEND, hyper_strategy=None, plot_type=None):
    if 'NO_CACHE' in strategies:
        strategies.remove('NO_CACHE')
    desc = {}
    desc['title'] = 'Path Stretch Mean: T=%s C=%s H=%s P=%s' % (topology, cache_size, hyper_strategy, plot_type)
    desc['ylabel'] = 'Path Stretch Mean'
    desc['xlabel'] = u'Content distribution \u03b1'
    desc['xparam'] = ('workload', 'alpha')
    desc['xvals'] = alpha_range
    if hyper_strategy is None:
        desc['filter'] = {'topology': {'name': topology},
                        'cache_placement': {'network_cache': cache_size}}
        desc['ycondnames'] = [('strategy', 'name')] * len(strategies)
    else:
        desc['filter'] = {'topology': {'name': topology},
                        'cache_placement': {'network_cache': cache_size},
                        'strategy': {'name': hyper_strategy}}
        desc['ycondnames'] = [('nnp', plot_type)] * len(strategies)
    desc['ymetrics'] = [('PATH_STRETCH', 'MEAN')] * len(strategies)
    desc['line_style'] = line_style
    desc['legend'] = legend
    desc['ycondvals'] = strategies
    desc['errorbar'] = True
    desc['legend_loc'] = 'upper left'
    desc['plotempty'] = PLOT_EMPTY_GRAPHS
    plot_lines(resultset, desc, 'PATH_STRETCH_T=%s@C=%s@H=%s@P=%s.pdf'
               % (topology, cache_size, hyper_strategy, plot_type), plotdir)


def plot_path_stretch_vs_cache_size(resultset, topology, alpha, cache_size_range, strategies, plotdir, line_style=STRATEGY_STYLE, legend=STRATEGY_LEGEND, hyper_strategy=None, plot_type=None):
    desc = {}
    if 'NO_CACHE' in strategies:
        strategies.remove('NO_CACHE')
    desc['title'] = 'Path Stretch Mean: T=%s A=%s H=%s P=%s' % (topology, alpha, hyper_strategy, plot_type)
    desc['xlabel'] = u'Cache to population ratio'
    desc['ylabel'] = 'Path Stretch Mean'
    desc['xscale'] = 'log'
    desc['xparam'] = ('cache_placement', 'network_cache')
    desc['xvals'] = cache_size_range
    if hyper_strategy is None:
        desc['filter'] = {'topology': {'name': topology},
                        'workload': {'name': 'STATIONARY', 'alpha': alpha}}
        desc['ycondnames'] = [('strategy', 'name')] * len(strategies)
    else:
        desc['filter'] = {'topology': {'name': topology},
                        'workload': {'name': 'STATIONARY', 'alpha': alpha},
                        'strategy': {'name': hyper_strategy}}
        desc['ycondnames'] = [('nnp', plot_type)] * len(strategies)
    desc['ymetrics'] = [('PATH_STRETCH', 'MEAN')] * len(strategies)
    desc['line_style'] = line_style
    desc['legend'] = legend
    desc['ycondvals'] = strategies
    desc['errorbar'] = True
    desc['legend_loc'] = 'upper left'
    desc['plotempty'] = PLOT_EMPTY_GRAPHS
    plot_lines(resultset, desc, 'PATH_STRETCH_T=%s@A=%s@H=%s@P=%s.pdf'
               % (topology, alpha, hyper_strategy, plot_type), plotdir)

def plot_cache_hits_vs_topology(resultset, alpha, cache_size, topology_range, strategies, plotdir, bar_color=STRATEGY_BAR_COLOR, bar_hatch = STRATEGY_BAR_HATCH, legend = STRATEGY_LEGEND, hyper_strategy=None, plot_type=None):
    """
    Plot bar graphs of cache hit ratio for specific values of alpha and cache
    size for various topologies.

    The objective here is to show that our algorithms works well on all
    topologies considered
    """
    if 'NO_CACHE' in strategies:
        strategies.remove('NO_CACHE')
    desc = {}
    desc['title'] = 'Cache hit ratio: A=%s C=%s H=%s P=%s' % (alpha, cache_size, hyper_strategy, plot_type)
    desc['ylabel'] = 'Cache hit ratio'
    desc['xparam'] = ('topology', 'name')
    desc['xvals'] = topology_range
    if hyper_strategy is None:
        desc['filter'] = {'cache_placement': {'network_cache': cache_size},
                      'workload': {'name': 'STATIONARY', 'alpha': alpha}}
        desc['ycondnames'] = [('strategy', 'name')] * len(strategies)
    else:
        desc['filter'] = {'cache_placement': {'network_cache': cache_size},
                      'workload': {'name': 'STATIONARY', 'alpha': alpha},
                      'strategy': {'name': hyper_strategy}}
        desc['ycondnames'] = [('nnp', plot_type)] * len(strategies)
    desc['ymetrics'] = [('CACHE_HIT_RATIO', 'MEAN')] * len(strategies)
    desc['bar_color'] = bar_color
    desc['bar_hatch'] = bar_hatch
    desc['legend'] = legend
    desc['ycondvals'] = strategies
    desc['errorbar'] = True
    desc['legend_loc'] = 'lower right'
    desc['plotempty'] = PLOT_EMPTY_GRAPHS
    plot_bar_chart(resultset, desc, 'CACHE_HIT_RATIO_A=%s_C=%s_H=%s_plot=%s.pdf'
                   % (alpha, cache_size, hyper_strategy, plot_type), plotdir)


def plot_path_stretch_vs_topology(resultset, alpha, cache_size, topology_range, strategies, plotdir, bar_color=STRATEGY_BAR_COLOR, bar_hatch = STRATEGY_BAR_HATCH, legend = STRATEGY_LEGEND, hyper_strategy=None, plot_type=None):
    """
    Plot bar graphs of path stretch mean for specific values of alpha and cache
    size for various topologies.

    The objective here is to show that our algorithms works well on all
    topologies considered
    """
    if 'NO_CACHE' in strategies:
        strategies.remove('NO_CACHE')
    desc = {}
    desc['title'] = 'Path Stretch Mean: A=%s C=%s H=%s P=%s' % (alpha, cache_size, hyper_strategy, plot_type)
    desc['ylabel'] = 'Path Stretch Mean'
    desc['xparam'] = ('topology', 'name')
    desc['xvals'] = topology_range
    if hyper_strategy is None:
        desc['filter'] = {'cache_placement': {'network_cache': cache_size},
                      'workload': {'name': 'STATIONARY', 'alpha': alpha}}
        desc['ycondnames'] = [('strategy', 'name')] * len(strategies)
    else:
        desc['filter'] = {'cache_placement': {'network_cache': cache_size},
                      'workload': {'name': 'STATIONARY', 'alpha': alpha},
                      'strategy': {'name': hyper_strategy}}
        desc['ycondnames'] = [('nnp', plot_type)] * len(strategies)
    desc['ymetrics'] = [('PATH_STRETCH', 'MEAN')] * len(strategies)
    desc['bar_color'] = bar_color
    desc['bar_hatch'] = bar_hatch
    desc['legend'] = legend
    desc['ycondvals'] = strategies
    desc['errorbar'] = True
    desc['legend_loc'] = 'lower right'
    desc['plotempty'] = PLOT_EMPTY_GRAPHS
    plot_bar_chart(resultset, desc, 'PATH_STRETCH_MEAN_A=%s_C=%s_H=%s_plot=%s.pdf'
                   % (alpha, cache_size, hyper_strategy, plot_type), plotdir)

def plot_link_load_vs_topology(resultset, alpha, cache_size, topology_range, strategies, plotdir, bar_color=STRATEGY_BAR_COLOR, bar_hatch = STRATEGY_BAR_HATCH, legend = STRATEGY_LEGEND, hyper_strategy=None, plot_type=None):
    """
    Plot bar graphs of link load for specific values of alpha and cache
    size for various topologies.

    The objective here is to show that our algorithms works well on all
    topologies considered
    """
    desc = {}
    desc['title'] = 'Internal link load: A=%s C=%s H=%s P=%s' % (alpha, cache_size, hyper_strategy, plot_type)
    desc['ylabel'] = 'Internal link load'
    desc['xparam'] = ('topology', 'name')
    desc['xvals'] = topology_range
    if hyper_strategy is None:
        desc['filter'] = {'cache_placement': {'network_cache': cache_size},
                      'workload': {'name': 'STATIONARY', 'alpha': alpha}}
        desc['ycondnames'] = [('strategy', 'name')] * len(strategies)
    else:
        desc['filter'] = {'cache_placement': {'network_cache': cache_size},
                      'workload': {'name': 'STATIONARY', 'alpha': alpha},
                      'strategy': {'name': hyper_strategy}}
        desc['ycondnames'] = [('nnp', plot_type)] * len(strategies)
    desc['ymetrics'] = [('LINK_LOAD', 'MEAN_INTERNAL')] * len(strategies)
    desc['bar_color'] = bar_color
    desc['bar_hatch'] = bar_hatch
    desc['legend'] = legend
    desc['ycondvals'] = strategies
    desc['errorbar'] = True
    desc['legend_loc'] = 'lower right'
    desc['plotempty'] = PLOT_EMPTY_GRAPHS
    plot_bar_chart(resultset, desc, 'LINK_LOAD_INTERNAL_A=%s_C=%s_H=%s_plot=%s.pdf'
                   % (alpha, cache_size, hyper_strategy, plot_type), plotdir)

def plot_latency_vs_topology(resultset, alpha, cache_size, topology_range, strategies, plotdir, bar_color=STRATEGY_BAR_COLOR, bar_hatch=STRATEGY_BAR_HATCH, legend=STRATEGY_LEGEND, hyper_strategy=None, plot_type=None):
    """
    Plot bar graphs of Latency for specific values of alpha and cache
    size for various topologies.

    The objective here is to show that our algorithms works well on all
    topologies considered
    """
    desc = {}
    desc['title'] = 'Latency: A=%s C=%s H=%s P=%s' % (alpha, cache_size, hyper_strategy, plot_type)
    desc['ylabel'] = 'Latency(ms)'
    desc['xparam'] = ('topology', 'name')
    desc['xvals'] = topology_range
    if hyper_strategy is None:
        desc['filter'] = {'cache_placement': {'network_cache': cache_size},
                      'workload': {'name': 'STATIONARY', 'alpha': alpha}}
        desc['ycondnames'] = [('strategy', 'name')] * len(strategies)
    else:
        desc['filter'] = {'cache_placement': {'network_cache': cache_size},
                      'workload': {'name': 'STATIONARY', 'alpha': alpha},
                      'strategy': {'name': hyper_strategy}}
        desc['ycondnames'] = [('nnp', plot_type)] * len(strategies)
    desc['ymetrics'] = [('LATENCY', 'MEAN')] * len(strategies)
    desc['bar_color'] = bar_color
    desc['bar_hatch'] = bar_hatch
    desc['legend'] = legend
    desc['ycondvals'] = strategies
    desc['errorbar'] = True
    desc['legend_loc'] = 'lower right'
    desc['plotempty'] = PLOT_EMPTY_GRAPHS
    plot_bar_chart(resultset, desc, 'LATENCY_A=%s_C=%s_H=%s_plot=%s.pdf'
                   % (alpha, cache_size, hyper_strategy, plot_type), plotdir)

def run(config, results, plotdir):
    """Run the plot script

    Parameters
    ----------
    config : str
        The path of the configuration file
    results : str
        The file storing the experiment results
    plotdir : str
        The directory into which graphs will be saved
    """
    settings = Settings()
    settings.read_from(config)
    config_logging(settings.LOG_LEVEL)
    resultset = RESULTS_READER[settings.RESULTS_FORMAT](results)
    # Create dir if not existsing
    if not os.path.exists(plotdir):
        os.makedirs(plotdir)
    # Parse params from settings
    topologies = ['TREE']
    cache_sizes = settings.NETWORK_CACHE
    alphas = settings.ALPHA
    strategies = settings.STRATEGIES
    global INDEX_THRESHOLD_D_LEGEND
    global INDEX_THRESHOLD_D_STYLE
    global INDEX_THRESHOLD_D_BAR_COLOR
    global INDEX_THRESHOLD_D_BAR_HATCH
    global INDEX_THRESHOLD_F_LEGEND
    global INDEX_THRESHOLD_F_STYLE
    global INDEX_THRESHOLD_F_BAR_COLOR
    global INDEX_THRESHOLD_F_BAR_HATCH
    global WINDOW_LEGEND
    global WINDOW_STYLE
    global WINDOW_BAR_COLOR
    global WINDOW_BAR_HATCH
    global LR_LEGEND
    global LR_STYLE
    global LR_BAR_COLOR
    global LR_BAR_HATCH
    global GAMMA_LEGEND
    global GAMMA_STYLE
    global GAMMA_BAR_COLOR
    global GAMMA_BAR_HATCH
    global UPDATE_FREQ_LEGEND
    global UPDATE_FREQ_STYLE
    global UPDATE_FREQ_BAR_COLOR
    global UPDATE_FREQ_BAR_HATCH
    
    if settings.HYPERPARAM_RUN == 1:
        windows = settings.WINDOW_SIZE
        lrs = settings.LR
        gammas = settings.GAMMA
        ufs = settings.UPDATE_FREQ
        tfs = settings.THRESHOLD_F
        tds = settings.THRESHOLD_D
        INDEX_THRESHOLD_D_LEGEND = {
            tds[0]: '1',
            tds[1]: '2',
            tds[2]: '3'
            }

        INDEX_THRESHOLD_D_STYLE = {
            tds[0]: 'b-o',
            tds[1]: 'r-o',
            tds[2]: 'y-o'
            }

        INDEX_THRESHOLD_D_BAR_COLOR = {
            tds[0]:   'k',
            tds[1]:   '0.4',
            tds[2]:    '0.5'
            }

        INDEX_THRESHOLD_D_BAR_HATCH = {
            tds[0]:   None,
            tds[1]:   '//',
            tds[2]:    'x'
            }

        INDEX_THRESHOLD_F_LEGEND = {
            tfs[0]: '1',
            tfs[1]: '2',
            tfs[2]: '3'
            }

        INDEX_THRESHOLD_F_STYLE = {
            tfs[0]: 'b-o',
            tfs[1]: 'r-o',
            tfs[2]: 'y-o'
            }

        INDEX_THRESHOLD_F_BAR_COLOR = {
            tfs[0]:   'k',
            tfs[1]:   '0.4',
            tfs[2]:    '0.5'
            }

        INDEX_THRESHOLD_F_BAR_HATCH = {
            tfs[0]:   None,
            tfs[1]:   '//',
            tfs[2]:    'x'
            }

        WINDOW_LEGEND = {
            windows[0]: '1',
            windows[1]: '2',
            windows[2]: '3'
            }

        WINDOW_STYLE = {
            windows[0]: 'b-o',
            windows[1]: 'r-o',
            windows[2]: 'y-o'
            }

        WINDOW_BAR_COLOR = {
            windows[0]:   'k',
            windows[1]:   '0.4',
            windows[2]:    '0.5'
            }

        WINDOW_BAR_HATCH = {
            windows[0]:   None,
            windows[1]:   '//',
            windows[2]:    'x'
            }

        LR_LEGEND = {
            lrs[0]: '1',
            lrs[1]: '2',
            lrs[2]: '3'
            }

        LR_STYLE = {
            lrs[0]: 'b-o',
            lrs[1]: 'r-o',
            lrs[2]: 'y-o'
            }

        LR_BAR_COLOR = {
            lrs[0]:   'k',
            lrs[1]:   '0.4',
            lrs[2]:    '0.5'
            }

        LR_BAR_HATCH = {
            lrs[0]:   None,
            lrs[1]:   '//',
            lrs[2]:    'x'
            }

        GAMMA_LEGEND = {
            gammas[0]: '1',
            gammas[1]: '2',
            gammas[2]: '3'
            }

        GAMMA_STYLE = {
            gammas[0]: 'b-o',
            gammas[1]: 'r-o',
            gammas[2]: 'y-o'
            }

        GAMMA_BAR_COLOR = {
            gammas[0]:   'k',
            gammas[1]:   '0.4',
            gammas[2]:    '0.5'
            }

        GAMMA_BAR_HATCH = {
            gammas[0]:   None,
            gammas[1]:   '//',
            gammas[2]:    'x'
            }

        UPDATE_FREQ_LEGEND = {
            ufs[0]: '1',
            ufs[1]: '2',
            ufs[2]: '3'
            }

        UPDATE_FREQ_STYLE = {
            ufs[0]: 'b-o',
            ufs[1]: 'r-o',
            ufs[2]: 'y-o'
            }

        UPDATE_FREQ_BAR_COLOR = {
            ufs[0]:   'k',
            ufs[1]:   '0.4',
            ufs[2]:    '0.5'
            }

        UPDATE_FREQ_BAR_HATCH = {
            ufs[0]:   None,
            ufs[1]:   '//',
            ufs[2]:    'x'
            }

    # Plot graphs
    for topology in topologies:
        for cache_size in cache_sizes:
            logger.info('Plotting cache hit ratio for topology %s and cache size %s vs alpha' % (topology, str(cache_size)))
            plot_cache_hits_vs_alpha(resultset, topology, cache_size, alphas, strategies, plotdir)
            logger.info('Plotting link load for topology %s vs cache size %s' % (topology, str(cache_size)))
            plot_link_load_vs_alpha(resultset, topology, cache_size, alphas, strategies, plotdir)
            logger.info('Plotting latency for topology %s vs cache size %s' % (topology, str(cache_size)))
            plot_latency_vs_alpha(resultset, topology, cache_size, alphas, strategies, plotdir)
            logger.info('Plotting path stretch for topology %s vs cache size %s' % (topology, str(cache_size)))
            plot_path_stretch_vs_alpha(resultset, topology, cache_size, alphas, strategies, plotdir)
            if settings.HYPERPARAM_RUN == 1:
                if 'INDEX_DIST' in strategies:
                    logger.info('Plotting cache hit ratio for INDEXD threshold and cache size %s vs alpha' % (str(cache_size)))
                    plot_cache_hits_vs_alpha(resultset, topology, cache_size, alphas, tds, plotdir, INDEX_THRESHOLD_D_STYLE, INDEX_THRESHOLD_D_LEGEND, 'INDEX_DIST', 'index_threshold_d')
                    logger.info('Plotting link load for INDEXD threshold and cache size %s vs alpha' % (str(cache_size)))
                    plot_link_load_vs_alpha(resultset, topology, cache_size, alphas, tds, plotdir, INDEX_THRESHOLD_D_STYLE, INDEX_THRESHOLD_D_LEGEND, 'INDEX_DIST', 'index_threshold_d')
                    logger.info('Plotting latency for INDEXD threshold and cache size %s vs alpha' % (str(cache_size)))
                    plot_latency_vs_alpha(resultset, topology, cache_size, alphas, tds, plotdir, INDEX_THRESHOLD_D_STYLE, INDEX_THRESHOLD_D_LEGEND, 'INDEX_DIST', 'index_threshold_d')
                    logger.info('Plotting path stretch for INDEXD threshold and cache size %s vs alpha' % (str(cache_size)))
                    plot_path_stretch_vs_alpha(resultset, topology, cache_size, alphas, tds, plotdir, INDEX_THRESHOLD_D_STYLE, INDEX_THRESHOLD_D_LEGEND, 'INDEX_DIST', 'index_threshold_d')
                if 'INDEX' in strategies:
                    logger.info('Plotting cache hit ratio for INDEX threshold and cache size %s vs alpha' % (str(cache_size)))
                    plot_cache_hits_vs_alpha(resultset, topology, cache_size, alphas, tfs, plotdir, INDEX_THRESHOLD_F_STYLE, INDEX_THRESHOLD_F_LEGEND, 'INDEX', 'index_threshold_f')
                    logger.info('Plotting link load for INDEX threshold and cache size %s vs alpha' % (str(cache_size)))
                    plot_link_load_vs_alpha(resultset, topology, cache_size, alphas, tfs, plotdir, INDEX_THRESHOLD_F_STYLE, INDEX_THRESHOLD_F_LEGEND, 'INDEX', 'index_threshold_f')
                    logger.info('Plotting latency for INDEX threshold and cache size %s vs alpha' % (str(cache_size)))
                    plot_latency_vs_alpha(resultset, topology, cache_size, alphas, tfs, plotdir, INDEX_THRESHOLD_F_STYLE, INDEX_THRESHOLD_F_LEGEND, 'INDEX', 'index_threshold_f')
                    logger.info('Plotting path stretch for INDEX threshold and cache size %s vs alpha' % (str(cache_size)))
                    plot_path_stretch_vs_alpha(resultset, topology, cache_size, alphas, tfs, plotdir, INDEX_THRESHOLD_F_STYLE, INDEX_THRESHOLD_F_LEGEND, 'INDEX', 'index_threshold_f')
                    logger.info('Plotting cache hit ratio for INDEX window and cache size %s vs alpha' % (str(cache_size)))
                    plot_cache_hits_vs_alpha(resultset, topology, cache_size, alphas, windows, plotdir, WINDOW_STYLE, WINDOW_LEGEND, 'INDEX', 'window')
                    logger.info('Plotting link load for INDEX window and cache size %s vs alpha' % (str(cache_size)))
                    plot_link_load_vs_alpha(resultset, topology, cache_size, alphas, windows, plotdir, WINDOW_STYLE, WINDOW_LEGEND, 'INDEX', 'window')
                    logger.info('Plotting latency for INDEX window and cache size %s vs alpha' % (str(cache_size)))
                    plot_latency_vs_alpha(resultset, topology, cache_size, alphas, windows, plotdir, WINDOW_STYLE, WINDOW_LEGEND, 'INDEX', 'window')
                    logger.info('Plotting path stretch for INDEX window and cache size %s vs alpha' % (str(cache_size)))
                    plot_path_stretch_vs_alpha(resultset, topology, cache_size, alphas, windows, plotdir, WINDOW_STYLE, WINDOW_LEGEND, 'INDEX', 'window')
                if 'RL_DEC_1' in strategies:
                    logger.info('Plotting cache hit ratio for RL_DEC_1 LR and cache size %s vs alpha' % (str(cache_size)))
                    plot_cache_hits_vs_alpha(resultset, topology, cache_size, alphas, lrs, plotdir, LR_STYLE, LR_LEGEND, 'RL_DEC_1', 'lr')
                    logger.info('Plotting link load for RL_DEC_1 LR and cache size %s vs alpha' % (str(cache_size)))
                    plot_link_load_vs_alpha(resultset, topology, cache_size, alphas, lrs, plotdir, LR_STYLE, LR_LEGEND, 'RL_DEC_1', 'lr')
                    logger.info('Plotting latency for RL_DEC_1 LR and cache size %s vs alpha' % (str(cache_size)))
                    plot_latency_vs_alpha(resultset, topology, cache_size, alphas, lrs, plotdir, LR_STYLE, LR_LEGEND, 'RL_DEC_1', 'lr')
                    logger.info('Plotting path stretch for RL_DEC_1 LR and cache size %s vs alpha' % (str(cache_size)))
                    plot_path_stretch_vs_alpha(resultset, topology, cache_size, alphas, lrs, plotdir, LR_STYLE, LR_LEGEND, 'RL_DEC_1', 'lr')
                    logger.info('Plotting cache hit ratio for RL_DEC_1 GAMMA and cache size %s vs alpha' % (str(cache_size)))
                    plot_cache_hits_vs_alpha(resultset, topology, cache_size, alphas, gammas, plotdir, GAMMA_STYLE, GAMMA_LEGEND, 'RL_DEC_1', 'gamma')
                    logger.info('Plotting link load for RL_DEC_1 GAMMA and cache size %s vs alpha' % (str(cache_size)))
                    plot_link_load_vs_alpha(resultset, topology, cache_size, alphas, gammas, plotdir, GAMMA_STYLE, GAMMA_LEGEND, 'RL_DEC_1', 'gamma')
                    logger.info('Plotting latency for RL_DEC_1 GAMMA and cache size %s vs alpha' % (str(cache_size)))
                    plot_latency_vs_alpha(resultset, topology, cache_size, alphas, gammas, plotdir, GAMMA_STYLE, GAMMA_LEGEND, 'RL_DEC_1', 'gamma') 
                    logger.info('Plotting path stretch for RL_DEC_1 GAMMA and cache size %s vs alpha' % (str(cache_size)))
                    plot_path_stretch_vs_alpha(resultset, topology, cache_size, alphas, gammas, plotdir, GAMMA_STYLE, GAMMA_LEGEND, 'RL_DEC_1', 'gamma')
                    logger.info('Plotting cache hit ratio for RL_DEC_1 UPDATE_FREQ and cache size %s vs alpha' % (str(cache_size)))
                    plot_cache_hits_vs_alpha(resultset, topology, cache_size, alphas, ufs, plotdir, UPDATE_FREQ_STYLE, UPDATE_FREQ_LEGEND, 'RL_DEC_1', 'update_freq')
                    logger.info('Plotting link load for RL_DEC_1 UPDATE_FREQ and cache size %s vs alpha' % (str(cache_size)))
                    plot_link_load_vs_alpha(resultset, topology, cache_size, alphas, ufs, plotdir, UPDATE_FREQ_STYLE, UPDATE_FREQ_LEGEND, 'RL_DEC_1', 'update_freq')
                    logger.info('Plotting latency for RL_DEC_1 UPDATE_FREQ and cache size %s vs alpha' % (str(cache_size)))
                    plot_latency_vs_alpha(resultset, topology, cache_size, alphas, ufs, plotdir, UPDATE_FREQ_STYLE, UPDATE_FREQ_LEGEND, 'RL_DEC_1', 'update_freq')
                    logger.info('Plotting path stretch for RL_DEC_1 UPDATE_FREQ and cache size %s vs alpha' % (str(cache_size)))
                    plot_path_stretch_vs_alpha(resultset, topology, cache_size, alphas, ufs, plotdir, UPDATE_FREQ_STYLE, UPDATE_FREQ_LEGEND, 'RL_DEC_1', 'update_freq')
                if 'RL_DEC_2D' in strategies:
                    logger.info('Plotting cache hit ratio for RL_DEC_2D LR and cache size %s vs alpha' % (str(cache_size)))
                    plot_cache_hits_vs_alpha(resultset, topology, cache_size, alphas, lrs, plotdir, LR_STYLE, LR_LEGEND, 'RL_DEC_2D', 'lr')
                    logger.info('Plotting link load for RL_DEC_2D LR and cache size %s vs alpha' % (str(cache_size)))
                    plot_link_load_vs_alpha(resultset, topology, cache_size, alphas, lrs, plotdir, LR_STYLE, LR_LEGEND, 'RL_DEC_2D', 'lr')
                    logger.info('Plotting latency for RL_DEC_2D LR and cache size %s vs alpha' % (str(cache_size)))
                    plot_latency_vs_alpha(resultset, topology, cache_size, alphas, lrs, plotdir, LR_STYLE, LR_LEGEND, 'RL_DEC_2D', 'lr')
                    logger.info('Plotting path stretch for RL_DEC_2D LR and cache size %s vs alpha' % (str(cache_size)))
                    plot_path_stretch_vs_alpha(resultset, topology, cache_size, alphas, lrs, plotdir, LR_STYLE, LR_LEGEND, 'RL_DEC_2D', 'lr')
                    logger.info('Plotting cache hit ratio for RL_DEC_2D GAMMA and cache size %s vs alpha' % (str(cache_size)))
                    plot_cache_hits_vs_alpha(resultset, topology, cache_size, alphas, gammas, plotdir, GAMMA_STYLE, GAMMA_LEGEND, 'RL_DEC_2D', 'gamma')
                    logger.info('Plotting link load for RL_DEC_2D GAMMA and cache size %s vs alpha' % (str(cache_size)))
                    plot_link_load_vs_alpha(resultset, topology, cache_size, alphas, gammas, plotdir, GAMMA_STYLE, GAMMA_LEGEND, 'RL_DEC_2D', 'gamma')
                    logger.info('Plotting latency for RL_DEC_2D GAMMA and cache size %s vs alpha' % (str(cache_size)))
                    plot_latency_vs_alpha(resultset, topology, cache_size, alphas, gammas, plotdir, GAMMA_STYLE, GAMMA_LEGEND, 'RL_DEC_2D', 'gamma') 
                    logger.info('Plotting path stretch for RL_DEC_2D GAMMA and cache size %s vs alpha' % (str(cache_size)))
                    plot_path_stretch_vs_alpha(resultset, topology, cache_size, alphas, gammas, plotdir, GAMMA_STYLE, GAMMA_LEGEND, 'RL_DEC_2D', 'gamma')
                    logger.info('Plotting cache hit ratio for RL_DEC_2D window and cache size %s vs alpha' % (str(cache_size)))
                    plot_cache_hits_vs_alpha(resultset, topology, cache_size, alphas, windows, plotdir, WINDOW_STYLE, WINDOW_LEGEND, 'RL_DEC_2D', 'window')
                    logger.info('Plotting link load for RL_DEC_2D window and cache size %s vs alpha' % (str(cache_size)))
                    plot_link_load_vs_alpha(resultset, topology, cache_size, alphas, windows, plotdir, WINDOW_STYLE, WINDOW_LEGEND, 'RL_DEC_2D', 'window')
                    logger.info('Plotting latency for RL_DEC_2D window and cache size %s vs alpha' % (str(cache_size)))
                    plot_latency_vs_alpha(resultset, topology, cache_size, alphas, windows, plotdir, WINDOW_STYLE, WINDOW_LEGEND, 'RL_DEC_2D', 'window')
                    logger.info('Plotting path stretch for RL_DEC_2D window and cache size %s vs alpha' % (str(cache_size)))
                    plot_path_stretch_vs_alpha(resultset, topology, cache_size, alphas, windows, plotdir, WINDOW_STYLE, WINDOW_LEGEND, 'RL_DEC_2D', 'window')
                if 'RL_DEC_2F' in strategies:
                    logger.info('Plotting cache hit ratio for RL_DEC_2F LR and cache size %s vs alpha' % (str(cache_size)))
                    plot_cache_hits_vs_alpha(resultset, topology, cache_size, alphas, lrs, plotdir, LR_STYLE, LR_LEGEND, 'RL_DEC_2F', 'lr')
                    logger.info('Plotting link load for RL_DEC_2F LR and cache size %s vs alpha' % (str(cache_size)))
                    plot_link_load_vs_alpha(resultset, topology, cache_size, alphas, lrs, plotdir, LR_STYLE, LR_LEGEND, 'RL_DEC_2F', 'lr')
                    logger.info('Plotting latency for RL_DEC_2F LR and cache size %s vs alpha' % (str(cache_size)))
                    plot_latency_vs_alpha(resultset, topology, cache_size, alphas, lrs, plotdir, LR_STYLE, LR_LEGEND, 'RL_DEC_2F', 'lr')
                    logger.info('Plotting path stretch for RL_DEC_2F LR and cache size %s vs alpha' % (str(cache_size)))
                    plot_path_stretch_vs_alpha(resultset, topology, cache_size, alphas, lrs, plotdir, LR_STYLE, LR_LEGEND, 'RL_DEC_2F', 'lr')
                    logger.info('Plotting cache hit ratio for RL_DEC_2F GAMMA and cache size %s vs alpha' % (str(cache_size)))
                    plot_cache_hits_vs_alpha(resultset, topology, cache_size, alphas, gammas, plotdir, GAMMA_STYLE, GAMMA_LEGEND, 'RL_DEC_2F', 'gamma')
                    logger.info('Plotting link load for RL_DEC_2F GAMMA and cache size %s vs alpha' % (str(cache_size)))
                    plot_link_load_vs_alpha(resultset, topology, cache_size, alphas, gammas, plotdir, GAMMA_STYLE, GAMMA_LEGEND, 'RL_DEC_2F', 'gamma')
                    logger.info('Plotting latency for RL_DEC_2F GAMMA and cache size %s vs alpha' % (str(cache_size)))
                    plot_latency_vs_alpha(resultset, topology, cache_size, alphas, gammas, plotdir, GAMMA_STYLE, GAMMA_LEGEND, 'RL_DEC_2F', 'gamma') 
                    logger.info('Plotting path stretch for RL_DEC_2F GAMMA and cache size %s vs alpha' % (str(cache_size)))
                    plot_path_stretch_vs_alpha(resultset, topology, cache_size, alphas, gammas, plotdir, GAMMA_STYLE, GAMMA_LEGEND, 'RL_DEC_2F', 'gamma')
                    logger.info('Plotting cache hit ratio for RL_DEC_2F window and cache size %s vs alpha' % (str(cache_size)))
                    plot_cache_hits_vs_alpha(resultset, topology, cache_size, alphas, windows, plotdir, WINDOW_STYLE, WINDOW_LEGEND, 'RL_DEC_2F', 'window')
                    logger.info('Plotting link load for RL_DEC_2F window and cache size %s vs alpha' % (str(cache_size)))
                    plot_link_load_vs_alpha(resultset, topology, cache_size, alphas, windows, plotdir, WINDOW_STYLE, WINDOW_LEGEND, 'RL_DEC_2F', 'window')
                    logger.info('Plotting latency for RL_DEC_2F window and cache size %s vs alpha' % (str(cache_size)))
                    plot_latency_vs_alpha(resultset, topology, cache_size, alphas, windows, plotdir, WINDOW_STYLE, WINDOW_LEGEND, 'RL_DEC_2F', 'window')
                    logger.info('Plotting path stretch for RL_DEC_2F window and cache size %s vs alpha' % (str(cache_size)))
                    plot_path_stretch_vs_alpha(resultset, topology, cache_size, alphas, windows, plotdir, WINDOW_STYLE, WINDOW_LEGEND, 'RL_DEC_2F', 'window')
    
    # Print Statements are not correct (too lazy to change it - DO IT LATER) 
    for topology in topologies:
        for alpha in alphas:
            logger.info('Plotting cache hit ratio for topology %s and alpha %s vs cache size' % (topology, str(alpha)))
            plot_cache_hits_vs_cache_size(resultset, topology, alpha, cache_sizes, strategies, plotdir)
            logger.info('Plotting link load for topology %s and alpha %s vs cache size' % (topology, str(alpha)))
            plot_link_load_vs_cache_size(resultset, topology, alpha, cache_sizes, strategies, plotdir)
            logger.info('Plotting latency for topology %s and alpha %s vs cache size' % (topology, str(alpha)))
            plot_latency_vs_cache_size(resultset, topology, alpha, cache_sizes, strategies, plotdir)
            logger.info('Plotting path stretch for topology %s and alpha %s vs cache size' % (topology, str(alpha)))
            plot_path_stretch_vs_cache_size(resultset, topology, alpha, cache_sizes, strategies, plotdir)
            if settings.HYPERPARAM_RUN == 1:
                if 'INDEX_DIST' in strategies:
                    logger.info('Plotting cache hit ratio for INDEXD threshold and alpha %s vs cache size' % (str(alpha)))
                    plot_cache_hits_vs_cache_size(resultset, topology, alpha, cache_sizes, tds, plotdir, INDEX_THRESHOLD_D_STYLE, INDEX_THRESHOLD_D_LEGEND, 'INDEX_DIST', 'index_threshold_d')
                    logger.info('Plotting link load for INDEXD threshold and alpha %s vs cache size' % (str(alpha)))
                    plot_link_load_vs_cache_size(resultset, topology, alpha, cache_sizes, tds, plotdir, INDEX_THRESHOLD_D_STYLE, INDEX_THRESHOLD_D_LEGEND, 'INDEX_DIST', 'index_threshold_d')
                    logger.info('Plotting latency for INDEXD threshold and alpha %s vs cache size' % (str(alpha)))
                    plot_latency_vs_cache_size(resultset, topology, alpha, cache_sizes, tds, plotdir, INDEX_THRESHOLD_D_STYLE, INDEX_THRESHOLD_D_LEGEND, 'INDEX_DIST', 'index_threshold_d')
                    logger.info('Plotting path stretch for INDEXD threshold and alpha %s vs cache size' % (str(alpha)))
                    plot_path_stretch_vs_cache_size(resultset, topology, alpha, cache_sizes, tds, plotdir, INDEX_THRESHOLD_D_STYLE, INDEX_THRESHOLD_D_LEGEND, 'INDEX_DIST', 'index_threshold_d')
                if 'INDEX' in strategies:
                    logger.info('Plotting cache hit ratio for INDEX threshold and alpha %s vs cache size' % (str(alpha)))
                    plot_cache_hits_vs_cache_size(resultset, topology, alpha, cache_sizes, tfs, plotdir, INDEX_THRESHOLD_F_STYLE, INDEX_THRESHOLD_F_LEGEND, 'INDEX', 'index_threshold_f')
                    #logger.info('Plotting link load for INDEX threshold and cache size %s vs alpha' % (str(cache_size)))
                    plot_link_load_vs_cache_size(resultset, topology, alpha, cache_sizes, tfs, plotdir, INDEX_THRESHOLD_F_STYLE, INDEX_THRESHOLD_F_LEGEND, 'INDEX', 'index_threshold_f')
                    #logger.info('Plotting latency for INDEX threshold and cache size %s vs alpha' % (str(cache_size)))
                    plot_latency_vs_cache_size(resultset, topology, alpha, cache_sizes, tfs, plotdir, INDEX_THRESHOLD_F_STYLE, INDEX_THRESHOLD_F_LEGEND, 'INDEX', 'index_threshold_f')
                    #logger.info('Plotting path stretch for INDEX threshold and cache size %s vs alpha' % (str(cache_size)))
                    plot_path_stretch_vs_cache_size(resultset, topology, alpha, cache_sizes, tfs, plotdir, INDEX_THRESHOLD_F_STYLE, INDEX_THRESHOLD_F_LEGEND, 'INDEX', 'index_threshold_f')
                    #logger.info('Plotting cache hit ratio for INDEX window and cache size %s vs alpha' % (str(cache_size)))
                    plot_cache_hits_vs_cache_size(resultset, topology, alpha, cache_sizes, windows, plotdir, WINDOW_STYLE, WINDOW_LEGEND, 'INDEX', 'window')
                    #logger.info('Plotting link load for INDEX window and cache size %s vs alpha' % (str(cache_size)))
                    plot_link_load_vs_cache_size(resultset, topology, alpha, cache_sizes, windows, plotdir, WINDOW_STYLE, WINDOW_LEGEND, 'INDEX', 'window')
                    #logger.info('Plotting latency for INDEX window and cache size %s vs alpha' % (str(cache_size)))
                    plot_latency_vs_cache_size(resultset, topology, alpha, cache_sizes, windows, plotdir, WINDOW_STYLE, WINDOW_LEGEND, 'INDEX', 'window')
                    #logger.info('Plotting path stretch for INDEX window and cache size %s vs alpha' % (str(cache_size)))
                    plot_path_stretch_vs_cache_size(resultset, topology, alpha, cache_sizes, windows, plotdir, WINDOW_STYLE, WINDOW_LEGEND, 'INDEX', 'window')
                if 'RL_DEC_1' in strategies:
                    #logger.info('Plotting cache hit ratio for RL_DEC_1 LR and cache size %s vs alpha' % (str(cache_size)))
                    plot_cache_hits_vs_cache_size(resultset, topology, alpha, cache_sizes, lrs, plotdir, LR_STYLE, LR_LEGEND, 'RL_DEC_1', 'lr')
                    #logger.info('Plotting link load for RL_DEC_1 LR and cache size %s vs alpha' % (str(cache_size)))
                    plot_link_load_vs_cache_size(resultset, topology, alpha, cache_sizes, lrs, plotdir, LR_STYLE, LR_LEGEND, 'RL_DEC_1', 'lr')
                    #logger.info('Plotting latency for RL_DEC_1 LR and cache size %s vs alpha' % (str(cache_size)))
                    plot_latency_vs_cache_size(resultset, topology, alpha, cache_sizes, lrs, plotdir, LR_STYLE, LR_LEGEND, 'RL_DEC_1', 'lr')
                    #logger.info('Plotting path stretch for RL_DEC_1 LR and cache size %s vs alpha' % (str(cache_size)))
                    plot_path_stretch_vs_cache_size(resultset, topology, alpha, cache_sizes, lrs, plotdir, LR_STYLE, LR_LEGEND, 'RL_DEC_1', 'lr')
                    #logger.info('Plotting cache hit ratio for RL_DEC_1 GAMMA and cache size %s vs alpha' % (str(cache_size)))
                    plot_cache_hits_vs_cache_size(resultset, topology, alpha, cache_sizes, gammas, plotdir, GAMMA_STYLE, GAMMA_LEGEND, 'RL_DEC_1', 'gamma')
                    #logger.info('Plotting link load for RL_DEC_1 GAMMA and cache size %s vs alpha' % (str(cache_size)))
                    plot_link_load_vs_cache_size(resultset, topology, alpha, cache_sizes, gammas, plotdir, GAMMA_STYLE, GAMMA_LEGEND, 'RL_DEC_1', 'gamma')
                    #logger.info('Plotting latency for RL_DEC_1 GAMMA and cache size %s vs alpha' % (str(cache_size)))
                    plot_latency_vs_cache_size(resultset, topology, alpha, cache_sizes, gammas, plotdir, GAMMA_STYLE, GAMMA_LEGEND, 'RL_DEC_1', 'gamma') 
                    #logger.info('Plotting path stretch for RL_DEC_1 GAMMA and cache size %s vs alpha' % (str(cache_size)))
                    plot_path_stretch_vs_cache_size(resultset, topology, alpha, cache_sizes, gammas, plotdir, GAMMA_STYLE, GAMMA_LEGEND, 'RL_DEC_1', 'gamma')
                    #logger.info('Plotting cache hit ratio for RL_DEC_1 UPDATE_FREQ and cache size %s vs alpha' % (str(cache_size)))
                    plot_cache_hits_vs_cache_size(resultset, topology, alpha, cache_sizes, ufs, plotdir, UPDATE_FREQ_STYLE, UPDATE_FREQ_LEGEND, 'RL_DEC_1', 'update_freq')
                    #logger.info('Plotting link load for RL_DEC_1 UPDATE_FREQ and cache size %s vs alpha' % (str(cache_size)))
                    plot_link_load_vs_cache_size(resultset, topology, alpha, cache_sizes, ufs, plotdir, UPDATE_FREQ_STYLE, UPDATE_FREQ_LEGEND, 'RL_DEC_1', 'update_freq')
                    #logger.info('Plotting latency for RL_DEC_1 UPDATE_FREQ and cache size %s vs alpha' % (str(cache_size)))
                    plot_latency_vs_cache_size(resultset, topology, alpha, cache_sizes, ufs, plotdir, UPDATE_FREQ_STYLE, UPDATE_FREQ_LEGEND, 'RL_DEC_1', 'update_freq')
                    #logger.info('Plotting path stretch for RL_DEC_1 UPDATE_FREQ and cache size %s vs alpha' % (str(cache_size)))
                    plot_path_stretch_vs_cache_size(resultset, topology, alpha, cache_sizes, ufs, plotdir, UPDATE_FREQ_STYLE, UPDATE_FREQ_LEGEND, 'RL_DEC_1', 'update_freq')
                if 'RL_DEC_2D' in strategies:
                    #logger.info('Plotting cache hit ratio for RL_DEC_2D LR and cache size %s vs alpha' % (str(cache_size)))
                    plot_cache_hits_vs_cache_size(resultset, topology, alpha, cache_sizes, lrs, plotdir, LR_STYLE, LR_LEGEND, 'RL_DEC_2D', 'lr')
                    #logger.info('Plotting link load for RL_DEC_2D LR and cache size %s vs alpha' % (str(cache_size)))
                    plot_link_load_vs_cache_size(resultset, topology, alpha, cache_sizes, lrs, plotdir, LR_STYLE, LR_LEGEND, 'RL_DEC_2D', 'lr')
                    #logger.info('Plotting latency for RL_DEC_2D LR and cache size %s vs alpha' % (str(cache_size)))
                    plot_latency_vs_cache_size(resultset, topology, alpha, cache_sizes, lrs, plotdir, LR_STYLE, LR_LEGEND, 'RL_DEC_2D', 'lr')
                    #logger.info('Plotting path stretch for RL_DEC_2D LR and cache size %s vs alpha' % (str(cache_size)))
                    plot_path_stretch_vs_cache_size(resultset, topology, alpha, cache_sizes, lrs, plotdir, LR_STYLE, LR_LEGEND, 'RL_DEC_2D', 'lr')
                    #logger.info('Plotting cache hit ratio for RL_DEC_2D GAMMA and cache size %s vs alpha' % (str(cache_size)))
                    plot_cache_hits_vs_cache_size(resultset, topology, alpha, cache_sizes, gammas, plotdir, GAMMA_STYLE, GAMMA_LEGEND, 'RL_DEC_2D', 'gamma')
                    #logger.info('Plotting link load for RL_DEC_2D GAMMA and cache size %s vs alpha' % (str(cache_size)))
                    plot_link_load_vs_cache_size(resultset, topology, alpha, cache_sizes, gammas, plotdir, GAMMA_STYLE, GAMMA_LEGEND, 'RL_DEC_2D', 'gamma')
                    #logger.info('Plotting latency for RL_DEC_2D GAMMA and cache size %s vs alpha' % (str(cache_size)))
                    plot_latency_vs_cache_size(resultset, topology, alpha, cache_sizes, gammas, plotdir, GAMMA_STYLE, GAMMA_LEGEND, 'RL_DEC_2D', 'gamma') 
                    #logger.info('Plotting path stretch for RL_DEC_2D GAMMA and cache size %s vs alpha' % (str(cache_size)))
                    plot_path_stretch_vs_cache_size(resultset, topology, alpha, cache_sizes, gammas, plotdir, GAMMA_STYLE, GAMMA_LEGEND, 'RL_DEC_2D', 'gamma')
                    #logger.info('Plotting cache hit ratio for RL_DEC_2D window and cache size %s vs alpha' % (str(cache_size)))
                    plot_cache_hits_vs_cache_size(resultset, topology, alpha, cache_sizes, windows, plotdir, WINDOW_STYLE, WINDOW_LEGEND, 'RL_DEC_2D', 'window')
                    #logger.info('Plotting link load for RL_DEC_2D window and cache size %s vs alpha' % (str(cache_size)))
                    plot_link_load_vs_cache_size(resultset, topology, alpha, cache_sizes, windows, plotdir, WINDOW_STYLE, WINDOW_LEGEND, 'RL_DEC_2D', 'window')
                    #logger.info('Plotting latency for RL_DEC_2D window and cache size %s vs alpha' % (str(cache_size)))
                    plot_latency_vs_cache_size(resultset, topology, alpha, cache_sizes, windows, plotdir, WINDOW_STYLE, WINDOW_LEGEND, 'RL_DEC_2D', 'window')
                    #logger.info('Plotting path stretch for RL_DEC_2D window and cache size %s vs alpha' % (str(cache_size)))
                    plot_path_stretch_vs_cache_size(resultset, topology, alpha, cache_sizes, windows, plotdir, WINDOW_STYLE, WINDOW_LEGEND, 'RL_DEC_2D', 'window')
                if 'RL_DEC_2F' in strategies:
                    #logger.info('Plotting cache hit ratio for RL_DEC_2F LR and cache size %s vs alpha' % (str(cache_size)))
                    plot_cache_hits_vs_cache_size(resultset, topology, alpha, cache_sizes, lrs, plotdir, LR_STYLE, LR_LEGEND, 'RL_DEC_2F', 'lr')
                    #logger.info('Plotting link load for RL_DEC_2F LR and cache size %s vs alpha' % (str(cache_size)))
                    plot_link_load_vs_cache_size(resultset, topology, alpha, cache_sizes, lrs, plotdir, LR_STYLE, LR_LEGEND, 'RL_DEC_2F', 'lr')
                    #logger.info('Plotting latency for RL_DEC_2F LR and cache size %s vs alpha' % (str(cache_size)))
                    plot_latency_vs_cache_size(resultset, topology, alpha, cache_sizes, lrs, plotdir, LR_STYLE, LR_LEGEND, 'RL_DEC_2F', 'lr')
                    #logger.info('Plotting path stretch for RL_DEC_2F LR and cache size %s vs alpha' % (str(cache_size)))
                    plot_path_stretch_vs_cache_size(resultset, topology, alpha, cache_sizes, lrs, plotdir, LR_STYLE, LR_LEGEND, 'RL_DEC_2F', 'lr')
                    #logger.info('Plotting cache hit ratio for RL_DEC_2F GAMMA and cache size %s vs alpha' % (str(cache_size)))
                    plot_cache_hits_vs_cache_size(resultset, topology, alpha, cache_sizes, gammas, plotdir, GAMMA_STYLE, GAMMA_LEGEND, 'RL_DEC_2F', 'gamma')
                    #logger.info('Plotting link load for RL_DEC_2F GAMMA and cache size %s vs alpha' % (str(cache_size)))
                    plot_link_load_vs_cache_size(resultset, topology, alpha, cache_sizes, gammas, plotdir, GAMMA_STYLE, GAMMA_LEGEND, 'RL_DEC_2F', 'gamma')
                    #logger.info('Plotting latency for RL_DEC_2F GAMMA and cache size %s vs alpha' % (str(cache_size)))
                    plot_latency_vs_cache_size(resultset, topology, alpha, cache_sizes, gammas, plotdir, GAMMA_STYLE, GAMMA_LEGEND, 'RL_DEC_2F', 'gamma') 
                    #logger.info('Plotting path stretch for RL_DEC_2F GAMMA and cache size %s vs alpha' % (str(cache_size)))
                    plot_path_stretch_vs_cache_size(resultset, topology,alpha, cache_sizes, gammas, plotdir, GAMMA_STYLE, GAMMA_LEGEND, 'RL_DEC_2F', 'gamma')
                    #logger.info('Plotting cache hit ratio for RL_DEC_2F window and cache size %s vs alpha' % (str(cache_size)))
                    plot_cache_hits_vs_cache_size(resultset, topology, alpha, cache_sizes, windows, plotdir, WINDOW_STYLE, WINDOW_LEGEND, 'RL_DEC_2F', 'window')
                    #logger.info('Plotting link load for RL_DEC_2F window and cache size %s vs alpha' % (str(cache_size)))
                    plot_link_load_vs_cache_size(resultset, topology, alpha, cache_sizes, windows, plotdir, WINDOW_STYLE, WINDOW_LEGEND, 'RL_DEC_2F', 'window')
                    #logger.info('Plotting latency for RL_DEC_2F window and cache size %s vs alpha' % (str(cache_size)))
                    plot_latency_vs_cache_size(resultset, topology, alpha, cache_sizes, windows, plotdir, WINDOW_STYLE, WINDOW_LEGEND, 'RL_DEC_2F', 'window')
                    #logger.info('Plotting path stretch for RL_DEC_2F window and cache size %s vs alpha' % (str(cache_size)))
                    plot_path_stretch_vs_cache_size(resultset, topology, alpha, cache_sizes, windows, plotdir, WINDOW_STYLE, WINDOW_LEGEND, 'RL_DEC_2F', 'window')
    
    # Print Statements are not correct (too lazy to change it - DO IT LATER)  
    for cache_size in cache_sizes:
        for alpha in alphas:
            logger.info('Plotting cache hit ratio for cache size %s vs alpha %s against topologies' % (str(cache_size), str(alpha)))
            plot_cache_hits_vs_topology(resultset, alpha, cache_size, topologies, strategies, plotdir)
            logger.info('Plotting link load for cache size %s vs alpha %s against topologies' % (str(cache_size), str(alpha)))
            plot_link_load_vs_topology(resultset, alpha, cache_size, topologies, strategies, plotdir)
            logger.info('Plotting latency for cache size %s vs alpha %s against topologies' % (str(cache_size), str(alpha)))
            plot_latency_vs_topology(resultset, alpha, cache_size, topologies, strategies, plotdir)
            logger.info('Plotting path stretch for cache size %s vs alpha %s against topologies' % (str(cache_size), str(alpha)))
            plot_path_stretch_vs_topology(resultset, alpha, cache_size, topologies, strategies, plotdir)
            if settings.HYPERPARAM_RUN == 1:
                if 'INDEX_DIST' in strategies:
                    logger.info('Plotting cache hit ratio for INDEXD threshold cache size %s vs alpha %s against topologies' % (str(cache_size), str(alpha)))
                    plot_cache_hits_vs_topology(resultset, alpha, cache_size, topologies, tds, plotdir, INDEX_THRESHOLD_D_BAR_COLOR, INDEX_THRESHOLD_D_BAR_HATCH, INDEX_THRESHOLD_D_LEGEND, 'INDEX_DIST', 'index_threshold_d')
                    logger.info('Plotting link load for INDEXD threshold cache size %s vs alpha %s against topologies' % (str(cache_size), str(alpha)))
                    plot_link_load_vs_topology(resultset, alpha, cache_size, topologies, tds, plotdir, INDEX_THRESHOLD_D_BAR_COLOR, INDEX_THRESHOLD_D_BAR_HATCH, INDEX_THRESHOLD_D_LEGEND, 'INDEX_DIST', 'index_threshold_d')
                    logger.info('Plotting latency for INDEXD threshold cache size %s vs alpha %s against topologies' % (str(cache_size), str(alpha)))
                    plot_latency_vs_topology(resultset, alpha, cache_size, topologies, tds, plotdir, INDEX_THRESHOLD_D_BAR_COLOR, INDEX_THRESHOLD_D_BAR_HATCH, INDEX_THRESHOLD_D_LEGEND, 'INDEX_DIST', 'index_threshold_d')
                    logger.info('Plotting path stretch for INDEXD threshold cache size %s vs alpha %s against topologies' % (str(cache_size), str(alpha)))
                    plot_path_stretch_vs_topology(resultset, alpha, cache_size, topologies, tds, plotdir, INDEX_THRESHOLD_D_BAR_COLOR, INDEX_THRESHOLD_D_BAR_HATCH, INDEX_THRESHOLD_D_LEGEND, 'INDEX_DIST', 'index_threshold_d')
                if 'INDEX' in strategies:
                    logger.info('Plotting cache hit ratio for INDEX threshold and alpha %s vs cache size' % (str(alpha)))
                    plot_cache_hits_vs_topology(resultset, alpha, cache_size, topologies, tfs, plotdir, INDEX_THRESHOLD_F_BAR_COLOR, INDEX_THRESHOLD_F_BAR_HATCH, INDEX_THRESHOLD_F_LEGEND, 'INDEX', 'index_threshold_f')
                    logger.info('Plotting link load for INDEX threshold and cache size %s vs alpha' % (str(cache_size)))
                    plot_link_load_vs_topology(resultset, alpha, cache_size, topologies, tfs, plotdir, INDEX_THRESHOLD_F_BAR_COLOR, INDEX_THRESHOLD_F_BAR_HATCH, INDEX_THRESHOLD_F_LEGEND,  'INDEX', 'index_threshold_f')
                    logger.info('Plotting latency for INDEX threshold and cache size %s vs alpha' % (str(cache_size)))
                    plot_latency_vs_topology(resultset, alpha, cache_size, topologies, tfs, plotdir, INDEX_THRESHOLD_F_BAR_COLOR, INDEX_THRESHOLD_F_BAR_HATCH, INDEX_THRESHOLD_F_LEGEND,  'INDEX', 'index_threshold_f')
                    logger.info('Plotting path stretch for INDEX threshold and cache size %s vs alpha' % (str(cache_size)))
                    plot_path_stretch_vs_topology(resultset, alpha, cache_size, topologies, tfs, plotdir, INDEX_THRESHOLD_F_BAR_COLOR, INDEX_THRESHOLD_F_BAR_HATCH, INDEX_THRESHOLD_F_LEGEND,  'INDEX', 'index_threshold_f')
                    logger.info('Plotting cache hit ratio for INDEX window and cache size %s vs alpha' % (str(cache_size)))
                    plot_cache_hits_vs_topology(resultset, alpha, cache_size, topologies, windows, plotdir, WINDOW_BAR_COLOR, WINDOW_BAR_HATCH, WINDOW_LEGEND,  'INDEX', 'window')
                    logger.info('Plotting link load for INDEX window and cache size %s vs alpha' % (str(cache_size)))
                    plot_link_load_vs_topology(resultset, alpha, cache_size, topologies, windows, plotdir, WINDOW_BAR_COLOR, WINDOW_BAR_HATCH, WINDOW_LEGEND, 'INDEX', 'window')
                    logger.info('Plotting latency for INDEX window and cache size %s vs alpha' % (str(cache_size)))
                    plot_latency_vs_topology(resultset, alpha, cache_size, topologies, windows, plotdir, WINDOW_BAR_COLOR, WINDOW_BAR_HATCH, WINDOW_LEGEND, 'INDEX', 'window')
                    logger.info('Plotting path stretch for INDEX window and cache size %s vs alpha' % (str(cache_size)))
                    plot_path_stretch_vs_topology(resultset, alpha, cache_size, topologies, windows, plotdir, WINDOW_BAR_COLOR, WINDOW_BAR_HATCH, WINDOW_LEGEND, 'INDEX', 'window')
                if 'RL_DEC_1' in strategies:
                    logger.info('Plotting cache hit ratio for RL_DEC_1 LR and cache size %s vs alpha' % (str(cache_size)))
                    plot_cache_hits_vs_topology(resultset, alpha, cache_size, topologies, lrs, plotdir, LR_BAR_COLOR, LR_BAR_HATCH, LR_LEGEND, 'RL_DEC_1', 'lr')
                    logger.info('Plotting link load for RL_DEC_1 LR and cache size %s vs alpha' % (str(cache_size)))
                    plot_link_load_vs_topology(resultset, alpha, cache_size, topologies, lrs, plotdir, LR_BAR_COLOR, LR_BAR_HATCH, LR_LEGEND, 'RL_DEC_1', 'lr')
                    logger.info('Plotting latency for RL_DEC_1 LR and cache size %s vs alpha' % (str(cache_size)))
                    plot_latency_vs_topology(resultset, alpha, cache_size, topologies, lrs, plotdir, LR_BAR_COLOR, LR_BAR_HATCH, LR_LEGEND, 'RL_DEC_1', 'lr')
                    logger.info('Plotting path stretch for RL_DEC_1 LR and cache size %s vs alpha' % (str(cache_size)))
                    plot_path_stretch_vs_topology(resultset, alpha, cache_size, topologies, lrs, plotdir, LR_BAR_COLOR, LR_BAR_HATCH, LR_LEGEND, 'RL_DEC_1', 'lr')
                    logger.info('Plotting cache hit ratio for RL_DEC_1 GAMMA and cache size %s vs alpha' % (str(cache_size)))
                    plot_cache_hits_vs_topology(resultset, alpha, cache_size, topologies, gammas, plotdir, GAMMA_BAR_COLOR, GAMMA_BAR_HATCH, GAMMA_LEGEND, 'RL_DEC_1', 'gamma')
                    logger.info('Plotting link load for RL_DEC_1 GAMMA and cache size %s vs alpha' % (str(cache_size)))
                    plot_link_load_vs_topology(resultset, alpha, cache_size, topologies, gammas, plotdir, GAMMA_BAR_COLOR, GAMMA_BAR_HATCH, GAMMA_LEGEND, 'RL_DEC_1', 'gamma')
                    logger.info('Plotting latency for RL_DEC_1 GAMMA and cache size %s vs alpha' % (str(cache_size)))
                    plot_latency_vs_topology(resultset, alpha, cache_size, topologies, gammas, plotdir, GAMMA_BAR_COLOR, GAMMA_BAR_HATCH, GAMMA_LEGEND, 'RL_DEC_1', 'gamma') 
                    logger.info('Plotting path stretch for RL_DEC_1 GAMMA and cache size %s vs alpha' % (str(cache_size)))
                    plot_path_stretch_vs_topology(resultset, alpha, cache_size, topologies, gammas, plotdir, GAMMA_BAR_COLOR, GAMMA_BAR_HATCH, GAMMA_LEGEND, 'RL_DEC_1', 'gamma')
                    logger.info('Plotting cache hit ratio for RL_DEC_1 UPDATE_FREQ and cache size %s vs alpha' % (str(cache_size)))
                    plot_cache_hits_vs_topology(resultset, alpha, cache_size, topologies, ufs, plotdir, UPDATE_FREQ_BAR_COLOR, UPDATE_FREQ_BAR_HATCH, UPDATE_FREQ_LEGEND, 'RL_DEC_1', 'update_freq')
                    logger.info('Plotting link load for RL_DEC_1 UPDATE_FREQ and cache size %s vs alpha' % (str(cache_size)))
                    plot_link_load_vs_topology(resultset, alpha, cache_size, topologies, ufs, plotdir, UPDATE_FREQ_BAR_COLOR, UPDATE_FREQ_BAR_HATCH, UPDATE_FREQ_LEGEND, 'RL_DEC_1', 'update_freq')
                    logger.info('Plotting latency for RL_DEC_1 UPDATE_FREQ and cache size %s vs alpha' % (str(cache_size)))
                    plot_latency_vs_topology(resultset, alpha, cache_size, topologies, ufs, plotdir,UPDATE_FREQ_BAR_COLOR,UPDATE_FREQ_BAR_HATCH,UPDATE_FREQ_LEGEND, 'RL_DEC_1', 'update_freq')
                    logger.info('Plotting path stretch for RL_DEC_1 UPDATE_FREQ and cache size %s vs alpha' % (str(cache_size)))
                    plot_path_stretch_vs_topology(resultset, alpha, cache_size, topologies, ufs, plotdir,UPDATE_FREQ_BAR_COLOR, UPDATE_FREQ_BAR_HATCH, UPDATE_FREQ_LEGEND, 'RL_DEC_1', 'update_freq')
                if 'RL_DEC_2D' in strategies:
                    logger.info('Plotting cache hit ratio for RL_DEC_2D LR and cache size %s vs alpha' % (str(cache_size)))
                    plot_cache_hits_vs_topology(resultset, alpha, cache_size, topologies, lrs, plotdir, LR_BAR_COLOR, LR_BAR_HATCH, LR_LEGEND, 'RL_DEC_2D', 'lr')
                    logger.info('Plotting link load for RL_DEC_2D LR and cache size %s vs alpha' % (str(cache_size)))
                    plot_link_load_vs_topology(resultset, alpha, cache_size, topologies, lrs, plotdir, LR_BAR_COLOR, LR_BAR_HATCH, LR_LEGEND, 'RL_DEC_2D', 'lr')
                    logger.info('Plotting latency for RL_DEC_2D LR and cache size %s vs alpha' % (str(cache_size)))
                    plot_latency_vs_topology(resultset, alpha, cache_size, topologies, lrs, plotdir, LR_BAR_COLOR, LR_BAR_HATCH, LR_LEGEND, 'RL_DEC_2D', 'lr')
                    logger.info('Plotting path stretch for RL_DEC_2D LR and cache size %s vs alpha' % (str(cache_size)))
                    plot_path_stretch_vs_topology(resultset, alpha, cache_size, topologies, lrs, plotdir, LR_BAR_COLOR, LR_BAR_HATCH, LR_LEGEND, 'RL_DEC_2D', 'lr')
                    logger.info('Plotting cache hit ratio for RL_DEC_2D GAMMA and cache size %s vs alpha' % (str(cache_size)))
                    plot_cache_hits_vs_topology(resultset, alpha, cache_size, topologies, gammas, plotdir, GAMMA_BAR_COLOR, GAMMA_BAR_HATCH,  GAMMA_LEGEND, 'RL_DEC_2D', 'gamma')
                    logger.info('Plotting link load for RL_DEC_2D GAMMA and cache size %s vs alpha' % (str(cache_size)))
                    plot_link_load_vs_topology(resultset, alpha, cache_size, topologies, gammas, plotdir, GAMMA_BAR_COLOR, GAMMA_BAR_HATCH, GAMMA_LEGEND, 'RL_DEC_2D', 'gamma')
                    logger.info('Plotting latency for RL_DEC_2D GAMMA and cache size %s vs alpha' % (str(cache_size)))
                    plot_latency_vs_topology(resultset, alpha, cache_size, topologies, gammas, plotdir, GAMMA_BAR_COLOR, GAMMA_BAR_HATCH, GAMMA_LEGEND, 'RL_DEC_2D', 'gamma') 
                    logger.info('Plotting path stretch for RL_DEC_2D GAMMA and cache size %s vs alpha' % (str(cache_size)))
                    plot_path_stretch_vs_topology(resultset, alpha, cache_size, topologies, gammas, plotdir, GAMMA_BAR_COLOR, GAMMA_BAR_HATCH, GAMMA_LEGEND, 'RL_DEC_2D', 'gamma')
                    logger.info('Plotting cache hit ratio for RL_DEC_2D window and cache size %s vs alpha' % (str(cache_size)))
                    plot_cache_hits_vs_topology(resultset, alpha, cache_size, topologies, windows, plotdir, WINDOW_BAR_COLOR, WINDOW_BAR_HATCH, WINDOW_LEGEND, 'RL_DEC_2D', 'window')
                    logger.info('Plotting link load for RL_DEC_2D window and cache size %s vs alpha' % (str(cache_size)))
                    plot_link_load_vs_topology(resultset, alpha, cache_size, topologies, windows, plotdir, WINDOW_BAR_COLOR, WINDOW_BAR_HATCH, WINDOW_LEGEND, 'RL_DEC_2D', 'window')
                    logger.info('Plotting latency for RL_DEC_2D window and cache size %s vs alpha' % (str(cache_size)))
                    plot_latency_vs_topology(resultset, alpha, cache_size, topologies, windows, plotdir, WINDOW_BAR_COLOR, WINDOW_BAR_HATCH, WINDOW_LEGEND, 'RL_DEC_2D', 'window')
                    logger.info('Plotting path stretch for RL_DEC_2D window and cache size %s vs alpha' % (str(cache_size)))
                    plot_path_stretch_vs_topology(resultset, alpha, cache_size, topologies, windows, plotdir, WINDOW_BAR_COLOR, WINDOW_BAR_HATCH, WINDOW_LEGEND, 'RL_DEC_2D', 'window')
                if 'RL_DEC_2F' in strategies:
                    logger.info('Plotting cache hit ratio for RL_DEC_2F LR and cache size %s vs alpha' % (str(cache_size)))
                    plot_cache_hits_vs_topology(resultset, alpha, cache_size, topologies, lrs, plotdir, LR_BAR_COLOR, LR_BAR_HATCH, LR_LEGEND, 'RL_DEC_2F', 'lr')
                    logger.info('Plotting link load for RL_DEC_2F LR and cache size %s vs alpha' % (str(cache_size)))
                    plot_link_load_vs_topology(resultset, alpha, cache_size, topologies, lrs, plotdir, LR_BAR_COLOR, LR_BAR_HATCH, LR_LEGEND, 'RL_DEC_2F', 'lr')
                    logger.info('Plotting latency for RL_DEC_2F LR and cache size %s vs alpha' % (str(cache_size)))
                    plot_latency_vs_topology(resultset, alpha, cache_size, topologies, lrs, plotdir, LR_BAR_COLOR, LR_BAR_HATCH, LR_LEGEND, 'RL_DEC_2F', 'lr')
                    logger.info('Plotting path stretch for RL_DEC_2F LR and cache size %s vs alpha' % (str(cache_size)))
                    plot_path_stretch_vs_topology(resultset, alpha, cache_size, topologies, lrs, plotdir, LR_BAR_COLOR, LR_BAR_HATCH, LR_LEGEND, 'RL_DEC_2F', 'lr')
                    logger.info('Plotting cache hit ratio for RL_DEC_2F GAMMA and cache size %s vs alpha' % (str(cache_size)))
                    plot_cache_hits_vs_topology(resultset, alpha, cache_size, topologies, gammas, plotdir, GAMMA_BAR_COLOR, GAMMA_BAR_HATCH, GAMMA_LEGEND, 'RL_DEC_2F', 'gamma')
                    logger.info('Plotting link load for RL_DEC_2F GAMMA and cache size %s vs alpha' % (str(cache_size)))
                    plot_link_load_vs_topology(resultset, alpha, cache_size, topologies, gammas, plotdir, GAMMA_BAR_COLOR, GAMMA_BAR_HATCH, GAMMA_LEGEND, 'RL_DEC_2F', 'gamma')
                    logger.info('Plotting latency for RL_DEC_2F GAMMA and cache size %s vs alpha' % (str(cache_size)))
                    plot_latency_vs_topology(resultset, alpha, cache_size, topologies, gammas, plotdir, GAMMA_BAR_COLOR, GAMMA_BAR_HATCH, GAMMA_LEGEND, 'RL_DEC_2F', 'gamma') 
                    logger.info('Plotting path stretch for RL_DEC_2F GAMMA and cache size %s vs alpha' % (str(cache_size)))
                    plot_path_stretch_vs_topology(resultset, alpha, cache_size, topologies, gammas, plotdir, GAMMA_BAR_COLOR, GAMMA_BAR_HATCH, GAMMA_LEGEND, 'RL_DEC_2F', 'gamma')
                    logger.info('Plotting cache hit ratio for RL_DEC_2F window and cache size %s vs alpha' % (str(cache_size)))
                    plot_cache_hits_vs_topology(resultset, alpha, cache_size, topologies, windows, plotdir, WINDOW_BAR_COLOR, WINDOW_BAR_HATCH,  WINDOW_LEGEND, 'RL_DEC_2F', 'window')
                    logger.info('Plotting link load for RL_DEC_2F window and cache size %s vs alpha' % (str(cache_size)))
                    plot_link_load_vs_topology(resultset, alpha, cache_size, topologies, windows, plotdir, WINDOW_BAR_COLOR, WINDOW_BAR_HATCH,  WINDOW_LEGEND, 'RL_DEC_2F', 'window')
                    logger.info('Plotting latency for RL_DEC_2F window and cache size %s vs alpha' % (str(cache_size)))
                    plot_latency_vs_topology(resultset, alpha, cache_size, topologies, windows, plotdir, WINDOW_BAR_COLOR, WINDOW_BAR_HATCH,  WINDOW_LEGEND, 'RL_DEC_2F', 'window')
                    logger.info('Plotting path stretch for RL_DEC_2F window and cache size %s vs alpha' % (str(cache_size)))
                    plot_path_stretch_vs_topology(resultset, alpha, cache_size, topologies, windows, plotdir, WINDOW_BAR_COLOR, WINDOW_BAR_HATCH,  WINDOW_LEGEND, 'RL_DEC_2F', 'window')
    logger.info('Exit. Plots were saved in directory %s' % os.path.abspath(plotdir))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("-r", "--results", dest="results",
                        help='the results file',
                        required=True)
    parser.add_argument("-o", "--output", dest="output",
                        help='the output directory where plots will be saved',
                        required=True)
    parser.add_argument("config",
                        help="the configuration file")
    args = parser.parse_args()
    run(args.config, args.results, args.output)


if __name__ == '__main__':
    main()
