#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 22:59:47 2021

@author: roberttoth

Based on code by Justin Kiggins, available at:
https://justinkiggins.com/blog/generating-channel-graphs \
-for-klustakwik-automatically # noqa

"""

import numpy as np

import os.path as op

from scipy import spatial
from scipy.spatial.qhull import QhullError

from matplotlib import pyplot as plt
from copy import deepcopy


# Generates a klustakwik .prb file
def write_probe_file(path, prb, channels_used):
    
    dead_channels = list(set(prb['site_map'].values()) - set(channels_used))
    
    channel_groups = probe_data(prb, dead_channels)
    
    # Construct system independent path
    path = op.join(op.abspath(path), 'probe.prb')
    with open(path, 'w') as f:
        print('channel_groups = ', end = '', file = f)
        print(pretty(channel_groups), file = f)
    
    return channel_groups


def probe_data(prb, dead_channels):
    
    s = deepcopy(prb['site_map'])
    
    inv_s = {v: k for k, v in s.items()}
    
    dead_sites = [inv_s.get(key) for key in dead_channels]
    
    for dead_site in dead_sites:
        s[dead_site] = None
    
    channel_groups = {
        # Shank index.
        0: {
            # List of channels to keep for spike detection.
            'channels' : list(s.values()),
            
            # 2D positions of the channels
            # channel: (x,y)
            'geometry' : dict(zip(s.values(),
                              [prb['site_xy'].get(key) for key in s.keys()]))
        }
    }
    
    new_groups = {}
    for gr, group in channel_groups.items():
        new_groups[gr] = {
            'channels': [],
            'geometry': {}
        }
        new_groups[gr]['channels'] = [ch for ch in group['channels']
                                      if ch is not None]
        new_groups[gr]['geometry'] = {ch:xy for (ch,xy)
                                      in group['geometry'].items()
                                      if ch is not None}
    
    if (len(prb['site_map']) - len(dead_channels)) >= 2:
      channel_groups = build_geometries(new_groups)
    else:
      channel_groups = new_groups
      ch = channel_groups[0]['channels'][0]
      channel_groups[0]['graph'] = [(ch, ch)]
    
    return channel_groups


def get_graph_from_geometry(geometry):
    
    # let's transform the geometry into lists of channel names and coordinates
    chans,coords = zip(*[(ch,xy) for ch,xy in geometry.items()])
    
    # we'll perform the triangulation and extract the triangle elements
    try:
        tri = spatial.Delaunay(coords)
    except QhullError:
        # oh no! we probably have a linear geometry.
        chans,coords = list(chans),list(coords)
        x,y = zip(*coords)
        # let's add a dummy channel and try again
        coords.append((max(x)+1,max(y)+1))
        tri = spatial.Delaunay(coords)
    
    # Alternatively:
    # the use of 4 extra points helps deal with linear geometries,
    # and removes 'thin' triangles occasionally created along outer
    # edges of the structure. While 3 extra points of a bounding
    # triangle would be sufficient, their positions are more 
    # difficult to compute
    # chans,coords = list(chans),list(coords)
    # x,y = zip(*coords)
    # wx = max(x) - min(x)
    # wy = max(y) - min(y)
    # coords.append((max(x)+wx,max(y)))
    # coords.append((max(x)+wx,max(y)))
    # coords.append((min(x)-wx,max(y)))
    # coords.append((min(x)-wx,max(y)))
    # tri = spatial.Delaunay(coords)
    
    # then build the list of edges from the triangulation
    indices, indptr = tri.vertex_neighbor_vertices
    edges = []
    for k in range(indices.shape[0]-1):
        for j in indptr[indices[k]:indices[k+1]]:
            try:
                edges.append((chans[k],chans[j]))
            except IndexError:
                # let's ignore anything connected to the dummy channel
                pass
    
    # Edges are listed in two directions, pruning
    graph = []
    seen = set()
    for item in edges:
        if item not in seen and tuple(reversed(item)) not in seen:
            seen.add(item)
            graph.append(item)
    
    return graph


def build_geometries(channel_groups):
    for gr, group in channel_groups.items():
        group['graph'] = get_graph_from_geometry(group['geometry'])
    return channel_groups


def plot_channel_groups(channel_groups):
    
    n_shanks = len(channel_groups)
    
    f,ax = plt.subplots(1,n_shanks,squeeze=False)
    for sh in range(n_shanks):
        coords = [xy for ch,xy in channel_groups[sh]['geometry'].items()]
        x,y = zip(*coords)
        ax[sh,0].scatter(x,y,color='0.2')
        
        for pr in channel_groups[sh]['graph']:
            points = [channel_groups[sh]['geometry'][p] for p in pr]
            ax[sh,0].plot(*zip(*points),color='k',alpha=0.2)
        
        ax[sh,0].set_xlim(min(x)-10,max(x)+10)
        ax[sh,0].set_ylim(min(y)-10,max(y)+10)
        ax[sh,0].set_xticks([])
        ax[sh,0].set_yticks([])
        ax[sh,0].set_title('group %i'%sh)
        
        plt.axis('equal')
    return


# Design pattern by y.petremann at https://stackoverflow.com/a/26209900
def pretty(value, htchar='    ', lfchar='\n', indent=0):
    nlch = lfchar + htchar * (indent + 1)
    if type(value) is dict:
        items = [
            nlch + repr(key) + ': ' +
            pretty(value[key],
                   '' if type(value[key]) is tuple else htchar,
                   '' if type(value[key]) is tuple else lfchar,
                    0 if type(value[key]) is tuple else indent + 1)
            for key in value
        ]
        return '{%s}' % (','.join(items) + lfchar + htchar * indent)
    elif type(value) is list:
        items = [
            nlch +
            pretty(item,
                   '' if type(item) is tuple else htchar,
                   '' if type(item) is tuple else lfchar,
                    0 if type(item) is tuple else indent + 1)
            for item in value
        ]
        return '[%s]' % (','.join(items) + lfchar + htchar * indent)
    elif type(value) is tuple:
        items = [
            nlch + pretty(item, '', '', 0)
            for item in value
        ]
        return '(%s)' % (','.join(items))
    else:
        return repr(value)
