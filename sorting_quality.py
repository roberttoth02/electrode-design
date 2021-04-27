#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 15:20:53 2021

@author: roberttoth
"""

import numpy as np

#import os
import os.path as op
import sys
import h5py

from tqdm.auto import tqdm
from six import exec_



def read_config(path):
  
  # Construct system independent path, and assert that path exists
  prm_file = op.realpath(op.expanduser(path))
  assert op.exists(prm_file), 'Invalid path to configuration file'
  
  # Reading core parameters
  prm = _read_python(prm_file)['params']
  
  # Expected core parameters
  expected_prm = {'channels', 'recordingRate', 'recordingLength',
                  'spikeTypes', 'spikeLength', 'spikesPerType',
                  'maxAmplitude', 'noiseStd', 'repetitions',
                  'completeness', 'purity', 'tolerance', 'penalty',
                  'mode'}
  
  # Mode specific parameters
  mode0_prm = {'ampSweep1st', 'ampSweep2nd'}
  
  # Ensure all core parameters are available
  missing_prm = expected_prm - set(prm)
  if len(missing_prm):
    sys.exit('Error - Missing parameters: ' +
             (', '.join(item for item in missing_prm)))
  
  # Ensure all mode specific parameters are available
  if prm['mode'] == 0:
    missing_mode0 = mode0_prm - set(prm)
    if len(missing_mode0):
      sys.exit('Error - Missing parameters: ' +
               (', '.join(item for item in missing_prm)))
  
  # Check if any unexpected parameters were provided
  unexpected_prm = expected_prm.union(mode0_prm) - set(prm)
  if len(unexpected_prm):
    print('Warning - Unexpected parameters: ' +
          (', '.join(item for item in unexpected_prm)))
  
  # Reading probe information
  prb = _read_python(prm_file)['probe']
  
  # Ensure that required probe information is available
  expected_prb = {'site_map', 'site_xy'}
  
  missing_prb = expected_prb - set(prb)
  if len(missing_prb):
    sys.exit('Error - Missing probe information: ' +
             (', '.join(item for item in missing_prb)))
  
  unexpected_prb = set(prb) - expected_prb
  if len(unexpected_prb):
    print('Warning - Unexpected probe information: ' +
          (', '.join(item for item in unexpected_prb)))
  
  # Generate convenience parameters
  
  # Minimum samples between two spikes
  prm['minDistance'] = 2 * prm['spikeLength']
  # Length of recording in [samples]
  prm['interval'] = prm['recordingLength'] * prm['recordingRate']
  # Total number of spikes
  prm['numberOfSpikes'] = prm['spikeTypes'] * prm['spikesPerType']
  
  # if prm['mode'] == 0
  #    channels_used = range(0, prm['channels'])
  
  return prm, prb


# From Klusta to provide familiar setup process to users
# https://github.com/kwikteam/klusta
def _read_python(path):
  path = op.realpath(op.expanduser(path))
  assert op.exists(path)
  with open(path, 'r') as f:
    contents = f.read()
  metadata = {}
  exec_(contents, {}, metadata)
  metadata = {k.lower(): v for (k, v) in metadata.items()}
  
  return metadata


# Normalise cluster labels to a regular integer sequence from 0, as
# default klusta class labels can skip values
def norm_clusters(clu):
  u = np.unique(clu)
  newclu = np.zeros_like(clu)
  
  for i in range(0, u.shape[0]):
    newclu[clu == u[i]] = i
  
  return newclu


# Constructs the confusion matrix of a clustering
def confusion(prm, gpos, gclu, apos, aclu_in):
  
  aclu = norm_clusters(aclu_in)
  
  M = prm['spikeTypes']  # Number of original clusters
  N = np.max(aclu) + 1   # Number of clusters found
  
  tol = prm['tolerance']
  
  c = np.zeros((M + 1, N + 1))
  
  # Generated positions mark the start of a spike, obtained positions are
  # the points of threshold crossing, need to correct for offset
  offset = (prm['spikeLength'] // 2) - 1
  apos -= offset
  
  for i in range(0, M):
    gp = gpos[np.where(gclu == i)]
    for j in range(0, N):
        ap = apos[np.where(aclu == j)].reshape(-1,1)
        pos = np.tile(ap, (1, 2 * tol + 1)) + \
            np.tile(np.arange(-tol, tol + 1, 1), (ap.shape[0], 1))
        tpos = np.sort(pos.flatten())
        c[i + 1, j + 1] = np.sum(np.isin(gp, tpos))
  
  # Fills out c(i,0), i>0
  # For each original cluster i, finds all timestamps in apos that are from i,
  # then calculates the size of the set difference between i and the acquired
  # elements
  for i in range(0, M):
    gp = gpos[np.where(gclu == i)]
    ap = apos.reshape(-1,1)
    pos = np.tile(ap, (1, 2 * tol + 1)) + \
          np.tile(np.arange(-tol, tol + 1, 1), (ap.shape[0], 1))
    tpos = np.sort(pos.flatten())
    c[i + 1, 0] = gp[np.isin(gp, tpos, invert=True)].shape[0]
  
  # Fills out c(0,j), j>0
  for j in range(0, N):
    c[0, j + 1] = np.sum(aclu == j) - np.sum(c[:, j + 1], 0)
  
  return c


# Returns the Completeness of each obtained cluster
def completeness(conf, prm):
  
  k = prm['penalty']
  
  com = np.zeros(np.shape(conf)[1] - 1)
  for j in range(1, np.shape(conf)[1]):
    com[j - 1] = np.max(conf[1:, j] / np.sum(conf[1:, k:], 1))
  
  return com


# Returns the Purity of each obtained cluster
def purity(conf, prm):
  
  k = prm['penalty']
  
  pur = np.zeros(np.shape(conf)[1] - 1)
  for j in range(1, np.shape(conf)[1]):
    pur[j - 1] = np.max(conf[1:, j] / np.sum(conf[k:, j], 0))
  
  return pur


def sorting_quality(path):
  
  # Read user params
  prm, prb = read_config(path)
  
  # Get path for base directory
  path = op.split(op.abspath(path))[0]
  
  # Mode 0 - Amplitude Sweep
  if prm['mode'] == 0:
    max_count = prm['repetitions'] * prm['ampSweep1st'] * prm['ampSweep2nd']
    
    pbar = tqdm(total=max_count, ncols=80)
    # add ascii=True on Windows, if display is incorrect
    
    # Number of clusters satisfying the purity and completeness criteria
    n = np.zeros((prm['repetitions'], prm['ampSweep1st'] * prm['ampSweep2nd']))
        
    for i in range(0, prm['repetitions']):
      for j in range(0, prm['ampSweep1st'] * prm['ampSweep2nd']):
        
        workdir = op.join(path, 'Set_' + str(i), str(j))
        
        # Read generated spike times
        posfile = op.join(workdir, 'pos.csv')
        assert op.exists(posfile)
        gpos = np.genfromtxt(posfile, delimiter=',', dtype=np.double)
        
        # Read generated spike clusters
        clufile = op.join(workdir, 'clu.csv')
        assert op.exists(clufile)
        gclu = np.genfromtxt(clufile, delimiter=',', dtype=np.double)
        
        # Read spike sorting results: aquired spike times and clusters
        kwikpath = op.join(workdir, 'test_signal.kwik')
        assert op.exists(kwikpath)
        with h5py.File(kwikpath, 'r') as kwikfile:
          apos = kwikfile.get('/channel_groups/0/spikes/time_samples').value
          aclu = kwikfile.get('/channel_groups/0/spikes/clusters/main').value
        assert apos.shape == aclu.shape
        
        conf = confusion(prm, gpos, gclu, apos, aclu)
        
        tmp = np.logical_and(purity(conf, prm) >= prm['purity'], \
                             completeness(conf, prm) >= prm['completeness'])
        n[i, j] = np.sum(tmp)
        
        pbar.update(1)
    
    np.savetxt(op.join(path, 'n_clusters.csv'), n, delimiter=',', fmt='%.4f')
    
    pbar.close()
  
  elif prm['mode'] == 1:
    
    channels = range(0, prm['channels'])
    max_count = int(sum(comb(prm['channels'], ch) for ch in channels))
    
    pbar = tqdm(total=(max_count * prm['repetitions']), ncols=80) 
    # add ascii=True on Windows, if display is incorrect
    
    n = np.zeros((prm['repetitions'], max_count))
    
    for i in range(0, prm['repetitions']):
      
      count = 0
      
      for j in channels:
        for k in range(0, len(ch_combs)):
          
          workdir = op.join(path, 'Set_' + str(i), str(count))
          
          # Read generated spike times
          posfile = op.join(workdir, 'pos.csv')
          assert op.exists(posfile)
          gpos = np.genfromtxt(posfile, delimiter=',', dtype=np.double)
          
          # Read generated spike clusters
          clufile = op.join(workdir, 'clu.csv')
          assert op.exists(clufile)
          gclu = np.genfromtxt(clufile, delimiter=',', dtype=np.double)
          
          # Read spike sorting results: aquired spike times and clusters
          kwikpath = op.join(workdir, 'test_signal.kwik')
          assert op.exists(kwikpath)
          with h5py.File(kwikpath, 'r') as kwikfile:
            apos = kwikfile.get('/channel_groups/0/spikes/time_samples').value
            aclu = kwikfile.get('/channel_groups/0/spikes/clusters/main').value
          assert apos.shape == aclu.shape
          
          conf = confusion(prm, gpos, gclu, apos, aclu)
          
          tmp = np.logical_and(purity(conf, prm) >= prm['purity'], \
                               completeness(conf, prm) >= prm['completeness'])
          n[i, count] = np.sum(tmp)
          
          count += 1
          pbar.update(1)
    
    np.savetxt(op.join(path, 'n_clusters.csv'), n, delimiter=',', fmt='%.4f')
    
    # save averaged result
    
    pbar.close()
  
  return


if __name__ == "__main__":
  # TODO: make path to config file into argument
  sorting_quality('config.txt')












