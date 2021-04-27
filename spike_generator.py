#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 15:20:53 2021

@author: roberttoth
"""

import numpy as np
import numpy.random as rnd
import colorednoise as cn
import os
import os.path as op
import sys
import h5py
import itertools
import csv

from scipy.special import erfinv, comb
from tqdm import tqdm
from six import exec_

from probe_setup import *
from copy import deepcopy



def read_config(path):
  
  # Construct system independent path, and assert that path exists
  prm_file = op.abspath(op.expanduser(path))
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
  path = op.abspath(op.expanduser(path))
  assert op.exists(path)
  with open(path, 'r') as f:
    contents = f.read()
  metadata = {}
  exec_(contents, {}, metadata)
  metadata = {k.lower(): v for (k, v) in metadata.items()}
  
  return metadata


# Generates a klustakwik .prm file
def write_param_file(path, prm):
  
  params = \
    '# pay attention to use \' and not ‘\n' \
    '\n' \
    'experiment_name = \'test_signal\'\n' \
    'prb_file = \'probe.prb\'\n' \
    '\n' \
    'traces = dict(\n' \
    '    raw_data_files=\'test_signal.raw.kwd\',\n' \
    '    voltage_gain=10.,\n' \
    '    sample_rate=' + str(prm['recordingRate']) + ',\n' \
    '    n_channels=' + str(prm['channels']) + ',\n' \
    '    dtype=\'int16\',\n' \
    ')\n' \
    '\n' \
    'spikedetekt = dict(\n' \
    '    filter_low=500.,  # Low pass frequency (Hz)\n' \
    '    filter_high_factor=0.95 * .5,\n' \
    '    filter_butter_order=3,  # Order of Butterworth filter.\n' \
    '\n' \
    '    filter_lfp_low=0,  # LFP filter low-pass frequency\n' \
    '    filter_lfp_high=300,  # LFP filter high-pass frequency\n' \
    '\n' \
    '    chunk_size_seconds=1,\n' \
    '    chunk_overlap_seconds=.015,\n' \
    '\n' \
    '    n_excerpts=50,\n' \
    '    excerpt_size_seconds=1,\n' \
    '    threshold_strong_std_factor=4.5,\n' \
    '    threshold_weak_std_factor=2.,\n' \
    '    detect_spikes=\'negative\',\n' \
    '\n' \
    '    connected_component_join_size=1,\n' \
    '\n' \
    '    extract_s_before=' + str(prm['spikeLength'] // 2) + ',\n' \
    '    extract_s_after=' + str(prm['spikeLength'] // 2) + ',\n' \
    '\n' \
    '    n_features_per_channel=3,  # Number of features per channel.\n' \
    '    pca_n_waveforms_max=10000,\n' \
    ')\n' \
    '\n' \
    'klustakwik2 = dict(\n' \
    '    num_starting_clusters=100,\n' \
    ')\n'
  
  path = op.join(op.abspath(path), 'params.prm')
  with open(path, 'w') as f:
    print(params, file=f)
  
  return

# Read and normalize template file if user params require
def read_templates(path, prm):
  
  # Construct system independent path, and assert that template file exists
  path = op.join(op.abspath(op.expanduser(path)), 'templates.csv')
  assert op.exists(path)
  
  # Read spike templates
  templates = np.genfromtxt(path, delimiter=',', dtype=np.double)
  
  # If params.maxAmplitude is defined, scale the templates
  
  if not np.isnan(prm['maxAmplitude']):
    ch = prm['channels']
    for i in range(0, prm['spikeTypes']):
      temp = templates[:, ch * i : ch * (i + 1)]
      scale = np.abs(prm['maxAmplitude'] / np.min(temp))
      templates[:, ch * i : ch * (i + 1)] =  scale * temp
  
  return templates


# Scales spike templates for amplitude sweep simulations
def scale_templates(templates, prm, f1, f2):
  
  ch = prm['channels']
  st = prm['spikeTypes']
  sl = prm['spikeLength']
  
  temps = np.zeros((sl, ch * st))
  temp = np.zeros((sl, ch))
  
  a1 = np.zeros(st)
  a2 = np.zeros(st)
  
  for i in range(0, st):
    # Take next spike in order
    temp = deepcopy(templates[:, i * ch : (i + 1) * ch])
    # Get amplitude order of selected spike over all channels
    order = np.argsort(np.min(temp, axis=0))
    # Calculating scale factor of secondary channel
    s = f2 * f1 * np.min(temp[:, order[0]]) / np.min(temp[:, order[1]])
    # Scale the largest channel
    temp[:, order[0]] *= f1
    # Scale all other channels
    temp[:, np.delete(np.arange(0, ch), order[0])] *= s
    # Save scaled spike template
    temps[:, i * ch : (i + 1) * ch] = temp
    # Save 1st and 2nd amplitudes
    a1[i] = -np.min(temp[:, order[0]])
    a2[i] = -np.min(temp[:, order[1]])
  
  amp1 = np.mean(a1)
  amp2 = np.mean(a2)
  
  return temps, amp1, amp2


# Generates timepoints spanning [0, prm['interval']), with minimum
# distance prm['minDistance'] to prevent overlaps
def timepoints(prm):
  
  tx = rnd.randint(low=0,
                   high=(prm['interval'] - prm['spikeLength'] + 1),
                   size=(prm['numberOfSpikes'] * 10))
  
  px = np.insert(np.zeros(prm['numberOfSpikes'] - 1), 0, tx[0])
  
  counter = 1
  
  for j in range(1, np.size(tx)):
    current = tx[j]  # Choose a point to test
    distances = np.abs(px[0:] - current)  # Check distance from stored points
    if np.min(distances) >= prm['minDistance']:
      px[counter] = current
      counter += 1
    if counter == prm['numberOfSpikes']:
      break
  
  if px[-1] == 0:
    inc = (prm['interval'] - prm['minDistance']) / prm['numberOfSpikes']
    
    if inc < prm['minDistance']:
      sys.exit('Error - Recording length insufficient ' \
               'to generate non-overlapping spikes')
    
    print('Warning - Recording length too short for random spike times\n' \
          'distributing spikes uniformly')
    
    px = np.arange(prm['spikeLength'],
                   prm['interval'] - prm['spikeLength'],
                   np.ceil(inc))
    
    # Mixing points to simplify cluster assignment
    rnd.shuffle(px)
  
  return px


# Exports neural signal in KlustaKwik's hdf5 input format
# Generates KlustaKwik setup and probe files
# Exports spike classes and spike times in csv
def write_data(path, prm, prb, channels_used, signal, pos, clu):
  
  path = op.abspath(op.expanduser(path))
  os.makedirs(path, exist_ok=True)
  
  order = np.argsort(pos)
  np.savetxt(op.join(path, 'pos.csv'), pos[order],
             delimiter=',', fmt='%d')
  np.savetxt(op.join(path, 'clu.csv'), clu[order],
             delimiter=',', fmt='%d')
  
  write_param_file(path, prm)
  write_probe_file(path, prb, channels_used)
  
  h5file = h5py.File(op.join(path, 'test_signal.raw.kwd'), 'w')
  h5file.create_dataset(name='recordings/0/data',
                        data=signal, dtype=np.int16)
  h5file.close()
  
  return


# First order high-pass
# fc = fs / (2pi(2^n - 1))
def hp_filter(x_in, n):
  
  x = np.append(x_in[0], x_in)
  y = np.zeros_like(x)
  
  for i in range(1, x.shape[0]):
    y[i] = (2**n - 1)/(2**n) * (y[i-1] + x[i] - x[i-1])
  
  return y[1:]


def spike_generator(path):
  
  # Read user params
  prm, prb = read_config(path)
  
  # Get path for base directory
  path = op.split(op.abspath(path))[0]
  
  # Read and normalize template file if user params require
  templates = read_templates(path, prm)
  
  # Generate spike times and associated clusters
  pos = timepoints(prm)
  clu = np.tile(range(0, prm['spikeTypes']), prm['spikesPerType'])
  
  # Mode 0 - Amplitude Sweep
  if prm['mode'] == 0:
    
    incr1 = 1 / prm['ampSweep1st']
    incr2 = 1 / prm['ampSweep2nd']
    
    max_count = prm['repetitions'] * prm['ampSweep1st'] * prm['ampSweep2nd']
    
    amp1 = np.zeros(prm['ampSweep1st'] * prm['ampSweep2nd'])
    amp2 = np.zeros(prm['ampSweep1st'] * prm['ampSweep2nd'])
    
    pbar = tqdm(total=max_count, ncols=80) 
    # add ascii=True on Windows, if display is incorrect
    
    for i in range(0, prm['repetitions']):
      
      # Gaussian white noise with zero mean, and given standard deviation
      #noise = rnd.normal(loc=0.0,
      #                   scale=prm['noiseStd'],
      #                   size=(prm['interval'], prm['channels']))
      
      
      #noise = np.zeros((prm['interval'], prm['channels'])) 
      #for ch in range(0, prm['channels']):
      #  tmp = pink_noise(prm['interval']) 
      #  tmp = tmp - np.mean(tmp)
      #  std = np.median(np.abs(tmp)) / erfinv(0.5) / np.sqrt(2)
      #  noise[:, ch] = tmp * prm['noiseStd'] / std
      
      # Generating multiple channels of (1/f)^beta noise,
      # with given broad-band standard deviation
      # First order High-pass filtering at 3Hz imitates recording equipment
      beta = 1.50  # Noise spectrum is (1/f)^beta
      noise = np.zeros((prm['interval'], prm['channels']))
      for ch in range(0, prm['channels']):
        tmp = hp_filter(cn.powerlaw_psd_gaussian(beta, prm['interval']), 10)
        noise[:, ch] = (tmp - np.mean(tmp)) * prm['noiseStd'] / np.std(tmp)
      
      count = 0
      
      for j in np.arange(incr1, 1 + incr1, incr1):
        for k in np.arange(incr2, 1 + incr2, incr2):
          # Scale template amplitudes
          temp, a1, a2 = scale_templates(templates, prm, j, k)
          
          # Add spikes at the generated positions
          spikes = np.zeros((prm['interval'], prm['channels']))
          for l in range(0, prm['numberOfSpikes']):
            spk_slice = np.arange(pos[l],
                                  pos[l] + prm['spikeLength'],
                                  dtype=np.int)
            tmp_slice = np.arange(clu[l] * prm['channels'],
                                  (clu[l] + 1) * prm['channels'],
                                  dtype=np.int)
            spikes[spk_slice, :] = temp[:, tmp_slice]
          
          signal = noise + spikes
          
          # No need to overwrite this multiple times
          if i == 0:
            amp1[count] = a1
            amp2[count] = a2
          
          workdir = op.join(op.join(path, 'Set_' + str(i)), str(count))
          channels_used = range(0, prm['channels'])
          write_data(workdir, prm, prb, channels_used, signal, pos, clu)
          
          count += 1
          pbar.update(1)
    
    np.savetxt(op.join(path, 'amp1.csv'),
               np.reshape(amp1, (prm['ampSweep1st'], prm['ampSweep2nd'])),
               delimiter=',', fmt='%.3f')
    np.savetxt(op.join(path, 'amp2.csv'),
               np.reshape(amp2, (prm['ampSweep1st'], prm['ampSweep2nd'])),
               delimiter=',', fmt='%.3f')
    
    pbar.close()
    
  elif prm['mode'] == 1:
    
    channels = range(0, prm['channels'])
    
    max_count = int(sum(comb(prm['channels'], ch) for ch in channels))
    
    prb_list = ['']*max_count
    
    pbar = tqdm(total=(max_count * prm['repetitions']), ncols=80) 
    # add ascii=True on Windows, if display is incorrect
    
    for i in range(0, prm['repetitions']):
      
      count = 0
      
      # Noise
      beta = 1.50  # Noise spectrum is (1/f)^beta
      noise = np.zeros((prm['interval'], prm['channels']))
      for ch in range(0, prm['channels']):
        tmp = hp_filter(cn.powerlaw_psd_gaussian(beta, prm['interval']), 10)
        noise[:, ch] = (tmp - np.mean(tmp)) * prm['noiseStd'] / np.std(tmp)

      # Signal
      # Add spikes at the generated positions
      spikes = np.zeros((prm['interval'], prm['channels']))
      for l in range(0, prm['numberOfSpikes']):
        spk_slice = np.arange(pos[l],
                              pos[l] + prm['spikeLength'],
                              dtype=np.int)
        tmp_slice = np.arange(clu[l] * prm['channels'],
                              (clu[l] + 1) * prm['channels'],
                              dtype=np.int)
        spikes[spk_slice, :] = templates[:, tmp_slice]
      
      signal = noise + spikes
      
      for j in channels:
        tuples = list(itertools.combinations(channels, j + 1))
        # Results is list of tuples, need to convert to list of lists
        ch_combs = [[x[0]] if len(x) == 1 else list(x) for x in tuples]
        for k in range(0, len(ch_combs)):
          workdir = op.join(path, 'Set_' + str(i), str(count))
          write_data(workdir, prm, prb, ch_combs[k], signal, pos, clu)
          
          # No need to overwrite this multiple times
          if i == 0:
            tmp = ['1' if ch in ch_combs[k] else '0' for ch in channels]
            prb_list[count] = ''.join(tmp)
          
          count += 1
          pbar.update(1)
    
    np.savetxt(op.join(path, 'probes.csv'), prb_list, delimiter=',', fmt='%s')
    
    pbar.close()
  
  return


if __name__ == "__main__":
  # TODO: make path to config file into argument
  spike_generator('config.txt')
  
  
  
  
