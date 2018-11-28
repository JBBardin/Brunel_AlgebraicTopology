# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 11:27:27 2017

@author: JB
"""

import numpy as np
import utils
import pyspike as spk

import sys
import getopt


opts, args = getopt.getopt(sys.argv[1:], "i:g:v:")

for opt, arg in opts:    
  if opt in ('-i'):
    inp = arg
  if opt in ('-g'):
    g = arg
  if opt in ('-v'):
    v = arg

for s in range(1):
  try:  
    d = np.load('save/brunel_inp={}_g={}_seed_{}.npy'.format(inp, g, s)).item()
    
    # synchronicity
    sp = d['sp']
    
    spike_list = []
    for train in sp:
      spike_list.append(spk.SpikeTrain(list(sp[train]), (0, 50), is_sorted=False))
    
    sync_dist = spk.spike_sync_matrix(spike_list, indices=None, interval=(1, 20))
    spike_dist = spk.spike_distance_matrix(spike_list, indices=None, interval=(1, 20))
    for i in range(sync_dist.shape[0]):
      sync_dist[i,i] = 1
    utils.Weight2txt(1-sync_dist, 'txt/brunel_inp={}_g={}_seed_{}_sync.txt'.format(inp, g, s))
    utils.Weight2txt(spike_dist, 'txt/brunel_inp={}_g={}_seed_{}_dist.txt'.format(inp, g, s))
    
    # Correlation
    corr = utils.Correlation_matrice(sp, interval=(1, 20), bin_by_sec=500, tau=1)
    for i in range(corr.shape[0]):
      corr[i, i] = 1 
    utils.Weight2txt(1-corr, 'txt/brunel_inp={}_g={}_seed_{}_corr.txt'.format(inp, g, s))
  except Exception as e:
    print('Error encounter: passed')
    print(str(e))
    