# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 16:03:48 2017

@author: jb
"""

import numpy as np
import ph_analysis as ph
import seaborn as sns
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def extract_features_betti(curve, dim):
  if dim in curve:
    c = curve[dim]
    max_idx = np.argmax(c[:, 1])
    if dim==0:
      max_idx += 1
    
    bins = np.diff(c[:, 0])
    area = np.sum(bins[::2] * c[::2, 1])
    return 1, c[max_idx, 0], c[max_idx, 1], area
  else:
    return 0, None, None, None


def dummy_labels(labels, n_class=4):
  out = np.zeros((len(labels), n_class))
  for i, l in enumerate(labels):
    out[i, l] = 1
  return out


def label_v1(inp, g):
  if inp == 1:
    if g <= 3:
      return 0 #SR
    elif g >=4 and g <= 5:
      return 2 #SI slow
    elif g == 8:
      return 2
    else:
      return 1 #AI
  if inp == 2:
    if g <= 3.5:
      return 0
    else:
      return 1
  if inp == 3:
    if g <= 3.5:
      return 0
    elif g > 6 :
      return 3 #SI fast
    else: 
      return 1
  if inp == 4:
    if g <= 3.5:
      return 0
    elif g > 5.5 :
      return 3 #SI fast
    else: 
      return 1   

def label_v2(inp, g):
  if g <= 4:
    return 0 #SR
  if inp == 1:
    if g > 4: 
      return 2 #SI slow
  if inp == 2:
    if g > 4: 
      return 1
  if inp == 3:
    if g < 6:
      return 1
    else:
      return 3 #SI fast
  if inp == 4:
    if g < 4:
      return 0
    elif g > 5.5 :
      return 3 #SI fast
    else: 
      return 1  


def label_v3(inp, g):
  if g < 4:
    return 0
  elif inp == 1:
    return 2
  else:
    if g <= 5:
      return 1
    else:
      return 2


def label_v1_clean(inp, g):
  if inp == 1:
    if g == 3.5:
      return -1
    elif g <= 3:
      return 0 #SR
    elif g >=4 and g <= 5:
      return 2 #SI slow
    elif g == 8:
      return 2
    else:
      return 1 #AI
  if inp == 2:
    if g == 3.5:
      return -1
    elif g <= 3:
      return 0
    else:
      return 1
  if inp == 3:
    if g == 3.5 or g == 6:
      return -1
    elif g <= 3:
      return 0
    elif g > 6 :
      return 3 #SI fast
    else: 
      return 1
  if inp == 4:
    if g == 3.5:
      return -1
    elif g <= 3:
      return 0
    elif g > 5.5 :
      return 3 #SI fast
    else: 
      return 1   

def label_v2_clean(inp, g):
  if g == 4:
    return -1
  if g < 4:
    return 0 #SR
  if inp == 1:
    if g > 4: 
      return 2 #SI slow
  if inp == 2:
    if g > 4: 
      return 1
  if inp == 3:
    if g == 6:
      return -1
    elif g < 6:
      return 1
    else:
      return 3 #SI fast
  if inp == 4:
    if g < 4:
      return 0
    elif g > 5.5 :
      return 3 #SI fast
    else: 
      return 1  


def label_v3_clean(inp, g):
  if g == 4:
    return -1
  if g < 4:
    return 0
  elif inp == 1:
    return 2
  elif inp == 3 or inp == 4:
    if g == 5:
      return -1
    else:
      if g < 5:
        return 1
      else:
        return 2
  else:
    if g <= 5:
      return 1
    else:
      return 2
    
    
def label_v1_brian(inp, g):
  if g <= 3:
    return 0
  elif inp == 1:
    return 2
  else:
    return 1
  
def label_v2_brian(inp, g):
  if g < 4:
    return 0
  elif inp == 1:
    if g ==4:
      return 3
    else:
      return None
  elif g >= 6:
    return 2
  elif inp == 2 and g==5:
    return 2
  else:
    return 3 #transition from SR to SI slow
  
def label_v3_brian(inp, g):
  if g < 4:
    return 0
  elif inp == 1:
    if g ==4:
      return 3
    else:
      return None
  elif inp == 4:
    if g >= 7:
      return 2
    else:
      return 3
  elif inp == 3:
    if g >= 6:
      return 2
    else:
      return 3
  elif inp == 2:
    if g >= 5:
      return 2
    else:
      return 3
   
  
for v, fun_label in zip(range(3), [label_v1_clean, label_v2_clean, label_v3_clean]):  
  samples = []
  labels = []
  inp_in = []
  g_in = []
  for inp in [1,2,3,4]:
    for g in [2,3,3.5,4,4.5,5,6,7,8]:    
      for s in range(10):
        sample = []
        for app in ['corr', 'sync', 'dist']:
          f_name = '../save/brunel_v{}_inp={}_g={}_seed_{}_{}_PH.npy'.format(v+1, inp, g, s, app)
          try:
            d = np.load(f_name).item()
          except FileNotFoundError:
            print('file not found: {}'.format(f_name))
            continue
           
          c = ph.Betti_curve(d)
        
          exist, peak_pos, peak_amp, area = extract_features_betti(c, 0)
          sample.append(exist)
          sample.append(peak_pos)
          sample.append(peak_amp)
          sample.append(area)
          exist, peak_pos, peak_amp, area = extract_features_betti(c, 1)
          sample.append(exist)
          sample.append(peak_pos)
          sample.append(peak_amp)
          sample.append(area)
          exist, peak_pos, peak_amp, area = extract_features_betti(c, 2)
          sample.append(exist)
          sample.append(peak_pos)
          sample.append(peak_amp)
          sample.append(area)
           
        label = fun_label(inp, g)
        if [] not in sample and not label == -1:
          labels.append(label)
          samples.append(sample)
          inp_in.append(inp)
          g_in.append(g)
        
  #remove samples created because of FileNotFoundError        
  assert([] not in samples), "Error empty samples"
    
  samples = np.array(samples, dtype=np.float)     
  labels = np.array(labels) 
          
  np.save('../save/brunel_v{}_features', {'samples':samples, 'labels':labels, 'inp': inp_in, 'g':g_in})

