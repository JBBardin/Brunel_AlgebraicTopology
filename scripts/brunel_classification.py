# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 16:03:48 2017

@author: jb
"""

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd


def dummy_labels(labels, n_class=4):
  out = np.zeros((len(labels), n_class))
  for i, l in enumerate(labels):
    l = min(n_class-1, l)
    out[i, l] = 1
  return out


def preprocess_features(samples, idx):
  samples = samples[:, idx]
  mean = np.nanmean(samples, axis=0)
  std = np.nanstd(samples, axis=0)
  samples = (samples - mean)/std
  
  #replace NaNs by 0
  samples = np.nan_to_num(samples)
  return samples


def file_name(v, clean=False):
  if clean:
    return 'brunel_v{}_features_clean.npy'.format(v)
  else:
    return 'brian_rate_{}_features.npy'.format(v)
  
  
def find_badly_classified(pred, labels, inp, g):
  bad = []
  for p, l, i, g_ in zip(pred, labels, inp, g):
    if not (p == l).all():
      bad.append([i, g_])
  return bad


def clean_None(samples, labels, inp, g):
  args = np.argwhere(labels==None)
  samples = np.delete(samples, args, axis=0)
  labels = np.delete(labels, args, axis=0)
  inp = np.delete(inp, args, axis=0)
  g = np.delete(g, args, axis=0)
  return np.array(samples, dtype=float), np.array(labels, dtype=float), inp, g

def plot_badly_classified(bad, v_v):
  def filterg(v, v_v):
    if v_v==1:
      return 114.65 + v*39.71
    elif v_v==2:
      return 64 + v*35.75
    elif v_v==3:
      return 55.61 + v*48.86
      
    
  def filterinp(v, v_v):
    if v_v==1:
      return 297.7 - v*69
    elif v_v==2:
      return 249 - v*61
    elif v_v==3:
      return 345 - v*81
  
  cmap = plt.get_cmap('Set1').colors
  
  plt.figure()
  img = plt.imread('../fig/phase-diagram-Brunel_v{}.png'.format(v_v))
  plt.imshow(img, zorder=0)
  plt.axis('off')
  
  bad_, counts = np.unique(bad, return_counts=True, axis=0)
  for b, c in zip(bad_, counts):
    plt.plot(filterg(b[1], v_v), filterinp(b[0], v_v), 'o', markersize=c, zorder=1, color=cmap[0], alpha=0.6)
  return None


def custom_accuracy_score(label, pred, same_regime=False, ignore_other=False):
  agree = np.where(label == pred)[0]
  if ignore_other:
    idx = np.argwhere(label % 2 == 0)
    agree = np.where(label[idx] == pred[idx])[0]
    return len(agree)/len(idx)
  if same_regime:
    idx = np.argwhere(label % 2 == 1)
    idx_2 = np.where(pred % 2 == 1)
    agree2 = np.intersect1d(idx, idx_2)
    idx = np.argwhere(label == pred)
    agree_3 = np.intersect1d(agree2, idx)
    return (len(agree) + len(agree2) - len(agree_3))/len(label)
  return len(agree)/len(label)


def confusion_matrix(true, pred):
  out = np.zeros((4, 4))
  for t, p in zip(true, pred):
    out[int(t), int(p)] += 1
  return out

def plot_predicted_PD(pred, inp, g, v_v):
  def filterg(v, v_v):
    if v_v==1:
      return 114.65 + v*39.71
    elif v_v==2:
      return 64 + v*35.75
    elif v_v==3:
      return 55.61 + v*48.86
      
    
  def filterinp(v, v_v):
    if v_v==1:
      return 297.7 - v*69
    elif v_v==2:
      return 249 - v*61
    elif v_v==3:
      return 345 - v*81
  
  cmap = plt.get_cmap('Set1').colors
  c = [cmap[1],cmap[2],cmap[0],cmap[3]]
  
  plt.figure()
  img = plt.imread('../fig/phase-diagram-Brunel_v{}.png'.format(v_v))
  plt.imshow(img, zorder=0)
  plt.axis('off')
  
  for inp_ in np.unique(inp):
    for g_ in np.unique(g):
      idx1 = np.argwhere(inp == inp_)
      idx2 = np.argwhere(g == g_)
      idx = []
      for i in idx1:
        if i in idx2:
          idx.append(i)
      if idx == []:
        continue
      
      idx = np.array(idx)
#      print(idx, pred[idx], np.unique(pred[idx], return_counts=True))
#      classes = np.argmax(pred[idx])
      uni, counts = np.unique(pred[idx], return_counts=True)
#      print(uni, counts)
      plt.plot(filterg(g_, v_v), filterinp(inp_, v_v), 'o', 
               markersize=20, zorder=1, color=c[int(uni[np.argmax(counts)])], alpha=0.6)
  
  m_sr = plt.Line2D((0, 1), (0, 0), marker='o', linestyle='', color=c[0], markersize=20, alpha=0.6, label='0' )
  m_ai = plt.Line2D((0, 1), (0, 0), marker='o', linestyle='', color=c[1], markersize=20, alpha=0.6, label='1' )
  m_si = plt.Line2D((0, 1), (0, 0), marker='o', linestyle='', color=c[2], markersize=20, alpha=0.6, label='2' )
  m_si_2 = plt.Line2D((0, 1), (0, 0), marker='o', linestyle='', color=c[3], markersize=20, alpha=0.6, label='3' )

  plt.legend(handles=[m_sr, m_ai, m_si, m_si_2], labelspacing=1.5)
  return None

classifier1 = svm.SVC(random_state=33, kernel='rbf', decision_function_shape='ovo')#, class_weight='balanced')
parameters1 = {'C': np.logspace(-6, 0, num=7) }

classifier2 = KNeighborsClassifier(n_neighbors=3)
parameters2 = {'n_neighbors': np.arange(1,10)}


fieldnames = ['features', 'training set', 'best_CV_accuracy', 'best_params', 
              'Accuracy on v1', 'Accuracy on v2', 'Accuracy on v3']

label_id = ['corr_exist', 'corr_peak_pos', 'corr_peak_amp', 'corr_area',
            'sync_exist', 'sync_peak_pos', 'sync_peak_amp', 'sync_area',
            'dist_exist', 'dist_peak_pos', 'dist_peak_amp', 'dist_area']
label_ids = []
for s in range(2):
  for l in label_id:
    label_ids.append(l + '_betti_{}'.format(s))
    

var = []
idx = [4,7,8,12,16,20,23]#[3,4,7,8,11,12, 15,16,19,20,23,24]#
for i in idx:
  var.append(label_ids[i-1])

  
out = np.load('../save/' + file_name(1, clean=False)).item()
samples_v1 = out['samples']
labels_v1 = out['labels']
inp_v1 = out['inp']
g_v1 = out['g']
samples_v2, labels_v2, inp_v2, g_v2 = np.load('../save/' + file_name(2, clean=False)).item().values()
out = np.load('../save/' + file_name(3, clean=False)).item()
samples_v3 = np.array(out['samples'], dtype=float)
labels_v3 = np.array(out['labels'])
inp_v3 = out['inp']
g_v3 = out['g']

samples_v1 = preprocess_features(samples_v1, np.array(idx)-1)
samples_v2 = preprocess_features(samples_v2, np.array(idx)-1)
samples_v3 = preprocess_features(samples_v3, np.array(idx)-1)
  
#labels_v1 = dummy_labels(labels_v1, n_class=3)
#labels_v2 = dummy_labels(labels_v2, n_class=3)
#labels_v3 = dummy_labels(labels_v3, n_class=3)

samples_v2, labels_v2, inp_v2, g_v2 = clean_None(samples_v2, labels_v2, inp_v2, g_v2)
samples_v3, labels_v3, inp_v3, g_v3 = clean_None(samples_v3, labels_v3, inp_v3, g_v3)

result = pd.DataFrame()
n=0

for classifier, parameters, cla_s in zip([classifier1, classifier2], [parameters1, parameters2],
                                         ['SVM', 'kNN']):
  cla = GridSearchCV(classifier, parameters, cv=10, refit=True, return_train_score=True)
  #        scores = cross_val_score(classifier, samples, labels, cv=10)
   
  for samples, labels, v_t, inp, g, samples_val1, labels_val1, v_v1, inp_v1, \
      g_v1, samples_val2, labels_val2, v_v2, inp_v2, g_v2 in zip(
      [samples_v1, samples_v2, samples_v3], 
      [labels_v1, labels_v2, labels_v3], [1,2,3],
      [inp_v1, inp_v2, inp_v3], [g_v1, g_v2, g_v3],
      [samples_v2, samples_v3, samples_v1],
      [labels_v2, labels_v3, labels_v1], [2,3,1],
      [inp_v2, inp_v3, inp_v1], [g_v2, g_v3, g_v1],
      [samples_v3, samples_v1, samples_v2],
      [labels_v3, labels_v1, labels_v2], [3,1,2],
      [inp_v3, inp_v1, inp_v2], [g_v3, g_v1, g_v2]):
        
    val_frac = 0.1
    s_tr, s_te, l_tr, l_te = train_test_split(samples, labels, test_size=val_frac,
                                              random_state=42)
    cla.fit(s_tr, l_tr)
        
    print("CV_mean accuracy={:.2%}".format(cla.best_score_))
    pred = cla.predict(s_te)
    print("Validation accuracy={:.2%}".format(accuracy_score(l_te, pred)))
    print(cla.best_params_)
    
#    cla.fit(samples, labels)
    
    #        if c:
    #          dic['Accuracy on clean(mean +/- std)'] = '{:.2%} +/- {:.2%}'.format(np.mean(scores), np.std(scores))
    #        else:
          
    for s_val, l_val, i_val, g_val, v in zip([samples, samples_val1, samples_val2],
                                              [labels, labels_val1, labels_val2],
                                              [inp, inp_v1, inp_v2],
                                              [g, g_v1, g_v2],
                                              [v_t, v_v1, v_v2]):
      
      
      dic = dict()
      dic['classifier'] = cla_s
      dic['train set'] = v_t
      dic['test set'] = v
      dic['param_str'] = cla.best_params_.keys()
      dic['param_v'] = cla.best_params_.values()
    
      
      pred = cla.predict(s_val)
      acc = custom_accuracy_score(l_val, pred)
      print('acc: ', acc)
      acc_s = custom_accuracy_score(l_val, pred, True)
      print('acc_same: ', acc_s)
      conf = confusion_matrix(l_val, pred)
      acc_i = custom_accuracy_score(l_val, pred, False, True)
      bad = find_badly_classified(pred, l_val, i_val, g_val)
      if bad:
        plot_badly_classified(bad, v)
        plt.savefig('../fig/bad_vt{}_vv{}_{}.png'.format(v_t, v, cla_s))
        plt.close()
      plot_predicted_PD(pred, i_val, g_val, v)
      plt.savefig('../fig/predicted_PD_vt{}_vv{}_{}.png'.format(v_t, v, cla_s))
      plt.close()
      
      dic['acc'] = acc
      dic['acc same'] = acc_s
      dic['acc ignore'] = acc_i
      dic['confusion_matrix'] = [conf.flatten()]
      dic['grid'] = [cla.grid_scores_]
        
      dic = pd.DataFrame(dic, index=[n])
      n+=1
  
      result = result.append(dic)

result.to_excel('../result_svc.xls')
  
# =============================================================================
# if idx == 1:
#   dic['Accuracy on v1'] = acc
# elif idx == 2:
#   dic['Accuracy on v2'] = acc
# elif idx == 3:
#   dic['Accuracy on v3'] = acc
# =============================================================================
 
   
    
    
def purge(prob):  
  p = cla.predict_proba(samples)
  pred = cla.predict(samples)
  p_ = np.max(p, axis=1)   
  a = np.argwhere(p_< prob)
  pred = np.delete(pred, a, axis=0)
  lab = np.delete(labels, a, axis=0)
  inp = inp_v1
  g = g_v1
  inp = np.delete(inp, a , axis=0)
  g = np.delete(g, a , axis=0)
  print(accuracy_score(lab, pred))
  bad = find_badly_classified(pred, lab, inp, g)
  plot_badly_classified(bad, 1)
# =============================================================================
#     for mv, string in zip([mv_betti12, mv_betti02, mv_betti01, mv_betti2, mv_betti1, mv_betti1, []],
#                           ['0', '1', '2', '0+1', '0+2', '1+2', 'all']):
#       idx_del = np.concatenate((idx_base, mv))
#       
#       dic = {'features' : string, 'training set' : v_t}
#       for c in [False]:
#         samples, labels, inp, g = np.load('../save/' + file_name(v_t, clean=c)).item().values()
#         samples = preprocess_features(samples, idx_del)
#         labels = dummy_labels(labels, n_class=3)
#         
#         bad = []
#         cla = GridSearchCV(classifier, parameters, cv=10, refit=True, return_train_score=True)
# #        scores = cross_val_score(classifier, samples, labels, cv=10)
#         cla.fit(samples, labels)
#           
#         print("Betti {}, Training on v{}, mean accuracy={:.2%}".format(string, v_t, cla.best_score_))
# 
# #        if c:
# #          dic['Accuracy on clean(mean +/- std)'] = '{:.2%} +/- {:.2%}'.format(np.mean(scores), np.std(scores))
# #        else:
#         dic['Accuracy on full(mean +/- std)'] = '{:.2%}'.format(cla.best_score_)
#         
#         writer.writerow(dic)
# =============================================================================


        
        
        
        
        
        
        
#        samples_V, labels_V, inp_V, g_V = np.load('../save/' + file_name(v_v, clean=c)).item().values()
#        samples_V = preprocess_features(samples_V, idx_del)
#        labels_V = dummy_labels(labels_V, n_class=3)
#          
#          score_v[it] = classifier.score(samples_V, labels_V)
#          pred = classifier.predict(samples_V)
#          bad.extend(find_badly_classified(pred, labels_V, inp_V, g_V))
#          
#        if c:
#          c_string = '_clean'
#          dic['Accuracy on clean(mean +/- std)'] = '{:.2%} +/- {:.2%}'.format(np.mean(score_v), np.std(score_v))
#        else:
#          c_string = ''
#          dic['Accuracy on full(mean +/- std)'] = '{:.2%} +/- {:.2%}'.format(np.mean(score_v), np.std(score_v))
#       
#        plot_badly_classified(bad, v_v=v_v)
#        plt.savefig('../fig/bad_classify_vt{}_vv{}{}_f{}.png'.format(v_t, v_v, c_string, string))
#        plt.close()
#          
#        print("Betti {}, Training on v{}, Testing on v{}{:<6}, mean accuracy={:.2%}, std={:.2%}".format(string, v_t, v_v, c_string, np.mean(score_v), np.std(score_v)))
#      writer.writerow(dic)

