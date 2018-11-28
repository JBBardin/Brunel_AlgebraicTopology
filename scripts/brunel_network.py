# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 11:48:22 2017

@author: jb
"""

import numpy as np
import brian2 as br
import random

# SIMULATION PARAMETERS:
sim_time = 20             #Simulation time of the network (in seconds)
downscale = 12.5          #Downscaling factor of the network size
connectivity = 0.4        #Probabilty to receive a connexion from a particular neuron
syn_delay = 1.5           #Synaptic delay (in ms)
Vm_jump_after_EPSP = 0.2  #increase of the posynaptic neuron membrane potential after EPSP (in mV)


def Simulate(g, inp, seed=1337):
  """Simulation of the Brunel network with
  relative strength between IPSP and EPSP g and 
  external input parameter inp.
  Inp=1 is the minimum amount of external stimulation to observe sustained 
  activity in the full network (12 500 neurons).
  """

  br.set_device('cpp_standalone')
  br.prefs.codegen.target = 'weave'
  
	# NETWORK INITIALIZATION
  np.random.seed(seed) # set seed for reproducibility of simulations
  random.seed(seed) # set seed for reproducibility of simulations

	# =============================================================================
	# PARAMETERS
	# Simulation parameters
  simdt = 0.01*br.ms
  simtime = sim_time*br.second
  br.defaultclock.dt = simdt         # Brian's default sim time step
  dt = br.defaultclock.dt/br.second  
  
  	# scaling parameter to the true on of Brunel 2000
  N_true = 12500
  delay = syn_delay * br.ms                   # Synaptic delay
  J = Vm_jump_after_EPSP * downscale * br.mV  # jump of membrane potential after EPSP
  
  	# Network numbers parameters
  N = int(N_true / downscale)		# number of Neuron
  C = int(connectivity * N)		   # number of connexions per neuron
  sparseness = C/float(N)
  f_exc = 0.8                     # fraction of exitatory synapses
  Ce = int(C * f_exc)             # number of exitatory connexions
  NE = int(N * f_exc)             # Number of excitatory cells
  NI = N-NE                       # Number of inhibitory cells 
  CE_true = int(N_true * connectivity * f_exc)
  
  	# Neurons parameters
  tau = 20.0 * br.ms                 # membrane time constant
  mu_0 = 0. * br.mV                  # offset to the membrane
  Vr = mu_0 + 10.0 * br.mV           # potential of reset after spiking
  theta = mu_0 + 20.0 * br.mV  	      # Spiking threshold
  taurefr = 2. * br.ms               # time constant of refractory period
  
  	# Synapse parameters
  g = float(g)                    # relative strength between IPSP and EPSP
  
  	# parameter of external population
  Inp = float(inp)                # coefficient multiplying nu_theta
  nu_theta = theta/(Ce * J * tau) # minimal required input frequency for firing
  
  	# =============================================================================
  	# INITIALIZE NEURONS GROUP
  	# Main group
  eqs_neurons= br.Equations('''
  	dV/dt = (-V +  mu_0)/tau : volt (unless refractory)
  	''')
  Group=br.NeuronGroup(N, model=eqs_neurons,\
                       threshold='V>=theta', reset='V=Vr', \
                       refractory=taurefr, method='euler')
  Group.V = np.linspace(mu_0/br.mV-20,theta/br.mV,N)*br.mV
  
  
  	# external group
  
  	# =============================================================================
  	# SYNAPSES DEFINITION AND CONNEXIONS
  	# The implementation of connections come from ExcInhNet_Ostojic2014_Brunel2000_brian2.py 
  	# written by Aditya Gilra to connect the synapses.
  
  	# Main group synapses
  sparseness_e = Ce/float(NE)
  sparseness_i = (1-f_exc)*C/float(NI)
  con = br.Synapses(Group,Group,'w:volt',on_pre='V_post += w',method='euler')
  	# Connections from some Exc/Inh neurons to each neuron
  random.seed(seed) # set seed for reproducibility of simulations
  conn_i = []
  conn_j = []
  for j in range(0,N):
		# sample Ce number of neuron indices out of NE neurons
    preIdxsE = random.sample(range(NE),Ce)
    	# sample Ci=C-excC number of neuron indices out of inhibitory neurons
    preIdxsI = random.sample(range(NE,N),C-Ce)
		# connect these presynaptically to i-th post-synaptic neuron
		# choose the synapses object based on whether post-syn nrn is exc or inh
    conn_i += preIdxsE
    conn_j += [j]*Ce
    conn_i += preIdxsI
    conn_j += [j]*(C-Ce)
  con.connect(i=conn_i, j=conn_j)
  con.delay = delay
  con.w['i<NE'] = J
  con.w['i>=NE'] = -g*J

  
#  # save connectivity matrix of the network for post-analysis purpose
#  # safe to comment if you are not interested in this information     
#  struct = np.zeros((N,N))
#  for i,j in zip(conn_i, conn_j):
#    struct[i,j] = 1
#  utils.save('brunel_struct_seed={}.npy'.format(seed), struct)
  
  # EXTERNAL POPULATION
  #correct external population firing rate for downscaling
  if g > 4:
    approx_rate = ((Inp - 1) * nu_theta)/(g*0.25 - 1)/N
    rate_ext_0 =  Ce * Inp * nu_theta
    rate_balance = Ce * ((1/downscale) - 1) *\
                        (Inp * nu_theta + approx_rate*(1 + 0.25*g**2)) /\
                        (1 + g**2)
    rate_corrected = (rate_ext_0 + rate_balance)/Ce
  else:
    rate_corrected=Inp*nu_theta
  print(Inp*nu_theta, rate_corrected)
  
  PI = br.PoissonInput(Group, 'V', N=Ce, rate=rate_corrected, weight=J)	

	# =============================================================================
	# SIMULATION

	#np.random.seed(seed)
	#random.seed(seed)

	# Setting up monitors
  M = br.SpikeMonitor(Group)
  LFP = br.PopulationRateMonitor(Group)
  
  br.run(simtime,report='text')
  
  # saving population firing rate and spike trains
  fr = [LFP.t/br.ms, LFP.smooth_rate(window='flat', width=0.5*br.ms)/br.Hz]
  sp = M.spike_trains()
  br.device.reinit()
  return sp, fr


if __name__=='__main__':


  input_ = [1,2,3,4]
  g_ = [2,3,3.5,4,4.5,5,6,7,8]
  
  for inp in input_:
    for g in g_:
      for s in range(10):
        print('seed: ', s)
        sp, fr = Simulate(g, inp, seed=s)
        for k, d in sp.items():
          sp[k] = d/br.second
    
        d = {'sp': sp, 'fr': fr}
        np.save('save/brunel_inp={}_g={}_seed_{}'.format(inp, g, s), d)



    
