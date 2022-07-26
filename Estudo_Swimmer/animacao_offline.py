#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 12:25:58 2022

@author: luciano
"""
import numpy as np
import matplotlib.pyplot as plt


def Animacao(nballs,current_state_dec,filetable):

    def ReadDeltasTables(ndeltas, nstates, nactions, filetable):            
        
      A = np.loadtxt(filetable)
      Deltas = []
      for d in range(ndeltas):
          Deltas.append(np.zeros(shape=(nstates, nactions)))
          for state in range(nstates):
              Deltas[d][state, :] = A[d * nstates + state, :]
      return Deltas  
    
    def rotation_matrix(angle):
       Q = np.identity(3)
       Q[0,0] =  np.cos(angle)
       Q[0,1] = -np.sin(angle)
       Q[1,0] =  np.sin(angle)
       Q[1,1] =  np.cos(angle)
       return Q
    
    def BLINKFORM(t,Nl,current_state,action):   
        tP = t/taction - np.floor(t/taction)
        if (np.abs(int(round(t/taction))-t/taction) < dt/10):
           if it%int(taction/dt) == 0:
              tP = 0.0
           else:
              tP = 1.0
        next_state = np.remainder(current_state+action,2*np.ones(nlinks, dtype=int))
        a0 = amin + (amax-amin)*current_state
        a1 = amin + (amax-amin)*next_state
        a  = a0 + tP*(a1-a0) 
        return a[Nl-1]
    
    def BLINK(t,Nl,current_state,action):   
        d = BLINKFORM(t,Nl,current_state,action)   
        return d
           
    nlinks = nballs - 1
    ndeltas = 6
    nstates = 2 ** nlinks
    nactions = nlinks
    ll = 3.0
    Lx = 450
    Ly = 450    
    
    Deltas = ReadDeltasTables(ndeltas, nstates, nactions, filetable)
    current_disp = np.zeros(2)

    freq_frame = 100
    
    e_ref = np.array([1,0,0])
    it = 0
    time = 0
    dt = 0.1
    amin = -45*np.pi/180
    amax = 45*np.pi/180
    taction = 1.0
    radii = [1.0 for ib in range(nballs)]
    radih = [0.2 for ib in range(nballs)]
    radii[0] = 1.5
    radih[0] = 0.25
    xc = [np.array([Lx/2.,Ly/2.,0.]) for ib in range(nballs)]
    th = [0. for ib in range(nballs)]
    th[0] = np.pi/4 #angulo inicial da cabeça
    Qr = [rotation_matrix(th[ib]) for ib in range(nballs)]
       
    #Lista de ações em decimais para ser executada
    alist = []    
    A = np.loadtxt('actions.txt')
    for i in range(len(A)):
        alist.append(int(A[i]))
    
    count = 0
    for it in range(len(alist)):
        
        current_state = np.array([int(i) for i in np.binary_repr(current_state_dec,width=nlinks)])    
        
        
        for ib in range(1,nballs): # Integrating with positional mapping
            th[ib] = th[ib-1] + BLINK(time,ib,current_state,np.zeros(nlinks))
            Qr[ib] = rotation_matrix(th[ib])
            xc[ib] = xc[ib-1] + 0.5*ll*(np.dot(Qr[ib-1],e_ref)+ np.dot(Qr[ib],e_ref))
        
        h = 0.1
        for ib in range(nballs):
            theta = np.arange(0.0, 2*np.pi+h, h)
            X = xc[ib][0] + Qr[ib][0,0]*radii[ib]*np.cos(theta) + Qr[ib][0,1]*radih[ib]*np.sin(theta)
            Y = xc[ib][1] + Qr[ib][1,0]*radii[ib]*np.cos(theta) + Qr[ib][1,1]*radih[ib]*np.sin(theta)
            color = 'gray' if ib != 0 else 'red'
            if(it%freq_frame == 0):
                plt.plot(X,Y,color=color)      
        
        if(it%freq_frame == 0):
            plt.xlim(150,300)
            plt.ylim(150,300)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.savefig("png/4b%06d.png" %(count), dpi=150)       
            plt.clf()
            count += 1
        
        action_dec = alist[it]
        action = np.zeros(nlinks)
        action[action_dec] = 1    
        
        current_disp[0] = Deltas[4][current_state_dec,action_dec]
        current_disp[1] = Deltas[5][current_state_dec,action_dec]
        
        current_dtheta  = Deltas[2][current_state_dec,action_dec]
        
        Rmatrix = np.array([[np.cos(th[0]), -np.sin(th[0])],
                            [np.sin(th[0]),  np.cos(th[0])]])
        
        xcaux = np.array([xc[0][0],xc[0][1]])
        xcaux += Rmatrix @ current_disp*1 
        xc[0][0] = xcaux[0]
        xc[0][1] = xcaux[1]
        th[0] += current_dtheta*1
        Qr[0] = rotation_matrix(th[0])
        
        current_state = np.remainder(current_state+action,2*np.ones(nlinks, dtype=int))
        current_state_dec = int(np.dot(current_state,2**np.arange(nlinks-1,-1,-1,dtype=int)))      

nballs = 4
stinicial = 0 #estado inicial dos links dos corpos
ang = 10
dir_ = './DELTAS/DELTASfino/theta'+str(ang)+'/deltas.txt'
Animacao(nballs,stinicial,dir_)
