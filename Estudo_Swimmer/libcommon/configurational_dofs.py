import numpy as np

###############################################################################
pi = np.pi
ome = 2*pi
pha = 90.0*pi/180.0
alp = 0.0  #.65#.75
lmax = 3.0 #1 #25
lmin = (1-alp)*lmax
eps = alp*lmax
cas = 0
P = 4.0
amin = -15.*pi/180.
amax = +15.*pi/180.

###############################################################################
def DecimalToBinary(num,A):
    if num > 1:
        A = DecimalToBinary(num // 2,A)
    A.append(num % 2)
    return A


def LLINK(params,t,Nl):
   if params.exactqc or np.abs(int(round(t/params.dt))-t/params.dt) < 1e-8:
      d = LLINKFORM(params,t,Nl)
   else:
      d = 0.5*LLINKFORM(params,t-0.5*params.dt,Nl)\
         +0.5*LLINKFORM(params,t+0.5*params.dt,Nl)
   
   return d   


def LLINKFORM(params,t,Nl):
   if True and params.solveQL:
      tP = t/params.taction - np.floor(t/params.taction)
      if (np.abs(int(round(t/params.taction))-t/params.taction) < 1e-8):
         if params.it%int(params.taction/params.dt) == 0:
            tP = 0.0
         else:
            tP = 1.0
      next_state = np.remainder(params.current_state+params.action,2*np.ones(params.nlinks, dtype=int))
      l0 = lmin + (lmax-lmin)*params.current_state
      l1 = lmin + (lmax-lmin)*next_state
      l  = l0 + tP*(l1-l0)
      return l[Nl-1]
   else:
      epm = params.dt/10**2
      tP = t - np.floor(t/P)*P
      if (np.abs(int(round(t/P))-t/P) < 1e-8):
         tP = 0.0
      d = 0
      if (cas == 0):
         d = lmax
      elif (cas == 1):
         d = +lmax*((1-alp) + (alp/2.0) * (np.cos(ome*t + 1*(Nl-1)*pha) + 1.0))
      elif (cas == 2):
         if (Nl-1 == 0):
           if  ((          0.0 <= tP)and(tP < P/4.0-epm    )):
              d = lmax - eps/(P/4.0)*tP
           elif((    P/4.0-epm <= tP)and(tP < P/2.0-epm    )):
              d = lmin
           elif((    P/2.0-epm <= tP)and(tP < 3.0*P/4.0-epm)):
              d = lmin + eps/(P/4.0)*(tP-P/2.0)
           elif((3.0*P/4.0-epm <= tP)and(tP < P            )):
              d = lmax
         elif (Nl-1 == 1):
           if  ((          0.0 <= tP)and(tP < P/4.0-epm    )):
              d = lmax
           elif((    P/4.0-epm <= tP)and(tP < P/2.0-epm    )):
              d = lmax - eps/(P/4.0)*(tP-P/4.0)
           elif((    P/2.0-epm <= tP)and(tP < 3.0*P/4.0-epm)):
              d = lmin
           elif((3.0*P/4.0-epm <= tP)and(tP < P            )):
              d = lmin + eps/(P/4.0)*(tP-3.0*P/4.0)
         else:
            d = lmax

      return d


def DLLINK(params,t,Nl):
   if params.exactdqc:# or np.abs(int(t/params.dt)-t/params.dt) < 1e-8: #t==0
      if params.fp == 0:
         d = DLLINKFORM(params,t,Nl)
      elif int(params.thmetstk) == 1:
         d = DLLINKFORM(params,t+params.dt,Nl)
         if ( t > 0 and t < 2 and (np.abs(int(round(t))-t) < 1e-8)):
            d = (LLINK(params,t+params.dt,Nl)-LLINK(params,t,Nl))/params.dt
      else:
         d = DLLINKFORM(params,t+params.dt/2,Nl)
   else:
      if False and params.solveQL and params.it%int(params.taction/params.dt) == int(params.taction/params.dt)-1:
         d = (LLINK(params,t,Nl)-LLINK(params,t-params.dt,Nl))/params.dt
      else:
         d = (LLINK(params,t+params.dt,Nl)-LLINK(params,t,Nl))/params.dt
   #print(d)
   return d   


def DLLINKFORM(params,t,Nl):
   if True and params.solveQL:
      next_state = np.remainder(params.current_state+params.action,2*np.ones(params.nlinks, dtype=int))
      l0 = lmin + (lmax-lmin)*params.current_state
      l1 = lmin + (lmax-lmin)*next_state
      l  = l1-l0
      return l[Nl-1]
   else:
      epm = params.dt/10**2
      tP = t - np.floor(t/P)*P
      if (np.abs(int(round(t/P))-t/P) < 1e-8):
         tP = 0.0
      d = 0
      if (cas == 0):
         d = 0
      elif (cas == 1):
         d = -lmax*(alp/2.0)*ome*np.sin(ome*t + 1*(Nl-1)*pha)
      elif (cas == 2):   
         if (Nl-1 == 0):
           if  ((          0.0 <= tP)and(tP <  P/4.0-epm    )):
              d = - eps/(P/4.0)
           elif((    P/4.0-epm <= tP)and(tP <  P/2.0-epm    )):
              d = 0.0
           elif((    P/2.0-epm <= tP)and(tP <  3.0*P/4.0-epm)):
              d = + eps/(P/4.0)
           elif((3.0*P/4.0-epm <= tP)and(tP <  P            )):
              d = 0.0
         elif (Nl-1 == 1):
           if  ((          0.0 <= tP)and(tP <  P/4.0-epm    )):
              d = 0.0
           elif((    P/4.0-epm <= tP)and(tP <  P/2.0-epm    )):
              d = - eps/(P/4.0)
           elif((    P/2.0-epm <= tP)and(tP <  3.0*P/4.0-epm)):
              d = 0.0
           elif((3.0*P/4.0-epm <= tP)and(tP <  P            )):
              d = + eps/(P/4.0)
         else:
            d = 0.0

      return d
   

def BLINK(params,t,Nl):
   if params.isaxis: return 0.
   
   if params.exactqc or np.abs(int(round(t/params.dt))-t/params.dt) < 1e-8:
      d = BLINKFORM(params,t,Nl)
   else:
      d = 0.5*BLINKFORM(params,t-0.5*params.dt,Nl)\
         +0.5*BLINKFORM(params,t+0.5*params.dt,Nl)
   
   return d
   
   
def BLINKFORM(params,t,Nl):
   if params.solveQL:
      tP = t/params.taction - np.floor(t/params.taction)
      if (np.abs(int(round(t/params.taction))-t/params.taction) < 1e-8):
         if params.it%int(params.taction/params.dt) == 0:
            tP = 0.0
         else:
            tP = 1.0
      next_state = np.remainder(params.current_state+params.action,2*np.ones(params.nlinks, dtype=int))
      a0 = amin + (amax-amin)*params.current_state
      a1 = amin + (amax-amin)*next_state
      a  = a0 + tP*(a1-a0) #0.5*(1.-np.cos(tP*np.pi))*(a1-a0) #  
      return a[Nl-1]
   elif True:
      #t = 0
      if ( t <= 1 ):
         d = 1 * amax * np.sin(ome*t + (Nl-1)*2.0*pi/params.nlinks)
      else:
         d = 1 * amax * np.sin(ome*(2-t) + (Nl-1)*2.0*pi/params.nlinks)
      
      '''
      if (Nl-1 == 1):
         d = np.pi/3.0 * cos(ome*t - pi/2.0 + 0*(Nl-1)*pha)
      elif (Nl-1 == 3):
         d = np.pi/3.0 * cos(ome*t + pi/2.0 + 0*(Nl-1)*pha)
      '''
      return d
   else:
      eps = amax-amin
      epm = params.dt/10**2
      tP = t - np.floor(t/P)*P
      if (np.abs(int(round(t/P))-t/P) < 1e-8):
         tP = 0.0
      d = 0
      if (Nl-1 == 0):
        if  ((          0.0 <= tP)and(tP < P/4.0-epm    )):
           d = amax - eps/(P/4.0)*tP
        elif((    P/4.0-epm <= tP)and(tP < P/2.0-epm    )):
           d = amin
        elif((    P/2.0-epm <= tP)and(tP < 3.0*P/4.0-epm)):
           d = amin + eps/(P/4.0)*(tP-P/2.0)
        elif((3.0*P/4.0-epm <= tP)and(tP < P            )):
           d = amax
      elif (Nl-1 == 1):
        if  ((          0.0 <= tP)and(tP < P/4.0-epm    )):
           d = amax
        elif((    P/4.0-epm <= tP)and(tP < P/2.0-epm    )):
           d = amax - eps/(P/4.0)*(tP-P/4.0)
        elif((    P/2.0-epm <= tP)and(tP < 3.0*P/4.0-epm)):
           d = amin
        elif((3.0*P/4.0-epm <= tP)and(tP < P            )):
           d = amin + eps/(P/4.0)*(tP-3.0*P/4.0)
           
      return d
   

def DBLINK(params,t,Nl):
   if params.isaxis: return 0.
   
   if params.exactdqc:# or np.abs(int(t/params.dt)-t/params.dt) < 1e-8: #t==0
      if params.fp == 0:
         d = DBLINKFORM(params,t,Nl)
      elif int(params.thmetstk) == 1:
         d = DBLINKFORM(params,t+params.dt,Nl)
         if ( t > 0 and t < 2 and (np.abs(int(round(t))-t) < 1e-8)):
            d = (BLINK(params,t+params.dt,Nl)-BLINK(params,t,Nl))/params.dt
      else:
         d = DBLINKFORM(params,t+params.dt/2,Nl)
   else:
      if False and params.solveQL and params.it%int(params.taction/params.dt) == int(params.taction/params.dt)-1:
         d = (BLINK(params,t,Nl)-BLINK(params,t-params.dt,Nl))/params.dt
      else:
         d = (BLINK(params,t+params.dt,Nl)-BLINK(params,t,Nl))/params.dt
   #print(d)
   return d


def DBLINKFORM(params,t,Nl):
   #return 0.0
   if params.solveQL:
      next_state = np.remainder(params.current_state+params.action,2*np.ones(params.nlinks, dtype=int))
      a0 = amin + (amax-amin)*params.current_state
      a1 = amin + (amax-amin)*next_state
      a  = a1-a0
      
      # a = pow(-1,params.current_state[Nl-1]+1)*amax*(pi/params.taction)*(-1)*np.sin(pi*tP/params.taction)*params.action
      return a[Nl-1]
   elif True:
      if ( t <= 1 ):
         d = 1 * amax * ome * np.cos(ome*t + (Nl-1)*2.0*pi/params.nlinks)
      else:
         d = 1 * amax *-ome * np.cos(ome*(2-t) + (Nl-1)*2.0*pi/params.nlinks)
      '''
      if (Nl-1 == 1):
         d = - np.pi/3.0 * ome * sin(ome*t - pi/2.0 + 0*(Nl-1)*pha)
      elif (Nl-1 == 3):
         d = - np.pi/3.0 * ome * sin(ome*t + pi/2.0 + 0*(Nl-1)*pha)
      '''
      return d
   
   elif False:
      eps = amax-amin
      epm = params.dt/10**2
      tP = t - np.floor(t/P)*P
      if (np.abs(int(round(t/P))-t/P) < 1e-8):
         tP = 0.0
      d = 0
      if (Nl-1 == 0):
        if  ((          0.0 <= tP)and(tP <  P/4.0-epm    )):
           d = - eps/(P/4.0)
        elif((    P/4.0-epm <= tP)and(tP <  P/2.0-epm    )):
           d = 0.0
        elif((    P/2.0-epm <= tP)and(tP <  3.0*P/4.0-epm)):
           d = + eps/(P/4.0)
        elif((3.0*P/4.0-epm <= tP)and(tP <  P            )):
           d = 0.0
      elif (Nl-1 == 1):
        if  ((          0.0 <= tP)and(tP <  P/4.0-epm    )):
           d = 0.0
        elif((    P/4.0-epm <= tP)and(tP <  P/2.0-epm    )):
           d = - eps/(P/4.0)
        elif((    P/2.0-epm <= tP)and(tP <  3.0*P/4.0-epm)):
           d = 0.0
        elif((3.0*P/4.0-epm <= tP)and(tP <  P            )):
           d = + eps/(P/4.0)
      else:
         d = 0.0

      return d
