import numpy as np
import copy


def integrate_translation(params,dt,ig,ib):
   xcn = params.xcn[ig][ib]
   vcn = params.vcn[ig][ib]
   vc  = params.vc[ig][ib]
   xc  = xcn + 0.5*dt*(vcn+vc)
   # if params.fp == 0:
   #    xc = xcn + dt*vcn
   # elif params.pcmet == 1 and params.fp == 1:
   #    xc = xcn + 0.5*dt*(vcn+vc)
   return xc


def integrate_rotation(params,dt,ig,ib):
   thn = params.thn[ig][ib]
   omn = params.omn[ig][ib][2]
   om  = params.om[ig][ib][2]
   th  = thn + 0.5*dt*(omn+om) 
   # if params.fp == 0:
   #    th = thn + dt*omn
   # elif params.pcmet == 1 and params.fp == 1:
   #    th = thn + 0.5*dt*(omn+om) 
   return th


def prepare_next_time_step(params,it):
   params.xcn = copy.deepcopy(params.xc)
   params.thn = copy.deepcopy(params.th)
   params.Qrn = copy.deepcopy(params.Qr)
   params.time = params.time + params.dt #(1-params.thmetcon)*params.dt
   params.VF = params.VF + params.movingframe*(params.dt/params.taud)*\
               np.array([params.vcn[0][0][0],params.vcn[0][0][1]])
   params.it = it+1
   
   
def copy_to_n(params):
   params.xcn = copy.deepcopy(params.xc)
   params.vcn = copy.deepcopy(params.vc)
   params.thn = copy.deepcopy(params.th)
   params.omn = copy.deepcopy(params.om)
   params.Qrn = copy.deepcopy(params.Qr)