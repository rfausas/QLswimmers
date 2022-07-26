import numpy as np
import random
import copy

def prepare_action(params,rank):
   params.xcnref = copy.deepcopy(params.xcn)
   params.fluxperbodyaction = [0.0]*params.nballs
   #print_reward(dir_,evolfilerew,itl,N_BALLS,xcnref[0],total_reward)
   if rank == 0: print_states(params)
   
   # Learning decisions #
   params.eps_greedy = params.min_eps + (params.max_eps - params.min_eps)*np.exp(-params.lambd * params.ita)
   params.action = params.actionimp[params.ita%params.actioncount] if params.impqlaction else choose_action(params)
   params.next_state = compute_next_state(params)   #params.current_state
   params.ita += 1
   #current_reward = 0.0
   
   
def choose_action(params):
   if(random.random() < (1-params.eps_greedy)):
      current_state_dec = np.dot(params.current_state,2**np.arange(params.nlinks-1,-1,-1,dtype=int))
      aux = np.zeros(params.nlinks, dtype=int)
      aux[np.argmax(params.MQL[current_state_dec,:])] = 1
      return aux
   else:
      aux = np.identity(params.nlinks, dtype=int)
      return aux[random.randint(0, params.nlinks-1),:]
   
   
def compute_next_state(params):
   return np.remainder(params.current_state+params.action,2*np.ones(params.nlinks, dtype=int))


def update_MQL(params,rank):
   current_state_dec = np.dot(params.current_state,2**np.arange(params.nlinks-1,-1,-1,dtype=int))
   next_state_dec    = np.dot(params.next_state,2**np.arange(params.nlinks-1,-1,-1,dtype=int))
   print(next_state_dec)
   next_MQLmax       = np.max(params.MQL[next_state_dec,:])
   current_MQLval    = params.MQL[current_state_dec,np.argmax(params.action)]
   params.current_reward = compute_reward(params)
   params.total_reward += params.current_reward

   params.MQL[current_state_dec,np.argmax(params.action)] = current_MQLval + \
      params.alpha*(params.current_reward + params.gamma*next_MQLmax - current_MQLval)
      
   if rank == 0:
      params.MQLdiffnorm = np.linalg.norm(params.MQL - params.MQLn)
      print('\n   ::> Norm of difference MQL-MQLn =', params.MQLdiffnorm,'\n')
      print('Total rewards at step %d = %g, Eps = %g' %(params.ita, params.total_reward, params.eps_greedy))
      print('\n', params.MQL, '\n')
      
   params.MQLn = copy.deepcopy(params.MQL)
   params.current_state = copy.deepcopy(params.next_state)


def compute_reward(params):
   r = 0.0
   if params.rewardtype == 0:
      r = compute_centroid_displacement(params)
   elif params.rewardtype == 1: #params.solvefood:
      for ig in range(1+0*len(params.fluxperbody)):
         r += params.fluxperbody[ig]
   else:
      for ig in range(1+0*len(params.fluxperbody)):
         r += params.fluxperbodyaction[ig]      

   return r


def compute_centroid_displacement(params):
   if True:
      r = 0.0
      for ib in range(params.nballs):
         r += params.xc[0][ib][0]-params.xcnref[0][ib][0] + params.xc[0][ib][1]-params.xcnref[0][ib][1]
      r = r/params.nballs
   else:
      r = np.array([0.,0.,0.])
      for ib in range(params.nballs):
         r += params.xc[ib]-params.xcn[ib]
      r = np.linalg.norm(r)/params.nballs
      
   return r


###############################################################################
def print_states(params):
   file_txt = open(params.dir_+params.evolstatefile, 'a+')
   #file_txt.write("%.5e " % (params.time))
   for i in range(params.current_state.shape[0]-1):
      file_txt.write("%d " % (params.current_state[i]) )
   file_txt.write("%d" % (params.current_state[params.current_state.shape[0]-1]) )
   
   #file_txt.write(" %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e" % (MQlearning[0,0],MQlearning[0,1],
   #                                                                    MQlearning[1,1],MQlearning[1,2],
   #                                                                    MQlearning[2,2],MQlearning[2,3],
   #                                                                    MQlearning[3,0],MQlearning[3,3]) )
   file_txt.write("\n")
   file_txt.close()
   
   file_txt = open(params.dir_+params.evolrewfile, 'a+')
   file_txt.write("%.5e " % (params.time))
   #for ig in range(params.nballs):
   #   file_txt.write(" %.10e " % (xc[ig][0]) )
   file_txt.write("%.10e " % (params.total_reward) )
   file_txt.write("%.10e" %(params.MQLdiffnorm))
   file_txt.write("\n")
   file_txt.close()
   
   if params.printMQL:
      file_txt = open(params.dir_+params.evolqmatrix, 'a+')
      for i in range(params.nstates):
         for j in range(params.nactions):
            file_txt.write("%.10e " %(params.MQL[i,j]))
      file_txt.write("\n")
      file_txt.close()   
   
