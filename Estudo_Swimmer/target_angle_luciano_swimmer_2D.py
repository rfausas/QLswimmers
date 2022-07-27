#from dolfin import *

import numpy as np
import copy
import random
import time as tim

#from libcommon.user_utils import *
#from libcommon.build_geometry import *
#from libcommon.configurational_dofs import *
#from libcommon.variational_formulation import *
from libcommon.ql import *

import psutil
import itertools
import math

# Once and for all (do not mix np.random and random)
np.random.seed()

# Global arrays
chosen_actions = []
visited_states = []

def ComputeStep(nballg, refbody, reffar, dt, state_dec, state_bin, action):

   comm = MPI.comm_world
   self = MPI.comm_self
   size = comm.Get_size()
   rank = comm.Get_rank()
   set_log_active(False) # dolfin verbose
   
   ###############################################################################
   # USER DEFINED PARAMETERS
   params = Params()
   
   # IO
   params.fileio = 'pvd'
   params.dir_   = './FEniCS2022'
   
   # Dimensions
   params.isaxis = 0
   params.nsd = 2 # space dimension
   params.npd = 3 if not params.isaxis else 1 # number positional dofs per body group reference
   params.nbg = 1 # number of body groups
   
   # Swimmers type: circle, square, ellipse
   params.swmtype = 'ellipse'
   
   # Number of immersed balls
   params.nballg = [nballg for ib in range(params.nbg)] # number of bodies per body group
   params.nballs = sum(params.nballg)
   
   # Computational domain
   params.Lx   = 3*150. #1500.0 #20 #1500.0
   params.Ly   = 3*150. #1500.0 #20 #1500.0
   params.xmin = 0.0
   params.ymin = 0.0
   
   # Balls initial dofs
   params.xc  = [[] for ig in range(params.nbg)]
   params.vc  = [[] for ig in range(params.nbg)]
   params.th  = [[] for ig in range(params.nbg)]
   params.om  = [[] for ig in range(params.nbg)]
   params.Qr  = [[] for ig in range(params.nbg)]
   params.xcn = [[] for ig in range(params.nbg)]
   params.vcn = [[] for ig in range(params.nbg)]
   params.thn = [[] for ig in range(params.nbg)]
   params.omn = [[] for ig in range(params.nbg)]
   params.Qrn = [[] for ig in range(params.nbg)]
   
   params.radii  = [[] for ig in range(params.nbg)] # major axis
   params.radih  = [[] for ig in range(params.nbg)] # minor axis
   params.e_ref  = [[] for ig in range(params.nbg)] # links directors
   params.MM     = [[] for ig in range(params.nbg)] # bodies masses
   
   for ig in range(params.nbg):
      params.xc[ig] = [np.array([params.Lx/2.,params.Ly/2.+4*ig,0.]) for ib in range(params.nballg[ig])]
      
      params.vc[ig] = [np.array([0.,0.,0.]) for ib in range(params.nballg[ig])]
      params.th[ig] = [0. for ib in range(params.nballg[ig])]
      params.om[ig] = [np.array([0.,0.,0.]) for ib in range(params.nballg[ig])]
      params.Qr[ig] = [rotation_matrix(params.th[ig][ib]) for ib in range(params.nballg[ig])]
   
      params.radii[ig] = [1.0 + 0*3.0+0*0.1+(2-ib)*0.0 for ib in range(params.nballg[ig])]
      params.radih[ig] = [0*1.0+1*0.2 + 0*0.2 for ib in range(params.nballg[ig])]
      if(True):
         params.radih[0][0] = 0.25
         params.radii[0][0] = 1.5
         
      params.e_ref[ig] = [np.array([1.,0.,0.]) for ib in range(params.nballg[ig])]
      params.MM[ig]    = [Constant(1.0) for ib in range(params.nballg[ig])]
   
      if params.isaxis:  params.xc[ig][0][1] = 0.
   
   params.xcn = copy.deepcopy(params.xc)
   params.vcn = copy.deepcopy(params.vc)
   params.thn = copy.deepcopy(params.th)
   params.omn = copy.deepcopy(params.om)
   params.Qrn = copy.deepcopy(params.Qr)
   
   # Finite elements
   params.orderU = 1
   params.orderP = 1
   
   # Solver
   params.stokes_system_as_linear = False # for debugging
   params.exactqc = False
   params.exactdqc = params.exactqc
   
   # Body force on fluid
   params.ff = Constant((0.0,0.0))
   
   # Slip condition
   params.noslip = Constant((0, 0))
   
   # Moving frame
   params.VF   = np.array([0.,0.])
   params.taud = 10.0
   params.movingframe = False
   
   # Time step ###################################################################
   params.dt       = dt
   if(len(chosen_actions)):
      nacts_policy = len(chosen_actions)
   else:
      nacts_policy = 1
      aux = np.zeros(nballg-1, dtype=int)
      aux[action] = 1
      params.actionimp = [aux]
      chosen_actions.append(np.array(aux))
      visited_states.append(np.array(state_bin))

   params.freq_out = 1
   params.taction = 1.0 # Each action take 1unit of time
   params.nsteps   = nacts_policy*int(params.taction/params.dt) 

   #if(len(chosen_actions) == 1):
   #   params.nsteps   -= 1
      
   params.nfp      = 2 #2 #if params.pcmet == 1 else 1 # fixed point iterations
   params.thmetstk = 0.5 #0.5 #0.5 to set RK2MP, 1 to set RK2ME, these need nfp >= 2
   params.thmetcon = params.thmetstk
   params.fp       = 0
   
   # Print to file options
   params.print_solutions = True
   params.print_dofs      = True
   
   # Exact solution to test convergences
   params.compute_errors = False
   if params.compute_errors:
      pass
   else:
      uexact = params.noslip
      pexact = Constant(0)
      
   # Mesh control ################################################################
   meshqual           = [0,0]
   meshquallim        = 0.4
   params.remeshingm  = True
   params.adaptmesh   = True
   params.gmshverbose = False
   params.lcar        = 150./2**(reffar)
   params.lbod        = 0.3/2**(refbody)
   params.lcor        = params.lcar
   params.laxi        = params.lbod*1
   params.btagini     = -1 # don't change
   params.it          = 0 # don't change
   mesh,mesh_facets,mesh_physic = [],[],[]
   
   # Compute physical quantities
   compute_forces = False
   params.visdiss = 0.0
   params.acc_visdiss = 0.0
   
   # Food transport problem ######################################################
   params.solvefood = False
   params.print_concentration = True
   params.Dfood = Constant(0.04)
   params.Rfood = Constant(0.0)
   params.Sfood = Constant(0.0)
   #sfexp = "1.0*exp(-pow(x[0]-(Lx/2),2)/2 - pow(x[1]-(3*Ly/5),2)/2)"
   #Sfood = Expression(sfexp, Lx=Lx, Ly=Ly, degree=4)
   params.Alphaeat  = Constant(1.0)
   params.Geat = Constant(0.0)
   params.fcpestab = 'SUPG'
   params.cestab = Constant(0.15) #Constant(0.015)
   params.iniFood = Constant(1.0) #initial_FoodC()
   #eaten = 0.5
   #net_displ_cm = 0.0
   params.fluxperbody = []
   params.intFoodCA = 0.0
   params.intFoodCG = 0.0
   params.intFoodCn = 0.0
   params.eatingbody = 4 # nummber of eating parts in each swimmer
   params.impzeroconvel = True
   params.dtdiffp       = 1 #50*params.dt
   params.foodininsteps = 100 #*int(params.dtdiffp) 
   
   # Q-Learning ##################################################################
   params.solveQL = 1 # 0 to ignore QL, 1 for QL, 2 for DQL
   params.nballs = params.nballs
   params.nlinks = params.nballg[0] - 1
   params.gamma, params.alpha, params.lambd = 0.8, 1.0, 0.003
   params.max_eps, params.min_eps, params.eps_greedy = 0.9, 0.1, 0.0
   params.taction = 1.0
   params.nstates, params.nactions, params.current_state_dec = (2**params.nlinks, params.nlinks, 0) if params.solveQL != 0 else (2, 1, 0)
   params.MQL, params.MQLn, params.MQLdiffnorm = np.zeros((params.nstates,params.nactions)), np.zeros((params.nstates,params.nactions)), 0.0
   params.ita = 0 # don't change

   #----------------
   # Set the action
   params.impqlaction  = True # True if basic action is imposed, False for QL

   # Examples:
   #params.actionimp = [np.array([1,0,0,0]),np.array([0,1,0,0]),np.array([0,0,1,0]),np.array([0,0,0,1])] # 4 balls, 3 links, 3 actions
   #params.actionimp = [np.array([0,1]),np.array([1,0])] # 3 balls, 2 links, 2 actions
   #params.actionimp = [np.array([0,1,0]), np.array([1,0,0])] # 4 balls, 3 links, 2 actions
   #params.actionimp = [np.array([0,1,0])] # 4 balls, 3 links, 1 actions
   aux = np.zeros(params.nlinks, dtype=int)
   aux[action] = 1
   params.actionimp = [aux]   
   params.action    = params.actionimp[0] if params.impqlaction else np.random.randint(2, size=params.nlinks)
   
   #----------------
   # Set the state
   # Examples:
   #params.current_state = 1*np.ones(params.nlinks, dtype=int) if True else np.random.randint(2, size=params.nlinks)
   #params.current_state = np.zeros(params.nlinks, dtype=int) if True else np.random.randint(2, size=params.nlinks)
   params.current_state = np.array(state_bin)
   params.next_state    = np.ones(params.nlinks, dtype=int) # not needed at this point
   params.total_reward  = 0.0
   params.actioncount   = len(params.actionimp)
   params.printMQL      = False
   params.rewardtype    = 0 # 0 for positional, 1 for final foodc , 2 for foodc in a whole action
   
   # Auxiliary zero vectors
   waux        = Constant([0.0 for i in range(params.nsd+1+0+params.nbg*params.npd)])
   params.vaux = Constant([0.0 for i in range(params.nsd)])
   #params.caux = Constant([0.0 for i in range(1)])
   if(params.nsd == 1):
      params.eaux = [Constant(1.),Constant(0.),Constant(0.)]
   elif(params.nsd == 2):
      params.eaux = [Constant((1.,0.)),Constant((0.,1.)),Constant((0.,0.))]
   else:
      params.eaux = [Constant((1.,0.,0.)),Constant((0.,1.,0.)),Constant((0.,0.,1.))]
   
   ###############################################################################
   # IO
   IO_definitions(params,rank)
   pid = os.getpid()
   memory_info = open(params.dir_+'/memoryinfo.txt','w')
   
   ###############################################################################
   # Fluid finite elements
   EU = VectorElement("Lagrange", 'triangle', params.orderU)
   EP = FiniteElement("Lagrange", 'triangle', params.orderP)
   ES = VectorElement("Real", 'triangle', 0, params.nbg*params.npd)
   EM = MixedElement([EU,EP,ES])
   
   ###############################################################################
   # Food concentration finite elements
   params.orderC = 2
   EC = FiniteElement("Lagrange", 'triangle', params.orderC)
   
   ###############################################################################
   # Compatible inital geometry at time 0+thetastokes*dt
   params.time = 0.0
   
   #for ig in range(params.nbg):
   update_dependent_dofs(params,params.time) #+0*params.thmetstk*params.dt)
   copy_to_n(params)
   
   # Domain meshing  #############################################################
   mesh, mesh_facets, meshqual = load_geometry(params,mesh,mesh_facets,meshqual,rank,size,comm,True)
   params.W = FunctionSpace(mesh, EM)
   
   ###############################################################################
   # Initial condition for food transport problem
   if params.solvefood:
      if rank == 0: print('\n   ::> Computing initial condition for transport problem...\n')
      FoodCn = SolveFoodDiffusiveProblem(params,mesh,mesh_facets,EC,EU,1)
      compute_foodc_quantities(params,FoodCn,mesh,mesh_facets)
      if rank == 0: print_food_info(params)
      
   ###############################################################################
   # Time stepping
   startTime = tim.time() #datetime.now()
   if rank == 0: print('\n   ::> Begin loop over time steps')

   # Save initial configuration to compute deltas
   params.xcini = copy.deepcopy(params.xc)
   params.thini = copy.deepcopy(params.th)
   
   xcm0 = np.array([0.0, 0.0, 0.0])
   params.xcini = copy.deepcopy(params.xc)
   for i in range(params.nballg[0]):
      xcm0 += params.xcini[0][i]/params.nballg[0]

   for it in range(params.nsteps):
      if rank == 0:
         print("\n##################################################")
         print("\n    |-Time #%.5e step #%5d begins" %(params.time,it))

      if params.solveQL and it%int(params.taction/params.dt) == 0:
          params.action = chosen_actions[params.ita]
          params.current_state = visited_states[params.ita]
          current_state_dec = int(np.dot(params.current_state,2**np.arange(nlinks-1,-1,-1,dtype=int)))
          params.ita += 1
          print('    |> Solving for state= ', current_state_dec, '>', params.current_state, ', action= ', params.action)
   
      # Initial shoot ############################################################
      if (it == 0): 
         w = interpolate(waux, params.W) # initial shoot for nonlinear problem
   
      if rank == 0: print("\n    |-Fixed point iteration", end = ' ')
      for fp in range(params.nfp): # Fixed point iteration #######################
         params.fp = fp
         if rank == 0: print("%3d" %(fp), end = '\n' if fp == params.nfp-1 else ' ')
         
         # Compute auxiliary phi functions #######################################
         phi = compute_phi_bodies(params,params.W,mesh_facets)
         
         # Fluid-Solid Interaction Problem #######################################
         w = SolveFSIProblem(params,mesh,mesh_facets,params.W,w,phi)
         
         # Split and recover solution ############################################
         (u, p, s) = w.split(True)
         if fp == 0: #params.nfp-1:       
            # Plotting current time ##############################################
            if params.print_solutions and it%params.freq_out == 0:
               ufinal, pfinal, visfin = recover_FSI_solution(params,u,p,s,phi)
               print_solutions_user(params,mesh,ufinal,pfinal,visfin)
            
         # Updating body's dofs to time n+thmetstk ###############################
         update_positional_dofs(params,mesh,s,params.dt)
         if int(params.thmetstk) == 1 or fp == params.nfp-1: update_dependent_dofs(params,params.time+params.dt)
         #print(params.xc[0][0][0],params.xc[0][0][1])
         if fp < params.nfp-1: #params.pcmet == 1 and fp < params.nfp-1:
            if int(params.thmetstk) != 1:
               update_positional_dofs_midpoint(params)
               update_dependent_dofs(params,params.time+0.5*params.dt)
            if fp > 0: # ALE mesh movement #######################################
               meshdisp.vector()[:] = -meshdisp.vector()[:] #negmeshdisp = project(-meshdisp,VectorFunctionSpace(mesh, "Lagrange", 1),solver_type="mumps")
               ALE.move(mesh, meshdisp)
               mesh.bounding_box_tree().build(mesh)
            meshdisp = ALE_mesh_movement(params,mesh,mesh_facets)
      
      
      # Split and recover solution at time n or n+1/2 ############################
      (u, p, s) = w.split(True)
      ufinal, pfinal, visfin = recover_FSI_solution(params,u,p,s,phi)
      
      # Compute quantities of interest ###########################################
      compute_physical_quantities(params,ufinal,pfinal,mesh)
      params.acc_visdiss += params.visdiss
      
      # Print body dofs information over time ####################################
      if params.print_dofs:
         if rank == 0: print_pdofs(params)
      
      # Recover mesh at time n ###################################################
      if params.nfp > 1:
         meshdisp.vector()[:] = -meshdisp.vector()[:]
         ALE.move(mesh, meshdisp)
         mesh.bounding_box_tree().build(mesh)
   
      # Initial condition for food transport problem #############################
      if params.solvefood:
         if it > 0: FoodCn = compute_projected_foodc(params,mesh,FoodC,EC,remeshingm,rank,comm)
         FoodCn.rename('foodc', 'foodc')
         if it%params.freq_out == 0 and params.print_concentration: params.foodcn_pvd << FoodCn
      
      # Mesh at time n+1 #########################################################
      meshdisp = ALE_mesh_movement(params,mesh,mesh_facets)
      
      # Solve food transport problem #############################################
      if params.solvefood:
         meshdisphalf = Function(FunctionSpace(mesh, EU))
         if int(params.thmetstk) != 1:
            meshdisphalf.vector()[:] = -0.5*meshdisp.vector()[:]
            ALE.move(mesh, meshdisphalf)
            mesh.bounding_box_tree().build(mesh)
      
         FoodC = SolveFoodTransportProblem(params,mesh,mesh_facets,EC,FoodCn,ufinal,meshdisp,params.thmetcon*params.dt,params.thmetcon)
         FoodC.set_allow_extrapolation(True)
            
         if int(params.thmetstk) != 1:
            meshdisphalf.vector()[:] = -meshdisphalf.vector()[:]
            ALE.move(mesh, meshdisphalf)
            mesh.bounding_box_tree().build(mesh)
             
         compute_foodc_quantities(params,FoodC,mesh,mesh_facets)
   
      # Remeshing test ###########################################################
      meshqual = MeshQuality.radius_ratio_min_max(mesh)
      remeshingm = False if ((params.adaptmesh and meshqual[0] > meshquallim)) else params.remeshingm
   
      # Next time step ###########################################################
      prepare_next_time_step(params,it)
      
      # Domain meshing  ##########################################################
      mesh, mesh_facets, meshqual = load_geometry(params,mesh,mesh_facets,meshqual,rank,size,comm,remeshingm)
      params.W = FunctionSpace(mesh, EM)
      
      process_memory = str(psutil.Process(pid).memory_info()[0]/1024**3) # bytes to Gb
      memory_info.write(str(it) + '\t'+ process_memory + '\n')
      memory_info.flush()
   
   # End time steps loop #########################################################
   if rank == 0: 
      print('\n   ::> End of loop over time steps\n   ::> Simulation time %.5e' %(tim.time()-startTime))

   #return params.xc, xcini, params.th, thini
   return params

#------------------------------------------------------------------------
# We assume hereafter just 1 group of bodies
# The idea is to build a table of displacements (delta_pos, delta_angle)
# and perform Q-learning with reward functions that depend on swimmer position

# Flags
run_mode = 2  # 1: Build the table only (to compute the deltas)
              # 2: Learn by using a precomputed table
              # 3: Learn by solving the full problem to compute the deltas (not programmed yet ...)
              # 4: Learn by using a precomputed table and execute the learned policy by solving FSI
              # 5: Extract policy from table and generate a sequence of actions (in progress ...)

# Number of balls
nballs = 4
nlinks = nballs - 1
nstates = 2**nlinks
nactions = nlinks

# Numerical parameter
refbody = 1 # Refinement close to bodies
reffar = 2  # Refinement far away from bodies
dt = 1e-1   # Time step
ndeltas = 6

if(run_mode == 1):

    Deltas = [np.zeros(shape=(nstates, nactions)),
              np.zeros(shape=(nstates, nactions)),
              np.zeros(shape=(nstates, nactions)),
              np.zeros(shape=(nstates, nactions)),
              np.zeros(shape=(nstates, nactions)),
              np.zeros(shape=(nstates, nactions))]

    lst_sts = list(map(list, itertools.product([0, 1], repeat=nlinks))) # List of possible states in order

    for state in range(nstates):
        for action in range(nactions):

           chosen_actions = []
           visited_states = []
           params = ComputeStep(nballs, refbody, reffar, dt, state, lst_sts[state], action)
           
           xcm0 = np.array([0.0, 0.0, 0.0])
           xcm = np.array([0.0, 0.0, 0.0])
           for i in range(params.nballg[0]):
              xcm0 += params.xcini[0][i]/params.nballg[0]
              xcm += params.xc[0][i]/params.nballg[0]
              
           deltaxc = (xcm - xcm0)
           deltath = params.th[0][0] - params.thini[0][0] # the angle, just for the reference head

           Deltas[0][state,action] = deltaxc[0]
           Deltas[1][state,action] = deltaxc[1]
           Deltas[2][state,action] = deltath
           Deltas[3][state,action] = params.acc_visdiss
           Deltas[4][state,action] = params.xc[0][0][0] - params.xcini[0][0][0]
           Deltas[5][state,action] = params.xc[0][0][1] - params.xcini[0][0][1]
            
    for i in range(ndeltas):
        print( Deltas[i] , '\n')

    print(params.dir_)
    deltas_file = open(params.dir_+'/deltas.txt', 'w+')
    for i in range(ndeltas):
       for state in range(nstates):
          for action in range(nactions):
             deltas_file.write("%.10e " % (Deltas[i][state,action]) )
          deltas_file.write("\n")
       deltas_file.write("\n\n")

    deltas_file.close()
            
elif(run_mode == 2 or run_mode == 4 or run_mode == 5):

    def ReadDeltasTables(ndeltas, nstates, nactions, filetable):
       ##print('     |> Loading table ', filetable)
       A = np.loadtxt(filetable)       
       Deltas = []
       for d in range(ndeltas):
          Deltas.append(np.zeros(shape=(nstates, nactions)))
          for state in range(nstates):
             Deltas[d][state,:] = A[d*nstates+state,:]
       return Deltas

    file_table = 'DELTAS/' + 'DELTASfino/'+ 'theta10/' +'deltas.txt'
    Deltas = ReadDeltasTables(ndeltas, nstates, nactions, file_table)      
     
    # Program your reward function    
    def compute_reward(angletarget, angle_n, angle):       
              
        dist = np.abs((angle)%(2*np.pi) - angletarget)
        dist = min(dist, 2*np.pi - dist) 
        dist_n = np.abs((angle_n)%(2*np.pi) - angletarget)
        dist_n = min(dist_n, 2*np.pi - dist_n)
        
        rew = -dist + dist_n       
        return rew

    def choose_action(current_state, eps_greedy, MQlearning):
        if(np.random.random() < (1-eps_greedy)):
            current_state_dec = int(np.dot(current_state,2**np.arange(nlinks-1,-1,-1,dtype=int)))            
            aux = np.zeros(nlinks, dtype=int)
            aux[np.argmax(MQlearning[current_state_dec,:])] = 1
            return aux
        else:
            aux = np.identity(nlinks, dtype=int)
            ret = aux[np.random.randint(0, nlinks),:]
            return ret

    def compute_next_state(current_state, action):
        return np.remainder(current_state+action,2*np.ones(nlinks, dtype=int))
    
    # Q-learning matrix
    MQlearning   = np.zeros((nstates,nactions))
    MQlearning_n = np.zeros((nstates,nactions))   
    
    '''
    #Hyperparameters rotação
    
    gamma = 0.999
    alpha = 1
    max_eps = 0.9
    min_eps = 0.05
    lambd = 1e-03
    n_QL_steps = 20000
    '''
    
    #Hyperparameters 
    GAMMA = 0.9999
    ALPHA = 1
    MAX_EPSILON = 0.2
    MIN_EPSILON = 0.2
    LAMBDA   = 0.1e-03 
    N_QL_steps = 400000
    
    # Geometry and position
    Lx, Ly        = 3*150., 3*150. # Check consistency with ComputeStep()
    xc_swimmer    = np.array([Lx/2.0, Ly/2.0])
    xc_swimmer_n  = np.array([Lx/2.0, Ly/2.0])
    theta_swimmer = -np.pi/4
    current_disp  = np.zeros(2)
    current_state = np.zeros(nlinks, dtype=int) # As a vector    
    #current_state = np.ones(nlinks, dtype=int) # As a vector    
    
    impqlaction = False
    actionimp   = [np.array([0,0,1]),np.array([1,0,0])] #[np.array([0,0,1]),np.array([0,1,0]),np.array([1,0,0])] #
    ##actionimp   = [np.array([0,1,0]),np.array([0,0,1])]
    ##actionimp   = [np.array([0,0,1]),np.array([0,1,0])]
    ##actionimp   = [np.array([1,0,0]),np.array([0,0,1])]
    ##actionimp   = [np.array([0,0,1]),np.array([1,0,0])]
    ##actionimp   = [np.array([1,0,0]),np.array([0,1,0])]
    #actionimp   = [np.array([0,1,0]),np.array([1,0,0])]
    actioncount = len(actionimp)

    advancing_mode = 'CM' # CM: Tracking center of mass
                          # LB: Tracking position of reference body (leftmost)   

    #-----------------------------
    #--- Loop over learning steps
    rew_file = open('reward.txt', 'a+')
    #sts_file = open('state.txt', 'a+')
    ##mtx_file = open('matrix.txt', 'a+')
    ac_file = open('actions.txt', 'w+')
    total_reward  = 0.0
    tt, taction = 0.0, 1.0
    angletarget = -np.pi/2
    nstepstest = 0
    alist = []
    for it in range(N_QL_steps):
   
        #print('it = ', it) 
        #print('theta= ', theta_swimmer)
        
        '''
        sts_file.write("%d %d %.10e %.10e %.10e\n" %
                       (it,
                        int(np.dot(current_state,2**np.arange(nlinks-1,-1,-1,dtype=int))),
                        xc_swimmer[0], xc_swimmer[1], theta_swimmer
                       ))
       
        sts_file.flush()       
        '''
        if(it < N_QL_steps):
            rew_file.write("%d %.10e %.10e %.10e %.10e\n" % (it, total_reward, theta_swimmer, xc_swimmer[0], xc_swimmer[1]) )
            rew_file.flush()
        
        # Learning decisions ----------------------
        eps_greedy  = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON)*np.exp(-LAMBDA * it)
        action2exec = actionimp[it%actioncount] if impqlaction else choose_action(current_state, eps_greedy, MQlearning)    
        action      = np.argmax(action2exec)
        alist.append(action)
        next_state  = compute_next_state(current_state, action2exec)

        # Save the history of states and actions
        chosen_actions.append(action2exec)
        visited_states.append(current_state)

        current_state_dec = int(np.dot(current_state,2**np.arange(nlinks-1,-1,-1,dtype=int)))
        if(advancing_mode == 'CM'):
           current_disp[0] = Deltas[0][current_state_dec,action]
           current_disp[1] = Deltas[1][current_state_dec,action]
        elif(advancing_mode == 'LB'):
           current_disp[0] = Deltas[4][current_state_dec,action]
           current_disp[1] = Deltas[5][current_state_dec,action]
           
        current_dtheta  = Deltas[2][current_state_dec,action]       

        
        Rmatrix = np.array([[np.cos(theta_swimmer), -np.sin(theta_swimmer)],
                            [np.sin(theta_swimmer),  np.cos(theta_swimmer)]])

        # Update dofs -------------------------
        xc_swimmer_n   = copy.deepcopy(xc_swimmer)
        xc_swimmer    += Rmatrix @ current_disp
        theta_swimmer_n = theta_swimmer
        theta_swimmer += current_dtheta 
        
        
        # Update Q learning matrix ----------------------
        
        next_state_dec = int(np.dot(next_state,2**np.arange(nlinks-1,-1,-1,dtype=int)))        
        next_Qmax      = np.max(MQlearning[next_state_dec,:])        
        current_Q      = MQlearning[current_state_dec,action]         
        ##current_reward = compute_reward(it, current_state)
        ##current_reward = compute_reward(it, xc_swimmer_n, xc_swimmer)
        ##current_reward = compute_reward(theta_swimmer_n, theta_swimmer)        
        current_reward = compute_reward(angletarget,theta_swimmer_n, theta_swimmer)        
        total_reward  += current_reward
        tt += taction        
        MQlearning[current_state_dec,np.argmax(action2exec)] = current_Q + ALPHA*(current_reward + GAMMA*next_Qmax - current_Q)

        print('NormDQ= ', np.linalg.norm(MQlearning - MQlearning_n))
        
        current_state = copy.deepcopy(next_state)
        MQlearning_n = copy.deepcopy(MQlearning)
        ac_file.write("%d\n" %(alist[it]))       


        #--- End loop over learning steps
        #--------------------------------
        
    def ExtractLearnedPolicy(Qmatrix):
       lst_sts = list(map(list, itertools.product([0, 1], repeat=nlinks)))
       print(lst_sts)
       action_sequence = []
       state_sequence = []
       sini = 0
       current_state_dec = sini
       current_state = np.array(lst_sts[current_state_dec])
       nturns = 100
       
       for i in range(nturns):
          aux = np.zeros(nlinks, dtype=int)
          aux[np.argmax(Qmatrix[current_state_dec,:])] = 1
          action2exec = aux
          next_state = np.remainder(current_state+action2exec,2*np.ones(nlinks, dtype=int))
          action_sequence.append(action2exec)
          state_sequence.append(current_state)
          print('step= ', i, 'state= ', '(',  current_state_dec , current_state, '), ',  'action= ', action2exec)
          current_state = next_state
          current_state_dec = int(np.dot(current_state,2**np.arange(nlinks-1,-1,-1,dtype=int)))
       return action_sequence, state_sequence
    
    #learned_policy, learned_states = ExtractLearnedPolicy(MQlearning)
    print(alist[N_QL_steps-100:-1])
    print(len(alist))
    
    if(run_mode == 4):
        params = ComputeStep(nballs, refbody, reffar, dt,
                             int(np.dot(visited_states[0],2**np.arange(nlinks-1,-1,-1,dtype=int))),
                             visited_states[0], np.argmax(chosen_actions[0]))

        
    if(run_mode == 5):
       
       def ExtractLearnedPolicy(Qmatrix):
          lst_sts = list(map(list, itertools.product([0, 1], repeat=nlinks)))
          print(lst_sts)
          action_sequence = []
          state_sequence = []
          sini = 0
          current_state_dec = sini
          current_state = np.array(lst_sts[current_state_dec])
          nturns = 10
          for i in range(nturns):
             aux = np.zeros(nlinks, dtype=int)
             aux[np.argmax(Qmatrix[current_state_dec,:])] = 1
             action2exec = aux
             next_state = np.remainder(current_state+action2exec,2*np.ones(nlinks, dtype=int))
             action_sequence.append(action2exec)
             state_sequence.append(current_state)
             print('step= ', i, 'state= ', '(',  current_state_dec , current_state, '), ',  'action= ', action2exec)
             current_state = next_state
             current_state_dec = int(np.dot(current_state,2**np.arange(nlinks-1,-1,-1,dtype=int)))
          return action_sequence, state_sequence

       learned_policy, learned_states = ExtractLearnedPolicy(MQlearning)

       if(True):
          visited_states = learned_states
          chosen_actions = learned_policy
          
          current_state = np.array(lst_sts[0])
          sts_file = open('state_test.txt', 'a+')
          xc_swimmer    = np.array([Lx/2.0, Ly/2.0])
          xc_swimmer_n   = copy.deepcopy(xc_swimmer)
          theta_swimmer = 0.0
          for it in range(10):
             current_state = visited_states[it]
             current_state_dec = int(np.dot(current_state,2**np.arange(nlinks-1,-1,-1,dtype=int)))
             action2exec = chosen_actions[it]
             action = np.argmax(action2exec)
             
             sts_file.write("%d %d %.10e %.10e %.10e\n" %
                            (it,
                             int(np.dot(current_state,2**np.arange(nlinks-1,-1,-1,dtype=int))),
                             xc_swimmer[0], xc_swimmer[1], theta_swimmer
                            ))
             sts_file.flush()             
             
             if(advancing_mode == 'CM'):
                current_disp[0] = Deltas[0][current_state_dec,action]
                current_disp[1] = Deltas[1][current_state_dec,action]
             elif(advancing_mode == 'LB'):
                current_disp[0] = Deltas[4][current_state_dec,action]
                current_disp[1] = Deltas[5][current_state_dec,action]
           
             current_dtheta  = Deltas[2][current_state_dec,action]
        
             Rmatrix = np.array([[np.cos(theta_swimmer), -np.sin(theta_swimmer)],
                                 [np.sin(theta_swimmer),  np.cos(theta_swimmer)]])

             # Update dofs -------------------------
             xc_swimmer_n   = copy.deepcopy(xc_swimmer)
             xc_swimmer    += Rmatrix @ current_disp
             theta_swimmer += current_dtheta 

          sts_file.close()
          ''' 
          # Execute the actual sequence by solving the fluid
          params = ComputeStep(nballs, refbody, reffar, dt,
                               int(np.dot(visited_states[0],2**np.arange(nlinks-1,-1,-1,dtype=int))),
                               visited_states[0], np.argmax(chosen_actions[0]))
          '''
      
       
elif(run_mode == 3):
    raise Exception ('Not program yet ...')
else:
    raise Exception ('Option not recognized ...')




