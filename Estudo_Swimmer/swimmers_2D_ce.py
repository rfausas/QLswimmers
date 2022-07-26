from dolfin import *

import numpy as np
import copy
import random
import time as tim

from libcommon.user_utils import *
from libcommon.build_geometry import *
from libcommon.configurational_dofs import *
from libcommon.variational_formulation import *
from libcommon.ql import *

import psutil

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
#params.dir_   = './test04ce'
#params.dir_   = './testcompleto2'
params.dir_   = './videoapresentacao2'

# Dimensions
params.isaxis = 0
params.nsd = 2 # space dimension
params.npd = 3 if not params.isaxis else 1 # number positional dofs per body group reference
params.nbg = 1 # number of body groups

# Swimmers type: circle, square, ellipse
params.swmtype = 'ellipse'

# Number of immersed balls
params.nballg = [4 for ib in range(params.nbg)] # number of bodies per body group
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

   params.radii[ig] = [1.5] + [1.0 for ib in range(params.nballg[ig]-1)] #[1.0] + [0.45 for ib in range(params.nballg[ig]-1)]
   params.radih[ig] = [0.25] + [0.2 for ib in range(params.nballg[ig]-1)]
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
params.dt       = 0.1 #0125
params.freq_out = 2*int(0.1/params.dt) #1*int(0.05/params.dt)
params.nsteps   = 4000*int(1/params.dt) + 1 #12 #25+2 #int(1/dt) +1 #1 #1001
#params.pcmet    = 0
params.nfp      = 2 #2 #if params.pcmet == 1 else 1 # fixed point iterations
params.thmetstk = 1.0 #0.5 #0.5 to set RK2MP, 1 to set RK2ME, these need nfp >= 2
params.thmetcon = params.thmetstk
#params.thmetcon = 1
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
params.lcar        = 150./2**(-1+2+1)
params.lbod        = 0.3/2**(1)
params.lcor        = params.lcar
params.laxi        = params.lbod*1
params.btagini     = -1 # don't change
params.it          = 0 # don't change
mesh,mesh_facets,mesh_physic = [],[],[]

# Compute physical quantities
compute_forces = False
params.visdiss = 0.0

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
params.eatingbody = params.nballs # nummber of eating parts in each swimmer
params.impzeroconvel = True
params.dtdiffp       = 1 #50*params.dt
params.foodininsteps = 10 #*int(params.dtdiffp) 

np.random.seed(0)
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
params.impqlaction   = True # True if basic action is imposed, False for QL
#params.actionimp = [np.array([1,0,0,0]),np.array([0,1,0,0]),np.array([0,0,1,0]),np.array([0,0,0,1])]
#params.actionimp = [np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])]
params.actionimp = 50*[np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])] + 50*[np.array([1,0,0]),np.array([0,0,1])] + 50*[np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])] + 40*[np.array([1,0,0]),np.array([0,0,1])] + 20*[np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])]
#params.actionimp = [np.array([1,0,0]),np.array([0,0,1])]
#params.actionimp = [np.array([0,0,1]),np.array([0,1,0]),np.array([1,0,0]),np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])]
params.current_state = 1*np.ones(params.nlinks, dtype=int) if True else np.random.randint(2, size=params.nlinks)
#params.current_state = 1*np.zeros(params.nlinks, dtype=int) if True else np.random.randint(2, size=params.nlinks)
params.action        = params.actionimp[0] if params.impqlaction else np.random.randint(2, size=params.nlinks)
params.next_state    = np.ones(params.nlinks, dtype=int)
params.total_reward  = 0.0
params.actioncount   = len(params.actionimp)
params.printMQL      = True
params.rewardtype    = 0 # 0 for positional, 1 for final foodc , 2 for foodc in a whole action
params.fluxperbodyaction = [0.0]*params.nballs

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

for it in range(params.nsteps):
   if rank == 0:
      print("\n##################################################")
      print("\n    |-Time #%.5e step #%5d begins" %(params.time,it))
      
   if params.solveQL and it%int(params.taction/params.dt) == 0:
      prepare_action(params,rank)

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
      
      # Updating current state ################################################
      #if False and fp == params.nfp-1 and params.solveQL \
      #           and it%int(params.taction/params.dt) == int(params.taction/params.dt)-1: #and int(params.thmetcon) == 1
      #   params.current_state = copy.deepcopy(params.next_state)
         
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
   
      FoodC = SolveFoodTransportProblem(params,mesh,mesh_facets,EC,FoodCn,ufinal,meshdisp,params.dt,params.thmetcon)
      FoodC.set_allow_extrapolation(True)
         
      if int(params.thmetstk) != 1:
         meshdisphalf.vector()[:] = -meshdisphalf.vector()[:]
         ALE.move(mesh, meshdisphalf)
         mesh.bounding_box_tree().build(mesh)
          
      compute_foodc_quantities(params,FoodC,mesh,mesh_facets)

   # Remeshing test ###########################################################
   meshqual = MeshQuality.radius_ratio_min_max(mesh)
   remeshingm = False if ((params.adaptmesh and meshqual[0] > meshquallim)) else params.remeshingm
   
   # Update Q learning matrix #################################################
   if params.solveQL and it%int(params.taction/params.dt) == int(params.taction/params.dt)-1:
      update_MQL(params,rank)

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
