from dolfin import *

import numpy as np
import time as tim
from .positional_dofs import *
from .configurational_dofs import LLINK,DLLINK,BLINK,DBLINK
from .build_geometry import rotation_matrix
from .user_utils import generalized_force, print_food_info, print_quantities

###############################################################################
def transpose_skew_matrix_expression(xc,nsd):
   R = []
   if (nsd == 2):
      tmp = Expression(("-(x[1]-x1)","+(x[0]-x0)"),x0=xc[0],x1=xc[1],degree=1)
      R.append(tmp)
   else:
      tmp = Expression(("0","-(x[2]-x2)","+(x[1]-x1)"),x1=xc[1],x2=xc[2],degree=1)
      R.append(tmp)
      tmp = Expression(("+(x[2]-x2)","0","-(x[0]-x0)"),x0=xc[0],x2=xc[2],degree=1)
      R.append(tmp)
      tmp = Expression(("-(x[1]-x1)","+(x[0]-x0)","0"),x0=xc[0],x1=xc[1],degree=1)
      R.append(tmp)

   return R


def transpose_skew_matrix_expression_full(xc,nsd,velBallvec):
   R = []
   if (nsd == 2):
      tmp = Expression(("-(x[1]-x1) + C0","+(x[0]-x0) + C1"),x0=xc[0],x1=xc[1],C0=velBallvec[0],C1=velBallvec[1],degree=1)
      R.append(tmp)
   else:
      tmp = Expression(("0","-(x[2]-x2)","+(x[1]-x1)"),x1=xc[1],x2=xc[2],degree=1)
      R.append(tmp)
      tmp = Expression(("+(x[2]-x2)","0","-(x[0]-x0)"),x0=xc[0],x2=xc[2],degree=1)
      R.append(tmp)
      tmp = Expression(("-(x[1]-x1)","+(x[0]-x0)","0"),x0=xc[0],x1=xc[1],degree=1)
      R.append(tmp)

   return R


def compute_phi_bodies(params,W,mesh_facets):#,nbstart,nbg,nsd,npd,nballg,eaux,vaux,xc,th):
   nbstart = params.btagini
   nbg     = params.nbg
   nsd     = params.nsd
   npd     = params.npd
   nballg  = params.nballg
   eaux    = params.eaux
   vaux    = params.vaux
   xc      = params.xc
   th      = params.th
   
   phi = [[] for i in range(nbg)]
   for ig in range(nbg):
      phi[ig] = [[] for i in range(nballg[ig])]
      
      # Building phi[ig][0] which has the dofs info for referential body in group ig
      
      # Translational dofs ####################################################
      for ic in range(nsd):
         
         aux = interpolate(vaux,W.sub(0).collapse())
         
         velBall = eaux[ic] # set velocity 1 at direction ic
         for ib in range(nballg[ig]):
            bcball = DirichletBC(W.sub(0).collapse(), velBall, mesh_facets, nbstart+ib)
            bcball.apply(aux.vector())
         phi[ig][0].append(aux)
         # aux.rename("phi","phi")
         # params.visfin_pvd << aux
         
      # Rotational dofs #######################################################
      if (nsd == 2 and npd > 2): # Rotational dofs 2D #########################
         nskm = transpose_skew_matrix_expression(xc[ig][0],nsd)
         velBall = nskm[0]
         aux = interpolate(vaux,W.sub(0).collapse())
         for ib in range(nballg[ig]):
            bcball = DirichletBC(W.sub(0).collapse(), velBall, mesh_facets, nbstart+ib)
            bcball.apply(aux.vector())
         phi[ig][0].append(aux)
         # aux.rename("phi","phi")
         # params.visfin_pvd << aux
      elif (nsd == 3 and npd > 3): # Rotational dofs 3D #######################
         nskm = transpose_skew_matrix_expression(xc[ig][0],nsd)
         Q0eref = np.dot(Qr[ig][0],e_ref[ig][0])
         askm = -skew_matrix(Q0eref)
         for ic in range(3): # 3 from 3D rotational positional dofs
            aux = interpolate(vaux,W.sub(0).collapse())
            for ib in range(nballg[ig]):
               velBall = nskm[ic]
               # Contribution from configurational dofs #######################
               ll = 0.0
               for il in range(0,ib):
                  ll += LLINK(time,il,nballs-1,current_state,action2exec,taction)
               velBall += Constant((ll*askm[0,ic],ll*askm[1,ic],ll*askm[2,ic]))
               ################################################################
               bcball = DirichletBC(W.sub(0).collapse(), velBall, mesh_facets, nbstart+ib)
               bcball.apply(aux.vector())
            phi[ig][0].append(aux)
           
      # Configurational dofs links ############################################
      for il in range(1,nballg[ig]):
         aux = interpolate(vaux,W.sub(0).collapse())
         sth = (np.sin(th[ig][il-1]) + np.sin(th[ig][il]))
         cth = (np.cos(th[ig][il-1]) + np.cos(th[ig][il]))
         velBall = Constant((0.5*cth,0.5*sth))
         for ib in range(il,nballg[ig]):
            bcball = DirichletBC(W.sub(0).collapse(), velBall, mesh_facets, nbstart+ib)
            bcball.apply(aux.vector())
         phi[ig][il].append(aux)
         # aux.rename("phi","phi")
         # params.visfin_pvd << aux
         
      # Configurational dofs angles ###########################################
      #nskm = transpose_skew_matrix_expression(xc[ig][0],nsd)
      for il in range(1,nballg[ig]):
         aux = interpolate(vaux,W.sub(0).collapse())
         laux = 0.5*LLINK(params,params.time,il)
         velBallvec = np.array([laux*np.sin(th[ig][il-1]),-laux*np.cos(th[ig][il-1])])
         #velBall = nskm[0] - Constant((-laux*np.sin(th[ig][il-1]),laux*np.cos(th[ig][il-1])))
         for jl in range(1,il):
            sth = (np.sin(th[ig][jl-1]) + np.sin(th[ig][jl]))
            cth = (np.cos(th[ig][jl-1]) + np.cos(th[ig][jl]))
            laux = 0.5*LLINK(params,params.time,jl)
            velBallvec += np.array([laux*sth,-laux*cth])
            #velBall -= Constant((-laux*sth,laux*cth))
         velBall = transpose_skew_matrix_expression_full(xc[ig][0],nsd,velBallvec)[0]
         for ib in range(il,nballg[ig]):
            #tini= tim.time()
            bcball = DirichletBC(W.sub(0).collapse(), velBall, mesh_facets, nbstart+ib)
            #print(f"angles: {tim.time()-tini}")
            bcball.apply(aux.vector())
         phi[ig][il].append(aux)
         # aux.rename("phi","phi")
         # params.visfin_pvd << aux
         
      # Updating tag starting for body group ##################################
      nbstart = nbstart+nballg[ig]
         
   return phi


def SolveFSIProblem(params,mesh,mesh_facets,W,w,phi):
   nbg    = params.nbg
   nsd    = params.nsd
   npd    = params.npd
   nballg = params.nballg
   time   = params.time
   
   domains = MeshFunction("size_t", mesh, mesh.topology().dim(), mesh.domains())
   dx = Measure("dx")(subdomain_data=domains)

   # Variational formulation ##################################################
   (u, p, s) = TrialFunctions(W)
   (v, q, d) = TestFunctions(W)   
   
   # Build trial uT (u Total) function ########################################
   uT = compute_uT(params, u, s, phi)

   # Build total residual #####################################################
   Res  = momentum_residual(params, uT, p, v, mesh)
   Res += mass_residual(params, uT, p, q, mesh)
   
   # Positional dofs contribution #############################################
   FT = generalized_force(time,nsd)
   #FT = -1e8*np.array([s[0],s[1],s[2]])
   scount = 0
   for ig in range(nbg):
      for ic in range(npd):
         Res += momentum_residual(params, uT, p, d[scount]*phi[ig][0][ic], mesh)
         Res -= d[scount]*FT[ic]*Constant(1.0 / assemble(1.0 * dx(mesh)))*dx
         scount += 1
   
   # Dirichlet boundary conditions ############################################
   bcs = set_dirichlet_bc(params, W, mesh_facets)

   # Compute solution #########################################################
   sol = Function(W)
   
   if params.stokes_system_as_linear:
      a, L = lhs(Res), rhs(Res)
      solve(a == L, sol, bcs, solver_parameters={'linear_solver': 'mumps'})
   else:
      if params.remeshingm:
         w.set_allow_extrapolation(True)
         sol.interpolate(w)
      else:
         sol.vector()[:] = w.vector()[:]
      Res = action(Res, sol)
      Jac = derivative(Res, sol)
      problem = NonlinearVariationalProblem(Res, sol, bcs, Jac)
      solver = NonlinearVariationalSolver(problem)
      solverType = 'newton' # 'snes'
      solver.parameters['nonlinear_solver'] = solverType
      nlparam = solver.parameters[solverType+'_solver']
      #nlparam['method'] = 'newtonls'
      nlparam['maximum_iterations'] = 100
      nlparam['absolute_tolerance'] = 1E-10
      nlparam['relative_tolerance'] = 1E-9
      nlparam['error_on_nonconvergence'] = False
      direct = True
      if(direct == True):
         nlparam['linear_solver'] = 'mumps'
      else:
         nlparam['linear_solver'] = 'gmres'
         nlparam['preconditioner'] = 'ilu' # or 'none'
         linparam = solver.parameters[solverType+'_solver']['krylov_solver']
         linparam['absolute_tolerance'] = 1E-10
         linparam['relative_tolerance'] = 1E-7
         linparam['monitor_convergence'] = True
      solver.solve()
   
   return sol
   

def viscosity(u):
   gammadot2 = 2*inner(sym(grad(u)),sym(grad(u))) #+ 1e-5
   muinf, mu0, lamb, n = 1.0, 2.0, 1.0, 0.5 
   return 1 #+ 0*gammadot2 #muinf+(mu0-muinf)*(1+(lamb)**2*gammadot2)**((n-1)/2) #0.035*gammadot**(-0.2)


def momentum_residual(params, u, p, v, mesh):
   mu = viscosity(u)
   afactor = Constant(1)
   visterm = Constant(0)*div(v)
   divterm = Constant(0)*div(v)
   if params.isaxis:
      x       = SpatialCoordinate(mesh)
      afactor = 2*pi*x[1]
      visterm = 2*mu*(u[1]*v[1]/x[1]**2)
      divterm = v[1]/x[1]
   a = (inner(2*mu*sym(grad(u)),grad(v))+visterm-(div(v)+divterm)*p)*afactor*dx
   L = inner(params.ff, v)*afactor*dx

   return a - L


def mass_residual(params, u, p, q, mesh):
   mu = viscosity(u)
   afactor = Constant(1)
   divterm = Constant(0)*div(u)
   if params.isaxis:
      x       = SpatialCoordinate(mesh)
      afactor = 2*pi*x[1]
      divterm = u[1]/x[1]
   a = q*(div(u)+divterm)*afactor*dx

   # Stabilization #
   if(params.orderU == params.orderP):
      hk = CellDiameter(mesh)
      tauk = hk*hk/(4*mu)
      a += tauk*inner(grad(q),grad(p))*afactor*dx
      L  = tauk*inner(params.ff,grad(q))*afactor*dx

      return a - L

   return a


def compute_uT(params, u, s, phi):
   uT = u
   scount = 0
   for ig in range(params.nbg):
      for ic in range(params.npd):
         uT += s[scount]*phi[ig][0][ic]
         scount += 1
      for ib in range(1,params.nballg[ig]):
         uT += Constant(DLLINK(params,params.time,ib))*phi[ig][ib][0]
         uT += Constant(DBLINK(params,params.time,ib))*phi[ig][ib][1]
         
   return uT


def set_dirichlet_bc(params, W, mesh_facets):
   nballg  = params.nballg
   vaux    = params.vaux
   VF      = params.VF
   btagini = params.btagini
   geomtype = 0
   bcs = []
   noslip = Constant((-VF[0], -VF[1]))
   if  (geomtype == 0):
      # Boundary conditions
      #bcsl = DirichletBC(W.sub(0), noslip, mesh_facets, 4) #left
      #bcs.append(bcsl)
      #bcsr = DirichletBC(W.sub(0), noslip, mesh_facets, 2) #right
      #bcs.append(bcsr)
      bcsd = DirichletBC(W.sub(0), noslip, mesh_facets, 1) #down
      if params.isaxis:
         bcsd = DirichletBC(W.sub(0).sub(1), Constant(0), mesh_facets, 1) #down
      bcs.append(bcsd)
      bcsu = DirichletBC(W.sub(0), noslip, mesh_facets, 3) #up
      bcs.append(bcsu)
      # Solids' boundary condition
      for ib in range(sum(nballg)):
         velBall = vaux
         bcball = DirichletBC(W.sub(0), velBall, mesh_facets, btagini+ib)
         bcs.append(bcball)

   else:
      raise Exception('check geomtype option')

   return bcs


def recover_FSI_solution(params, u, p, s, phi):
   uT = compute_uT(params, u, s, phi)
   ufinal = uT
   pfinal = p
   visfin = viscosity(uT)
   
   return ufinal, pfinal, visfin


def update_positional_dofs(params,mesh,s,dt):
   nbg  = params.nbg
   nsd  = params.nsd
   npd  = params.npd
   time = params.time
   meshc = mesh.coordinates()
   dpdt = s(meshc[0,:])
   
   # recover dofs derivatives from s
   if not params.isaxis:
      scount = 0
      for ig in range(nbg):
         # uptating velocity dofs (reference body in group ig)
         for ic in range(nsd):
            params.vcn[ig][0][ic] = dpdt[scount]
            scount += 1
         if (npd > 2):
            if (nsd == 2):
               params.omn[ig][0][2] = dpdt[scount]
               scount += 1
            else:
               for ic in range(nsd):
                  params.omn[ig][0][ic] = dpdt[scount]
                  scount += 1           
   else:
      for ig in range(nbg):
         # uptating velocity dofs (reference body in group ig)
         params.vcn[ig][0][0] = dpdt
               
   for ig in range(params.nbg):
      for ib in range(1,params.nballg[ig]):
         params.omn[ig][ib][2] = params.omn[ig][ib-1][2] + DBLINK(params,params.time,ib)
         sth = (np.sin(params.th[ig][ib-1]) + np.sin(params.th[ig][ib]))
         cth = (np.cos(params.th[ig][ib-1]) + np.cos(params.th[ig][ib]))
         dsth =  (params.omn[ig][ib-1][2]*np.cos(params.th[ig][ib-1]) + params.omn[ig][ib][2]*np.cos(params.th[ig][ib]))
         dcth = -(params.omn[ig][ib-1][2]*np.sin(params.th[ig][ib-1]) + params.omn[ig][ib][2]*np.sin(params.th[ig][ib]))
         params.vcn[ig][ib] = params.vcn[ig][ib-1] + \
                              0.5*DLLINK(params,time,ib)*np.array([cth,sth,0]) +\
                              0.5*LLINK(params,time,ib)*np.array([dcth,dsth,0])
   
   if params.thmetstk != 1 or params.fp == 0:
      params.vc = copy.deepcopy(params.vcn)
      params.om = copy.deepcopy(params.omn)

   for ig in range(nbg):
      # updating dofs (reference body in group ig)
      params.xc[ig][0] = integrate_translation(params,dt,ig,0)
      params.th[ig][0] = integrate_rotation(params,dt,ig,0)
      params.Qr[ig][0] = rotation_matrix(params.th[ig][0])
      
   #params.vc = copy.deepcopy(params.vcn)
   #params.om = copy.deepcopy(params.omn)
                                
                  
def update_dependent_dofs(params,time):
   if True:
      for ig in range(params.nbg):
         for ib in range(1,params.nballg[ig]): # Integrating with positional mapping
            params.th[ig][ib] = params.th[ig][ib-1] + BLINK(params,time,ib)
            params.Qr[ig][ib] = rotation_matrix(params.th[ig][ib])
            params.xc[ig][ib] = params.xc[ig][ib-1] + 0.5*LLINK(params,time,ib)*\
                                (np.dot(params.Qr[ig][ib-1],params.e_ref[ig][0])\
                                 + np.dot(params.Qr[ig][ib],params.e_ref[ig][0]))
   else:
      for ig in range(params.nbg):
         for ib in range(1,params.nballg[ig]): # Integrating with velocity given by SFI
            params.xc[ig][ib] = integrate_translation(params,dt,ig,ib)
            params.th[ig][ib] = integrate_rotation(params,dt,ig,ib)
            params.Qr[ig][ib] = rotation_matrix(params.th[ig][ib])   
                                
         
def update_positional_dofs_midpoint(params):
   for ig in range(params.nbg):
      # updating dofs (reference body in group ig)
      params.xc[ig][0] = (params.xc[ig][0]+params.xcn[ig][0])/2.0
      params.th[ig][0] = (params.th[ig][0]+params.thn[ig][0])/2.0
      params.Qr[ig][0] = rotation_matrix(params.th[ig][0]) 
                  
             
def SolveFoodTransportProblem(params,mesh,mesh_facets,EC,FoodCn,ufinal,meshdisp,dtc,thmetcon):
   nbg = params.nbg
   nballg = params.nballg
   btagini = params.btagini
   Dfood = params.Dfood
   Rfood = params.Rfood
   Sfood = params.Sfood
   Geat  = params.Geat
   Alphaeat = params.Alphaeat
   fcpestab = params.fcpestab
   #thmetcon = params.thmetcon
   beta = convective_velocity(params,mesh,mesh_facets,ufinal,meshdisp,dtc)
   
   WF    = FunctionSpace(mesh, EC)
   FoodC = Function(WF)
   uf    = TrialFunction(WF)
   vf    = TestFunction(WF)
   
   afactor = Constant(1)
   if params.isaxis:
      x       = SpatialCoordinate(mesh)
      afactor = 2*pi*x[1]
   
   # Variational formulation #
   uft = thmetcon*uf + (1-thmetcon)*FoodCn
   
   ds = Measure("ds", domain=mesh, subdomain_data=mesh_facets)
   a  = (uf/dtc)*vf*afactor*dx
   a += inner(Dfood*grad(uft), grad(vf))*afactor*dx
   a += dot(beta,grad(uft))*vf*afactor*dx
   a += Rfood*uft*vf*afactor*dx
   L  = (FoodCn/dtc)*vf*afactor*dx
   L += Sfood*vf*afactor*dx
   
   # Define which balls eat #
   bcount = 0
   for ig in range(nbg):
      for ib in range(params.eatingbody if (params.eatingbody <= nballg[ig]) else nballg[ig]):
         a += Alphaeat*uft*vf*afactor*ds(btagini+bcount)
         L += Geat*vf*afactor*ds(btagini+bcount)
         bcount += 1

   # Estabilization #
   if(fcpestab == 'SUPG'):
      hk      = CellDiameter(mesh)/Constant(params.orderC)
      tauk    = params.cestab * 1.0/(4.0*Dfood/hk**2 + 2.0*(dot(beta,beta)**0.5)/hk + Rfood)
      lhs_eq  = (uf-FoodCn)/dtc
      lhs_eq += (-div(Dfood*grad(uft)) + dot(beta,grad(uft)) + Rfood*uft)
      rhs_eq  = Sfood 
      a      += tauk*lhs_eq*dot(beta,grad(vf))*afactor*dx
      L      += tauk*rhs_eq*dot(beta,grad(vf))*afactor*dx
      
   # Boundary conditions: [] means natural BC over the Box #
   bcs = []

   # Solve problem #
   Res = a - L
   if True:
      a, L = lhs(Res), rhs(Res)
      solve(a == L, FoodC, bcs, solver_parameters={'linear_solver': 'mumps'})
   
   # Cumputing flux per body in groups and mass
   # bcount = 0
   # params.fluxperbody = []
   # for ig in range(nbg):
   #    for ib in range(params.eatingbody if (params.eatingbody <= nballg[ig]) else nballg[ig]):
   #       params.fluxperbody.append(assemble(Alphaeat*FoodC*ds(btagini+bcount)))
   #       bcount += 1
      
   #params.intFoodCA = assemble(FoodC*dx)
   
   return FoodC


def compute_projected_foodc(params,mesh,FoodC,EC,remeshingm,rank,comm):

   if remeshingm:
      hk = CellDiameter(mesh)
      WF = FunctionSpace(mesh, EC)
      FoodCn = Function(WF)
      # interpolating on high pol degree space
      qdeg = 5
      WFN = FunctionSpace(mesh, FiniteElement("Lagrange", 'triangle', qdeg))
      FoodCI = interpolate(FoodC,WFN)
      # projecting on PN space
      w = TestFunction(WF)
      v = TrialFunction(WF)
      A = assemble( (inner(v,w) + 0*hk**2*inner(grad(v),grad(w)) )*dx(mesh))
      b = assemble(inner(FoodCI,w)*dx(mesh),form_compiler_parameters={'quadrature_degree': qdeg,'quadrature_rule': 'default'})      
      solve(A, FoodCn.vector(), b, "lu")
      #FoodCn = project(FoodCI,WF,form_compiler_parameters={'quadrature_degree': qdeg,'quadrature_rule': 'default'})
         
   else:
      FoodCn = FoodC
   
   afactor = Constant(1)
   if params.isaxis:
      x       = SpatialCoordinate(mesh)
      afactor = 2*pi*x[1]
   
   params.intFoodCG = assemble(FoodCn*afactor*Measure("dx", domain=mesh))#,form_compiler_parameters={'quadrature_degree': 20,'quadrature_rule': 'special_quadrature'})
   if rank == 0: print_food_info(params)
   params.intFoodCn = params.intFoodCG
   
   return FoodCn


def compute_physical_quantities(params,u,p,mesh):
   mu = viscosity(u)
   params.visdiss = assemble(inner(2*mu*sym(grad(u)), sym(grad(u)))*dx)
   params.numvisdiss = assemble(p*div(u)*dx)
   print_quantities(params)


def convective_velocity(params,mesh,mesh_facets,ufinal,meshdisp,dtc):
   beta = ufinal - meshdisp/dtc
   beta_proj = project(beta,VectorFunctionSpace(mesh, "Lagrange", params.orderU),solver_type="mumps")
   if params.impzeroconvel:
      for ig in range(params.nbg):
         for ib in range(params.nballg[ig]):
            bcball = DirichletBC(params.W.sub(0).collapse(), Constant((0, 0)), mesh_facets, params.btagini+ib)
            bcball.apply(beta_proj.vector())
   
   if False:
      beta_proj.rename("beta","beta")
      params.dfinal_pvd << beta_proj
      
   return beta_proj
   

def SolveFoodDiffusiveProblem(params,mesh,mesh_facets,EC,EU,thmetcon):
   FoodCn  = interpolate(params.iniFood, FunctionSpace(mesh, EC))
   zerovel = interpolate(Constant((0.0,0.0)), FunctionSpace(mesh, EU))
   
   for it in range(params.foodininsteps):
      FoodCn = SolveFoodTransportProblem(params,mesh,mesh_facets,EC,FoodCn,zerovel,zerovel,params.dtdiffp,thmetcon)
      if False:
         FoodCn.rename("foodini","foofini")
         params.dfinal_pvd << FoodCn
      
   afactor = Constant(1)
   if params.isaxis:
      x       = SpatialCoordinate(mesh)
      afactor = 2*pi*x[1]

   params.intFoodCn = assemble(FoodCn*afactor*Measure("dx", domain=mesh))
   params.intFoodCG = params.intFoodCn
   return FoodCn


def compute_foodc_quantities(params,FoodC,mesh,mesh_facets):
   afactor = Constant(1)
   if params.isaxis:
      x       = SpatialCoordinate(mesh)
      afactor = 2*pi*x[1]
   ds = Measure("ds", domain=mesh, subdomain_data=mesh_facets)
   bcount = 0
   params.fluxperbody = []
   for ig in range(params.nbg):
      for ib in range(params.eatingbody if (params.eatingbody <= params.nballg[ig]) else params.nballg[ig]):
         params.fluxperbody.append(assemble(params.Alphaeat*FoodC*afactor*ds(params.btagini+bcount)))
         bcount += 1
   params.intFoodCA = assemble(FoodC*afactor*Measure("dx", domain=mesh))
   
   if params.solveQL:
      for ib in range(len(params.fluxperbody)):
         params.fluxperbodyaction[ib] += params.fluxperbody[ib]