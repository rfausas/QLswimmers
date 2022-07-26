from dolfin import *

import numpy as np
import os
import gmsh
import time as tim
import meshio
import random
from .user_utils import print_meshinfo

###############################################################################
def rotation_matrix(angle):
   Q = np.identity(3)
   Q[0,0] =  np.cos(angle)
   Q[0,1] = -np.sin(angle)
   Q[1,0] =  np.sin(angle)
   Q[1,1] =  np.cos(angle)
   return Q


def load_geometry(params,mesh,mesh_facets,meshqual,rank,size,comm,remeshingm): ####################
   if remeshingm: # From gmsh #
      meshfile = params.dir_+"/mesh/meshdom"
      create_gmshmesh(params,meshfile,rank,size,comm)
      if rank == 0: create_xdmf(meshfile)
      comm.Barrier()
      mesh = Mesh()
      with XDMFFile(f"{meshfile}_2d.xdmf") as infile:
         infile.read(mesh)
      mvc = MeshValueCollection("size_t", mesh, 1) 
      with XDMFFile(f"{meshfile}_1d.xdmf") as infile:
         infile.read(mvc)
      mesh_facets = cpp.mesh.MeshFunctionSizet(mesh, mvc)
      if rank == 0: print_meshinfo(params)

   #meshqual = MeshQuality.radius_ratio_min_max(mesh)
   # Mesh info #
   if params.gmshverbose and rank == 0:
      print("    |-Mesh done")
      print("    |--Number of vertices = "+str(mesh.num_vertices()))
      print("    |--Number of edges = "+str(mesh.num_edges()))
      print("    |--Number of cells = "+str(mesh.num_cells()))
      print("    |--Cell size hmax,hmin = %.3g %.3g" % (mesh.hmax(), mesh.hmin()))
      print("    |--Mesh quality radius ratio: min = "+str(meshqual[0])+" and max = "+str(meshqual[1]))
   
   return mesh, mesh_facets, meshqual


def create_mesh(mesh, cell_type, prune_z=False):
   cells = mesh.get_cells_type(cell_type)
   cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
   out_mesh = meshio.Mesh(points=mesh.points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
   if prune_z:
      out_mesh.prune_z_0()
   return out_mesh


def create_gmshmesh(params,meshfile,rank,size,comm):
   random.seed(params.it)
   xmin, ymin = params.xmin, params.ymin
   Lx, Ly = params.Lx, params.Ly
   lcar, lcor, lbod = params.lcar, params.lcor, params.lbod #* (random.uniform(0.85,1.))
   gmsh.initialize()
   if rank == 0:
      gmsh.model.add("gmsh model")
      
      # Boundary box #
      pb0 = gmsh.model.geo.addPoint(xmin,   ymin,   0,lcor)
      pb1 = gmsh.model.geo.addPoint(xmin+Lx,ymin,   0,lcor)
      pb2 = gmsh.model.geo.addPoint(xmin+Lx,ymin+Ly,0,lcor)
      pb3 = gmsh.model.geo.addPoint(xmin,   ymin+Ly,0,lcor)
      lb0 = gmsh.model.geo.addLine(pb0,pb1)
      lb1 = gmsh.model.geo.addLine(pb1,pb2)
      lb2 = gmsh.model.geo.addLine(pb2,pb3)
      lb3 = gmsh.model.geo.addLine(pb3,pb0)
      llb = gmsh.model.geo.addCurveLoop([lb0,lb1,lb2,lb3])
      
      # Holes - swimmer bodies #
      balls = [llb]
      btags = []
      for ig in range(params.nbg):
         for ib in range(params.nballg[ig]):
            xc = params.xc[ig][ib].tolist()
            radii = params.radii[ig][ib]
            radih = params.radih[ig][ib]
            if params.swmtype == 'circle':
               pass
            elif params.swmtype == 'ellipse':
               pxmin = gmsh.model.geo.addPoint(xc[0]-radii,xc[1],0,lbod)
               pxmax = gmsh.model.geo.addPoint(xc[0]+radii,xc[1],0,lbod)
               pymin = gmsh.model.geo.addPoint(xc[0],xc[1]-radih,0,lbod)
               pymax = gmsh.model.geo.addPoint(xc[0],xc[1]+radih,0,lbod)
               pcent = gmsh.model.geo.addPoint(xc[0],xc[1],      0,lbod)
               angle, axis = angle_axis_from_rotation(params.Qr[ig][ib],params.nsd)
               gmsh.model.geo.rotate([(0,pxmin),(0,pxmax),(0,pymin),(0,pymax)],xc[0],xc[1],xc[2],axis[0],axis[1],axis[2],angle)
               ea1 = gmsh.model.geo.addEllipseArc(pxmax,pcent,pxmin,pymax)
               ea2 = gmsh.model.geo.addEllipseArc(pymax,pcent,pymin,pxmin)
               ea3 = gmsh.model.geo.addEllipseArc(pxmin,pcent,pxmax,pymin)
               ea4 = gmsh.model.geo.addEllipseArc(pymin,pcent,pymax,pxmax)
               balls.append(gmsh.model.geo.addCurveLoop([ea1,ea2,ea3,ea4]))
               btags.append([ea1,ea2,ea3,ea4])
            elif params.swmtype == 'square':
               pass
      
      # Fluid domain #
      fsurf = gmsh.model.geo.addPlaneSurface(balls)
      
      # Physical groups #
      gmsh.model.geo.synchronize()
      ptag = 1
      gmsh.model.addPhysicalGroup(1, [lb0], ptag)
      gmsh.model.setPhysicalName(1, ptag, "dwall")
      ptag += 1
      gmsh.model.addPhysicalGroup(1, [lb1], ptag)
      gmsh.model.setPhysicalName(1, ptag, "rwall")
      ptag += 1
      gmsh.model.addPhysicalGroup(1, [lb2], ptag)
      gmsh.model.setPhysicalName(1, ptag, "uwall")
      ptag += 1
      gmsh.model.addPhysicalGroup(1, [lb3], ptag)
      gmsh.model.setPhysicalName(1, ptag, "lwall")
      
      params.btagini = ptag+1  
      for ib in range(sum(params.nballg)):
         ptag += 1
         gmsh.model.addPhysicalGroup(1, btags[ib], ptag)
         gmsh.model.setPhysicalName(1, ptag, f"body{ib}")
      
      ptag += 1      
      gmsh.model.addPhysicalGroup(2, [fsurf], ptag)
      gmsh.model.setPhysicalName(2, ptag, "volume")
      
      # Meshing options #
      #np.random.seed(0)
      rn = 1 #np.random.randint(2)
      gmsh.option.setNumber("Mesh.Algorithm", 6*rn + 1*(1-rn))
      gmsh.option.setNumber("Mesh.ElementOrder", 1)
      gmsh.option.setNumber("General.Terminal", 1 if params.gmshverbose else 0)
      # gmsh.option.setNumber("Mesh.RandomFactor", 1e-5)
      # gmsh.option.setNumber("Mesh.RandomSeed", random.randint(1, 10000))
      gmsh.model.mesh.generate(params.nsd)
      
      # Write msh file #
      gmsh.write(f"{meshfile}.msh")
   
   gmsh.finalize()
   params.btagini = comm.scatter([params.btagini]*size, root=0)


def create_xdmf(meshfile):
   mesh_from_file = meshio.read(f"{meshfile}.msh")
   lines_mesh = create_mesh(mesh_from_file, "line",     prune_z=True)
   trian_mesh = create_mesh(mesh_from_file, "triangle", prune_z=True)
   
   meshio.write(f"{meshfile}_1d.xdmf", lines_mesh)    
   meshio.write(f"{meshfile}_2d.xdmf", trian_mesh)


def angle_axis_from_rotation(Q,nsd):
   if (nsd == 2):
      alpha = np.arctan2(Q[1,0],Q[0,0])
      rotax = np.array([0.,0.,1.])
   else:
      alpha = np.arccos((np.trace(Q)-1.0)/2.0)
      r = Rotation.from_matrix(Q)
      rotax = r.as_vector()
      if (np.sin(alpha) != 0):
         rotax = (0.5/np.sin(alpha))*np.array([Q[3,2]-Q[2,3],Q[1,3]-Q[3,1],Q[2,1]-Q[1,2]])
      else:
         M = Q+np.identity(3)
         for j in range(3):
            nrm = np.linalg.norm(M[:,j])
            if (nrm > 0):
               rotax = M[:,j]/nrm

   return alpha, rotax


def ALE_mesh_movement(params,mesh,mesh_facets):
   meshdisp = SolveLinearElasticityProblem(params,mesh,mesh_facets)
   ALE.move(mesh, meshdisp)
   # new_mesh_facets = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
   # new_mesh_facets.set_values(mesh_facets.array())
   # mesh_facets = new_mesh_facets
   # if False and params.print_solutions:
   #    meshdisp.rename("d","d")
   #    params.dfinal_pvd << meshdisp
   mesh.bounding_box_tree().build(mesh)
   if False and params.print_solutions:
      meshdisp.rename("d","d")
      params.dfinal_pvd << meshdisp
      
   return meshdisp


def SolveLinearElasticityProblem(params,mesh,mesh_facets):
   nsd     = params.nsd
   nbg     = params.nbg
   nballg  = params.nballg
   btagini = params.btagini
   xc      = params.xc
   xcn     = params.xcn
   Qr      = params.Qr
   Qrn     = params.Qrn

   # Finete element #
   orderD = 1
   ED = VectorElement("Lagrange", 'triangle', orderD)
   W = FunctionSpace(mesh, ED)
   u = TrialFunction(W)
   v = TestFunction(W)
   
   # Variational fomulation #
   E, nu = 10.0, 0.3
   mu = E/(2.0*(1.0 + nu))
   lmbda = E*nu/((1.0 + nu)*(1.0 -2.0*nu))
   sig   = 2*mu*sym(grad(u)) + lmbda*tr(grad(u))*Identity(nsd)
   F = inner(sig, grad(v))*dx
   a, L = lhs(F), rhs(F)
   
   # Boundary conditions: Box
   noslip = Constant((0, 0))
   bcs = []
   bcsl = DirichletBC(W, noslip, mesh_facets, 4) #left
   bcs.append(bcsl)
   bcsr = DirichletBC(W, noslip, mesh_facets, 2) #right
   bcs.append(bcsr)
   bcsd = DirichletBC(W, noslip, mesh_facets, 1) #down
   bcs.append(bcsd)
   bcsu = DirichletBC(W, noslip, mesh_facets, 3) #up
   bcs.append(bcsu)   
   
   # Boundary conditions: Bodies
   bcount = 0
   for ig in range(nbg):
      for ib in range(nballg[ig]):
         Qaux = np.dot(Qr[ig][ib],Qrn[ig][ib].transpose())-np.identity(3)
         daux = xc[ig][ib]-xcn[ig][ib]
         dcx  = "da0 + Qa00*(x[0]-xnc0) + Qa01*(x[1]-xnc1)"
         dcy  = "da1 + Qa10*(x[0]-xnc0) + Qa11*(x[1]-xnc1)"
         dct = (dcx,dcy)
         if nsd == 3:
            dcx += " + Qa02*(x[2]-xnc2)"
            dcy += " + Qa12*(x[2]-xnc2)"
            dcz = "da2 + Qa20*(x[0]-xnc0) + Qa21*(x[1]-xnc1) + Qa22*(x[2]-xnc2)"
            dct = (dcx,dcy,dcz)
         dispBall = Expression(dct,da0=daux[0],da1=daux[1],da2=daux[2],Qa00=Qaux[0,0],Qa01=Qaux[0,1],Qa02=Qaux[0,2]
                                                                            ,Qa10=Qaux[1,0],Qa11=Qaux[1,1],Qa12=Qaux[1,2]
                                                                            ,Qa20=Qaux[2,0],Qa21=Qaux[2,1],Qa22=Qaux[2,2]
                                                                            ,xnc0=xcn[ig][ib][0],xnc1=xcn[ig][ib][1],xnc2=xcn[ig][ib][2]
                                                                            ,degree=4)
         # dispBall = rigid_body_displacement(xcn=xcn[ig][ib],xc=xc[ig][ib],Qrn=Qrn[ig][ib],Qr=Qr[ig][ib],nsd=nsd)
         bcball = DirichletBC(W, dispBall, mesh_facets, btagini+bcount)
         bcs.append(bcball)
         bcount += 1
   
   displacement = Function(W)
   solve(a==L, displacement, bcs, solver_parameters={'linear_solver': 'mumps'})

   return displacement
