from dolfin import *
import numpy as np

import os

###############################################################################
class Params(dict):
    """ https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    Example:
    m = Params({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs):
        super(Params, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Params, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Params, self).__delitem__(key)
        del self.__dict__[key]

###############################################################################
def IO_definitions(params,rank):
   if rank == 0:
      if not os.path.exists(params.dir_):
          os.mkdir(params.dir_)
          
      if not os.path.exists(params.dir_+"/outs"):
          os.mkdir(params.dir_+"/outs")
          
      if not os.path.exists(params.dir_+"/mesh"):
          os.mkdir(params.dir_+"/mesh")
   
      params.evoldofsfile      = '/evoldofs.txt'
      file_txt = open(params.dir_+params.evoldofsfile, 'w')
      file_txt.write("time n      ")
      for ig in range(1): #range(params.nballs):
         file_txt.write(f"pos body {ig:5d}                    angle body {ig:5d}")
         file_txt.write(f" vel body {ig:5d}                    vel ang body {ig:5d}")
      file_txt.write("\n")
      file_txt.close()
      params.evolrewfile   = '/evolreward.txt'
      file_txt = open(params.dir_+params.evolrewfile, 'w')
      file_txt.close()
      params.evolstatefile = '/evolstates.txt'
      file_txt = open(params.dir_+params.evolstatefile, 'w')
      file_txt.close()
      params.evolqmatrix   = '/evolqmatrix.txt'
      file_txt = open(params.dir_+params.evolrewfile, 'w')
      file_txt.close()
      params.evolfoodcfile = '/evolfoodc.txt'
      file_txt = open(params.dir_+params.evolfoodcfile, 'w')
      file_txt.write("time n      ")
      for ig in range(params.nballs):
         file_txt.write(f"food body {ig:5d}  ")
      file_txt.write("food m. n ALE    ")
      file_txt.write("food m. n GMSH   ")
      file_txt.write("food m. n-1 GMSH ")
      file_txt.write("food mass lost   ")
      file_txt.write("\n")
      file_txt.close()
      params.evolmeshinfo = '/evolmeshi.txt'
      file_txt = open(params.dir_+params.evolmeshinfo, 'w')
      file_txt.write("remeshing time\n")
      file_txt.close()
      params.evolquantities = '/evolquantities.txt'
      file_txt = open(params.dir_+params.evolquantities, 'w')
      file_txt.write("time n      Viscous dissipation\n")
      file_txt.close()
   
   if params.fileio == 'xdmf':
      filex = XDMFFile(params.dir_+'/sol.xdmf')
      filex.parameters['functions_share_mesh'] = True
      filex.parameters['rewrite_function_mesh'] = True
      filex.parameters["flush_output"] = True
   else:
      params.ufinal_pvd = File(params.dir_+"/outs/ufinal.pvd")
      params.pfinal_pvd = File(params.dir_+"/outs/pfinal.pvd")
      params.phifin_pvd = File(params.dir_+"/outs/phifin.pvd")
      params.visfin_pvd = File(params.dir_+"/outs/visfin.pvd")
      params.dfinal_pvd = File(params.dir_+"/outs/dfinal.pvd")
      params.foodcn_pvd = File(params.dir_+"/outs/foodcn.pvd")

###############################################################################
def generalized_force(t, nsd):
   if True:
      FF = [Constant(0.0),Constant(0.0),Constant(0.0)]
      TT = [Constant(0.0),Constant(0.0),Constant(0.0)]
   else:
      A = 1
      omega = 1
      phi = 0
      FF = [Constant(0.0),Constant(0.0),Constant(0.0)]
      TT = [Constant(0.0),Constant(0.0),Constant(A*cos(omega*t+phi))]

   if (nsd == 2):
      FT = [FF[0],FF[1],TT[2]]
   else:
      FT = FF+TT

   return FT

###############################################################################
def print_solutions_user(params,mesh,ufinal,pfinal,visfin):   
   ufinal_proj = project(ufinal,VectorFunctionSpace(mesh, "Lagrange", params.orderU),solver_type="mumps")
   pfinal_proj = pfinal #project(pfinal,FunctionSpace(mesh, "Lagrange", params.orderP),solver_type="mumps")
   visfin_proj = project(visfin,FunctionSpace(mesh, "Lagrange", params.orderP),solver_type="mumps")
   ufinal_proj.rename("u","u")
   pfinal_proj.rename("p","p")
   visfin_proj.rename("mu","mu")

   if params.fileio == 'xdmf':
      filex.write(ufinal_proj, it)
      filex.write(pfinal_proj, it)
   elif params.fileio == 'pvd':
      params.ufinal_pvd << ufinal_proj
      params.pfinal_pvd << pfinal_proj
      #params.visfin_pvd << visfin_proj

###############################################################################
def print_food_info(params):
      file_txt = open(params.dir_+params.evolfoodcfile, 'a+')
      file_txt.write("%.5e " % (params.time))
      bcount = 0
      for ig in range(params.nbg):
         for ib in range(params.eatingbody if (params.eatingbody <= params.nballg[ig]) else params.nballg[ig]):
            file_txt.write("%.10e " % (params.fluxperbody[ib]))
      file_txt.write("%.10e " % (params.intFoodCA))
      file_txt.write("%.10e " % (params.intFoodCG))
      file_txt.write("%.10e " % (params.intFoodCn))
      file_txt.write("%.10e "  % (params.intFoodCA-params.intFoodCG))
      file_txt.write("\n")
      file_txt.close()

############################################################################### 
def print_pdofs(params):
   file_txt = open(params.dir_+params.evoldofsfile, 'a+')
   file_txt.write("%.5e " % (params.time))
   for ig in range(params.nbg):
      for ib in range(params.nballg[ig]):
         file_txt.write("%.10e %.10e %.10e" % (params.xcn[ig][ib][0], params.xcn[ig][ib][1], params.thn[ig][ib]))
         file_txt.write(" %.10e %.10e %.10e " % (params.vcn[ig][ib][0], params.vcn[ig][ib][1], params.omn[ig][ib][2]))
   if params.movingframe:
      file_txt.write("%.10e %.10e" % (params.VF[0], params.VF[1]))
   file_txt.write("\n")
   file_txt.close()
   
############################################################################### 
def print_meshinfo(params):
   file_txt = open(params.dir_+params.evolmeshinfo, 'a+')
   file_txt.write("%.5e" % (params.time))
   file_txt.write("\n")
   file_txt.close()
   
############################################################################### 
def print_quantities(params):
   file_txt = open(params.dir_+params.evolquantities, 'a+')
   file_txt.write("%.5e " % (params.time))
   file_txt.write("%.10e "  % (params.visdiss))
   file_txt.write("%.10e"  % (params.numvisdiss))
   file_txt.write("\n")
   file_txt.close()

###############################################################################
class initial_FoodC(UserExpression):
   def __init__(self):
      UserExpression.__init__(self) # Call base class constructor!

   def eval(self, values, x):         
      m = np.floor((x[0]-2)/6)
      n = np.floor((x[1]-2)/6)
      
      if (m >= (x[0]-6)/6) and (n >= (x[1]-6)/6):
         values[0] = 1.0
      else:
         values[0] = 0.0

   def value_shape(self):
      return ()