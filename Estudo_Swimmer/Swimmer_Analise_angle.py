import numpy as np
import copy
import random

# Once and for all (do not mix np.random and random)
np.random.seed(seed=0)

# Partition of the circle
Nangs = 4
delta_ang = 2*np.pi/Nangs

# Number of balls
nballs = 4
nlinks = nballs - 1

# States-Actions
nstates      = 2 ** nlinks          # internals
nstates_glob = 2 ** nlinks + Nangs  # Total
nactions = nlinks

# Xposition_cm, Yposition_cm, Xposition_head, Yposition_head, Theta_head, Viscous_dissipation
ndeltas = 6
   
def ReadDeltasTables(ndeltas, nstates, nactions, filetable):
    ##print('     |> Loading table ', filetable)
    A = np.loadtxt(filetable)
    Deltas = []
    for d in range(ndeltas):
        Deltas.append(np.zeros(shape=(nstates, nactions)))
        for state in range(nstates):
            Deltas[d][state, :] = A[d * nstates + state, :]
    return Deltas

file_table = "DELTAS/" + "DELTASfino/" + "theta45/" + "deltas.txt"
Deltas = ReadDeltasTables(ndeltas, nstates, nactions, file_table)

def choose_action():
    aux = np.identity(nlinks, dtype=int)
    ret = aux[np.random.randint(0, nlinks), :]
    return ret

def compute_next_state(current_state, action):
    return np.remainder(current_state + action, 2 * np.ones(nlinks, dtype=int))   

# Geometry and position
Lx, Ly = 3 * 150.0, 3 * 150.0  # Check consistency with ComputeStep()
xc_swimmer = np.array([Lx / 2.0, Ly / 2.0])
xc_swimmer_n = np.array([Lx / 2.0, Ly / 2.0])
theta_swimmer = 0.0
current_disp = np.zeros(2)
current_state = np.zeros(nlinks, dtype=int)  # As a vector

for i in range(nlinks):
    current_state[i] = random.randint(0, 1)

current_state_dec_ini = int(np.dot(current_state, 2 ** np.arange(nlinks - 1, -1, -1, dtype=int)))        

advancing_mode = "CM"  # CM: Tracking center of mass
# LB: Tracking position of reference body (leftmost)

# -----------------------------
# --- Loop over steps

evolqlquantities_file = open('evolqlquantities.txt', 'a+')
steps = 1000000

evolqlquantities_file.write("Statei,Action,Staten,xcmi,ycmi,xcmn,ycmn,thetai,thetan\n")
evolqlquantities_file.flush()
for it in range(steps):              
    
    # Learning decisions ----------------------    
    action2exec = choose_action()
    action = np.argmax(action2exec)
    next_state = compute_next_state(current_state, action2exec)

    # Internal states
    current_state_dec = int(np.dot(current_state, 2 ** np.arange(nlinks - 1, -1, -1, dtype=int)))      
    current_state_decn = int(np.dot(next_state, 2 ** np.arange(nlinks - 1, -1, -1, dtype=int)))
        
    if advancing_mode == "CM":
        current_disp[0] = Deltas[0][current_state_dec, action]
        current_disp[1] = Deltas[1][current_state_dec, action]
    elif advancing_mode == "LB":
        current_disp[0] = Deltas[4][current_state_dec, action]
        current_disp[1] = Deltas[5][current_state_dec, action]

    current_dtheta = Deltas[2][current_state_dec, action]       

    Rmatrix = np.array(
        [
            [np.cos(theta_swimmer), -np.sin(theta_swimmer)],
            [np.sin(theta_swimmer), np.cos(theta_swimmer)],
        ]
    )

    # Update dofs -------------------------
    xc_swimmer_n = copy.deepcopy(xc_swimmer)
    xc_swimmer += Rmatrix @ current_disp
    theta_swimmer_n = theta_swimmer
    theta_swimmer += current_dtheta

    def ComputeAngleBody(theta_head, internal_state, opt_ang):

        if(opt_ang == 1):
            angle_state =
        elif(opt_ang == 2):
            angle_state = 
            
        itheta = int()
        
        return angle_state, itheta

    angle_state,  itheta = ComputeAngleBody()
    angle_staten, ithetan = ComputeAngleBody()
    
    current_state_dec_glob = current_state_dec + itheta * nstates
    current_state_decn_glob = current_state_decn + ithetan * nstates
    
    evolqlquantities_file.write("%d,%d,%d,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e\n" % 
                                              (current_state_dec,action,
                                              current_state_decn,
                                              xc_swimmer_n[0],xc_swimmer_n[1],
                                              xc_swimmer[0],xc_swimmer[1],
                                              theta_swimmer_n,theta_swimmer))               
    evolqlquantities_file.flush()
    current_state = copy.deepcopy(next_state) 

    # --- End loop over steps
    # --------------------------------  
    
    
  
    
    






  
