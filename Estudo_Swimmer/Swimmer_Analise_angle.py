import numpy as np
import copy
import random

# Once and for all (do not mix np.random and random)
np.random.seed(seed=0)

# Partition of the circle
Nangs = 2
delta_ang = 2*np.pi/Nangs
opt_ang = 2

# Number of balls
nballs = 4
nlinks = nballs - 1

# States-Actions
nstates      = 2 ** nlinks           # internals
nstates_glob = (2 ** nlinks) * Nangs # Total
nactions = nlinks

min_angle_links = -np.pi/4.0
max_angle_links = np.pi/4.0

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

def ComputeAngleStateBody(theta_head, internal_state, opt_ang):

    if(opt_ang == 1):
        angle_swimmer = theta_head
    elif(opt_ang == 2):
        angle_swimmer = 0.0
        for i in range(nlinks):
            if(internal_state[i] == 0):
                xi = min_angle_links
            else:
                xi = max_angle_links
            angle_swimmer += (nlinks - i)*xi / (nlinks+1)
        angle_swimmer += theta_head

    angle_aux = np.abs(angle_swimmer) - int(np.abs(angle_swimmer)/(2*np.pi))*2*np.pi
    if(angle_swimmer < 0.0):
        angle_aux = -angle_aux
        
    if(angle_aux < 0.0):
        angle_swimmer += 2*np.pi
        
    thetap = angle_swimmer - int(angle_swimmer/(2*np.pi))*2*np.pi
    itheta = int(thetap/delta_ang)
    
    return angle_swimmer, itheta

# Geometry and initial position
Lx, Ly = 3 * 150.0, 3 * 150.0  # Check consistency with ComputeStep()
current_xc_swimmer = np.array([Lx / 2.0, Ly / 2.0])
current_theta_swimmer = -np.pi/4
current_disp = np.zeros(2)
current_state = np.zeros(nlinks, dtype=int)  # As a vector
for i in range(nlinks):
    current_state[i] = random.randint(0, 1)

advancing_mode = "CM"  # CM: Tracking center of mass
                       # LB: Tracking position of reference body (leftmost)

# -----------------------------
# --- Loop over steps

evolqlquantities_file = open('evolqlquantities.txt', 'a+')
steps = 200000

evolqlquantities_file.write("current_state, action, next_state, current_xcm, current_ycm, next_xcm, next_ycm, current_theta, next_theta\n")
evolqlquantities_file.flush()
for it in range(steps):              
    
    # Learning decisions ----------------------    
    action2exec = choose_action()
    action = np.argmax(action2exec)
    next_state = compute_next_state(current_state, action2exec)

    # Internal states
    current_state_dec = int(np.dot(current_state, 2 ** np.arange(nlinks - 1, -1, -1, dtype=int)))      
    next_state_dec    = int(np.dot(next_state,    2 ** np.arange(nlinks - 1, -1, -1, dtype=int)))
        
    if advancing_mode == "CM":
        current_disp[0] = Deltas[0][current_state_dec, action]
        current_disp[1] = Deltas[1][current_state_dec, action]
    elif advancing_mode == "LB":
        current_disp[0] = Deltas[4][current_state_dec, action]
        current_disp[1] = Deltas[5][current_state_dec, action]

    current_dtheta = Deltas[2][current_state_dec, action]       

    theta = current_theta_swimmer
    Rmatrix = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )

    # Update dofs -------------------------
    next_xc_swimmer = current_xc_swimmer + Rmatrix @ current_disp
    next_theta_swimmer = current_theta_swimmer + current_dtheta

    current_angle_swimmer, current_itheta  = ComputeAngleStateBody(current_theta_swimmer, current_state, opt_ang)
    next_angle_swimmer   , next_itheta     = ComputeAngleStateBody(next_theta_swimmer,    next_state,    opt_ang)

    current_state_dec_glob  = current_state_dec  + current_itheta  * nstates
    next_state_dec_glob     = next_state_dec     + next_itheta * nstates
    
    evolqlquantities_file.write("%d,%d,%d,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e\n" % 
                                (current_state_dec_glob, action,
                                 next_state_dec_glob,
                                 current_xc_swimmer[0], current_xc_swimmer[1],
                                 next_xc_swimmer[0], next_xc_swimmer[1],
                                 current_theta_swimmer, next_theta_swimmer)) # head angles               
    evolqlquantities_file.flush()
    current_state = copy.deepcopy(next_state)
    current_xc_swimmer = copy.deepcopy(next_xc_swimmer)
    current_theta_swimmer = next_theta_swimmer

    # --- End loop over steps
    # --------------------------------  
    
    
  
    
    






  
