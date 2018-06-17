# loading the required libraries
import matplotlib
# this is required to 'plot' inside the CLI
matplotlib.use('AGG')

# load other libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import sys
import os
from tqdm import tqdm
import warnings

# avoid matplotlib warnings
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

################################################################################
#                                    INPUTS                                    # 
################################################################################

# Get the inputs from the terminal line
filename = str(sys.argv[1])
route = str(sys.argv[2])

# colormap definition
levels = [0, 0.5, 1.5, 2.5]
colors = ['white', 'red', [0.4,0.4,0.4]]
grainStates, norm = matplotlib.colors.from_levels_and_colors(levels, colors)

################################################################################
#                                  FUNCTIONS                                   # 
################################################################################

def cellUpdate(target, neighbors, pD = 0.99, pB = 0.99):
    '''Function to update the cells of the CA 
    
    INPUTS:
       target: target cell value
       neighbors: numpy.ndarray with the corresponding neighbors
       state: state matrix with the initial state of the grain
       pD: probability of the fire of dying out (default = 0.99)
       pB: burning probability of the fire (default = 0.99)
       
    OUTPUTS:
       newState: state of the target cell after update'''
    
    # if the cell is empty
    if target == 0:
        # it will remain empty
        newState = 0
        
    # if the cell is fire
    if target == 1:
        # it may go to empty
        if np.random.rand() < pD:
            newState = 0
        # or it may stay as fire
        else:
            newState = 1
            
    # if the cell is fuel
    if target == 2:
        # neighbor comparison, if there are any 0 or 1
        if np.logical_or(neighbors == 0, neighbors == 1).sum() != 0:
            # it may get in fire
            if np.random.rand() < pB:
                newState = 1
            # or it may stay as fuel 
            else:
                newState = 2
        # if there are no 'fire' nor 'empty', it will remain as fuel
        else:
            newState = 2
            
    return newState

################################################################################

def initialization(filename, route):
    '''Function to update the cells of the CA 
    
    INPUTS:
        filename: name of the file with initialState as squared
                  numpy.ndarray with 0 and 1
        route: folder directory of the location

    OUTPUTS:
       initialState: corrected file to have only 0 and 2'''
    
    # load the textfile with the desired name
    initialState = np.loadtxt(route+filename)
    
    # check that the array is squared
    if initialState.shape[0] != initialState.shape[1]:
        # if it is not squared, raise an error
        raise ValueError('Array has not a squared shape')
    # otherwise, return the corrected fuel-empty array
    return 2*initialState

################################################################################

def initialState(r, theta, state, filename = None, route = None, savefigure = False):
    '''Plotting of the initial state of the grain distribution 
    
    INPUTS:
       r: 1D array with the coordinates of the radial position
       theta: 1D array with the coordinates of the tangential position
       state: state matrix with the initial state of the grain
       route: folder directory of the location
       
    OPTIONAL INPUTS:
        filename (default=None): name of the output file with the initial state    
        savefigure (default=False): flag to save the iteration figure
       
    OUTPUTS:
       just a plot with the figure'''
    
    # using squared matrix, the dimension will be NxN, so get it from R
    N = len(r)
    
    # create the meshgrid with the 1D arrays
    R, TH = np.meshgrid(r,theta)
    
    # given that the array is created without 2*np.pi, let's complete the 
    # circle to create the contour by setting the same values in the state
    # for 0 and for 2*np.pi (correcting also the other arrays)
    FTH = np.vstack((TH, np.ones(N)*2*np.pi))
    FR = np.vstack((R, np.ones(N)*R[0,:]))
    Fst = np.reshape(np.append(state,state[0,:]),(N+1,N))

    # plot the initial state with a binary colormap
    fig = plt.figure(1, figsize = (15,15), dpi = 50)
    ax = plt.subplot(111, projection = 'polar')
    CS = ax.pcolormesh(FTH, FR, Fst, cmap = plt.cm.get_cmap('binary'))
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(False)
    
    # save the figure if it explicitilly said
    if savefigure:
        plt.savefig( route + filename + '.png', bbox_inches='tight')
        
################################################################################

def burningState(r, theta, state, iteration = None, filename = None, savefigure = False):
    '''Plotting of a state (t) of the grain distribution whereas burning

    INPUTS:
       r: 1D array with the coordinates of the radial position
       theta: 1D array with the coordinates of the tangential position
       state: state matrix with the current state of the grain

    OPTIONAL INPUTS:
        iteration (default=None): iteration for saving the figure
        filename (default=None): name of the output file with the initial state    
        savefigure (default=False): flag to save the iteration figure

    OUTPUTS:
       just a plot with the figure'''
    
    # using squared matrix, the dimension will be NxN, so get it from R
    N = len(r)
    
    # create the meshgrid with the 1D arrays
    R, TH = np.meshgrid(r,theta)
    
    # given that the array is created without 2*np.pi, let's complete the 
    # circle to create the contour by setting the same values in the state
    # for 0 and for 2*np.pi (correcting also the other arrays)
    FTH = np.vstack((TH, np.ones(N)*2*np.pi))
    FR = np.vstack((R, np.ones(N)*R[0,:]))
    Fst = np.reshape(np.append(state,state[0,:]),(N+1,N))

    # # plot the state with a custom colormap
    fig = plt.figure(1, figsize = (15,15), dpi = 50)
    ax = plt.subplot(111, projection = 'polar')
    ax.contourf(FTH, FR, Fst, 2, cmap = grainStates, norm = norm)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(False)
    
    # save the figure if it explicitilly said
    if savefigure:
        plt.savefig('state%s%i.png' %(filename, iteration), bbox_inches='tight')
    
    # before ending the function, clear all figures
    plt.clf()

################################################################################
    
def burningLength(state):
    '''Calculation of the burning (area) length of each state
    
    INPUTS:
        state: numpy.ndarray with the states (0,1,2) 
        
    OUTPUTS:
       burningLength: true burning length of the state'''
    
    # initial state have a burning length of zero
    if len(np.unique(state)) == 2:
        return 0
    
    # when there is burning length
    else:
        # get the number of divisions of the grid
        N = state.shape[0]
        # create the inter-cell radius 
        Rinter = np.linspace(0,1,N+1)
        # copy the state matrix
        mat = np.copy(state)
        # replace the fuel cells with 'fire' cells 
        np.place(mat, mat==2, 1)
        # radial direction variation
        radDir = (np.abs(mat[:,0:-1]-mat[:,1:])*2*np.pi/N*Rinter[1:-1]).sum()
        # tangential direction variation
        tanDir = (np.abs(mat-np.vstack((mat[1:,:],mat[0,:])))/N).sum()
        # return the sum of both fire fronts
        return tanDir + radDir
    
################################################################################
#                               CELLULAR AUTOMATA                              # 
################################################################################

# create a list to store the states along time
states = []

# select the type of neighborhood ('moore' or 'vonNeumann')
neighborhoodType = 'vonNeumann'

# load the desired file
states.append(initialization(filename, route))

# preallocate space for the intermediate state
intState = np.zeros((states[0].shape))

# time counter
t = 0

# print in CLI current state
print('Beginning CA process ...')

# create a progress bar with an approximation of the time
# by computing the maximum fuel cells in the radial direction
with tqdm(total=np.floor(1.1*states[0].sum(axis=1).mean()/2),desc='(estimated time)') as pbar:
    # loop until there are only zeros in the state
    while np.sum(states[t] == 0) != states[t].shape[0]**2:

        # loop over all the cells of the array
        # i direction is the tangential direction
        for i in range(states[t].shape[0]):

            # j direction is the ratial way
            for j in range(states[t].shape[1]):

                # get te neighbors for the two possible types of neighborhood

                # abs((i+1) % states[t].shape[0]) is there for the case where
                # i = states[t].shape[0]-1 which will return error for invalid index 

                # More neighborhood type selection
                if neighborhoodType == 'moore':
                    # if the index is zero, it will be the center    
                    if j == 0:
                        neighbors = np.array([states[t][i-1,j],states[t][i-1,j+1],
                                              states[t][i,j+1],
                                              states[t][abs((i+1) % states[t].shape[0]),j],
                                              states[t][abs((i+1) % states[t].shape[0]),j+1]])
                    # if the index is maximum, it will be the rocket wall
                    elif j == states[t].shape[1]-1:
                        neighbors = np.array([states[t][i-1,j-1],states[t][i-1,j],
                                              states[t][i,j-1],
                                              states[t][abs((i+1) % states[t].shape[0]),j-1],
                                              states[t][abs((i+1) % states[t].shape[0]),j]])
                    # for the 'i' direction the boundaries are connected
                    # so all the other cases will follow the same procedure
                    else:
                        neighbors = np.array([states[t][i-1,j-1],states[t][i-1,j],states[t][i-1,j+1],
                                              states[t][i,j-1],states[t][i,j+1],
                                              states[t][abs((i+1) % states[t].shape[0]),j-1],
                                              states[t][abs((i+1) % states[t].shape[0]),j],
                                              states[t][abs((i+1) % states[t].shape[0]),j+1]])    

                # von Neumann neighborhood type selection
                else:
                    # if the index is zero, it will be the center    
                    if j == 0:
                        neighbors = np.array([states[t][i-1,j],
                                              states[t][abs((i+1) % states[t].shape[0]),j],
                                              states[t][i,j+1]])
                    # if the index is maximum, it will be the rocket wall
                    elif j == states[t].shape[1]-1:
                        neighbors = np.array([states[t][i-1,j],
                                              states[t][abs((i+1) % states[t].shape[0]),j],
                                              states[t][i,j-1]])
                    # for the 'i' direction the boundaries are connected (no constraints)
                    # so all the other cases will follow the same procedure
                    else:
                        neighbors = np.array([states[t][i-1,j],
                                              states[t][abs((i+1) % states[t].shape[0]),j],
                                              states[t][i,j-1],states[t][i,j+1]])

                # compute the new value of the target cell
                intState[i,j] = cellUpdate(states[t][i,j],neighbors)

        # once the whole array has been updatedd, append it to the list
        states.append(np.copy(intState))

        # increase the time counter
        t += 1
    
        # increase one the progress bar
        pbar.update(1);
    
# print in CLI current state
print('CA process has ended ...')
    
################################################################################
#                                    PLOTTING                                  # 
################################################################################

# get the size of the array
N = states[0].shape[0]

# linspace for the radial direction
r = np.linspace(0,1,N)

# tangential direction (avoiding computation of the 0 = 2*np.pi)
theta = np.linspace(0,2*np.pi*(N-1)/N,N)

# plot the initial state
initialState(r, theta, states[0], filename=filename, route=route, savefigure=True)

# create a bar to show progress
with tqdm(total=len(states),desc='State plotting') as pbar:
    # plot also every other state
    for i in range(len(states)):
        burningState(r,theta,states[i],iteration=i,filename=filename,savefigure=True);
        pbar.update(1);
    
# print in CLI current state
print('Creation of the gif...')
    
# with all those pictures create a fancy gif with the burning process
os.system('convert -delay 2 $(ls -v state%s*.png) %s.gif' %(filename, filename))

# also delete the individual pics in favour of the gif
os.system('rm state%s*.png' %filename)

################################################################################
#                             THRUST CALCULATIONS                              # 
################################################################################

# compute the normalized thrust
Ttilda = np.array([burningLength(states[i]) for i in range(len(states))])

# plot the thrust profile
fig, ax = plt.subplots(1, figsize=(8,6))
ax.set_xlabel('Iteration time', fontsize=14)
ax.set_ylabel('$\widetilde{T}$', fontsize=14)
ax.set_title(filename, fontsize=16)
ax.plot(Ttilda, 'k', lw=2)
plt.savefig('%sTP.png' %filename, dpi = 200, bbox_inches = 'tight')

# move the files to the corresponding folder
os.system('mv ./%sTP.png %s%sTP.png' %(filename, route, filename))
os.system('mv ./%s.gif %s%s.gif' %(filename, route, filename))