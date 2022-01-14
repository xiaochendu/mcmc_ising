from time import perf_counter
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

"""Performs Monte Carlo Markov Chain (MCMC) simulation of the 2D Ising model without field.
Plots the Energy per spin and the Heat capacity per spin.
"""

def state_energy(state):
    """Calculate state energy"""
    energy = 0
    lattice_width = state.shape[0]
    for ii in range(lattice_width):
        for jj in range(lattice_width):
            site = state[ii, jj]
            # count number of  neighbors with periodic BC
            neighbors = state[(ii + 1) % lattice_width, jj] + state[(ii - 1) % lattice_width, jj] + \
            state[ii, (jj + 1) % lattice_width] + state[ii, (jj - 1) % lattice_width]
            # account for double counting
            energy += -0.5*site*neighbors
    return energy

def state_energy_sq(state):
    """Calculate the square of state energy"""
    return state_energy(state)**2

def state_net_mag(state):
    """Calculate net magnetization of state"""
    return abs(np.sum(state))

def state_mag_sq(state):
    """Calculate square of state magnetization"""
    return state_net_mag(state)**2

def transition_prob(curr_state, next_state, temp):
    """Calculate transition probability of moving to proposed state at given temperature"""
    # takes in the current lattice and next lattice states and temperature
    energy_diff = state_energy(next_state) - state_energy(curr_state)
    # print(f"current state:")
    # print(curr_state)
    # print(f"next state:")
    # print(next_state)
    # print(f"energy difference of {energy_diff} between current and proposed states")
    # set J/k = 1
    return min(1, np.exp(-energy_diff/temp))

def mcmc_run(num_runs=1000, temp=1, lattice_width=5, lattice=None):
    """Performs MCMC run with given parameters, initializing with a random lattice if not given an input.
    Each run is defined as one complete sweep through the lattice. Each sweep consists of randomly picking
    a cell and proposing (and accept/reject) a flip for a total number of times equals to the number of cells
    in the lattice. Only the resulting lattice after one run is appended to the history. Corresponding
    obversables are calculated also after each run.
    """
    if type(lattice) is not np.ndarray:
        # randomly initialize lattice
        print("initializing lattice")
        lattice = 2*np.random.randint(2, size=(lattice_width, lattice_width)) - 1
    history = deque()
    energy = deque()
    energy_sq = deque()
    net_mag = deque()
    mag_sq = deque()

    # sweep over # of lattices
    sweep_size = lattice_width*lattice_width
    
    for i in range(num_runs):
        for j in range(sweep_size):
            # random get one cell
            x = np.random.randint(0, high=lattice_width)
            y = np.random.randint(0, high=lattice_width)

            proposed_lattice = lattice.copy()
            # flip state
            proposed_lattice[x, y] = -proposed_lattice[x, y]
            # print(f"proposed change at ({x},{y})")

            # import pdb; pdb.set_trace()
            # check if transition succeeds
            if np.random.rand(1) < transition_prob(lattice, proposed_lattice, temp):
                # succeeds! change state
                # print("state at ({},{}) switched from {} to {}".format(x, y, lattice[x, y], proposed_lattice[x, y]))
                lattice[x, y] = proposed_lattice[x, y]
            else:
                # failed, keep current state
                # print("state kept the same")
                continue
        
        # end of sweep; append to history
        history.append(lattice.copy())

        # append values 
        energy.append(state_energy(lattice))
        energy_sq.append(state_energy_sq(lattice))
        net_mag.append(state_net_mag(lattice))
        mag_sq.append(state_mag_sq(lattice))

    return history, energy, energy_sq, net_mag, mag_sq

if __name__ == "__main__":
    # run for a range of temperatures with chain size = 1000 and using a 10 x 10 lattice
    temps = np.arange(0.2, 5.2, 0.2)
    num_runs = 1000
    lattice_width = 5
    N = lattice_width**2

    energy_results = np.random.rand(len(temps), num_runs)
    energy_sq_results = np.random.rand(len(temps), num_runs)
    starting_lattice = None
    for i, temp in reversed(list(enumerate(temps))):
        start = perf_counter()
        history, energy, energy_sq, net_mag, mag_sq = mcmc_run(num_runs=num_runs, temp=temp, \
            lattice_width=lattice_width, lattice=starting_lattice)
        # intitialize with previous temp simulation
        starting_lattice = history[-1]
        stop = perf_counter()
        print(f"Time taken = {stop - start} seconds for temp = {temp}")
        energy_results[i, :] = energy
        energy_sq_results[i, :] = energy_sq
    
    energy_sq_mean = np.mean(energy_sq_results, axis=1)
    energy_mean = np.mean(energy_results, axis=1)
    energy_mean_sq = np.power(energy_mean, 2)

    energy_per_spin = energy_mean/N

    # for normalizing the activities
    inverse_kbT_sq = 1/np.square(temps)
    C_v = (energy_sq_mean - energy_mean_sq)*inverse_kbT_sq
    C_v_per_spin = C_v/N

    import pdb; pdb.set_trace()

    fig, ax = plt.subplots()
    ax.plot(temps, energy_per_spin)
    ax.set_xlabel("Temperature (T)")
    ax.set_ylabel("Energy per spin (E/N)")
    ax.set_title("Energy per spin (E/N) vs Temperature (T)")
    fig.show()
    fig.savefig("energy_per_spin")

    fig, ax = plt.subplots()
    ax.plot(temps, C_v_per_spin)
    ax.set_xlabel("Temperature (T)")
    ax.set_ylabel("Heat capacity per spin (C/N)")
    ax.set_title("Heat capacity per spin (C/N) vs Temperature (T)")
    fig.show()
    fig.savefig("c_v_per_spin")
    import pdb; pdb.set_trace()
