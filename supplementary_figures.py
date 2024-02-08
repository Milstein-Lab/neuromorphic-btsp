import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gs
import pickle
import os

from utils import update_plot_defaults, get_scaled_sigmoid, get_BTSP_function, get_simple_BTSP_function, Volatile_Resistor
from figure_4 import VO2_LIF_BTSP_simulation
from figure_5 import simulate_linear_track_SR, plot_linear_track_SR, plot_SR_correlation, simulate_gridworld_multiple_seeds, plot_grid, plot_SR_place_fields, plot_pathlength_over_trials, plot_navigation_summary, plot_cumulative_reward, GridWorld

########################################################################################################################
# Supplementary Figures
########################################################################################################################

def generate_Figure_S3(save):
    '''IV relationship for device-to-device and cycle-to-cycle variability'''
    dev2dev_variability_IV = pd.read_excel('data/device-to-device-data.xlsx', sheet_name='i-V', header=1)
    dev2dev_IV_voltage = np.array(dev2dev_variability_IV.iloc[:,0::2])
    dev2dev_IV_current = np.array(dev2dev_variability_IV.iloc[:,1::2])*1000

    cyc2cyc_variability_IV = pd.read_excel('data/cycle-to-cycle-data.xlsx', sheet_name='i-V', header=2)
    cyc2cyc_IV_voltage = np.array(cyc2cyc_variability_IV.iloc[:,0::2])
    cyc2cyc_IV_current = np.array(cyc2cyc_variability_IV.iloc[:,1::2])*1000

    mm = 1 / 25.4  # millimeters in inches
    fig, (ax1, ax2)  = plt.subplots(1,2,figsize=(180*mm, 80*mm))

    ax1.plot(dev2dev_IV_voltage, dev2dev_IV_current, c='k', alpha=0.3)
    ax1.set_xlabel('Voltage (V)')
    ax1.set_ylabel('Current (mA)')
    ax1.set_title('Device-to-device variability')

    ax2.plot(cyc2cyc_IV_voltage, cyc2cyc_IV_current, c='k', alpha=0.3)
    ax2.set_xlabel('Voltage (V)')
    ax2.set_ylabel('Current (mA)')
    ax2.set_title('Cycle-to-cycle variability') 

    plt.tight_layout()
    if save:
        fig.savefig('figures/0-Supplementary Figures/Supplementary_variability_IV.png',dpi=300)
        fig.savefig('figures/0-Supplementary Figures/Supplementary_variability_IV.svg',dpi=300)


########################################################################################################################

def generate_Figure_S5(save):
    '''VO2 model characterization (data vs sim for 70C relaxations)'''

    mm = 1 / 25.4  # millimeters in inches    
    fig = plt.figure(figsize=(180*mm, 100*mm))
    axes = gs.GridSpec(nrows=1, ncols=3, bottom=0.6, top=0.9, left=0.08, right=0.98, wspace=0.4)

    Isteps_data = pd.read_excel('data/VO2_data_currents.xlsx',header=0)

    # Extract relevant variables 
    data_time = np.array(Isteps_data['Time'][2:])
    Isteps = np.array(Isteps_data.iloc[1][1::4])
    voltage_traces = np.array(Isteps_data.iloc[2:,1::4]).T
    resistance_traces = np.array(Isteps_data.iloc[2:,2::4]).T
    current_traces = np.array(Isteps_data.iloc[2:,3::4]).T

    sim_time, R_hist, controlI_hist = VO2_test_pulse(dt=10,T=10000, stim_time=(0,3000), temperature=70)
    sim_time /= 1000

    ax = fig.add_subplot(axes[0])
    ax.plot(sim_time, controlI_hist, 'k', linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Control current (mA)')
    ax.set_title('Stimulation current')

    ax = fig.add_subplot(axes[1])
    for i,R in enumerate(resistance_traces):
        ax.plot(data_time[100:10000], 1/R[100:10000]*1000, label=Isteps[i], c='k', linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Conductance (mS)')
    ax.set_title('VO2 data')

    ax = fig.add_subplot(axes[2])
    ax.plot(sim_time, 1/R_hist*1000, c='r', linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Conductance (mS)')
    ax.set_title('VO2 model simulation')


    '''Plot properties of VO2 emulation'''
    # Simulation runtime parameters
    dt = 0.1  # time step (ms)
    T = 300   # simulation time (ms)
    time = np.arange(0., T, dt)
    axes = gs.GridSpec(nrows=1, ncols=4, bottom=0.1, top=0.4, left=0.08, right=0.98, wspace=0.7)

    line_color = [0.2,0.2,0.2]

    g_volatile = Volatile_Resistor(dt, stim_scaling=100.)
    controlI = np.linspace(0,1,1000)
    R_eq = g_volatile.transfer_func(controlI*g_volatile.stim_scaling)
    g_eq = 1/ R_eq

    ax = fig.add_subplot(axes[0])   
    ax.plot(controlI,R_eq, c=line_color)
    ax.set_xlabel('Control current (mA)')
    ax.set_ylabel('R_eq (Ω)')

    ax = fig.add_subplot(axes[1])
    ax.plot(controlI,1000*g_eq, c=line_color)
    ax.set_xlabel('Control current (mA)')
    ax.set_ylabel('Conductance_eq (mS)')

    ax = fig.add_subplot(axes[2])
    ax.plot(controlI,g_volatile.decay_tau(controlI=controlI*g_volatile.stim_scaling,temperature=70),c='r')
    ax.set_title('Temp = 70 °C')
    ax.set_xlabel('Control current (mA)')
    ax.set_ylabel('Conductance\ndecay tau (ms)')

    ax = fig.add_subplot(axes[3])
    ax.plot(controlI,g_volatile.decay_tau(controlI=controlI*g_volatile.stim_scaling,temperature=64),c='b')
    ax.set_title('Temp = 64 °C')
    ax.set_xlabel('Control current (mA)')
    ax.set_ylabel('Conductance\ndecay tau (ms)')

    if save:
        fig.savefig('figures/0-Supplementary Figures/Supplementary_VO2_emulation_properties.png',dpi=300)
        fig.savefig('figures/0-Supplementary Figures/Supplementary_VO2_emulation_properties.svg',dpi=300)

def VO2_test_pulse(dt, T, stim_time, temperature):
    '''
    Simulate a test pulse of a VO2 volatile resistor
    :param dt: time step (ms)
    :param T: simulation time (ms)
    '''

    time = np.arange(0., T, dt)

    R_hist_ls = []
    controlI_hist_ls = []
    for pulseI in np.arange(20,110,10):
        g_volatile = Volatile_Resistor(dt, temperature=temperature)
        for t in time:
            if t>stim_time[0] and t<stim_time[1]:
                g_volatile.controlI = pulseI  
            else:
                g_volatile.controlI = 0
            g_volatile.time_step()
        R_hist_ls.append(g_volatile.R_history)
        controlI_hist_ls.append(g_volatile.controlI_history)

    R_hist = np.array(R_hist_ls).T
    controlI_hist = np.array(controlI_hist_ls).T
    return time, R_hist, controlI_hist

########################################################################################################################

def generate_Figure_S6(save):
    '''Plot data from multiple simultaneous VO2 devices'''

    mm = 1 / 25.4  # millimeters in inches
    fig, ax = plt.subplots(1,1,figsize=(183*mm, 60*mm))

    multiVO2_data = pd.read_csv('data/three_vo2_devices Run 24 2023-05-23T15.14.52.csv',header=0)
    time = np.array(multiVO2_data['SMU-1 Time (s)'][2000:39_000])
    current = np.array(multiVO2_data['SMU-1 Current (A)'][2000:39_000])
    voltage = 0.1
    time -= time[0]

    g = current / voltage
    omit = np.where(g < 0)[0]
    th = 0.004
    transition_start_indexes = [0]
    jump = 6
    i = jump
    while i < len(g)-jump:
        if i in omit:
            i += 1
        elif i + jump not in omit:
            if (g[i+jump] - g[i] > th) and abs(g[i+jump+1] - g[i+jump]) < 0.002:
                transition_start_indexes.append(i+jump+2)
                i += jump
            else:
                i += 1
        else:
            i += 1
    transition_start_indexes = np.array(transition_start_indexes)

    g *= 1e3
    ax.plot(time[transition_start_indexes][::3], g[transition_start_indexes][::3], c='r')
    ax.plot(time[transition_start_indexes][1:][::3], g[transition_start_indexes][1:][::3], c='y')
    ax.plot(time[transition_start_indexes][2:][::3], g[transition_start_indexes][2:][::3], c='k')
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Conductance (mS)')

    plt.tight_layout()

    if save:
        fig.savefig('figures/0-Supplementary Figures/Supplementary_multiVO2.png',dpi=300)
        fig.savefig('figures/0-Supplementary Figures/Supplementary_multiVO2.svg',dpi=300)

########################################################################################################################

def generate_Figure_S9(save):
    """BTSP rule: heatmap and plasticity kernel"""

    # BTSP parameters from (Milstein et al., 2021, eLife) Fig.7
    sig_pot = get_scaled_sigmoid(slope=4.405, threshold=0.415)
    sig_dep = get_scaled_sigmoid(slope=20.0, threshold=0.026)
    k_dep = 0.425
    k_pot = 1.1097
    Wmax = 4.68
    btsp_func = get_BTSP_function(Wmax, k_pot, k_dep, sig_pot, sig_dep)

    mm = 1 / 25.4 # convert mm to inches
    fig, axes = plt.subplots(1,3,figsize=(180*mm,50*mm))

    plot_VO2_btsp_learning_rule(axes, btsp_func, dwell_time=400, plateau_dur=300, lr=0.012, ET_temp=74.34, IS_temp=70.82) # Regular ET
    plt.tight_layout()

    if save:
        fig.savefig('figures/0-Supplementary Figures/Supplementary_BTSP_heatmap.png',dpi=300)
        fig.savefig('figures/0-Supplementary Figures/Supplementary_BTSP_heatmap.svg',dpi=300)

def plot_VO2_btsp_learning_rule(axes, btsp_func, dwell_time, plateau_dur, lr, ET_temp, IS_temp, ET_scale=0.4, IS_scale=0.8, norm=False):
    ''' Plot BTSP learning rule (dW vs time from plateau) 
    
    :param T: float, dwell time (s)
    :param ET_rise_tau: float, ET rise time constant (s)
    :param ET_decay_tau: float, ET decay time constant (s)
    :param IS_rise_tau: float, IS rise time constant (s)
    :param IS_decay_tau: float, IS decay time constant (s)
    '''

    # Set parameters
    dt = 1 # ms
    tmax = 20_000  # ms
    t_resolution = 50 # ms
    w_resolution = 0.05
    w_range = (1,3)
    ET_controlI = 100
    IS_controlI = 100

    # Generate ET and IS template traces
    VO2_ET = Volatile_Resistor(dt, temperature=ET_temp, metalR=100, stim_scaling=1)
    VO2_IS = Volatile_Resistor(dt, temperature=IS_temp, metalR=100, stim_scaling=1)
    for t in range(0, tmax, dt):
        VO2_ET.controlI = 0
        VO2_IS.controlI = 0
        if t>0 and t<dwell_time:
            VO2_ET.controlI = ET_controlI
        if t>0 and t<plateau_dur:
            VO2_IS.controlI = IS_controlI
        VO2_ET.time_step()
        VO2_IS.time_step()
    ET = np.array(VO2_ET.g_history)
    ET_min = 1/VO2_ET.insulatorR
    ET_max = 1/VO2_ET.metalR * ET_scale
    ET = (ET-ET_min)/(ET_max-ET_min)
    ET = np.roll(ET, int(tmax/2))
    ET = np.minimum(ET, 1) # clip values to [0,1]

    IS = np.array(VO2_IS.g_history)
    IS_min = 1/VO2_IS.insulatorR
    IS_max = 1/VO2_IS.metalR * IS_scale
    IS = (IS-IS_min)/(IS_max-IS_min)
    IS = np.roll(IS, int(tmax/2))
    IS[0:int(tmax/2)] = 0 # zero values before the IS start
    IS = np.minimum(IS, 1) # clip values to [0,1]

    # print(f"ET min: {ET.min()}, ET max: {ET.max()}")
    # print(f"IS min: {IS.min()}, IS max: {IS.max()}")

    # Compute dW matrix
    all_delta_t = np.arange(-5000, 5000, t_resolution)
    ET_all = []
    for delta_t in all_delta_t:
        shifted_ET = np.roll(ET, int(delta_t))
        shifted_ET[0:int(delta_t+tmax/2)] = 0 # zero values before the ET start to prevent spillover
        ET_all.append(shifted_ET)
    ET_all = np.array(ET_all)

    num_ETs = ET_all.shape[0]
    num_timesteps = ET_all.shape[1]

    w_init = np.arange(w_range[0], w_range[1], w_resolution) # vector of initial weights
    W = np.tile(w_init, (num_ETs,1)).T # initialize W matrix
    W_init = np.copy(W)
    for t in range(0,num_timesteps,10):
        dWdt = btsp_func(ET_all[:,t]*IS[t], W_init)
        W += lr*dWdt

    dW = W - np.tile(w_init, (num_ETs,1)).T    
    dW = np.flip(dW, axis=0)

    # Plot dW vs time from plateau
    i=0
    if len(axes)==3:
        ax = axes[0]
        colorscale = np.max(np.abs(dW))
        im = ax.imshow(dW, extent=[-5, 5, w_range[0], w_range[1]], aspect='auto', cmap='bwr', vmin=-colorscale, vmax=colorscale)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('$\Delta$ Weight', rotation=270, labelpad=8, fontsize=8)
        ax.set_xlim([-5,3])
        # ax.set_ylim([1,3])
        ax.set_xlabel('Time from dendritic spike (s)')
        ax.set_ylabel('Initial weight')
        i=1

    ax = axes[0+i]
    time = np.arange(-tmax/2, tmax/2, dt)/1000
    ET[0:int(tmax/2)] = 0 # zero values before the ET start
    ax.plot(time, ET, c='b', label='Eligibility trace (ET)')
    ax.plot(time, IS, c='r', label='Instructive signal (IS)')
    ax.set_xlim([-5,10])
    ax.set_ylim([-0.2,1])
    ax.set_xlabel('Time from dendritic spike (s)')
    ax.set_ylabel('Signal amplitude')
    ax.legend(loc='upper left', bbox_to_anchor=(0., 1.1), fontsize=8, frameon=False, ncol=1, handlelength=1)
    
    ax = axes[1+i]
    time = all_delta_t/1000
    dW = np.flip(dW, axis=0)
    w1_row = np.where(w_init>=1)[0][0]
    if norm:
        ax.plot(time, dW[w1_row,:]/np.max(dW[w1_row,:]), label=f'ET temp.={ET_temp}')
    else:
        ax.plot(time, dW[w1_row,:], label='W_init=1', c='k')
        w1p5_row = np.where(w_init>=1.5)[0][0]
        ax.plot(time, dW[w1p5_row,:], label='W_init=1.5', c='gray')
        w2_row = np.where(w_init>=2)[0][0]
        ax.plot(time, dW[w2_row,:], label='W_init=2', c='lightgray')

    ax.hlines(0, -5, 5, colors='gray', linestyles='dashed', alpha=0.5)
    ax.vlines(0, -0.5, 1.5, colors='r', linestyles='dashed', alpha=0.5)

    ax.set_xlabel('Time from dendritic spike (s)')
    ax.set_ylabel('$\Delta$ Weight')
    ax.set_xlim([-5,3])
    ax.set_ylim([-0.5,1.5])
    ax.legend(loc='upper left', bbox_to_anchor=(0., 1.1), fontsize=8, frameon=False, ncol=1, handlelength=1)

########################################################################################################################

def generate_Figure_S10(save):
    '''Plot simulations of VO2 variability (jitter tau values)'''

    mm = 1 / 25.4 # convert mm to inches
    fig = plt.figure(figsize=(180*mm, 80*mm))
    axes = gs.GridSpec(nrows=2, ncols=4, left=0.08, right=0.95, top=0.95, bottom=0.1, wspace=0.4, hspace=0.5)

    plt.rcParams.update({'font.size': 8,
                    'axes.spines.right': False,
                    'axes.spines.top': False,
                    'axes.linewidth':0.5,
                    'xtick.major.size': 3,
                    'xtick.major.width': 0.5,
                    'ytick.major.size': 3,
                    'ytick.major.width': 0.5,
                    'legend.frameon': False,
                    'legend.handletextpad': 0.1,
                    'figure.figsize': [10.0, 3.0],
                    'svg.fonttype': 'none',
                    'text.usetex': False,
                    'font.sans-serif': "Helvetica",
                    'font.family': "sans-serif",
                    'font.weight': "normal"})
    

    # BTSP parameters from (Milstein et al., 2021, eLife) Fig.7
    sig_pot = get_scaled_sigmoid(slope=4.405, threshold=0.415)
    sig_dep = get_scaled_sigmoid(slope=20.0, threshold=0.026)
    k_dep = 0.425
    k_pot = 1.1097
    Wmax = 4.68
    btsp_func = get_BTSP_function(Wmax, k_pot, k_dep, sig_pot, sig_dep)

    # Plot BTSP learning rule for different tau values
    IS_reference_temp = 70.82 # tau=0.44s
    IS_min_temp = IS_reference_temp - 0.66 # reduce tau by ~25% -> 0.33s
    IS_max_temp = IS_reference_temp + 0.49 # increase tau by ~25% -> 0.55s

    ET_reference_temp = 74.34 # tau=1.66s
    ET_min_temp = ET_reference_temp - 1.01 # reduce tau by ~25% -> 1.245s
    ET_max_temp = ET_reference_temp + 1.32 # increase tau by ~25% -> 2.075s

    ET_tau = [1.245, 1.66, 2.075]

    ax1 = fig.add_subplot(axes[0,0])
    # three different shades of red: light, medium, dark
    colors = [(1,0.,0.), (1,0.4,0.3), (1,0.6,0.6)]
    for column,ET_temp in enumerate([ET_min_temp, ET_reference_temp, ET_max_temp]):
        ax2 = fig.add_subplot(axes[0,column+1])
        ax2.set_title(f'ET tau={ET_tau[column]}s')
        for c,IS_temp in enumerate([IS_min_temp, IS_reference_temp, IS_max_temp]):
            plot_VO2_btsp_learning_rule_single(ax1, ax2, colors[c], btsp_func, dwell_time=400, plateau_dur=300, lr=0.012, ET_temp=ET_temp, IS_temp=IS_temp, save=save)

    ax1 = fig.add_subplot(axes[1,0])
    ax2 = fig.add_subplot(axes[1,1])
    plot_linear_track_VO2_variability(ax1, ax2)

    if save:
        fig.savefig('figures/0-Supplementary Figures/Supplementary_LIF_variability.png',dpi=300)
        fig.savefig('figures/0-Supplementary Figures/Supplementary_LIF_variability.svg',dpi=300)

def plot_VO2_btsp_learning_rule_single(ax1, ax2, color, btsp_func, dwell_time, plateau_dur, lr, ET_temp, IS_temp, save=False):
    ''' Plot BTSP learning rule (dW vs time from plateau) 
    
    :param T: float, dwell time (s)
    :param ET_rise_tau: float, ET rise time constant (s)
    :param ET_decay_tau: float, ET decay time constant (s)
    :param IS_rise_tau: float, IS rise time constant (s)
    :param IS_decay_tau: float, IS decay time constant (s)
    '''

    # Set parameters
    dt = 1 # ms
    tmax = 20_000  # ms
    t_resolution = 50 # ms
    ET_controlI = 100
    IS_controlI = 100

    # Generate ET and IS template traces
    VO2_ET = Volatile_Resistor(dt, temperature=ET_temp, metalR=100, stim_scaling=1)
    VO2_IS = Volatile_Resistor(dt, temperature=IS_temp, metalR=100, stim_scaling=1)
    for t in range(0, tmax, dt):
        VO2_ET.controlI = 0
        VO2_IS.controlI = 0
        if t>0 and t<dwell_time:
            VO2_ET.controlI = ET_controlI
        if t>0 and t<plateau_dur:
            VO2_IS.controlI = IS_controlI
        VO2_ET.time_step()
        VO2_IS.time_step()
    ET = np.array(VO2_ET.g_history)
    ET_min = 1/VO2_ET.insulatorR
    ET_max = 1/VO2_ET.metalR * 0.4
    ET = (ET-ET_min)/(ET_max-ET_min)
    ET = np.minimum(ET, 1) # clip values to [0,1]
    ET = np.roll(ET, int(tmax/2))

    IS = np.array(VO2_IS.g_history)
    IS_min = 1/VO2_IS.insulatorR
    IS_max = 1/VO2_IS.metalR * 0.8
    IS = (IS-IS_min)/(IS_max-IS_min)
    IS = np.minimum(IS, 1) # clip values to [0,1]
    IS = np.roll(IS, int(tmax/2))
    IS[0:int(tmax/2)] = 0 # zero values before the IS start

    # Compute dW matrix
    all_delta_t = np.arange(-5000, 5000, t_resolution)
    ET_all = []
    for delta_t in all_delta_t:
        shifted_ET = np.roll(ET, int(delta_t))
        shifted_ET[0:int(delta_t+tmax/2)] = 0 # zero values before the ET start to prevent spillover
        ET_all.append(shifted_ET)
    ET_all = np.array(ET_all)

    num_ETs = ET_all.shape[0]
    num_timesteps = ET_all.shape[1]

    w_init = 1
    W_init = np.ones(num_ETs) * w_init
    W = np.copy(W_init)
    for t in range(0,num_timesteps,10):
        dWdt = btsp_func(ET_all[:,t]*IS[t], W_init)
        W += lr*dWdt
    dW = W - w_init

    # Plot ET and IS traces
    ax = ax1
    time = np.arange(-tmax/2, tmax/2, dt)/1000
    ET[0:int(tmax/2)] = 0 # zero values before the ET start
    ax.plot(time, ET, c='b', label='Eligibility trace (ET)', alpha=0.3)
    ax.plot(time, IS, c='r', label='Instructive signal (IS)', alpha=0.3)
    ax.set_xlim([-5,10])
    ax.set_ylim([-0.2,1])
    ax.set_xlabel('Time from dendritic spike (s)')
    ax.set_ylabel('Signal amplitude')
    # ax.legend(loc='upper left', bbox_to_anchor=(0., 1.1), fontsize=8, frameon=False, ncol=1, handlelength=1)
    
    # Plot dW vs time from plateau
    ax = ax2
    time = all_delta_t/1000
    ax.plot(time, dW, label=f'ET temp.={ET_temp}', c=color, alpha=0.8)
    ax.hlines(0, -5, 5, colors='gray', linestyles='dashed', alpha=0.3)
    ax.vlines(0, -0.5, np.max(dW), colors='gray', linestyles='dashed', alpha=0.3)

    ax.set_xlabel('Time from dendritic spike (s)')
    ax.set_ylabel('$\Delta$ Weight')
    ax.set_xlim([-5,3])
    # ax.set_ylim([-0.4,2])
    ax.set_ylim(bottom=-0.4)

    # ax.legend(loc='upper left', bbox_to_anchor=(0., 1.1), fontsize=8, frameon=False, ncol=1, handlelength=1)

def plot_linear_track_VO2_variability(ax1, ax2):
    filename = "supplementary_VO2_BTSP_variability_simulations.pkl"    
    overwrite = False

    if os.path.exists(f'sim_data/{filename}') and not overwrite:
        with open(f'sim_data/{filename}', 'rb') as f:
            simulation_data_all = pickle.load(f)
    else:
        simulation_data_all = {}

    random_seeds = [12, 42, 4321 , 674, 974, 295, 2763, 809, 2349, 7862]
    for seed in random_seeds:
        if seed not in simulation_data_all:
            print('Running simulation for seed =', seed)
            if seed==12: # Include the example trace with no tau jitter
                randomly_sample_tau = False
            else:
                randomly_sample_tau = True
            simulation_data_all[seed] = VO2_LIF_BTSP_simulation(VO2_temp=62, num_synapses=30, firing_rate=30, jitter=randomly_sample_tau)

            with open(f'sim_data/{filename}', 'wb') as f:
                pickle.dump(simulation_data_all, f)

    example_color = 'steelblue'
    example_synapse = 6
    linewidth = 1
    simulation_results = simulation_data_all[random_seeds[0]] # use first seed as example

    ## ET and IS traces
    ax = ax1
    IS_all = []
    ET_all = []
    plateau_time = simulation_results['plateau_time']
    time = (simulation_results['time'] - plateau_time)/1000
    for seed in random_seeds:
        IS = simulation_data_all[seed]['dendrite_IS']
        IS_baseline = 1/IS.insulatorR
        IS_peak = 1/IS.metalR
        IS_norm = (np.array(IS.g_history) - IS_baseline) / (IS_peak - IS_baseline) *1.2
        IS_all.append(IS_norm)

        ET = simulation_data_all[seed]['synapse_ETs'][example_synapse]
        ET_baseline = 1/ET.insulatorR
        ET_peak = 1/ET.metalR
        ET_norm = (np.array(ET.g_history) - ET_baseline) / (ET_peak - ET_baseline) *1.9
        ET_all.append(ET_norm)
    IS_all = np.array(IS_all)
    ET_all = np.array(ET_all)

    IS_avg = np.mean(IS_all, axis=0)
    IS_std = np.std(IS_all, axis=0)
    ET_avg = np.mean(ET_all, axis=0)
    ET_std = np.std(ET_all, axis=0)

    step_size = 10 # downsample to reduce output image size
    time = time[::step_size]
    IS_avg = IS_avg[::step_size]
    IS_std = IS_std[::step_size]
    ET_avg = ET_avg[::step_size]
    ET_std = ET_std[::step_size]
    ax.plot(time, IS_avg, color='r', linewidth=linewidth, label='IS')
    ax.fill_between(time, IS_avg-IS_std, IS_avg+IS_std, color='r', alpha=0.5, linewidth=0)
    ax.plot(time, ET_avg, color=example_color, linewidth=linewidth, label='ET')
    ax.fill_between(time, ET_avg-ET_std, ET_avg+ET_std, color=example_color, alpha=0.5, linewidth=0)

    ax.set_ylabel('Signal amplitude')
    ax.set_xlabel('Time from dendritic spike (s)')
    ax.legend(loc='upper right', bbox_to_anchor=(0.9, 1.), handlelength=1, frameon=False, fontsize=6, handletextpad=0.5)
    ax.set_xlim([-3,3])

    ## Weight update plot
    ax = ax2
    w_init_all = [simulation_data_all[seed]['weights'] for seed in simulation_data_all]
    w_final_all = [simulation_data_all[seed]['new_weights'] for seed in simulation_data_all]
    w_init_avg = np.mean(w_init_all, axis=0)-1 # subtract 1 to make baseline 0
    w_final_avg = np.mean(w_final_all, axis=0)-1
    w_final_std = np.std(w_final_all, axis=0)

    num_synapses = len(simulation_results['weights'])
    T = simulation_results['time'][-1]

    field_width = simulation_results['field_width']
    position = (np.linspace(0, 6000, num_synapses) + field_width/2 - plateau_time)/1000

    ax.plot(position, w_init_avg, c='gray', alpha=0.5, label='Initial synaptic weights', linewidth=1.5*linewidth)
    ax.plot(position, w_final_avg, c='k', label='Weights after dendritic spike', linewidth=1.5*linewidth)
    ax.fill_between(position, w_final_avg-w_final_std, w_final_avg+w_final_std, color='gray', alpha=0.5, linewidth=0)
    ax.scatter(position[example_synapse], w_final_avg[example_synapse], s=30, facecolors='none', edgecolors=example_color, linewidth=1, zorder=10)

    ax.set_ylabel('$\Delta W$ (norm.)')
    ax.set_xlabel('Time from dendritic spike (s)')
    # ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.), handlelength=1, frameon=False)
    ax.set_xlim([-3,3])

########################################################################################################################

def generate_Figure_S11(save):
    '''BTSP variants (accelerated tau and simplified BTSP) for RL linear track and gridworld simulations'''

    mm = 1 / 25.4  # millimeters in inches
    fig = plt.figure(figsize=(180*mm, 210*mm))
    colors = {'TD': 'k', 'Hebb': 'gray', 'BTSP': 'r', 'accelerated_BTSP': 'm', 'simple_BTSP': 'c', 'simple_BTSPu': 'c'}
    learning_rules = ['TD', 'Hebb', 'BTSP', 'accelerated_BTSP', 'simple_BTSP']

    # BTSP parameters from (Milstein et al., 2021, eLife) Fig.7
    sig_pot = get_scaled_sigmoid(slope=4.405, threshold=0.415)
    sig_dep = get_scaled_sigmoid(slope=20.0, threshold=0.026)
    k_dep = 0.425
    k_pot = 1.1097
    Wmax = 4.68
    btsp_func = get_BTSP_function(Wmax, k_pot, k_dep, sig_pot, sig_dep)

    ##################################################
    ## 0. BTSP learing rule variations
    ##################################################

    axes = gs.GridSpec(nrows=1, ncols=2, left=0.06, right=0.6, top=0.99, bottom=0.85, wspace=0.5)
    ax1 = fig.add_subplot(axes[0])
    ax2 = fig.add_subplot(axes[1])

    plot_VO2_btsp_learning_rule((ax1,ax2), btsp_func, dwell_time=400, plateau_dur=300, lr=0.012, ET_temp=74.34, IS_temp=70.82, norm=True) # Regular ET
    plot_VO2_btsp_learning_rule((ax1,ax2), btsp_func, dwell_time=400, plateau_dur=300, lr=0.012, ET_temp=70.68, ET_scale=1.1, IS_temp=70.82, norm=True) #ET/4

    btsp_func = get_simple_BTSP_function()
    plot_VO2_btsp_learning_rule((ax1,ax2), btsp_func, dwell_time=400, plateau_dur=300, lr=0.0015, ET_temp=74.34, IS_temp=70.82, norm=True)


    ##################################################
    ## 1. Linear track simulation
    ##################################################
    overwrite = False
    filename = 'Figure5_linear_track_SR_matrices.pkl'

    M_dict, size = simulate_linear_track_SR(filename, btsp_func, overwrite)

    # Plot learned successor weights
    axes = gs.GridSpec(nrows=1, ncols=2, left=0.06, right=0.5, top=0.78, bottom=0.62, wspace=0.06)
    ax1 = fig.add_subplot(axes[0])
    ax2 = fig.add_subplot(axes[1])
    plot_linear_track_SR(M_dict, learning_rules[3:], fig, (ax1, ax2))

    # Plot quantification of SR correlation
    axes = gs.GridSpec(nrows=1, ncols=1, left=0.7, right=0.99, top=0.78, bottom=0.62)
    ax = fig.add_subplot(axes[0])
    plot_SR_correlation(M_dict, learning_rules[2:], size, ax, colors)


    ##################################################
    ## 2. Gridworld simulation
    ##################################################
    overwrite = False
    filename = 'Figure6_norect_1wi_ET4_round3.pkl'
    random_seeds = [42, 12, 4321, 674, 974, 295, 2763, 809, 2349, 7862]

    environment = GridWorld((5, 6), walls=(np.array([1, 2, 3]), 2), rewards=(1, 5), initial_state=(3, 1)) # 5x6 environment with simple wall
    grid_simulations_data = simulate_gridworld_multiple_seeds(learning_rules, random_seeds, filename, btsp_func, environment, overwrite)    

    # Plot example trajectories
    axes = gs.GridSpec(nrows=2, ncols=3, left=0.01, right=0.6, top=0.55, bottom=0.22, wspace=0.06)
    example_seed = 42
    for i, rule in enumerate(learning_rules[3:]):
        ax = fig.add_subplot(axes[i, 0])
        plot_grid(ax, environment, grid_simulations_data[example_seed][rule]['trajectories'], legend=(i==0))
        ax.set_title(f'{rule}')
    
    # Plot example SR place fields
    cell_nr = 11
    end_idx = -1 # end of the last trial
    for i, rule in enumerate(learning_rules[3:]):
        ax1 = fig.add_subplot(axes[i, 1])
        ax2 = fig.add_subplot(axes[i, 2])
        trial_num = 1
        start_idx = np.sum([len(grid_simulations_data[example_seed][rule]['trajectories'][i]) for i in range(trial_num)]) # index at the end of trial 0
        plot_SR_place_fields(fig, ax1, ax2, cell_nr, grid_simulations_data[example_seed][rule]['SR_weight_history'], start_idx, end_idx, environment.grid_size)

    # Plot navigation summary quantifications
    axes = gs.GridSpec(nrows=1, ncols=6, left=0.05, right=0.99, top=0.2, bottom=0.05, hspace=0.5, wspace=1)
    ax = fig.add_subplot(axes[0:2])
    plot_pathlength_over_trials(ax, grid_simulations_data, random_seeds, learning_rules[2:], colors)

    ax1 = fig.add_subplot(axes[2])
    ax2 = fig.add_subplot(axes[3])
    plot_navigation_summary(ax1, ax2, grid_simulations_data, random_seeds, learning_rules[2:], colors)
    
    ax = fig.add_subplot(axes[4:6])
    plot_cumulative_reward(ax, grid_simulations_data, learning_rules[2:], colors)

    if save:
        fig.savefig('figures/0-Supplementary Figures/Supplementary_BTSP_variations.png',dpi=300)
        fig.savefig('figures/0-Supplementary Figures/Supplementary_BTSP_variations.svg',dpi=300)

########################################################################################################################

if __name__ == '__main__':
    save = True
    update_plot_defaults()

    generate_Figure_S3(save)
    # generate_Figure_S5(save)
    # generate_Figure_S6(save)
    # generate_Figure_S9(save)
    # generate_Figure_S10(save)
    # generate_Figure_S11(save)


    plt.show()