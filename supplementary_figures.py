import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils import update_plot_defaults, get_scaled_sigmoid, get_BTSP_function, Volatile_Resistor

########################################################################################################################
# Supplementary Figures
########################################################################################################################

def generate_Supplementary1(save):
    # Plot heatmap for BTSP rule

    # BTSP parameters from (Milstein et al., 2021, eLife) Fig.7
    sig_pot = get_scaled_sigmoid(slope=4.405, threshold=0.415)
    sig_dep = get_scaled_sigmoid(slope=20.0, threshold=0.026)
    k_dep = 0.425
    k_pot = 1.1097
    Wmax = 4.68
    btsp_func = get_BTSP_function(Wmax, k_pot, k_dep, sig_pot, sig_dep)
    plot_VO2_btsp_learning_rule(btsp_func, dwell_time=400, plateau_dur=300, lr=0.012, ET_temp=74.34, IS_temp=70.82, save=save)


def plot_VO2_btsp_learning_rule(btsp_func, dwell_time, plateau_dur, lr, ET_temp, IS_temp, save=False):
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
    ET_max = 1/VO2_ET.metalR * 0.4
    ET = (ET-ET_min)/(ET_max-ET_min)
    ET = np.roll(ET, int(tmax/2))


    IS = np.array(VO2_IS.g_history)
    IS_min = 1/VO2_IS.insulatorR
    IS_max = 1/VO2_IS.metalR * 0.8
    IS = (IS-IS_min)/(IS_max-IS_min)
    IS = np.roll(IS, int(tmax/2))
    IS[0:int(tmax/2)] = 0 # zero values before the IS start

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
    mm = 1 / 25.4 # convert mm to inches
    fig, axes = plt.subplots(1,3,figsize=(180*mm,50*mm))

    ax = axes[0]
    colorscale = np.max(np.abs(dW))
    im = ax.imshow(dW, extent=[-5, 5, w_range[0], w_range[1]], aspect='auto', cmap='bwr', vmin=-colorscale, vmax=colorscale)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('$\Delta$ Weight', rotation=270, labelpad=8, fontsize=8)
    ax.set_xlim([-5,3])
    # ax.set_ylim([1,3])
    ax.set_xlabel('Time from dendritic spike (s)')
    ax.set_ylabel('Initial weight')
    ax.set_title('BTSP learning rule')

    ax = axes[1]
    time = np.arange(-tmax/2, tmax/2, dt)/1000
    ET[0:int(tmax/2)] = 0 # zero values before the ET start
    ax.plot(time, ET, c='b', label='Eligibility trace (ET)')
    ax.plot(time, IS, c='r', label='Instructive signal (IS)')
    ax.set_xlim([-5,10])
    ax.set_xlabel('Time from dendritic spike (s)')
    ax.set_ylabel('Signal amplitude')
    ax.legend(loc='upper left', bbox_to_anchor=(0., 1.1), fontsize=8, frameon=False, ncol=1, handlelength=1)
    
    ax = axes[2]
    time = all_delta_t/1000
    dW = np.flip(dW, axis=0)

    w0_row = np.where(w_init>=1)[0][0]
    ax.plot(time, dW[w0_row,:], label='W_init=1', c='k')
    w05_row = np.where(w_init>=1.5)[0][0]
    ax.plot(time, dW[w05_row,:], label='W_init=1.5', c='gray')
    w1_row = np.where(w_init>=2)[0][0]
    ax.plot(time, dW[w1_row,:], label='W_init=2', c='lightgray')

    ax.hlines(0, -5, 5, colors='gray', linestyles='dashed', alpha=0.5)
    ax.vlines(0, -0.5, 1.5, colors='r', linestyles='dashed', alpha=0.5)

    ax.set_xlabel('Time from dendritic spike (s)')
    ax.set_ylabel('$\Delta$ Weight')
    ax.set_xlim([-5,3])
    ax.set_ylim([-0.5,1.5])
    ax.legend(loc='upper left', bbox_to_anchor=(0., 1.1), fontsize=8, frameon=False, ncol=1, handlelength=1)

    plt.tight_layout()

    if save:
        fig.savefig('figures/0-Supplementary Figures/Supplementary_BTSP_heatmap.png',dpi=300)
        fig.savefig('figures/0-Supplementary Figures/Supplementary_BTSP_heatmap.svg',dpi=300)


def generate_Supplementary2(save):
    '''Plot data vs sim for 70C relaxations'''
    # Plot data vs simulation 

    mm = 1 / 25.4  # millimeters in inches    
    fig,ax = plt.subplots(1,3,figsize=(183*mm, 50*mm))

    Isteps_data = pd.read_excel('data/VO2_data_currents.xlsx',header=0)

    # Extract relevant variables 
    data_time = np.array(Isteps_data['Time'][2:])
    Isteps = np.array(Isteps_data.iloc[1][1::4])
    voltage_traces = np.array(Isteps_data.iloc[2:,1::4]).T
    resistance_traces = np.array(Isteps_data.iloc[2:,2::4]).T
    current_traces = np.array(Isteps_data.iloc[2:,3::4]).T


    plot_nr = (1)
    for i,R in enumerate(resistance_traces):
        ax[plot_nr].plot(data_time[100:10000], 1/R[100:10000], label=Isteps[i], c='k', linewidth=1)
    ax[plot_nr].set_xlabel('Time (s)')
    ax[plot_nr].set_ylabel('R (Ω)')
    # ax[plot_nr].set_xlim([1.,1.05])
    # ax[plot_nr].set_ylim(top=1)
    ax[plot_nr].set_title('VO2 data')

    # plot_nr = (1,0)
    # for i,I in enumerate(current_traces):
    #     ax[plot_nr].plot(data_time, I*1000, label=Isteps[i], c='k')
    # ax[plot_nr].set_xlabel('Time (s)')
    # ax[plot_nr].set_ylabel('Control I (mA)')
    # # ax[0,1].legend()

    sim_time, R_hist, controlI_hist = VO2_test_pulse(dt=10,T=10000, stim_time=(0,3000), temperature=70)
    sim_time /= 1000
    plot_nr = (2)
    ax[plot_nr].plot(sim_time, 1/R_hist, c='r', linewidth=1)
    ax[plot_nr].set_xlabel('Time (ms)')
    ax[plot_nr].set_ylabel('R (Ω)')
    ax[plot_nr].set_title('VO2 model simulation')

    plot_nr = 0
    ax[plot_nr].plot(sim_time, controlI_hist, 'k', linewidth=1)
    ax[plot_nr].set_xlabel('Time (s)')
    ax[plot_nr].set_ylabel('Control I (mA)')
    ax[plot_nr].set_title('Stimulation current')
    plt.tight_layout()

    if save:
        fig.savefig('figures/0-Supplementary Figures/Supplementary_Rrelaxation.png',dpi=300)
        fig.savefig('figures/0-Supplementary Figures/Supplementary_Rrelaxation.svg',dpi=300)
    

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


def generate_Supplementary3(save):
    '''Plot properties of VO2 emulation'''

    # Simulation runtime parameters
    dt = 0.1  # time step (ms)
    T = 300   # simulation time (ms)
    time = np.arange(0., T, dt)

    line_color = [0.2,0.2,0.2]

    mm = 1 / 25.4  # millimeters in inches    
    fig,ax = plt.subplots(1,4,figsize=(183*mm, 60*mm))

    g_volatile = Volatile_Resistor(dt, stim_scaling=100)
    controlI = np.linspace(0,1,1000)
    R_eq = g_volatile.transfer_func(controlI*g_volatile.stim_scaling)
    g_eq = 1/ R_eq

    ax[0].plot(controlI,R_eq, c=line_color)
    ax[0].set_xlabel('Control current (mA)')
    ax[0].set_ylabel('R$_{eq}$ (Ω)')

    ax[1].plot(controlI,g_eq, c=line_color)
    ax[1].set_xlabel('Control current (mA)')
    ax[1].set_ylabel('g$_{eq}$ (S)')

    ax[2].plot(controlI,g_volatile.decay_tau(controlI=controlI*g_volatile.stim_scaling,temperature=70),c='r')
    ax[2].set_title('Temp=70°C')
    ax[2].set_xlabel('Control current (mA)')
    ax[2].set_ylabel('g decay tau (ms)')

    ax[3].plot(controlI,g_volatile.decay_tau(controlI=controlI*g_volatile.stim_scaling,temperature=64),c='b')
    ax[3].set_title('Temp=64°C')
    ax[3].set_xlabel('Control current (mA)')
    ax[3].set_ylabel('g decay tau (ms)')

    plt.suptitle('VO$_{2}$ volatile resistor properties (modeled on data)')
    plt.tight_layout()

    if save:
        fig.savefig('figures/0-Supplementary Figures/Supplementary_VO2_emulation_properties.png',dpi=300)
        fig.savefig('figures/0-Supplementary Figures/Supplementary_VO2_emulation_properties.svg',dpi=300)


def generate_Supplementary4(save):

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



if __name__ == '__main__':
    save = False

    update_plot_defaults()

    generate_Supplementary1(save)
    generate_Supplementary2(save)
    generate_Supplementary3(save)
    generate_Supplementary4(save)

    plt.show()