import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import os
import pickle

from utils import Volatile_Resistor, Memristor, poisson_spike_train, get_scaled_sigmoid, get_BTSP_function, update_plot_defaults



########################################################################################################################
# Figure 4: BTSP data and simulation
########################################################################################################################


def VO2_LIF_BTSP_simulation(VO2_temp, num_synapses, firing_rate, w_init=None, jitter=False):
    ##############################################################
    ## Setup
    ##############################################################

    # Units
    kOhm = 1e3
    uF = 1e-6

    # Simulation runtime parameters
    dt = 0.1  # time step (ms)
    T = 14000   # simulation time (ms)
    time = np.arange(0., T, dt)

    input_stim_duration = 5 # ms
    V_low = 0.04 # V
    VO2_stim_amp = 1. # mA
    VO2_pulse_dur = 3. # ms

    field_width = 2000 # ms
    input_spike_times = [poisson_spike_train(firing_rate, field_width/1e3, refractory_period=0.01)*1e3+field_start for field_start in np.linspace(0, 6000, num_synapses)]
    # field_centers = np.linspace(0, T-field_width, num_synapses) + field_width/2
    plateau_time = 4000

    # Neuron model parameters
    VO2_cell = Volatile_Resistor(dt, temperature=VO2_temp, metalR=100, insulatorR=10*kOhm, stim_scaling=100)
    C = 1*uF
    V_th = 0.02 # V

    # BTSP parameters from eLife Fig.7
    slope_pot = 4.405
    threshold_pot = 0.415
    slope_dep = 20.0 
    threshold_dep = 0.026
    k_dep = 0.425
    k_pot = 1.1097
    Wmax = 4.68
    sig_pot = get_scaled_sigmoid(slope_pot, threshold_pot)
    sig_dep = get_scaled_sigmoid(slope_dep, threshold_dep)
    btsp_func = get_BTSP_function(Wmax, k_pot, k_dep, sig_pot, sig_dep)

    # Synapse/weight parameters
    synapses = [Memristor(conductingR=15*kOhm, insulatorR=200*kOhm) for _ in range(num_synapses)]
    for memristor in synapses:
        memristor.R = 155*kOhm # initialize to high resistance
    min_weight = 1/synapses[0].insulatorR
    max_weight = 1/synapses[0].conductingR

    memristor_R_array = np.array([memristor.R for memristor in synapses])
    baseline = 1/memristor.R

    target_baseline = 1.
    peak_w = 2.2
    if w_init is None:
        # convert from R to weight (in range [0,1])
        weights = 1/memristor_R_array
        weights = (weights - baseline) / (max_weight - baseline) * (peak_w-target_baseline) + target_baseline # normalize to [1,4]
    else:
        weights = w_init
        # convert from weight to R
        memristor_R_array = (peak_w-target_baseline) / ((weights - target_baseline) * (max_weight - baseline) + (peak_w-target_baseline)*baseline)
        for synapse_id, memristor in enumerate(synapses):
            memristor.R = memristor_R_array[synapse_id]

    # BTSP parameters
    temperature_jitter = np.random.uniform(-0.32, 0.49) * jitter
    dendrite_IS = Volatile_Resistor(dt, temperature=70.82+temperature_jitter, metalR=100, insulatorR=10*kOhm, stim_scaling=100)
    IS_pulse_dur = 300 # ms

    synapse_ETs = []
    for _ in range(num_synapses):
        temperature_jitter = np.random.uniform(-1.01, 1.32) * jitter
        synapse_ETs.append(Volatile_Resistor(dt, temperature=74.34+temperature_jitter, metalR=100, insulatorR=10*kOhm, stim_scaling=100))
    ET_pulse_dur = 20 # ms
    read_dur = 2000 # ms
    # learning_rate = 0.03
    learning_rate = 0.012

    ##############################################################
    ## Initialize variables
    ##############################################################

    V = np.zeros(len(time))
    I = np.zeros(len(time))
    I_syn = np.zeros([num_synapses, len(time)])
    ETxIS = np.zeros([num_synapses])
    dW = np.zeros([num_synapses, len(time)])
    W = np.zeros([num_synapses, len(time)])
    W[:,0] = weights

    output_spike_times = []
    VO2_cell_switch = np.zeros(len(time))
    VO2_pulse_dur_idx = np.round(VO2_pulse_dur/dt).astype(int)

    ET_switches = np.zeros([num_synapses,len(time)])
    ET_pulse_dur_idx = np.round(ET_pulse_dur/dt).astype(int)
    measured_ETs = np.zeros([num_synapses,len(time)])

    IS_switch = np.zeros(len(time))
    IS_pulse_dur_idx = np.round(IS_pulse_dur/dt).astype(int)
    measured_IS = np.zeros(len(time))
    read_switch = np.zeros(len(time))
    read_dur_idx = np.round(read_dur/dt).astype(int)

    # Create a stimulus trigger for each synapse
    stim_dur_idx = np.round(input_stim_duration/dt).astype(int)
    kernel = np.ones(stim_dur_idx)
    spike_trains = np.zeros([num_synapses,len(time)])
    stim_triggers = np.zeros([num_synapses,len(time)])
    for synapse_id,times in enumerate(input_spike_times):
        stim_times_idx = (times/dt).astype(int)
        spike_trains[synapse_id, stim_times_idx] = 1
        stim_trigger_train = np.convolve(spike_trains[synapse_id], kernel)[0:len(time)]
        stim_trigger_train = np.minimum(stim_trigger_train, 1) # clip to [0,1] 
        stim_triggers[synapse_id] = stim_trigger_train


    ##############################################################
    ## Simulation
    ##############################################################
    for t in range(len(time)):
        if V[t]>V_th and V[t-1]<V_th:
            output_spike_times.append(t)
            VO2_cell_switch[t:t+VO2_pulse_dur_idx] = 1    
            V[t] = V_th

        # Update synapse ETs from input spikes
        if np.sum(spike_trains[:,t]) > 0:
            for synapse_id in np.where(spike_trains[:,t])[0]:
                ET_switches[synapse_id, t:t+ET_pulse_dur_idx] = 1

        for synapse_id in np.where(ET_switches[:,t]!=ET_switches[:,t-1])[0]:
            synapse_ETs[synapse_id].controlI = ET_switches[synapse_id,t] * VO2_stim_amp

        # Update IS
        if t == int(plateau_time/dt):
            IS_pulse_end = t+IS_pulse_dur_idx
            IS_switch[t:IS_pulse_end] = 1
            # read_switch[IS_pulse_end:IS_pulse_end+read_dur_idx] = 1
            read_switch[IS_pulse_end:] = 1

        dendrite_IS.controlI = IS_switch[t] * VO2_stim_amp

        # Update somatic VO2 and compute dVdt
        if VO2_cell_switch[t] == 1:
            VO2_cell.controlI = VO2_stim_amp
            dVdt = 0
        else:
            VO2_cell.controlI = 0
            V_probe = stim_triggers[:,t] * V_low
            I_syn[:,t] = (V_probe - V[t]) / memristor_R_array
            I_syn[:,t] = np.maximum(I_syn[:,t], 0)
            I[t] = np.sum(I_syn[:,t])
            dVdt = I[t]/C - V[t]/(VO2_cell.R*C)

        # Update soma Vm
        if t < len(time)-1:
            V[t+1] = V[t] + dVdt*dt*1e-3 # convert to ms
        
        # Update weights with BTSP rule
        if read_switch[t]==1 and t%(10/dt)==0: # update weights every 10 ms
            # Normalize eligibility+instructive signals
            IS_baseline = 1 / dendrite_IS.insulatorR
            IS_peak = 1 / dendrite_IS.metalR
            measured_IS[t] = (dendrite_IS.g - IS_baseline) / (IS_peak - IS_baseline) *1.2
            measured_IS[t] = np.minimum(measured_IS[t], 1) # clip to [0,1]
            for i, ET in enumerate(synapse_ETs):
                ET_baseline = 1 / ET.insulatorR
                ET_peak = 1 / ET.metalR
                measured_ETs[i, t] = (ET.g - ET_baseline) / (ET_peak - ET_baseline) *1.9
                measured_ETs[i, t] = np.minimum(measured_ETs[i, t], 1) # clip to [0,1]

            dW[:,t] = btsp_func(measured_ETs[:,t] * measured_IS[t], weights)

            if measured_IS[t] < 0.01:
                read_switch[t:] = 0

        if t < len(time)-1:
            W[:,t+1] = W[:,t] + dW[:,t] *learning_rate

        # Step VO2 elements
        VO2_cell.time_step()
        dendrite_IS.time_step()
        for synapse in synapse_ETs:
            synapse.time_step()

    output_spike_times = np.array(output_spike_times)*dt
    new_weights = W[:,-1]

    simulation_results = {
        'time': time, 'V': V, 'I': I, 'I_syn': I_syn, 'V_th': V_th, 'field_width': field_width,
        'input_spike_times': input_spike_times, 'stim_triggers': stim_triggers, 'plateau_time': plateau_time,
        'output_spike_times': output_spike_times, 'VO2_cell': VO2_cell, 'VO2_pulse_dur': VO2_pulse_dur,
        'synapse_ETs': synapse_ETs, 'measured_ETs': measured_ETs, 'ET_pulse_dur': ET_pulse_dur, 'ET_switches': ET_switches,
        'dendrite_IS': dendrite_IS, 'measured_IS': measured_IS, 'IS_pulse_dur': IS_pulse_dur, 'IS_switch': IS_switch,
        'memristors': synapses, 'weights': weights, 'new_weights': new_weights, 'dW': dW, 'W': W,
    }

    return simulation_results


def generate_Figure4(show=True, save=False):
    np.random.seed(12)
    simulation_results = VO2_LIF_BTSP_simulation(VO2_temp=62, num_synapses=30, firing_rate=30)
    simulation_results2 = VO2_LIF_BTSP_simulation(VO2_temp=62, num_synapses=30, firing_rate=30, w_init=simulation_results['new_weights'])
    
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

    mm = 1/25.4
    fig = plt.figure(figsize=(183*mm, 80*mm))

    example_color = 'steelblue'
    example_synapse = 6

    ###########################
    # Column 1
    linewidth = 1
    axes = gs.GridSpec(nrows=2, ncols=3, left=0.05, right=1, top=0.95, bottom=0.1, wspace=0.4, hspace=0.5)

    ## Raster plot
    ax = fig.add_subplot(axes[0,0])

    plateau_time = simulation_results['plateau_time']

    for synapse_id, spike_times in enumerate(simulation_results['input_spike_times']):
        spike_times = (spike_times - plateau_time)/1000
        if synapse_id == example_synapse:
            ax.scatter(spike_times, synapse_id*np.ones(len(spike_times)), s=10, color=example_color, marker='|', linewidth=0.5)
        else:
            ax.scatter(spike_times, synapse_id*np.ones(len(spike_times)), s=10, color='gray', marker='|', linewidth=0.3)
    ax.set_ylabel('Synapse')
    ax.set_ylim(top=len(simulation_results['input_spike_times']))
    ax.set_ylim(ax.get_ylim()[::-1]) # flip y axis
    ax.set_xlim([-3,3])

    ## ET/IS example traces
    time = (simulation_results['time'] - plateau_time)/1000
    ax = fig.add_subplot(axes[1,0])
    IS = simulation_results['dendrite_IS']
    IS_baseline = 1/IS.insulatorR
    IS_peak = 1/IS.metalR
    IS_norm = (np.array(IS.g_history) - IS_baseline) / (IS_peak - IS_baseline) *1.2
    ax.plot(time, IS_norm, color='r', linewidth=linewidth, label='IS')

    ET = simulation_results['synapse_ETs'][example_synapse]
    ET_baseline = 1/ET.insulatorR
    ET_peak = 1/ET.metalR
    ET_norm = (np.array(ET.g_history) - ET_baseline) / (ET_peak - ET_baseline) *1.9
    ax.plot(time, ET_norm, color=example_color, linewidth=linewidth, label='ET')
    ax.set_xlim([-3,3])

    measurement_timepoints = np.where(simulation_results['measured_IS']>0)[0]
    ax.hlines(y=1, xmin=time[measurement_timepoints][0], xmax=time[measurement_timepoints][-1], color='k', alpha=0.5, linewidth=2*linewidth)

    ax.set_ylabel('Signal amplitude')
    ax.set_xlabel('Time from dendritic spike (s)')
    ax.legend(loc='upper right', bbox_to_anchor=(0.9, 1.), handlelength=1, frameon=False, fontsize=6, handletextpad=0.5)

    axes_sub = gs.GridSpec(nrows=1, ncols=3, left=0.05, right=1, top=0.5, bottom=0.45, wspace=0.4, hspace=0.4)
    ax = fig.add_subplot(axes_sub[0])
    ET_switch = simulation_results['ET_switches'][example_synapse]/10 + 1
    IS_switch = simulation_results['IS_switch']/10 + 1
    ax.plot(time, ET_switch, color=example_color, linewidth=linewidth)
    ax.plot(time, IS_switch, color='r', linewidth=linewidth)
    ax.axis('off')
    ax.set_xlim([-3,3])

            
    ###########################
    # Column 2

    ## Weight update plot
    ax = fig.add_subplot(axes[1,1])
    w_init = simulation_results['weights']
    w_final = simulation_results['new_weights']
    num_synapses = len(w_init)
    T = simulation_results['time'][-1]

    field_width = simulation_results['field_width']
    position = (np.linspace(0, 6000, num_synapses) + field_width/2 - plateau_time)/1000
    ax.plot(position, w_init, c='gray', alpha=0.5, label='Initial synaptic weights', linewidth=1.5*linewidth)
    ax.plot(position, w_final, c='k', label='Weights after dendritic spike', linewidth=1.5*linewidth)
    ax.scatter(position[example_synapse], w_final[example_synapse], s=30, facecolors='none', edgecolors=example_color, linewidth=1, zorder=10)

    ax.set_ylabel('$\Delta W$ (norm.)')
    ax.set_xlabel('Time from dendritic spike (s)')
    # ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.), handlelength=1, frameon=False)
    ax.set_xlim([-3,3])


    ## Vm ramp from eLife paper (Fig.1 example neuron)
    import pickle
    filename = 'data/20230918_BTSP_induction_example.pkl'
    with open(filename, 'rb') as f:
        BTSP_induction_recording = pickle.load(f)

    ax = fig.add_subplot(axes[0,1])
    time = BTSP_induction_recording['t']
    dV = BTSP_induction_recording['delta_ramp']
    ax.plot(time/1000, dV, color='k', linewidth=linewidth)
    ax.set_xlabel('Time from dendritic spike (s)')
    ax.set_ylabel('$\Delta V_m$ (mV)')
    # ax.set_xlim([-3,3])

    ###########################
    # Column 3
    axes = gs.GridSpec(nrows=2, ncols=3, left=0, right=0.96, top=0.5, bottom=0.1)

    linewidth = 0.3
    ax = fig.add_subplot(axes[0,2])
    shift = 5
    ax.plot(simulation_results['time'], simulation_results['V']*1e3+shift, color='k', linewidth=linewidth)
    # ax.axhline(y=simulation_results['V_th']*1e3, color='k', linestyle='--', alpha=0.3, linewidth=1.5)
    ax.set_ylim([0,25])
    ax.axis('off')
    ax.set_xlim([500,7500])

    ax = fig.add_subplot(axes[1,2])
    ax.plot(simulation_results2['time'], simulation_results2['V']*1e3, color='k', linewidth=linewidth)
    ax.hlines(y=simulation_results2['V_th']*1e3, xmin=1500, xmax=5200, color='k', linestyle='--', alpha=0.3, linewidth=1.5)
    ax.text(6500, simulation_results2['V_th']*1e3+1, 'Spike threshold', ha='center', va='bottom', fontsize=6, color='k', alpha=0.5, fontname='Helvetica')
    ax.set_xlim([500,7500])

    spike_times = simulation_results2['output_spike_times']
    ax.scatter(spike_times, np.ones(len(spike_times))*simulation_results2['V_th']*1e3+3, s=30, color='r', marker='|', linewidth=1)
    ax.set_ylim([0,25])
    ax.axis('off')
    ax.set_xlim([500,7500])

    # make electrophysiology-style simple black scale bars
    scalebar_x = 6500
    scalebar_width = 1000
    scalebar_y = 12
    scalebar_height = 5
    ax.vlines(x=scalebar_x, ymin=scalebar_y, ymax=scalebar_y+scalebar_height, color='k', linewidth=1)
    ax.text(scalebar_x-200, scalebar_y+scalebar_height/2, f'{scalebar_height} mV', ha='right', va='center', fontsize=6, fontname='Helvetica')
    ax.hlines(y=scalebar_y, xmin=scalebar_x, xmax=scalebar_x+scalebar_width, color='k', linewidth=1)
    ax.text(scalebar_x+scalebar_width/2, scalebar_y-1, f'{int(scalebar_width/1e3)} s', ha='center', va='top', fontsize=6, fontname='Helvetica')

    if show:
        plt.show()

    if save:
        fig.savefig('figures/Fig4-BTSP/spiking_BTSP_plots.svg', transparent=True, bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    update_plot_defaults()
    generate_Figure4(show=True, save=False)