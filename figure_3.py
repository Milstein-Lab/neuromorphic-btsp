import numpy as np
import matplotlib.pyplot as plt

from utils import Volatile_Resistor,  Memristor, poisson_spike_train


########################################################################################################################
# Figure 4: Spiking VO2
########################################################################################################################



def VO2_LIF_simulation(VO2_temp, num_synapses, firing_rate, VO2_pulse_dur):
    ##############################################################
    ## Setup
    ##############################################################

    # Units
    mV = 1e-3
    kOhm = 1e3
    MOhm = 1e6
    pF = 1e-12
    nF = 1e-9
    uF = 1e-6
    mF = 1e-3
    mA = 1e-3
    uA = 1e-6
    nA = 1e-9

    # Simulation runtime parameters
    dt = 0.1  # time step (ms)
    T = 800   # simulation time (ms)
    time = np.arange(0., T, dt)

    input_stim_duration = 5 # ms
    V_low = 0.1 # V
    # V_high = 1. # V
    VO2_stim_amp = 1. # mA

    spike_times = [poisson_spike_train(firing_rate, T/1e3, refractory_period=0.003)*1e3 for _ in range(num_synapses)]

    # spike_times1 = np.array([50,100])
    # spike_times2 = np.array([30,120, 200])
    # spike_times = [spike_times1, spike_times2]

    # Neuron model parameters
    VO2_cell = Volatile_Resistor(dt, temperature=VO2_temp, metalR=100, insulatorR=10*kOhm, stim_scaling=100)
    C = 1*uF
    V_th = 0.02 # V

    # Synapse parameters
    synapses = [Memristor(conductingR=15*kOhm, insulatorR=200*kOhm) for _ in range(num_synapses)]
    for R_mem in synapses:
        R_mem.R = 100*kOhm
    synapses[0].R = 50*kOhm


    ##############################################################
    ## Simulation
    ##############################################################

    V = np.zeros(len(time))
    I = np.zeros(len(time))
    I_syn = np.zeros([num_synapses, len(time)])
    output_spike_times = []
    VO2_cell_switch = np.zeros(len(time))
    VO2_pulse_dur_idx = np.round(VO2_pulse_dur/dt).astype(int)

    # Create a stimulus trigger for each synapse
    stim_dur_idx = np.round(input_stim_duration/dt).astype(int)
    kernel = np.ones(stim_dur_idx)
    stim_triggers = np.zeros([num_synapses,len(time)])
    for synapse_id,times in enumerate(spike_times):
        stim_idx = (times/dt).astype(int)
        spike_train = np.zeros(len(time))
        spike_train[stim_idx] = 1
        stim_trigger_train = np.convolve(spike_train, kernel)[0:len(time)]
        stim_trigger_train = np.minimum(stim_trigger_train, 1) # clip to [0,1]
        stim_triggers[synapse_id] = stim_trigger_train

    # Run main simulation loop
    for t in range(len(time)):
        if V[t]>V_th and V[t-1]<V_th:
            output_spike_times.append(t)
            VO2_cell_switch[t:t+VO2_pulse_dur_idx] = 1    
            V[t] = V_th

        if VO2_cell_switch[t] == 1:
            VO2_cell.controlI = VO2_stim_amp
            # VO2_cell.controlI = V_high / VO2_cell.R
            dVdt = 0
        else:
            VO2_cell.controlI = 0
            V_probe = stim_triggers[:,t] * V_low
            I_syn[:,t] = (V_probe - V[t]) / [memristor.R for memristor in synapses]
            I_syn[:,t] = np.maximum(I_syn[:,t], 0)
            I[t] = np.sum(I_syn[:,t])
            dVdt = I[t]/C - V[t]/(VO2_cell.R*C)

        if t < len(time)-1:
            V[t+1] = V[t] + dVdt*dt*1e-3 # convert to ms
        
        VO2_cell.time_step()

    output_spike_times = np.array(output_spike_times)*dt

    simulation_results = {
        'time': time,
        'V': V,
        'I': I,
        'I_syn': I_syn,
        'output_spike_times': output_spike_times,
        'VO2_cell': VO2_cell,
        'synapses': synapses,
        'stim_triggers': stim_triggers,
        'spike_times': spike_times,
        'V_th': V_th,
        'VO2_pulse_dur': VO2_pulse_dur,
    }

    return simulation_results


def generate_Figure3(show=False, save=False):
    np.random.seed(123)
    linewidth = 0.5

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
                    'text.usetex': False})

    mm = 1/25.4
    fig, axes = plt.subplots(4,2, figsize=(130*mm,95*mm))
    temperatures = [62.01, 69.25] # 1ms and 100ms time constants
    pulsewidth = [3., 10.]

    # titles = [f'Soma (fasrtVO$_{2}$ temp. = {temperatures[0]} C)', f'Dendrite (VO$_{2}$ temp. = {temperatures[1]} C)']
    example_colors = [['#FF6754', '#2D93DC'], ['#FF7F00', '#984EA3']]
    for col, VO2_temp in enumerate(temperatures):
        simulation_results = VO2_LIF_simulation(VO2_temp, num_synapses=10, firing_rate=30, VO2_pulse_dur=pulsewidth[col])

        ax = axes[0,col]
        for synapse_id, spike_times in enumerate(simulation_results['spike_times']):
            if synapse_id in [0,1]:
                color = example_colors[col][synapse_id]
            else:
                color = 'gray'
            ax.scatter(spike_times, synapse_id*np.ones(len(spike_times)), s=10, color=color, marker='|', linewidth=linewidth)
        ax.set_ylabel('Synapse')
        ax.set_ylim(top=len(simulation_results['spike_times']))
        ax.set_ylim(ax.get_ylim()[::-1]) # flip y axis
        ax.set_yticks(np.arange(0, len(simulation_results['spike_times']), 3))
        ax.set_yticklabels(np.arange(1, len(simulation_results['spike_times'])+1, 3))
        # ax.set_title(titles[col], fontsize=10, y=1.)
        ax.set_xticklabels([])
        
        ax = axes[1,col]
        # ax.plot(simulation_results['time'], simulation_results['I'])
        ax.plot(simulation_results['time'], simulation_results['I_syn'][0]*1e6, label=f'Synapse 1 (R={simulation_results["synapses"][0].R/1e3:.0f} k$\Omega$)', color=example_colors[col][0], linewidth=linewidth)
        ax.plot(simulation_results['time'], simulation_results['I_syn'][1]*1e6, label=f'Synapse 2 (R={simulation_results["synapses"][1].R/1e3:.0f} k$\Omega$)', color=example_colors[col][1], linewidth=linewidth)
        ax.set_ylabel('Current ($\mu$A)')
        ax.set_xticklabels([])
        # ax.legend(loc='upper right',ncol=1, frameon=False, handlelength=0.8, handletextpad=0.3, bbox_to_anchor=(1., 1.1))
        
        ax = axes[2,col]
        ax.plot(simulation_results['time'], simulation_results['V']*1e3, color='k', linewidth=linewidth)
        ax.axhline(y=simulation_results['V_th']*1e3, color='k', linestyle='--', alpha=0.3, linewidth=1.5)
        spike_times = simulation_results['output_spike_times']
        ax.scatter(spike_times, 1.1*np.ones(len(spike_times))*simulation_results['V_th']*1e3, s=30, color='r', marker='|', linewidth=2*linewidth)
        ax.set_ylabel('Voltage (mV)')
        ax.set_xticklabels([])

        ax = axes[3,col]
        ax.plot(simulation_results['time'], np.array(simulation_results['VO2_cell'].g_history)*1000, color='k', linewidth=linewidth)
        ax.set_ylabel('Conductance (mS)')
        g_max = np.max(simulation_results['VO2_cell'].g_history)*1000
        for t in spike_times:
            ax.plot([t, t+simulation_results['VO2_pulse_dur']], [g_max*1.1, g_max*1.1], color='r', linewidth=1.5)
        ax.set_xlabel('Time (ms)')

    plt.tight_layout(h_pad=1, w_pad=1)

    if show:
        plt.show()

    if save:
        fig.savefig('figures/Fig3-spiking_VO2/spiking_VO2_plots.svg', transparent=True, bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    generate_Figure3(show=True, save=False)