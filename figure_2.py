import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.ticker as mticker
import pandas as pd
import os
import scipy

from utils import update_plot_defaults, Volatile_Resistor


def generate_Figure2(show=False, save=False):
    mm = 1 / 25.4  # millimeters in inches
    fig = plt.figure(figsize=(180 * mm, 190 * mm))
    axes = gs.GridSpec(nrows=4, ncols=3,
                       left=0.05,right=0.98,
                       top=0.95, bottom = 0.08,
                       wspace=0.4, hspace=0.4)

    ax = fig.add_subplot(axes[0, 1])
    plot_RV_hysteresis(ax)

    ax = fig.add_subplot(axes[0, 2])
    plot_RV_hysteresis_simulation(ax)

    ax = fig.add_subplot(axes[1, 0])
    plot_relaxation_tree(ax)

    ax = fig.add_subplot(axes[1, 1])
    plot_relaxation_timescales(ax)

    ax = fig.add_subplot(axes[1, 2])
    plot_relaxation_timescales_simulation(ax)

    ax = fig.add_subplot(axes[2, 0])
    plot_short_steps(ax)

    ax = fig.add_subplot(axes[2, 1])
    plot_tau_scatter(ax)

    ax = fig.add_subplot(axes[2, 2])
    plot_tau_scatter_simulation(ax)

    ax = fig.add_subplot(axes[3, 0])
    plot_VO2_dev2dev_variability_data(ax)

    ax1 = fig.add_subplot(axes[3, 1])
    ax2 = fig.add_subplot(axes[3, 2])
    plot_VO2_cyc2cyc_variability_data(ax1, ax2)

    if save:
        fig.savefig('figures/Fig2-VO2-properties/Figure2_plots.svg', dpi=300)
        fig.savefig('figures/Fig2-VO2-properties/Figure2_plots.png', dpi=300)

    if show:
        plt.show()



def fit_single_exponential(time, trace, tau_est=0.1, verbose=False):
    monoExp = lambda x,m,t,b: m * np.exp(-t * x) + b

    p0 = (1., 1/tau_est, 0.01) # (m, t, b), start with values near those we expect
    params, cv = scipy.optimize.curve_fit(monoExp, time, trace, p0)
    m, t, b = params
    tau = 1/t

    # Determine quality of the fit
    squaredDiffs = np.square(trace - monoExp(time, m, t, b))
    squaredDiffsFromMean = np.square(trace - np.mean(trace))
    rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
    if verbose:
        print(f"R² = {rSquared}")
        print(f"tau = {tau}")

    # if rSquared < 0.9:
    #     print("WARNING: Fit is poor. R² < 0.9")

    return tau


def plot_RV_hysteresis(ax):
    RV_data = pd.read_csv('data/VO2_data_RvsV_transition.csv', header=34)
    time_data = RV_data['SMU-1 Time (s)']
    V = RV_data['SMU-1 Voltage (V)']
    I = RV_data['SMU-1 Current (A)']*1000
    R = RV_data['SMU-1 Resistance (Ω)']/1000

    ax.scatter(V[209:], 1/R[209:], s=10, c='steelblue', linewidth=0)
    ax.scatter(V[:209], 1/R[:209], s=10, c='red', linewidth=0)
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Conductance (mS)')


def plot_RI_hysteresis(ax):
    RV_data = pd.read_csv('data/VO2_data_RvsV_transition.csv', header=34)
    time_data = RV_data['SMU-1 Time (s)']
    V = RV_data['SMU-1 Voltage (V)']
    I = RV_data['SMU-1 Current (A)']*1000
    R = RV_data['SMU-1 Resistance (Ω)']/1000

    ax.scatter(I[209:], R[209:], s=10)
    ax.scatter(I[:209], R[:209], s=10, c='r')
    ax.set_xlabel('Current (mA)')
    ax.set_ylabel('Resistance (kΩ)')

    # Plot R-I curve fit
    x = np.linspace(0,100,1000)
    y = lambda x: 5000*np.exp(-0.12*x)+220
    ax.plot(x,y(x)/1000, 'k--')


def plot_RV_hysteresis_simulation(ax):
    dt = 100  # ms
    T = 150000  # ms
    time = np.arange(0, T, dt)
    v1 = np.linspace(0, 22, len(time) // 2)
    v2 = np.flip(v1)
    V = np.concatenate([v1, v2])

    g_volatile = Volatile_Resistor(dt, temperature=70.7, metalR=220)

    for t in range(len(time)):
        g_volatile.controlI = V[t] / g_volatile.R * 1000
        g_volatile.time_step()
    time = time / 1000
    g_volatile.R_history = np.array(g_volatile.R_history) / 1000

    ax.scatter(V[len(v1):], 1/g_volatile.R_history[len(v1):], s=10, c='steelblue', linewidth=0)
    ax.scatter(V[:len(v1)], 1/g_volatile.R_history[:len(v1)], s=10, c='red', linewidth=0)
    ax.set_xlabel('V')
    ax.set_ylabel('Conductance (mS)')


def plot_relaxation_timescales(ax):
    slow_relaxation_data1 = pd.read_excel('data/VO2_data_currents.xlsx', header=0)  # Large current steps at 70C
    time_slow = np.array(slow_relaxation_data1['Time'][2:])[100:]
    resistance_traces = np.array(slow_relaxation_data1.iloc[2:, 2::4]).T
    g_slow = 1 / resistance_traces[0][100:]
    g_slow = (g_slow - np.min(g_slow)) / (np.max(g_slow) - np.min(g_slow))

    g_medium = 1 / resistance_traces[5][100:]
    g_medium = (g_medium - np.min(g_medium)) / (np.max(g_medium) - np.min(g_medium))

    fast_relaxation_data = pd.read_excel('data/relaxation_timescales.xlsx', header=0, sheet_name='slow')  # Large current steps at 64C
    time_fast = np.array(fast_relaxation_data['Time (s)'][2:])[10:]
    resistance_traces = np.array(fast_relaxation_data.iloc[2:, 2:]).T
    current_steps = list(fast_relaxation_data.iloc[1][2:])
    current_steps = np.array([float(pw[:-2]) for pw in current_steps])
    g_fast = 1 / resistance_traces[0][10:]
    g_fast = (g_fast - np.min(g_fast)) / (np.max(g_fast) - np.min(g_fast))

    ax.plot(time_fast-0.1, g_fast, label='64°C, 70mA', c='darkgray')
    ax.plot(time_slow-3.05, g_medium, label='70°C, 40mA', c='gray')
    ax.plot(time_slow-3.05, g_slow, label='70°C, 100mA', c='k')

    ax.set_xlim([-0.1, 2.1])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Conductance [g] (norm.)')
    ax.legend(loc='upper right', bbox_to_anchor=(0.9, 1), fontsize=8, handlelength=0.7, handletextpad=0.4)


def plot_tau_scatter(ax):
    pointsize = 6

    # Plot dataset 1 (ultraslow relaxation, variable current steps)
    ultraslow_relaxation_data1 = pd.read_excel('data/VO2_data_currents.xlsx', header=0)  # Large current steps at 70C
    current_steps = np.array([100, 90, 80, 70, 50, 40, 30, 20])
    time_ultraslow = np.array(ultraslow_relaxation_data1['Time'][2:])
    resistance_traces = np.array(ultraslow_relaxation_data1.iloc[2:, 2::4]).T

    time = time_ultraslow[:-3000].astype(float)
    tau_g = np.zeros(len(current_steps))
    for i, R in enumerate(resistance_traces):
        g = 1 / R[3000:].astype(float)
        g_norm = (g - np.min(g)) / (np.max(g) - np.min(g))
        tau_g[i] = fit_single_exponential(time, g_norm, tau_est=1., verbose=False)
    ax.scatter(current_steps, tau_g*1000, s=pointsize, label='70°C', c='k')

    # Plot dataset 2 (slow relaxation)
    ultraslow_relaxation_data2 = pd.read_excel('data/relaxation_timescales.xlsx', header=0, sheet_name='ultraslow')
    time_ultraslow2 = np.array(ultraslow_relaxation_data2['Time (s)'][2:])[105:3000] - 0.1
    resistance_traces = np.array(ultraslow_relaxation_data2.iloc[2:, 2::2]).T
    R = resistance_traces[1][105:3000]
    g = 1 / R
    g_norm = (g - np.min(g)) / (np.max(g) - np.min(g))
    tau_g = fit_single_exponential(time_ultraslow2, g_norm, tau_est=0.01, verbose=False)
    ax.scatter(100, tau_g*1000, s=pointsize, c='darkgray')

    # Plot dataset 2 (medium-slow relaxation, variable current steps)
    slow_relaxation_data = pd.read_excel('data/relaxation_timescales.xlsx', header=0, sheet_name='slow')
    time_data = np.array(slow_relaxation_data['Time (s)'][2:])
    resistance_traces = np.array(slow_relaxation_data.iloc[2:, 2:]).T[0:6]
    current_steps = list(slow_relaxation_data.iloc[1][2:])[0:6]
    current_steps = np.array([float(pw[:-2]) for pw in current_steps])

    tau_g = []
    for i in range(6):
        if i in [0, 1]:
            time = time_data[110:] - 0.11
            R = resistance_traces[i][110:]
        else:
            time = time_data[101:] - 0.101
            R = resistance_traces[i][101:]
        g = 1 / R
        g_norm = (g - np.min(g)) / (np.max(g) - np.min(g))
        tau = fit_single_exponential(time, g_norm, tau_est=0.001, verbose=False)
        tau_g.append(tau)
    tau_g = np.array(tau_g)
    ax.scatter(current_steps, tau_g*1000, s=pointsize, label='64°C', c='darkgray')

    # Format axes
    ax.set_ylim([0.1, 1000])
    ax.set_xlim([0, 102])
    ax.legend(loc='lower right', bbox_to_anchor=(1.05, -0.05), labelspacing=0, handletextpad=-0.6, frameon=False,
              fontsize=8)
    ax.set_yscale('log')
    label_format = '{:,.1f}'
    ticks_loc = ax.get_yticks().tolist()
    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    ax.set_yticklabels([label_format.format(x) for x in ticks_loc])
    ax.set_xlabel('Current (mA)')
    ax.set_ylabel('Conductance \ndecay tau (ms)',  labelpad=-9)


def plot_relaxation_tree(ax):
    tree_data = pd.read_excel('data/relaxation_tree.xlsx', header=0)
    resistance_traces = np.array(tree_data.iloc[2:, 1:]).astype(float).T /1000
    time = np.array(tree_data['Time '][2:]).astype(float)

    for i,R in enumerate(resistance_traces):
        if i == 0:
            ax.plot(time, R, color='k', linewidth=1.5)
        else:
            ax.plot(time, R, color=[.4,.4,.4], linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Resistance (kΩ)')


def plot_short_steps(ax):
    Vsteps_data = {}
    for file in os.listdir('data/short_Vpulses_data'):
        if file.endswith('.csv'):
            Vsteps_data[file] = pd.read_csv('data/short_Vpulses_data/' + file, header=0)

    for i, pulse_amp in enumerate(Vsteps_data):
        if i==0:
            time = Vsteps_data[pulse_amp]['SMU-1 Time (s)'][:300] - 0.01
        R = Vsteps_data[pulse_amp]['SMU-1 Resistance (Ω)'][500:800]
        g = 1 / R * 1000
        ax.plot(time, g, c=[.1, .1, .1], label=pulse_amp, linewidth=1)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('g (mS)')


def plot_relaxation_timescales_simulation(ax):
    dt = 1
    T = 5000
    time = np.arange(0., T, dt)
    stim_time = (0, 2000)

    g_list = []
    temp_list = [64, 70, 70]
    stim_list = [70, 40, 100]
    VO2_list = [Volatile_Resistor(dt, temperature=temp, metalR=100) for temp in temp_list]

    for i,VO2 in enumerate(VO2_list):
        for t in time:
            if t >= stim_time[0] and t < stim_time[1]:
                VO2.controlI = stim_list[i]
            else:
                VO2.controlI = 0
            VO2.time_step()

        # Convert from VO2_R to norm. conductance
        g = np.array(VO2.g_history)
        g_norm = (g-g.min()) / (g.max()-g.min())
        g_list.append(g_norm)

    g_all = np.array(g_list).T
    time = (time-stim_time[1])/1000
    ax.plot(time, g_all[:,0], color='darkgray', linewidth=1.5, label=f'{temp_list[0]}°C, {stim_list[0]}mA')
    ax.plot(time, g_all[:,1], color='gray', linewidth=1.5, label=f'{temp_list[1]}°C, {stim_list[1]}mA')
    ax.plot(time, g_all[:,2], color='k', linewidth=1.5, label=f'{temp_list[2]}°C, {stim_list[2]}mA')

    ax.legend(loc='upper right', bbox_to_anchor=(0.9, 1), fontsize=8, handlelength=0.7, handletextpad=0.4)
    ax.set_xlabel('Time (s)')
    ax.set_xlim([-0.1, 2.1])
    ax.set_ylabel('Conductance [g] (norm.)')


def plot_tau_scatter_simulation(ax):
    dt = 0.1
    T = 5000
    time = np.arange(0., T, dt)
    stim_time = (0, 2000)
    current_steps = np.arange(5, 105, 5)
    temp_list = [70, 67, 64]

    g_traces = {temp: [] for temp in temp_list}
    for temp in temp_list:
        for stim_current in current_steps:
            VO2 = Volatile_Resistor(dt, temperature=temp, metalR=100)
            for t in time:
                if t >= stim_time[0] and t < stim_time[1]:
                    VO2.controlI = stim_current
                else:
                    VO2.controlI = 0
                VO2.time_step()

            # Convert from VO2_R to norm. conductance
            g = np.array(VO2.g_history)
            g_norm = (g - g.min()) / (g.max() - g.min())
            g_traces[temp].append(g_norm)

    time_scaled = (time - stim_time[1])  # set stim end at time 0

    tau_g = []
    for i, trace in enumerate(g_traces[temp_list[0]]):
        tau = fit_single_exponential(time_scaled[int(stim_time[1]/dt):], trace[int(stim_time[1]/dt):], tau_est=0.1)
        tau_g.append(tau)
    ax.scatter(current_steps, tau_g, label=str(temp_list[0]) + '°C', s=6, color='k')

    tau_g = []
    for i, trace in enumerate(g_traces[temp_list[1]]):
        tau = fit_single_exponential(time_scaled[int(stim_time[1]/dt):], trace[int(stim_time[1]/dt):], tau_est=0.1)
        tau_g.append(tau)
    ax.scatter(current_steps, tau_g, label=str(temp_list[1]) + '°C', s=6, color='gray')

    tau_g = []
    for i, trace in enumerate(g_traces[temp_list[2]]):
        tau = fit_single_exponential(time_scaled[int(stim_time[1]/dt):], trace[int(stim_time[1]/dt):], tau_est=0.1)
        tau_g.append(tau)
    ax.scatter(current_steps, tau_g, label=str(temp_list[2]) + '°C', s=6, color='darkgray')

    ax.set_ylim([0.1, 1000])
    ax.legend(loc='lower right', bbox_to_anchor=(1.05, -0.05), labelspacing=0, handletextpad=-0.6, frameon=False,
              fontsize=8)
    ax.set_yscale('log')
    label_format = '{:,.1f}'
    ticks_loc = ax.get_yticks().tolist()
    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    ax.set_yticklabels([label_format.format(x) for x in ticks_loc])
    ax.set_xlabel('Current (mA)')
    ax.set_ylabel('Conductance \ndecay tau (ms)', labelpad=-9)
    ax.set_xlim([0, 102])


def plot_VO2_dev2dev_variability_data(ax):
    # Device to device variability
    dev2dev_variability_data = pd.read_csv('data/dev2dev_gv_data.csv',header=2)
    time = dev2dev_variability_data.iloc[:,0]
    conductance_10mA = np.array(dev2dev_variability_data.iloc[:,1:10])
    conductance_20mA = np.array(dev2dev_variability_data.iloc[:,11:20])
    conductance_30mA = np.array(dev2dev_variability_data.iloc[:,21:30])
    conductance_40mA = np.array(dev2dev_variability_data.iloc[:,31:40])

    all_traces = {'10 mA':conductance_10mA, '20 mA':conductance_20mA, '30 mA':conductance_30mA, '40 mA':conductance_40mA}
    colors = ['c','m','b','k']
    for i, (stim_amplitude, traces) in enumerate(all_traces.items()):
        ax.plot(time, traces[:,0], label=stim_amplitude, color=colors[i], alpha=0.4, linewidth=0.5)
        ax.plot(time,traces[:,1:], color=colors[i], alpha=0.4, linewidth=0.5)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Conductance (mS)', labelpad=-1)
    ax.set_xlim(-0.1, 3)
    ax.legend()


def plot_VO2_cyc2cyc_variability_data(ax2, ax1):
    cyc2cyc_variability_data = pd.read_excel('data/cycle-to-cycle-data-long.xlsx', sheet_name='decay time constant', header=2)
    device_time_constants = np.array(cyc2cyc_variability_data.iloc[:,1:4])

    mean_tau = np.mean(device_time_constants, axis=1)*1000
    SD_tau = np.std(device_time_constants, axis=1)*1000
    ax1.plot(mean_tau, '-o', c='k', markersize=2, linewidth=0.5)
    ax1.fill_between(np.arange(len(mean_tau)), mean_tau-SD_tau, mean_tau+SD_tau, alpha=0.2, color='k')
    ax1.set_ylabel('Decay time constant (ms)', labelpad=-0.3)
    ax1.set_xlabel('Sample measurement time (min)')
    ax1.set_ylim(0,60)

    dev1 = pd.read_excel('data/cycle-to-cycle-data-long.xlsx', sheet_name='d1 10 mA', header=2)
    dev2 = pd.read_excel('data/cycle-to-cycle-data-long.xlsx', sheet_name='d2 10 mA', header=2)
    dev3 = pd.read_excel('data/cycle-to-cycle-data-long.xlsx', sheet_name='d3 10 mA', header=2)
    time = dev1.iloc[:,0]

    dev1_conductance = np.array(dev1.iloc[:,1:])
    dev2_conductance = np.array(dev2.iloc[:,1:])
    dev3_conductance = np.array(dev3.iloc[:,1:])

    pastel_colors = ['lightgreen', 'steelblue', 'lightcoral']
    ax2.plot(time, dev1_conductance, alpha=0.5, linewidth=0.3)
    # ax2.plot(time, dev1_conductance, c=pastel_colors[0], alpha=0.5, linewidth=0.3)
    # ax2.plot(time, dev2_conductance, c=pastel_colors[1], alpha=0.5, linewidth=0.3)
    # ax2.plot(time, dev3_conductance, c=pastel_colors[2], alpha=0.5, linewidth=0.3)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Conductance (mS)', labelpad=-0.3)
    ax2.set_xlim(-0.1, 3)



if __name__=="__main__":
    update_plot_defaults()
    generate_Figure2(show=True, save=True)


