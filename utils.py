import numpy as np
import matplotlib.pyplot as plt

def update_plot_defaults():
    plt.rcParams.update({'font.size': 6,
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


class Volatile_Resistor:
    def __init__(self, dt, temperature=70, metalR=100, insulatorR=None, stim_scaling=1):
        '''
        Class for simulating a volatile resistor
        dt: time step in ms
        temperature: ambient temperature in degrees C
        metalR: lowest resistance (metallic state) in Ohms
        insulatorR: highest resistance (insulating state) in Ohms
        stim_scaling: scaling factor for the control current
        '''
        self.dt = dt
        self.amb_temp = temperature  # ambient temperature in degrees C
        if insulatorR is None:
            self.insulatorR = self.temperature_sigmoid(self.amb_temp)
        else:
            self.insulatorR = insulatorR
        self.metalR = metalR
        self.R = self.insulatorR
        self.g = 1 / self.R

        self.peak_g = self.g
        self.controlI = 0
        self.controlI_timer = 0
        self.decay_tau_multiplier = 1
        self.stim_scaling = stim_scaling

        self.R_history = []
        self.g_history = []
        self.controlI_history = []

    def time_step(self):
        self.R_history.append(self.R)
        self.g_history.append(self.g)
        self.controlI_history.append(self.controlI)

        R_eq = self.transfer_func(self.controlI * self.stim_scaling)
        g_eq = 1 / R_eq

        if self.g <= g_eq:  # rising phase
            self.peak_g = self.g
            rise_tau = np.maximum(self.dt, self.rise_tau(self.controlI * self.stim_scaling, temperature=self.amb_temp))  # tau can't be less than dt
            dgdt = (g_eq - self.g) / rise_tau
        else:  # decay/relaxation phase
            decay_tau = self.decay_tau(g=self.peak_g, temperature=self.amb_temp)
            decay_tau = np.maximum(self.dt, decay_tau)  # tau can't be less than dt
            dgdt = (g_eq - self.g) / decay_tau

        self.g += dgdt * self.dt
        self.R = 1 / self.g

    def temperature_sigmoid(self, temp, a=0.001):
        slow_component = 2.3e7 / (temp + 200)
        sigm = (1 - a) / (1 + np.exp((temp - 62) / 3)) + a
        minR = 10
        R = slow_component * sigm + minR
        return R

    def transfer_func(self, controlI):
        '''
        Function to convert between controlI and resistance
        controlI: current in units of mA
        '''
        
        R_eq = (self.insulatorR - self.metalR) * np.exp(-0.11 * controlI) + self.metalR  # (Ohm)
        # g_eq = 1 / R_eq * 1e9  # convert to conductance (nS)
        return R_eq

    def inv_transfer_func(self, R):
        controlI = np.log((R - self.metalR) / (self.insulatorR - self.metalR)) / (-0.12)
        return controlI

    def decay_tau(self, controlI=None, g=None, temperature=70):
        '''
        Decay timeconstant for the VO2 conductance, fit to slow relaxation data
        '''
        if controlI is None:
            R = 1 / g
            R = np.maximum(R, self.metalR+0.1)
            assert R > self.metalR, 'Maximum current exceeded'
            controlI = self.inv_transfer_func(R)
        m = 0.2277
        t = 86.81
        b = -0.1167
        tau = (m * np.exp(controlI * 1 / t) + b)
        tau *= self.temperature_scaling(temperature)
        tau *= 1e3  # convert to ms
        return tau

    def rise_tau(self, controlI, temperature=70):
        tau = (-6e-4 * controlI + 0.175)
        tau *= self.temperature_scaling(temperature)
        tau = tau * 1e3  # convert to ms
        return tau

    def temperature_scaling(self, temperature):
        '''
        Sigmoid function to scale time constants with temperature, calibrated to = 1 at 70 C
        '''
        sigmoid = 20 / (1+np.exp((74-temperature)/1.359))
        return sigmoid


class Memristor:
    def __init__(self, conductingR=15_000., insulatorR=200_000., alpha=0.01, dt=0.1):
        self.conductingR = conductingR  # Initial conducting value in ohms
        self.insulatorR = insulatorR  # Insulating value in ohms
        self.alpha = alpha  # Rate of resistance change (a tuning parameter)
        self.dt = dt  # Time step for simulation

        self.R = self.insulatorR  # Initial state (high resistance)
        self.R_history = []

    def update(self, current):
        # Update the memristor state based on current passed through it
        delta_R = self.alpha * self.R * current * self.dt
        self.R -= delta_R
        self.R_history.append(self.R)

        # Ensure that resistance values stay within bounds
        self.R = max(self.R, self.conductingR)
        self.R = min(self.R, self.insulatorR)

    @property
    def g(self):
        return 1/self.R
    

def poisson_spike_train(rate, t, refractory_period):
    '''
    Generates a spike train from a Poisson process with a given rate and refractory period.
    rate: average firing rate in Hz
    t: length of spike train in seconds
    refractory_period: minimum time between spikes in seconds
    '''
    n_spikes = np.random.poisson(rate * t)
    
    inter_spike_intervals = np.random.exponential(1 / rate, n_spikes)    
    inter_spike_intervals[inter_spike_intervals < refractory_period] = refractory_period    
    spike_times = np.cumsum(inter_spike_intervals)
    spike_times = spike_times[spike_times <= t] # remove spike times beyond t
    return spike_times


def get_scaled_sigmoid(slope, threshold):
    '''
    Returns a callable function for a scaled sigmoid with the given slope and threshold.
    The sigmoid is scaled so that it goes from 0 to 1 for x in [0,1]
    '''
    peak =  1. / (1. + np.exp(-slope * (1 - threshold)))    # value of sigmoid at x=1
    baseline = 1. / (1. + np.exp(-slope * (0 - threshold))) # value of sigmoid at x=0
    scaled_sigmoid = lambda x: (1. / (1. + np.exp(-slope * (x - threshold))) - baseline) / (peak-baseline)
    return scaled_sigmoid


def get_BTSP_function(Wmax, k_pot, k_dep, sig_pot, sig_dep):
    '''
    Returns a callable function that computes the BTSP weight update as a function of the current synaptic weight and
    the overlap between eligibility traces and instructive signals.
    The BTSP function is parameterized by the maximum synaptic weight and sigmoid functions for potentiation and
    depression (scaled by potentiation and depression coefficients).
    '''
    dW = lambda ETxIS, W: (Wmax - W) * k_pot * sig_pot(ETxIS) - W * k_dep * sig_dep(ETxIS)
    return dW


def get_simple_BTSP_function():
    dW = lambda ETxIS, W: ETxIS
    return dW