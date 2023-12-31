import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import os
import scipy
from tqdm import tqdm
import pickle

from utils import update_plot_defaults, get_scaled_sigmoid, get_BTSP_function



########################################################################################################################
# Figure 5: Reinforcement learning
########################################################################################################################


def generate_Figure5(show=True, save=False):
    mm = 1 / 25.4  # millimeters in inches
    fig = plt.figure(figsize=(183*mm, 190*mm))
    
    # BTSP parameters from (Milstein et al., 2021, eLife) Fig.7
    sig_pot = get_scaled_sigmoid(slope=4.405, threshold=0.415)
    sig_dep = get_scaled_sigmoid(slope=20.0, threshold=0.026)
    k_dep = 0.425
    k_pot = 1.1097
    Wmax = 4.68
    btsp_func = get_BTSP_function(Wmax, k_pot, k_dep, sig_pot, sig_dep)

    ##################################################
    ## 1. Linear track simulation
    ##################################################
    axes = gs.GridSpec(nrows=1, ncols=3, left=0.01, right=0.6, top=0.89, bottom=0.72, wspace=0.03)
    M_dict, size = simulate_linear_track_SR(btsp_func, overwrite=False)

    # Plot learned successor weights
    ax1 = fig.add_subplot(axes[0])
    ax2 = fig.add_subplot(axes[1])
    ax3 = fig.add_subplot(axes[2])
    plot_linear_track_SR(M_dict, fig, ax1, ax2, ax3)

    # Plot quantification of SR correlation
    axes = gs.GridSpec(nrows=1, ncols=1, left=0.73, right=0.98, top=0.91, bottom = 0.74)
    ax = fig.add_subplot(axes[0])
    plot_SR_correlation(M_dict, size, ax)
 

    ##################################################
    ## 2. Gridworld simulation
    ##################################################
    grid_top = 0.64
    grid_bottom = 0.05
    axes = gs.GridSpec(nrows=3, ncols=3, left=0.01, right=0.59, top=grid_top, bottom=grid_bottom, wspace=0.06, hspace=0.5)

    # 5x6 environment with simple wall
    filename = 'Figure4_grid_simulations_56_simple_seed.pkl'
    common_seed = True
    environment = GridWorld((5, 6), walls=(np.array([1, 2, 3]), 2), rewards=(1, 5), initial_state=(3, 1))

    random_seeds = [42, 12, 4321 , 674, 974, 295, 2763, 809, 2349, 7862]
    # random_seeds = random_seeds[:3]
    grid_simulations_data = simulate_gridworld_multiple_seeds(random_seeds, filename, common_seed, btsp_func, environment, overwrite=False)


    # Plot example trajectories
    example_seed = 42
    ax1 = plt.subplot(axes[0, 0])
    ax2 = plt.subplot(axes[1, 0])
    ax3 = plt.subplot(axes[2, 0])
    trajectories = grid_simulations_data[example_seed]['trajectories']
    plot_grid(ax1, environment, trajectories, learning_rule='TD', legend=True)
    plot_grid(ax2, environment, trajectories, learning_rule='Hebb')
    plot_grid(ax3, environment, trajectories, learning_rule='BTSP')


    # Plot example SR place fields (beginnning of learning)
    M_history = grid_simulations_data[example_seed]['SR_weight_histories']
    cell_nr = 11
    end_idx = -1 # end of the last trial
    for i, rule in enumerate(['TD', 'Hebb', 'BTSP']):
        ax1 = plt.subplot(axes[i, 1])
        ax2 = plt.subplot(axes[i, 2])
        start_idx = len(trajectories[rule][0]) # index at the end of trial 1
        plot_SR_place_fields(fig, ax1, ax2, cell_nr, M_history, start_idx, end_idx, environment.grid_size, learning_rule=rule)


    # Plot navigation summary quantifications
    axes = gs.GridSpec(nrows=3, ncols=2, left=0.73, right=0.985, top=grid_top, bottom=grid_bottom, hspace=0.3, wspace=0.8)
    trajectory_lengths = extract_trajectory_lengths(grid_simulations_data)

    ax = plt.subplot(axes[0, 0:2])
    plot_pathlength_over_trials(ax, trajectory_lengths)

    ax1 = plt.subplot(axes[1, 0])
    ax2 = plt.subplot(axes[1, 1])
    plot_navigation_summary(ax1, ax2, trajectory_lengths)
    
    ax = plt.subplot(axes[2, 0:2])
    # plot_place_field_size(ax, M_history_all, trial_indexes, environment.grid_size)
    plot_cumulative_reward(ax, trajectory_lengths)

    if show:
        plt.show()

    if save:
        fig.savefig('figures/Fig5-reinforcement-learning/Figure5_plots.png',dpi=300)
        fig.savefig('figures/Fig5-reinforcement-learning/Figure5_plots.svg',dpi=300)



def simulate_linear_track_SR(btsp_func, overwrite=False):
    linear_track_size = 20

    if os.path.exists('sim_data/Figure4_linear_track_SR_matrices.pkl') and not overwrite:
        with open('sim_data/Figure4_linear_track_SR_matrices.pkl', 'rb') as f:
            M_dict = pickle.load(f)
    else:
        M_dict = {}

        environment = GridWorld(grid_size=(1, linear_track_size), wraparound=True)
        
        dwell_time = 400
        learning_rate = 0.012
        
        agent = Agent(environment.num_states, ET_temp=74.34, IS_temp=70.82, learning_rule=btsp_func, policy=deterministic_policy)
        RL_run_loop(agent, environment, learning_rule='BTSP', num_episodes=1, dwell_time=dwell_time, lr=learning_rate, max_steps=linear_track_size*linear_track_size)
        M_dict['BTSP'] = agent.M_history

        agent = Agent(environment.num_states, ET_temp=74.34, IS_temp=70.82, learning_rule=btsp_func, policy=deterministic_policy)
        RL_run_loop(agent, environment, learning_rule='TD', num_episodes=1, dwell_time=dwell_time, lr=learning_rate, max_steps=linear_track_size*linear_track_size)
        M_dict['TD'] = agent.M_TD_history

        agent = Agent(environment.num_states, ET_temp=74.34, IS_temp=70.82, learning_rule=btsp_func, policy=deterministic_policy)
        RL_run_loop(agent, environment, learning_rule='Hebb', num_episodes=1, dwell_time=dwell_time, lr=learning_rate, max_steps=linear_track_size*linear_track_size)
        M_dict['Hebb'] = agent.M_hebb_history

        with open('sim_data/Figure4_linear_track_SR_matrices.pkl', 'wb') as f:
            pickle.dump(M_dict, f)

    return M_dict, linear_track_size

def simulate_gridworld_multiple_seeds(random_seeds, filename, common_seed, btsp_func, environment, overwrite=False):
    num_trials = 30
    max_steps = 60

    if os.path.exists(f'sim_data/{filename}') and not overwrite:
        with open(f'sim_data/{filename}', 'rb') as f:
            grid_simulations_data = pickle.load(f)
    else:
        grid_simulations_data = {}

    for seed in random_seeds:
        if seed not in grid_simulations_data:
            print('Running simulation for seed = ', seed)
            grid_simulations_data[seed] = simulate_gridworld(environment, btsp_func, seed, num_trials, max_steps, common_seed)

            with open(f'sim_data/{filename}', 'wb') as f:
                pickle.dump(grid_simulations_data, f)

    return grid_simulations_data

def simulate_gridworld(environment, btsp_func, seed, num_episodes, max_steps, common_seed=True):
    trajectory_lengths = {}
    trajectories = {}
    SR_weight_histories = {}

    np.random.seed(seed)

    learning_rules = ['TD', 'Hebb', 'BTSP']
    for rule in learning_rules:
        if common_seed:
            np.random.seed(seed)
        agent = Agent(environment.num_states, ET_temp=74.2, IS_temp=71.25,
                      learning_rule=btsp_func, policy=epsilon_greedy_policy, policy_successor=rule, epsilon=0.2)
        
        RL_run_loop(agent, environment, rule, num_episodes, dwell_time=400, lr=0.012, max_steps=max_steps,
                    initial_state=environment.initial_state)

        trajectory_lengths[rule] = [len(trajectory) for trajectory in agent.trajectories]
        trajectories[rule] = agent.trajectories
        if rule == 'BTSP':
            SR_weight_histories[rule] = agent.M_history
        elif rule == 'TD':
            SR_weight_histories[rule] = agent.M_TD_history
        elif rule == 'Hebb':
            SR_weight_histories[rule] = agent.M_hebb_history

    return {'trajectory_lengths': trajectory_lengths, 'trajectories': trajectories,
            'SR_weight_histories': SR_weight_histories}

def RL_run_loop(agent, environment, learning_rule='BTSP', num_episodes=1, max_steps=None, initial_state=(0,0), random_start=False, dwell_time=500, lr=1., field_size=1., field_type='binary'):
    """Run agent in environment for given number of episodes.
    
    :param agent: Agent object
    :param environment: Environment object
    :param num_episodes: Number of episodes to run
    :param max_steps: Maximum number of steps per episode
    :param initial_state: Initial state of the agent
    :param random_start: Whether to randomly sample initial state
    :param dwell_time: Dwell time in each state (ms)
    :param lr: Learning rate
    :param field_size: Size of place fields (in grid cell units)
    :param field_type: Type of place fields ('gaussian' or 'binary')
    """

    environment.initial_state = initial_state

    if max_steps is None:
        max_steps = environment.grid_size[1]

    dt = 1 # (ms)
    time_extension = 10_000 # (ms)
    place_fields_2D, agent.flat_place_fields = generate_place_fields(environment, field_size, field_type)

    for episode in tqdm(range(num_episodes)):
        if random_start:
            environment.initial_state = environment.get_random_state()
                     
        agent.reset(environment.initial_state)        
        state_counter = 0
        
        while state_counter < max_steps:
            state_nr = np.ravel_multi_index(agent.state, environment.grid_size)
            agent.plateau_timer[state_nr] += 1
            input_activities = agent.flat_place_fields[state_nr]
            agent.input_activity = input_activities
            agent.post_pop_activity = agent.M_hebb.T @ agent.flat_place_fields[state_nr]
            agent.post_pop_activity = rectified_sigmoid(agent.post_pop_activity)
            agent.post_pop_activity[state_nr] += 1 # Nudge post-pop activity of current state to induce Hebbian plasticity

            if learning_rule == 'BTSP':
                for t in np.arange(0, dwell_time, dt):
                    # Update ET and IS for BTSP and short-timescale plasticity rules
                    for i, ET_VO2 in enumerate(agent.ET):
                        if input_activities[i] > 0:
                            ET_VO2.controlI = 100 * input_activities[i]

                    agent.plateau_timer[agent.plateau_timer > agent.plateau_dur/dt] = 0 # Reset plateau timer for states that have reached plateau duration
                    agent.plateau_timer[agent.plateau_timer > 0] += 1 # Increment plateau timer for active states
                    for i, IS_VO2 in enumerate(agent.IS):
                        if agent.plateau_timer[i] > 0:
                            IS_VO2.controlI = 100

                    for (ET_VO2, IS_VO2) in zip(agent.ET, agent.IS):
                        ET_VO2.time_step()
                        IS_VO2.time_step()
                        ET_VO2.controlI = 0
                        IS_VO2.controlI = 0

                    # Update weights every 10 ms
                    if t%10 == 0:
                        agent.compute_BTSP_update()

            agent.update_M_BTSP(lr) # Buffer the weight updates and apply them at the end of the episode
            agent.reset_dW()
            
            if state_counter == max_steps-1: # add time to the end of episode to let traces decay
                agent.reset_dW()
                for t in np.arange(0, time_extension, dt):
                    for (ET_VO2, IS_VO2) in zip(agent.ET, agent.IS):
                        ET_VO2.time_step()
                        IS_VO2.time_step()

                    # Update weights every 10 ms
                    if t%10 == 0:
                        agent.compute_BTSP_update()

                    if t%dwell_time == 0:
                        agent.update_M_BTSP(lr) # Buffer the weight updates and apply them at the end of the episode
                        agent.reset_dW()

            agent.state_history.append(agent.state)
            agent.action = agent.select_action(environment.possible_actions(agent.state), environment.get_neighboring_states(agent.state))            
            next_state = environment.next_state(agent.state, agent.action)
            next_state_nr = np.ravel_multi_index(next_state, environment.grid_size)

            if learning_rule == 'BTSP':
                agent.M_history.append(agent.M.copy())
            elif learning_rule == 'TD':
                agent.update_M_TD(state_nr, next_state_nr)
                agent.M_TD_history.append(agent.M_TD.copy())
            elif learning_rule == 'Hebb':
                agent.update_M_hebb(lr=0.04)
                agent.M_hebb_history.append(agent.M_hebb.copy())
            
            agent.input_activity_history.append(agent.input_activity.copy())
            agent.post_pop_activity_history.append(agent.post_pop_activity.copy())

            if environment.reward(agent.state) > 0:
                agent.update_R(state_nr, environment.reward(agent.state))
                agent.update_V()
                break

            agent.state = next_state
            state_counter += 1
            
        agent.trajectories.append(agent.state_history)

    agent.ET_history = np.array(agent.ET_history)
    agent.IS_history = np.array(agent.IS_history)
    agent.post_pop_activity_history = np.array(agent.post_pop_activity_history)
    agent.input_activity_history = np.array(agent.input_activity_history)



def gaussian_place_field(grid_size, center, sigma):
    x = np.linspace(0, grid_size[0] - 1, grid_size[0])
    y = np.linspace(0, grid_size[1] - 1, grid_size[1])
    xv, yv = np.meshgrid(y, x)  # Create grid of x and y values
    return np.exp(-((yv - center[0]) ** 2 + (xv - center[1]) ** 2) / (2 * sigma ** 2))

def binary_place_field(grid_size, center, radius):
    x = np.linspace(0, grid_size[0] - 1, grid_size[0])
    y = np.linspace(0, grid_size[1] - 1, grid_size[1])
    xv, yv = np.meshgrid(y, x)  # Create grid of x and y values
    distance_from_center = np.sqrt((yv - center[0]) ** 2 + (xv - center[1]) ** 2)
    return (distance_from_center <= radius).astype(int)

def generate_1D_place_fields(track_size, field_size=2., field_type='gaussian'):
    ''' Generate place fields for each state in the environment.

    :param environment: Environment object
    :param field_size: Size of place fields (in grid cell units)
    :return: List of place fields (2D numpy arrays) and flattened place fields (1D numpy arrays)
    '''

    if field_type == 'gaussian':
        field_size = field_size / (2 * np.sqrt(2 * np.log(2)))  # Convert from FWHM to SD
        generate_field = gaussian_place_field
    elif field_type == 'binary':
        generate_field = binary_place_field
        field_size -= 1
    else:
        raise ValueError('Invalid field type')

    place_fields = []
    for location in range(track_size):
        place_field = generate_field([1, track_size], [0, location], field_size)
        place_fields.append(place_field.flatten())
    place_fields = np.array(place_fields)

    return place_fields.T

def generate_place_fields(environment, field_size=0., field_type='binary'):
    ''' Generate place fields for each state in the environment.

    :param environment: Environment object
    :param field_size: Size of place fields (in grid cell units)
    :return: List of place fields (2D numpy arrays) and flattened place fields (1D numpy arrays)
    '''

    place_fields = []
    flat_place_fields = []

    if field_type == 'gaussian':
        field_size = field_size / (2 * np.sqrt(2 * np.log(2)))  # Convert from FWHM to SD
        generate_field = gaussian_place_field
    elif field_type == 'binary':
        generate_field = binary_place_field
        field_size -= 1
    else:
        raise ValueError('Invalid field type')

    for state_nr in range(environment.num_states):
        state = environment.get_state(state_nr)
        place_field = generate_field(environment.grid_size, state, field_size)
        place_fields.append(place_field)
        flat_place_fields.append(place_field.flatten())
    place_fields = np.array(place_fields)
    flat_place_fields = np.array(flat_place_fields)

    return place_fields, flat_place_fields.T

def rectified_sigmoid(x):
    '''
    Rectified sigmoid function, crossing origin and asymptoting to 1 for x>0
    '''
    sigm = 2/(1+np.exp(-x)) -1
    sigm = np.maximum(sigm, 0)
    return sigm

class GridWorld():
    def __init__(self, grid_size, walls=None, rewards=None, wraparound=False, initial_state=None):
        """Build a grid environment with given size."""

        self.grid_size = grid_size
        self.grid_rows = grid_size[0]
        self.grid_cols = grid_size[1]
        self.num_states = self.grid_rows * self.grid_cols
        self.wraparound = wraparound
        self.initial_state = initial_state

        # Build the grid with walls
        self.grid = -1 * np.ones([self.grid_rows + 2, self.grid_cols + 2])
        self.grid[1:-1, 1:-1] = 0
        if type(walls) == list:
            for wall in walls:
                rows = np.array(wall[0])+1
                cols = np.array(wall[1])+1
                self.grid[rows, cols] = -1
        elif type(walls) == tuple:
            self.grid[walls[0] + 1, walls[1] + 1] = -1

        # Add rewards
        self.rewards = rewards
        if type(rewards) == list:
            for reward_loc in rewards:
                self.grid[reward_loc[0] + 1, reward_loc[1] + 1] = 1
        elif type(rewards) == tuple:
            self.grid[rewards[0] + 1, rewards[1] + 1] = 1

    def possible_actions(self, state):
        state_row = state[0] + 1
        state_col = state[1] + 1
        possible_actions = ['up', 'right', 'down', 'left']

        if self.grid[state_row, state_col + 1] == -1:
            possible_actions.remove('right')
        if self.grid[state_row, state_col - 1] == -1:
            possible_actions.remove('left')
        if self.grid[state_row + 1, state_col] == -1:
            possible_actions.remove('down')
        if self.grid[state_row - 1, state_col] == -1:
            possible_actions.remove('up')

        if self.wraparound:
            if state_row == 1:
                possible_actions.append('up')
            if state_row == self.grid_rows:
                possible_actions.append('down')
            if state_col == 1:
                possible_actions.append('left')
            if state_col == self.grid_cols:
                possible_actions.append('right')

        return possible_actions

    def next_state(self, state, action):
        state_row = state[0]
        state_col = state[1]

        if action == 'up':
            next_state = (state_row - 1, state_col)
        elif action == 'right':
            next_state = (state_row, state_col + 1)
        elif action == 'down':
            next_state = (state_row + 1, state_col)
        elif action == 'left':
            next_state = (state_row, state_col - 1)
        elif action is None:
            next_state = state
        else:
            raise ValueError('Invalid action')

        if self.wraparound:
            if next_state[0] < 0:
                next_state = (self.grid_rows - 1, next_state[1])
            elif next_state[0] >= self.grid_rows:
                next_state = (0, next_state[1])
            elif next_state[1] < 0:
                next_state = (next_state[0], self.grid_cols - 1)
            elif next_state[1] >= self.grid_cols:
                next_state = (next_state[0], 0)

        # if action is invalid
        if self.grid[next_state[0] + 1, next_state[1] + 1] == -1:
            raise ValueError('Invalid action')

        return next_state

    def get_state(self, state_nr):
        '''Get 2D state coordinates from state number.'''
        return (state_nr // self.grid_cols, state_nr % self.grid_cols)

    def get_neighboring_states(self, state):
        """Return neighboring states."""
        neighbors = {}
        for action in self.possible_actions(state):
            next_state = self.next_state(state, action)
            next_state_nr = np.ravel_multi_index(next_state, self.grid_size)
            neighbors[action] = (next_state, next_state_nr)

        return neighbors

    def get_random_state(self):
        """Return a random state."""
        possible_states = np.where(self.grid != -1)
        random_state_idx = np.random.randint(len(possible_states[0]))
        random_state = (possible_states[0][random_state_idx] - 1, possible_states[1][random_state_idx] - 1)
        return random_state

    def reward(self, state):
        state = (state[0] + 1, state[1] + 1)
        return self.grid[state]

    def plot_grid(self, agent=None, plot_trajectories=False):
        """Plot grid world."""

        plt.figure(figsize=(10, 10))
        # plt.imshow(self.grid >= 0, interpolation='nearest', cmap='Pastel1')
        plt.imshow(self.grid >= 0, interpolation='nearest', cmap='gray', vmin=-3, vmax=1)

        ax = plt.gca()
        if hasattr(self, 'initial_state'):
            plt.text(self.initial_state[1] + 1, self.initial_state[0] + 1.08,
                     r'$\mathbf{S}$', fontsize=16, ha='center', va='center')

        if self.rewards is not None:
            if type(self.rewards) == list:
                for reward_loc in self.rewards:
                    plt.text(reward_loc[1] + 1, reward_loc[0] + 1.08,
                             r'$\mathbf{R}$', fontsize=16, ha='center', va='center')
            elif type(self.rewards) == tuple:
                plt.text(self.rewards[1] + 1, self.rewards[0] + 1.08,
                         r'$\mathbf{R}$', fontsize=16, ha='center', va='center')

        for x in range(self.grid_cols + 1):
            plt.plot([x + 0.5, x + 0.5], [0.5, self.grid_rows + 0.5], '-k', lw=1)
        for y in range(self.grid_rows + 1):
            plt.plot([0.5, self.grid_cols + 0.5], [y + 0.5, y + 0.5], '-k', lw=1)
        ax.set_axis_off()

        if agent is not None:
            if plot_trajectories:
                for i, trajectory in enumerate(agent.trajectories):
                    plt.plot([state[1] + 1 + np.random.uniform(-0.2, 0.2) for state in trajectory],
                             [state[0] + 1 + np.random.uniform(-0.2, 0.2) for state in trajectory], 'o-', c='gray',
                             alpha=0.15, lw=2, ms=8)
                if len(agent.trajectories) > 2:
                    # Plot first two trajectories in red and blue
                    plt.plot([state[1] + 1 + 0.05 for state in agent.trajectories[0]],
                             [state[0] + 1 + 0.05 for state in agent.trajectories[0]], 'o-', c=[1, .3, .3], lw=2, ms=8)
                    plt.plot([state[1] + 1 - 0.05 for state in agent.trajectories[1]],
                             [state[0] + 1 - 0.05 for state in agent.trajectories[1]], 'o-', c=[.3, .5, 1], lw=2, ms=8)
            else:  # Plot agent position
                plt.plot(agent.state[0] + 1, agent.state[1] + 1, 'o', c='C0', ms=10)

        ax.set_xlim(0.1, self.grid_cols + 0.9)
        ax.set_ylim(0.1, self.grid_rows + 0.9)

        plt.show()

class Agent():
    def __init__(self, num_states, ET_temp, IS_temp, learning_rule, policy, policy_successor='BTSP', epsilon=None):
        
        self.M = np.ones((num_states, num_states)) # Successor matrix
        self.M_TD = np.zeros((num_states, num_states)) # Successor matrix for TD
        self.M_hebb = np.zeros((num_states, num_states)) # Successor matrix for Hebbian/STDP-like short timescale rule
        self.learning_rule = learning_rule
        self.dW = np.zeros_like(self.M)

        self.R = np.zeros(num_states) # Reward vector
        self.V = np.zeros(num_states) # V values
        self.action = None
        self.policy = policy # Policy function to use
        self.policy_successor = policy_successor # Which successor matrix to consult for policy
        self.epsilon = epsilon

        self.ET_temp = ET_temp # Target tau = 1.66s
        self.IS_temp = IS_temp # Target tau = 0.44s
        self.ET = [Volatile_Resistor(dt=1, temperature=self.ET_temp, metalR=100) for i in range(num_states)] # Eligibility traces
        self.IS = [Volatile_Resistor(dt=1, temperature=self.IS_temp, metalR=100) for i in range(num_states)] # Instructive signals
        self.plateau_dur = 300 # (ms), Duration of plateau potential (width of IS rise)
        self.plateau_timer = np.zeros(num_states) # For tracking which units are in plateau

        self.post_pop_activity = np.zeros(num_states)
        self.input_activity = np.zeros(num_states)

        self.state_history = []
        self.trajectories = []
        self.ET_history = []
        self.IS_history = []
        self.input_activity_history = [self.input_activity.copy()]
        self.post_pop_activity_history = [self.post_pop_activity.copy()]
        self.M_history = [self.M.copy()]
        self.M_TD_history = [self.M_TD.copy()]
        self.M_hebb_history = [self.M_hebb.copy()]
    
    def select_action(self, possible_actions, neighbours):
        prev_action = self.action
        values = {}
        
        # Compute value of each action
        for action in possible_actions:
            state_nr = neighbours[action][1]
            values[action] = self.V[state_nr]

        # If policy is epsilon-greedy, pass values to policy
        if self.policy == epsilon_greedy_policy and hasattr(self, 'epsilon'):
            action = self.policy(prev_action, possible_actions, values, self.epsilon)
        else:
            action = self.policy(prev_action, possible_actions, values)

        return action

    def reset(self, initial_state):
        self.state = initial_state
        self.action = None
        self.state_history = []
        for ET, IS in zip(self.ET, self.IS):
            ET.controlI = 0
            IS.controlI = 0
            ET.R = ET.insulatorR
            IS.R = IS.insulatorR

    def reset_history(self):
        self.post_pop_activity_history = []
        self.trajectories = []
        self.ET_history = []
        self.IS_history = []

    def reset_dW(self):
        self.dW = np.zeros_like(self.M)
        
    def compute_BTSP_update(self):
        # Convert from VO2_R to norm. eligibility signals
        g_ET_baseline = 1/self.ET[0].insulatorR
        g_ET_peak = 1/self.ET[0].metalR *0.4
        ET = np.array([[1/VO2.R for VO2 in self.ET]])
        ET = (ET - g_ET_baseline)/(g_ET_peak-g_ET_baseline)
        
        g_IS_baseline = 1/self.IS[0].insulatorR
        g_IS_peak = 1/self.IS[0].metalR *0.8
        IS = np.array([[1/VO2.R for VO2 in self.IS]])
        IS = (IS - g_IS_baseline)/(g_IS_peak-g_IS_baseline)
        
        self.ET_history.append(ET[0]) # 0 index removes empty dimension
        self.IS_history.append(IS[0])

        # Compute dW matrix
        # self.dW += self.learning_rule(ET.T@IS, self.M_init)
        self.dW += self.learning_rule(ET.T@IS, self.M)

    def update_M_BTSP(self, lr):
        # Update successor matrix
        self.M += lr * self.dW
        self.M = np.maximum(self.M, 0) # Clip negative values to 0

    def update_M_TD(self, state_nr, next_state_nr, lr=1., gamma=0.75):
        """
        Computes the delta for learning the Successor matrix M using the TD learning algorithm.
        """
        I_state = np.eye(self.M_TD.shape[0])[state_nr]
        delta_TD = I_state + gamma*self.M_TD[next_state_nr] - self.M_TD[state_nr]
        self.M_TD[state_nr] += lr * delta_TD

    def update_M_hebb(self, lr):
        presyn = self.input_activity
        presyn_prev = self.input_activity_history[-1]
        postsyn = self.post_pop_activity
        postsyn_prev = self.post_pop_activity_history[-1]

        k_pot = 0.5
        k_dep = 0.5
        dW = np.outer(presyn, postsyn) + k_pot*np.outer(presyn_prev, postsyn) - k_dep*np.outer(presyn, postsyn_prev) 
        
        self.M_hebb += lr * dW
        self.M_hebb = np.maximum(self.M_hebb, 0) # Clip negative values to 0

    def update_R(self, state_nr, reward, lr=1.):
        self.R[state_nr] += lr * (reward - self.R[state_nr])

    def update_V(self):
        if self.policy_successor == 'BTSP':
            self.V = np.dot(self.M, self.R)
        elif self.policy_successor == 'TD':
            self.V = np.dot(self.M_TD, self.R)
        elif self.policy_successor == 'short_timescale':
            self.V = np.dot(self.M_hebb, self.R)

    def plot_eligibility_traces(self):
        dt = 0.001
        time = np.arange(0,agent.ET_history.shape[0]) * dt

        fig, ax = plt.subplots(1,2, figsize=(12,4))
        ax[0].plot(time, agent.ET_history)
        ax[0].set_ylim(-0.1,1)
        ax[0].set_title('Eligibility traces')

        ax[1].plot(time, agent.IS_history)
        ax[1].set_ylim(-0.1,1)
        ax[1].set_title('Instructive signals')
        plt.show()

    def plot_successor_matrix(self, rule='BTSP'):
        if hasattr(self, 'flat_place_fields'):
            fig, ax = plt.subplots(1,2, figsize=(12,4))
            im = ax[0].imshow(self.flat_place_fields)
            cax = fig.add_axes([ax[0].get_position().x1 + 0.01, ax[0].get_position().y0, 0.012, ax[0].get_position().height])
            cbar = plt.colorbar(im, ax=ax[0], cax=cax)
            ax[0].set_title('Input place fields')
            ax[0].set_ylabel('Cell nr.')
            ax[0].set_xlabel('State')

            if rule == 'BTSP':
                im = ax[1].imshow(self.M, interpolation='nearest')
            elif rule == 'TD':
                im = ax[1].imshow(self.M_TD, interpolation='nearest')
            elif rule == 'short_timescale':
                im = ax[1].imshow(self.M_hebb, interpolation='nearest')
            cax = fig.add_axes([ax[1].get_position().x1 + 0.01, ax[1].get_position().y0, 0.012, ax[1].get_position().height])
            cbar = plt.colorbar(im, ax=ax[1], cax=cax)
            ax[1].set_title('Successor matrix')
            ax[1].set_xlabel('State')
            ax[1].set_ylabel('State')
            plt.show()

        else:
            plt.figure(figsize=(5,5))
            if rule == 'BTSP':
                im = plt.imshow(self.M, interpolation='nearest')
            elif rule == 'TD':
                im = plt.imshow(self.M_TD, interpolation='nearest')
            elif rule == 'short_timescale':
                im = plt.imshow(self.M_hebb, interpolation='nearest')
            plt.imshow(self.M, interpolation='nearest')
            plt.colorbar()
            plt.title('Successor matrix')
            plt.xlabel('State')
            plt.ylabel('State')
            plt.show()

    def plot_place_field(self, cell_id, grid_size):
        place_field = self.M[:,cell_id].reshape(grid_size)
        plt.imshow(place_field)
        plt.title('Output place field')
        plt.colorbar()
        plt.show()

    def plot_value_function(self, grid_size):
        plt.imshow(self.V.reshape(grid_size))
        plt.colorbar()
        plt.title('Value function')
        plt.show()

def random_policy(prev_action, possible_actions, momentum=True, *args):
    """Return random integer between [0, len(num_actions))."""

    if momentum and len(possible_actions) > 1 and prev_action is not None:  # Don't go back to previous state
        opposite_action = {'up': 'down', 'right': 'left', 'down': 'up', 'left': 'right'}
        possible_actions.remove(opposite_action[prev_action])
    return np.random.choice(possible_actions)

def deterministic_policy(prev_action, possible_actions, momentum=True, *args):
    """Always run from left to right."""
    if 'right' in possible_actions:
        return 'right'

def epsilon_greedy_policy(prev_action, possible_actions, values, epsilon, momentum=True):
    """Return action with highest Q value with probability 1-epsilon, else return random action."""

    values = {key: value for key, value in values.items() if key in possible_actions}

    if momentum and len(possible_actions) > 1 and prev_action is not None:  # Don't go back to previous state
        opposite_action = {'up': 'down', 'right': 'left', 'down': 'up', 'left': 'right'}
        possible_actions.remove(opposite_action[prev_action])
        values.pop(opposite_action[prev_action])

    if np.random.rand() < epsilon:
        # print('random action!')
        return np.random.choice(possible_actions)
    else:
        max_value = max(values.values())
        max_keys = [key for key, value in values.items() if value == max_value]
        return np.random.choice(max_keys)

def softmax_policy(prev_action, possible_actions, values, momentum=True):
    """Select action according to softmax probability distribution"""
    if momentum and len(possible_actions) > 1 and prev_action is not None:  # Don't go back to previous state
        opposite_action = {'up': 'down', 'right': 'left', 'down': 'up', 'left': 'right'}
        possible_actions.remove(opposite_action[prev_action])
        values.pop(opposite_action[prev_action])

    Qs = np.array([values[action] for action in possible_actions])
    Qs = np.exp(20 * Qs) / np.sum(np.exp(20 * Qs))
    action = np.random.choice(possible_actions, p=Qs)
    # if action != max(values, key=values.get):
    #     print('random action!')
    return action


def extract_trajectory_lengths(grid_simulations_data):
    trajectory_lengths = {}
    for seed in grid_simulations_data:
        for rule in grid_simulations_data[seed]['trajectory_lengths']:
            if rule not in trajectory_lengths:
                trajectory_lengths[rule] = {}
            trajectory_lengths[rule][seed] = grid_simulations_data[seed]['trajectory_lengths'][rule]
    return trajectory_lengths

def extract_SR_weight_history(grid_simulations_data):
    M_history = {}
    for seed in grid_simulations_data:
        for rule in grid_simulations_data[seed]['SR_weight_histories']:
            if rule not in M_history:
                M_history[rule] = {}
            M_history[rule][seed] = grid_simulations_data[seed]['SR_weight_histories'][rule]
    return M_history

def plot_linear_track_SR(M_dict, fig, ax1, ax2, ax3):
    ax = ax1
    delta_M_TD = M_dict['TD'][-1] - M_dict['TD'][0]
    im = ax.imshow(delta_M_TD, interpolation='nearest', aspect='equal', vmin=0, vmax=1)
    ax.set_title('TD')
    ax.axis('off')

    ax = ax2
    delta_M_hebb = M_dict['Hebb'][-1] - M_dict['Hebb'][0]
    im = ax.imshow(delta_M_hebb, interpolation='nearest', aspect='equal', vmin=0, vmax=1)
    ax.set_title('Hebbian')
    ax.axis('off')

    ax = ax3
    delta_M_BTSP = M_dict['BTSP'][-1] - M_dict['BTSP'][0]
    im = ax.imshow(delta_M_BTSP, interpolation='nearest', aspect='equal', vmin=0, vmax=1)
    ax.set_title('BTSP')
    ax.axis('off')
    
    cax = fig.add_axes([ax.get_position().x1+0.01, ax.get_position().y0, 0.01, ax.get_position().height])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('$\Delta$ Weight', rotation=270, labelpad=8, fontsize=8)
    cbar.ax.tick_params(labelsize=8)

def plot_SR_correlation(M_dict, size, ax):
    correlations = {'BTSP':[], 'TD':[], 'Hebb':[]}

    M_history = M_dict['BTSP']
    M_TD_history = M_dict['TD']
    M_hebb_history = M_dict['Hebb']
    final_SR = M_TD_history[-1]
    for M_BTSP,M_TD,M_hebb in zip(M_history, M_TD_history, M_hebb_history):
        # Compute the correlation coefficient
        correlations['TD'].append(np.corrcoef(M_TD.flatten(), final_SR.flatten())[0,1]**2)
        correlations['Hebb'].append(np.corrcoef(M_hebb.flatten(), final_SR.flatten())[0,1]**2)
        correlations['BTSP'].append(np.corrcoef(M_BTSP.flatten(), final_SR.flatten())[0,1]**2)

    ax.plot(correlations['TD'], label='TD', color='k', linewidth=1)
    ax.plot(correlations['Hebb'], label='Hebbian', color='gray', linewidth=1)
    ax.plot(correlations['BTSP'], label='BTSP', color='r', linewidth=1)
    ax.plot([0, len(correlations['BTSP'])], [1, 1], '--', color='k', linewidth=1)
    ax.plot([size,size], [0,0.85], '--', color='r', alpha=0.5, linewidth=1.5)
    ax.set_xticks(np.arange(0, len(correlations['BTSP']), size), np.arange(size+1))
    ax.set_xlim([0,size*10])
    ax.set_xlabel('Number of trials')
    ax.set_ylabel('$R^2$', labelpad=-2)
    ax.set_ylim([0,1.01])
    ax.legend(loc='center right', bbox_to_anchor=(1.05, 0.3), handlelength=1, fontsize=8, frameon=False, handletextpad=0.4)


def compute_tau(temp, current, duration, dt=1, plot=False):
    duration = int(duration/dt)
    sample_ET = Volatile_Resistor(dt, temperature=temp, metalR=100)
    sample_ET.controlI = current
    for i in range(duration):
        sample_ET.time_step()
    sample_ET.controlI = 0
    for t in range(10_000):
        sample_ET.time_step()

    g = np.array(sample_ET.g_history)
    g_max = 1/sample_ET.metalR
    g_min = 1/sample_ET.insulatorR
    g_norm = (g-g_min)/(g_max-g_min)

    if plot:
        time = np.arange(0, len(sample_ET.g_history))/1000
        plt.plot(time, g_norm)
        plt.show()

    ET = g_norm[duration:]
    time = np.arange(0, len(ET))/1000
    tau = fit_single_exponential(time,ET, verbose=False)
    tau = tau*dt *1000    
    print(f"tau = {tau} ms")

    return tau

def plot_btsp_learning_rule(ax1, ax2, dwell_time, plateau_dur, btsp_func, ET_rise_tau, IS_rise_tau, ET_decay_tau, IS_decay_tau):
    # Set parameters
    dt = 0.001  # s
    tmax = 10.  # s
    Wmax = 5.
    w_init = np.arange(0, Wmax, 0.05)
    ET_start = 0  # s

    # Generate ET template trace
    t1 = np.arange(0, dwell_time, dt)
    ET_rise = 1 - np.exp(-t1 / ET_rise_tau)
    t2 = np.arange(0, tmax - dwell_time, dt)
    ET_decay = ET_rise[-1] * np.exp(-t2 / ET_decay_tau)
    ET = np.concatenate((ET_rise, ET_decay))
    ET = np.roll(ET, int((tmax / 2 + ET_start) / dt))  # start ET in the center of the trace
    # ET = np.roll(ET, int((tmax/2)/dt)) # start ET in the center of the trace

    # Generate IS template trace
    t1 = np.arange(0, plateau_dur, dt)
    IS_rise = 1 - np.exp(-t1 / IS_rise_tau)
    t2 = np.arange(0, tmax - plateau_dur, dt)
    IS_decay = IS_rise[-1] * np.exp(-t2 / IS_decay_tau)
    IS = np.concatenate((IS_rise, IS_decay))
    IS = np.roll(IS, int((tmax / 2) / dt))  # start plateau in the center of the trace
    IS[0:int((tmax / 2) / dt)] = 0  # zero values before the start of plateau

    # Compute dW matrix
    all_delta_t = np.arange(-tmax / 2, tmax / 2, 0.05)
    dW = np.zeros((len(w_init), len(all_delta_t)))
    w_init = np.expand_dims(w_init, axis=1)

    for i, delta_t in enumerate(all_delta_t):
        shifted_ET = np.roll(ET, int(delta_t / dt))
        shifted_ET[0:int((delta_t + ET_start + tmax / 2) / dt)] = 0  # zero values before the start of ET rise
        dW_all_timepoints = btsp_func(shifted_ET * IS, w_init)
        dW[:, i] = np.mean(dW_all_timepoints, axis=1)
    dW = np.flip(dW, axis=0)

    # Plot dW matrix, with time relative to plateau on the x-axis and initial weight on the y-axis (going from 0 at the bottom to Wmax at the top)
    colorscale = np.max(np.abs(dW))
    im = ax1.imshow(dW, extent=[-tmax / 2, tmax / 2, 0, Wmax], aspect='auto', cmap='bwr', vmin=-colorscale,
                      vmax=colorscale)
    plt.colorbar(im, ax=ax1)
    ax1.set_xlim([-5, 2])
    ax1.set_ylim([0,3])
    # ax1.set_xlabel('Time from plateau (s)')
    ax1.set_ylabel('Initial weight')
    ax1.set_title('BTSP learning rule')

    ax2.plot(np.linspace(-tmax / 2, tmax / 2, len(all_delta_t)), dW[-1, :], label='W_init=0', c='lightgray', linewidth=1)
    # ax2.plot(np.linspace(-tmax / 2, tmax / 2, len(all_delta_t)), dW[80, :], label='W_init=1', c='darkgray', linewidth=1)
    # ax2.plot(np.linspace(-tmax / 2, tmax / 2, len(all_delta_t)), dW[70, :], label='W_init=1.5', c='k', linewidth=1)
    # ax2.plot(np.linspace(-tmax / 2, tmax / 2, len(all_delta_t)), dW[60, :], label='W_init=2', c='k', linewidth=1)
    ax2.plot([0, 0], [min(dW[60, :]), max(dW[-1, :])], 'r--', alpha=0.5, linewidth=1.5)
    ax2.plot([-5, 5], [0, 0], 'gray', alpha=0.5)
    ax2.set_xlim([-5, 2])
    ax2.set_xlabel('Time from plateau (s)')
    ax2.set_ylabel('dW')
    # ax2.legend()


def plot_grid(ax, environment, trajectories, learning_rule, legend=False):
    """Plot gridworld environment and trajectories on it."""

    trajectories = trajectories[learning_rule]

    ax.imshow(environment.grid >= 0, interpolation='nearest', cmap='gray', vmin=-2, vmax=1)

    # Print initial state on grid
    if hasattr(environment, 'initial_state'):
        ax.text(environment.initial_state[1] + 1, environment.initial_state[0] + 1.1,
                 r'$\mathbf{S}$', fontsize=12, ha='center', va='center')

    # Print reward locations on grid
    if environment.rewards is not None:
        if type(environment.rewards) == list:
            for reward_loc in environment.rewards:
                ax.text(reward_loc[1] + 1, reward_loc[0] + 1.08,
                         r'$\mathbf{R}$', fontsize=12, ha='center', va='center')
        elif type(environment.rewards) == tuple:
            ax.text(environment.rewards[1] + 1, environment.rewards[0] + 1.1,
                     r'$\mathbf{R}$', fontsize=12, ha='center', va='center')

    # Print grid lines
    for x in range(environment.grid_cols + 1):
        ax.plot([x+0.5, x+0.5], [0.5, environment.grid_rows+0.5], '-k', lw=0.5)
    for y in range(environment.grid_rows + 1):
        ax.plot([0.5, environment.grid_cols+0.5], [y+0.5, y+0.5], '-k', lw=0.5)

    # Plot trajectories
    for trajectory in trajectories:
        # All trajectories in gray
        ax.plot([state[1] + 1 + np.random.uniform(-0.2, 0.2) for state in trajectory],
                [state[0] + 1 + np.random.uniform(-0.2, 0.2) for state in trajectory],
                'o-', c='gray', alpha=0.1, lw=0.5, ms=2)
    if len(trajectories) > 2:
        # # Plot first two trajectories in red and blue
        # ax.plot([state[1] + 1 + np.random.uniform(0., 0.3) for state in trajectories[0]],
        #         [state[0] + 1 + np.random.uniform(0., 0.3) for state in trajectories[0]], 'o-', c=[1, .3, .3], lw=0.5, ms=2, label='Trial 1', alpha=0.8)
        # ax.plot([state[1] + 1 - np.random.uniform(0., 0.3) for state in trajectories[1]],
        #         [state[0] + 1 - np.random.uniform(0., 0.3) for state in trajectories[1]], 'o-', c=[.3, .5, 1], lw=0.5, ms=2, label='Trial 2', alpha=0.8)
        # Plot first two trajectories in red and blue
        ax.plot([state[1] + 1 + np.random.uniform(0., 0.3) for state in trajectories[0]],
                [state[0] + 1 + np.random.uniform(0., 0.3) for state in trajectories[0]], 'o-', c=[1, .0, .0], lw=0.5, ms=2, label='Trial 1', alpha=0.6)
        ax.plot([state[1] + 1 - np.random.uniform(0., 0.3) for state in trajectories[1]],
                [state[0] + 1 - np.random.uniform(0., 0.3) for state in trajectories[1]], 'o-', c=[.0, .2, 1], lw=0.5, ms=2, label='Trial 2', alpha=0.6)


    ax.set_xlim(0.05, environment.grid_cols + 0.9)
    ax.set_ylim(environment.grid_rows + 0.9, 0.05)
    if legend:
        ax.legend(loc='lower left', bbox_to_anchor=(-0.05, -0.35), fontsize=8, frameon=False, ncol=1, labelspacing=0.2)
    ax.axis('off')
    ax.set_title(f'{learning_rule}')

def plot_SR_place_fields(fig, ax1, ax2, cell_nr, M_history, start_idx, end_idx, grid_size, learning_rule):
    M_start = M_history[learning_rule][start_idx] - M_history[learning_rule][0]
    place_field = M_start[:, cell_nr].reshape(grid_size)
    im = ax1.imshow(place_field)
    ax1.set_title('Trial 1')
    ax1.axis('off')

    M_end = M_history[learning_rule][end_idx] - M_history[learning_rule][0]
    place_field = M_end[:, cell_nr].reshape(grid_size)
    im = ax2.imshow(place_field)
    ax2.set_title('Trial 30')
    ax2.axis('off')

    ax = ax2
    cax = fig.add_axes([ax.get_position().x1+0.01, ax.get_position().y0, 0.01, ax.get_position().height])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('$\Delta$ Weight', rotation=270, labelpad=8, fontsize=8)
    cbar.ax.tick_params(labelsize=8)

def plot_navigation_summary(ax1, ax2, trajectory_lengths):
    target_length = 10
    learning_rules = ['TD', 'Hebb', 'BTSP']
    colors = ['k', 'gray', 'r']
    
    # min_length = np.min([value for key, value in trajectory_lengths.items()])
    # print(min_length)

    lengths = {rule: [] for rule in learning_rules}
    trials_to_optimal = {rule: [] for rule in learning_rules}

    for i, rule in enumerate(learning_rules):
        for seed in trajectory_lengths[rule]:
            lengths[rule].append(np.array(trajectory_lengths[rule][seed]))
        lengths[rule] = np.array(lengths[rule]) -1 # -1 because we don't count the initial state

        mean_lengths = np.mean(lengths[rule], axis=1)
        ax1.boxplot(mean_lengths, positions=[i], widths=0.5, showfliers=False, patch_artist=True, boxprops=dict(facecolor=colors[i], alpha=0.5))
        # ax1.bar(i, np.mean(lengths[rule]), label=rule, alpha=0.5, color=colors[i])
        # ax1.scatter(np.ones(lengths[rule].shape[0]) * i, np.mean(lengths[rule], axis=1), color='k', alpha=0.3, s=5)

        for trajectory in lengths[rule]:
            if min(trajectory) <= target_length:
                trials_to_optimal[rule].append(np.where(trajectory <= target_length)[0][0])
            else:
                trials_to_optimal[rule].append(len(trajectory))
        trials_to_optimal[rule] = np.array(trials_to_optimal[rule])

        ax2.boxplot(trials_to_optimal[rule], positions=[i], widths=0.5, showfliers=False, patch_artist=True, boxprops=dict(facecolor=colors[i], alpha=0.5))
        # ax2.bar(i, np.mean(trials_to_optimal[rule]), label=rule, alpha=0.5, color=colors[i])
        # ax2.scatter(np.ones(trials_to_optimal[rule].shape[0]) * i, trials_to_optimal[rule], color='k', alpha=0.3, s=5)

    ax1.set_ylabel('Avg steps to reward', labelpad=0)
    ax1.set_ylim(bottom=-1)
    ax2.set_ylabel('Trials to learn \npath to reward', labelpad=-1)
    ax2.set_ylim(bottom=-1)

    ax1.set_xticks(np.arange(len(learning_rules)))
    ax1.set_xticklabels(learning_rules, rotation=-45, ha='left', fontsize=8)
    ax2.set_xticks(np.arange(len(learning_rules)))
    ax2.set_xticklabels(learning_rules, rotation=-45, ha='left', fontsize=8)

    # Compute p-values
    mean_lengths_Hebb = np.mean(lengths['Hebb'], axis=1)
    mean_lengths_TD = np.mean(lengths['TD'], axis=1)
    mean_lengths_BTSP = np.mean(lengths['BTSP'], axis=1)

    print('Avg steps to reward comparison:')
    print('Mean TD: ', np.mean(mean_lengths_TD))
    print('Mean Hebb: ', np.mean(mean_lengths_Hebb))
    print('Mean BTSP: ', np.mean(mean_lengths_BTSP))
    print('TD vs Hebb: p=', scipy.stats.ttest_ind(mean_lengths_TD, mean_lengths_Hebb)[1])
    print('TD vs BTSP: p=', scipy.stats.ttest_ind(mean_lengths_TD, mean_lengths_BTSP)[1])
    print('Hebb vs BTSP: p=', scipy.stats.ttest_ind(mean_lengths_Hebb, mean_lengths_BTSP)[1])

    print('---------------------------------')
    print('Trials to learn short path comparison:')
    print('Mean TD: ', np.mean(trials_to_optimal['TD']))
    print('Mean Hebb: ', np.mean(trials_to_optimal['Hebb']))
    print('Mean BTSP: ', np.mean(trials_to_optimal['BTSP']))
    print('TD vs Hebb: p=', scipy.stats.ttest_ind(trials_to_optimal['TD'], trials_to_optimal['Hebb'])[1])
    print('TD vs BTSP: p=', scipy.stats.ttest_ind(trials_to_optimal['TD'], trials_to_optimal['BTSP'])[1])
    print('Hebb vs BTSP: p=', scipy.stats.ttest_ind(trials_to_optimal['Hebb'], trials_to_optimal['BTSP'])[1])

def plot_pathlength_over_trials(ax, trajectory_lengths):
    target_length = 10
    learning_rules = ['TD', 'Hebb', 'BTSP']
    colors = ['k', 'gray', 'r']

    ax.plot([0, 30], [target_length, target_length], '--', c='gray', alpha=0.5, linewidth=1)
    for i, rule in enumerate(learning_rules):
        lengths = []
        for seed in trajectory_lengths[rule]:
            lengths.append(trajectory_lengths[rule][seed])
        lengths = np.array(lengths) -1 # number of steps is number of states visited -1
        trials = np.arange(lengths.shape[1])
        ax.plot(trials, np.mean(lengths, axis=0), label=rule, linewidth=1, c=colors[i])
        SEM = np.std(lengths, axis=0) / np.sqrt(lengths.shape[0])
        ax.fill_between(trials, np.mean(lengths, axis=0) - SEM,
                                    np.mean(lengths, axis=0) + SEM, alpha=0.3, color=colors[i])
        
    # ax.legend(loc='upper right', bbox_to_anchor=(1., 1), fontsize=8, frameon=False, ncol=3, labelspacing=0.01, handlelength=1.2)
    ax.set_xlabel('Trial', labelpad=-1)
    ax.set_ylabel('Number of steps to reward', labelpad=0)
    ax.set_ylim(bottom=5)
    ax.set_xlim(right=30)

def plot_place_field_size(ax, M_history_all, trial_indexes, grid_size):
    learning_rules = ['TD', 'Hebb', 'BTSP']
    colors = ['k', 'gray', 'r']

    
    for i, rule in enumerate(learning_rules):
        fraction_learned_all = []
        for seed in trial_indexes[rule]:
            fraction_learned = []
            for trial in trial_indexes[rule][seed]:
                M = M_history_all[rule][seed][trial]
                fraction_learned.append(np.count_nonzero(M)/M.size)
            fraction_learned_all.append(fraction_learned)

        fraction_learned_all = np.array(fraction_learned_all)

        ax.plot(np.arange(30), np.mean(fraction_learned_all, axis=0), label=rule, linewidth=1, c=colors[i])
        SEM = np.std(fraction_learned_all, axis=0) / np.sqrt(fraction_learned_all.shape[0])
        ax.fill_between(np.arange(30), np.mean(fraction_learned_all, axis=0) - SEM,
                                       np.mean(fraction_learned_all, axis=0) + SEM, alpha=0.3, color=colors[i])
        


    ax.legend(loc='upper right', bbox_to_anchor=(1., 1.1), fontsize=8, frameon=False, ncol=3, labelspacing=0.01, handlelength=1.2)
    ax.set_xlabel('Trial', labelpad=-1)
    ax.set_ylabel('Fraction of weights learned', labelpad=0)
    ax.set_ylim(bottom=0)
    ax.set_xlim(right=30)

def plot_cumulative_reward(ax, trajectory_lengths):

    learning_rules = ['TD', 'Hebb', 'BTSP']
    labels = ['TD', 'Hebbian', 'BTSP']
    colors = ['k', 'gray', 'r']

    for i, rule in enumerate(learning_rules):
        cumulative_rewards = []
        for seed in trajectory_lengths[rule]:
            lengths = np.array(trajectory_lengths[rule][seed])-1 # number of steps is number of states visited -1
            total_num_steps = np.sum(lengths)
            rewards = np.zeros(total_num_steps)
            rewards[np.cumsum(lengths)-1] = 1
            cumulative_rewards.append(np.cumsum(rewards))

        min_length = min(len(rewards_seed) for rewards_seed in cumulative_rewards)
        cumulative_rewards = [rewards_seed[:min_length] for rewards_seed in cumulative_rewards]
        cumulative_rewards = np.array(cumulative_rewards)
        steps = np.arange(cumulative_rewards.shape[1])

        ax.plot(steps, np.mean(cumulative_rewards, axis=0), label=labels[i], linewidth=1, c=colors[i])
        SEM = np.std(cumulative_rewards, axis=0) / np.sqrt(cumulative_rewards.shape[0])
        ax.fill_between(steps, np.mean(cumulative_rewards, axis=0) - SEM,
                                        np.mean(cumulative_rewards, axis=0) + SEM, alpha=0.3, color=colors[i])
        
    ax.legend(loc='center right', bbox_to_anchor=(0.5, 0.7), fontsize=8, frameon=False, handlelength=1, handletextpad=0.4)
    ax.set_xlabel('Number of steps', labelpad=0)
    ax.set_ylabel('Cumulative reward', labelpad=0)
    # ax.set_ylim(bottom=0)
    ax.set_xlim(0,cumulative_rewards.shape[1])




if __name__ == '__main__':
    update_plot_defaults()
    generate_Figure5(show=True, save=False)