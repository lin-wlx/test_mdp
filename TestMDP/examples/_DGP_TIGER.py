#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% packages
#%% 
import os, sys
package_path = os.path.dirname(os.path.abspath(os.getcwd()))

sys.path.insert(0, package_path + "/test_func")
from _core_test_fun import *
#############################################################################
#############################################################################

def list2Matrix(List):
    # return a n * 1 matrix
    return np.array(np.expand_dims(np.array(List),1))

#%%

def TIGER_dynamics(state, action):
    """
    simulates the environment dynamics. i.e. what happens when you take an action in a given state
    state= -1: tiger on the left
    state= 1: tiger on the right
    action = 0: listen
    action = 1: open left door
    action = 2: open right door ?
    """

    p_correct = 0.7 # larger -> more POMDP
    #probability of hearing the right door
    #1.0 first order markov
    #0.5 max uncertainty


    # obs -> action -> obs, reward
    if action == 0: # listen
        p = rbin(1, p_correct) #Random binary draw with probability p_correct
        #=1 with prob 0.7, correct observation

        obs = p * state +  (1-p) * (0-state) #correct state if p is correct

        reward = -1 #each listen -1

    else: # action = -1 or 1
        if action == state: #open the tiger
            reward = -100 #lose
        else: # no tiger door
            reward = 10
        obs = 3 # end status
        #termination criterion

    return reward, obs #obs = listened state if choose to listen / = 3 if take an action


def TIGER_choose_action(obs, behav_def = 0):
    """
    implements different behavioral policies for choosing actions
    behav_def:
        0. always listen
        1. random
        2. adaptive
    (default: 0)
    """
    p_listen = 0.9 # for random policy
    #0.9 listen in random policy

    T_must_obs = 10 # for adaptive plicy
    #must listen 10 steps before deciding in adaptive policy
    
    if behav_def == 0:
        return 0 # always listen

    elif behav_def == 1: #random policy
        if rbin(1, p_listen): #0.9 chance to listen
            return 0 #chooses action = 0, to listen

        elif rbin(1, 0.5): #if chooses not to listen, open left door with 0.5 chance
            return 1

        else: #if not listen, nor open left door
            return -1

    elif behav_def == 2: #smart adaptive policy
        """ based on obs, Chengchun's approach
        1. if n <= T_must_obs: obs
        2. else: n > T_must_obs 时 p_listen = (1- max(p_left,p_right)) * 2, o.w. open the door accourding to the prob.
        """
        if obs[1] <= T_must_obs: #always observe until some amounts done

        #note: obs here have become an array, 0 proportion, 1 counts

            return 0


        else:
            p_l = obs[0]

            p_listen = (1- max(p_l,1 - p_l)) * 2 #calculates listening probability based on uncertainty
            #max(p_l, 1-p_l): Your confidence level (between 0.5 and 1.0)
            #1 - max(...): Your uncertainty (between 0.5 and 0)
            #* 2: Scale uncertainty to probability (between 1.0 and 0)

            if rbin(1, p_listen): #choose whether not to listen based on prob

                return 0

            elif rbin(1, p_l): #choose the left door based on proportion observed

                return -1

            else:

                return 1
        
def simu_tiger(N = 1, T = 20, seed = 1, behav_def = 0, obs_def = "alt", T_def = 0, include_reward = True, fixed_state_comp = False):
    """
    Main simulation function that generates N trajectories of tiger problem data

    T: spycify the game here
    A: "listen"/ "open_l" / "open_r"  ---- 0 / -1 / +1
    State:  "l" / "r" : -1 / +1
    Obervation: hear "l" / "r"
    Reward: -1, 10, - 100
    Returns: a list (len = N) of [$O_{T*dim_O},A_{T*1}$] or [O,A,R]
    
    behav_def:
        0. always listen
        1. random
        2. adaptive
    obs_def:
        "alt": [1,-1]
        1: [p]
        2: [p,n]
    T_def:
        0: length = T with always listen
        1: truncation
    """   
    # gamma = .9 
    
    MDPs = []
    rseed(seed); npseed(seed)
    init_state = rbin(1, .5, N) * 2 - 1 #Generate N binary values (0 or 1) with 50% probability each, scale to 2 then -1 to get 1 or -1
    true_states = [] #list of true tiger locaiton, Only used when fixed_state_comp = True
    
    if T_def == 1: #trunction mode, stops when action taken
        def stop(obs,t):
            return obs != 3
    else: #default, keeps going for fixed t no matter what
        def stop(obs,t):
            return t < T
    
    for i in range(N):
        ## Initialization
        state = init_state[i]
        obs, obs_hist = 0, [0] #hist: history of raw observations
        A = [] #list to store actions
        R = [0] # for alignment purpose, starts with 0, no reward before first action
        O, O_1 = [[0.5, 0]], [0.5]  #format for obs_def = 2, obs_def = 1
        t, left_cnt = 0, 0 #timestep, counter for left as well
        
        while(stop(obs,t)): # not in the Terminal state
            ## choose actiom, receive reward and state trainsition [observations]
            action = TIGER_choose_action(obs = O[-1], behav_def = behav_def) # obs = [p,n], old version
            reward, obs = TIGER_dynamics(state,action)
            
            ## record
            left_cnt += (obs == -1)
            t += 1
            # for obs_def_0
            obs_hist.append(obs)
            # for obs_def_1
            O_1.append(left_cnt/t)
            # for action choosing and obs_def_2
            if obs == 3: #Use t-1 to avoid including the terminal observation in proportion
                # Append [proportion_before_terminal, current_time]
                O.append([left_cnt/(t-1),t])
            else:
                O.append([left_cnt/t,t])  
            A.append(action)
            R.append(reward)
        A.append(3) #Add terminal marker to action sequence
        
        if obs_def == "alt":
            O =  list2Matrix(obs_hist)
        elif obs_def == "null":
#             O =  list2Matrix(obs_hist)        
            if fixed_state_comp:
                O = list2Matrix(obs_hist)
                true_states.append(state)
            else:
                O = np.array([[a,state] for a in obs_hist])
#             print(O.shape)
        elif obs_def == 1:
            O = list2Matrix(O_1)
        elif obs_def == 2:
            O = np.array(O)
        if include_reward:
            MDP = [O, list2Matrix(A), list2Matrix(R)]
        else:
            MDP = [O, list2Matrix(A)]
        MDPs.append(MDP)
    if fixed_state_comp:
        return [MDPs,true_states]
    return MDPs


# ============================================================================
# K-TH ORDER MOVING TIGER MDP (For k-th Order Detection)
# ============================================================================

class MovingTigerKthOrder:
    """
    Tiger that moves in deterministic k-cycle pattern.

    **Key Design Decisions:**
    1. Perfect observations (obs_accuracy = 1.0)
       - Required for k-th order MDP property
       - With noise, becomes POMDP regardless of k

    2. Deterministic k-cycle
       - Tiger position at time t: pattern[t mod k]
       - Completely predictable given k history

    3. True k-th order MDP
       - J < k: Cannot determine cycle position
       - J = k: Can perfectly determine position and predict next
       - J > k: Still works (redundant history)

    This is used to demonstrate that the Markov test can detect
    k-th order structure (not just 1st order).

    Parameters
    ----------
    k : int, default=4
        Order of Markov process (length of cycle)
    pattern : list, optional
        Specific k-length pattern. If None, generates one.
    policy : str, default='always_listen'
        Behavioral policy
    random_state : int, optional
        Random seed

    Examples
    --------
    """

    def __init__(self, k=6, pattern=None, policy='always_listen',
                 random_state=None):
        """Initialize k-th order Moving Tiger."""
        self.k = k
        self.policy = policy
        self.obs_accuracy = 1.0  # MUST be perfect for k-th order MDP!

        if random_state is not None:
            np.random.seed(random_state)

        # Generate or validate pattern
        if pattern is None:
            self.pattern = self._generate_pattern(k)
        else:
            assert len(pattern) == k, f"Pattern length must be {k}"
            assert all(p in [-1, 1] for p in pattern), "Pattern must be -1 or +1"
            self.pattern = list(pattern)

    def _generate_pattern(self, k):
        """Generate a non-trivial k-length pattern."""
        if k == 2:
            return [-1, 1]
        elif k == 3:
            return [-1, 1, -1]
        elif k == 4:
            return [-1, 1, 1, -1]  # L, R, R, L
        elif k == 5:
            return [-1, 1, -1, 1, 1]  # L, R, L, R, R
        elif k == 6:
            return [-1, 1, 1, -1, -1, 1]

        elif k == 7:
            return [-1, 1, -1, -1, 1, -1, 1]

        elif k ==11:
            return [-1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1]
        else:
            # General pattern: mostly left with some right
            pattern = [-1] * k
            for i in range(k // 3, k, 2):
                pattern[i] = 1
            return pattern

    def _get_state_at_time(self, t):
        """
        Get tiger state at time t (deterministic cycle).

        This is the KEY FUNCTION for k-th order:
        state(t) = pattern[t mod k]
        """
        return self.pattern[t % self.k]

    def _get_observation(self, state):
        """
        Get observation of state.

        CRITICAL: obs_accuracy = 1.0 (perfect observation)
        This is REQUIRED for k-th order MDP property!
        """
        # With perfect observation, just return state
        return state

    def _choose_action(self, obs_hist, t):
        """Choose action based on policy."""
        if self.policy == 'always_listen':
            return 0
        elif self.policy == 'random':
            r = np.random.rand()
            return 0 if r < 0.9 else (-1 if r < 0.95 else 1)
        else:  # adaptive
            if t < 10:
                return 0
            left = sum(1 for o in obs_hist if o == -1)
            total = len([o for o in obs_hist if o != 0])
            if total == 0:
                return 0
            prop = float(left) / total
            return 1 if prop > 0.6 else (-1 if prop < 0.4 else 0)

    def generate_trajectory(self, length):
        """Generate one trajectory with k-th order dynamics."""
        obs_list = [0]  # Initial observation
        act_list = []
        states = []

        for t in range(length + 1):
            # Get state from deterministic cycle
            state = self._get_state_at_time(t)
            states.append(state)

            if t < length:
                # Perfect observation
                obs = self._get_observation(state)
                obs_list.append(obs)

                # Choose action
                action = self._choose_action(obs_list, t)
                act_list.append(action)

                # Check terminal
                if action != 0:
                    act_list.append(3)
                    break

        # Pad actions
        while len(act_list) < len(obs_list):
            act_list.append(3)

        O = np.array(obs_list).reshape(-1, 1)
        A = np.array(act_list).reshape(-1, 1)

        return (O, A), states

    def generate(self, n_trajectories, trajectory_length):
        """
        Generate multiple trajectories.

        Returns data in same format as simu_tiger for compatibility.
        """
        data = []
        for i in range(n_trajectories):
            traj, states = self.generate_trajectory(trajectory_length)
            data.append(traj)
        return data

    def get_pattern_info(self):
        """Get information about the pattern."""
        names = {-1: "LEFT", 1: "RIGHT"}
        pattern_str = " → ".join([names[p] for p in self.pattern])
        return {
            'k': self.k,
            'pattern': self.pattern,
            'pattern_str': pattern_str,
            'obs_accuracy': self.obs_accuracy,
        }


# ============================================================================
# WRAPPER FUNCTION (For Compatibility)
# ============================================================================

def simu_moving_tiger(N=1, T=20, seed=1, k=4, pattern=None,
                      behav_def=0, include_reward=False):
    """
    Wrapper function for k-th order Moving Tiger.

    Provides similar interface to simu_tiger() for easy integration.

    Parameters
    ----------
    N : int
        Number of trajectories
    T : int
        Trajectory length
    seed : int
        Random seed
    k : int, default=4
        Order of Markov process
    pattern : list, optional
        Specific k-length pattern
    behav_def : int
        Behavioral policy (0=always listen, 1=random, 2=adaptive)
    include_reward : bool
        Include rewards (not implemented, for compatibility)

    Returns
    -------
    trajectories : list
        List of (observations, actions) tuples

    Examples
    --------
    """
    policy_map = {0: 'always_listen', 1: 'random', 2: 'adaptive'}
    policy = policy_map.get(behav_def, 'always_listen')

    sim = MovingTigerKthOrder(k=k, pattern=pattern, policy=policy,
                              random_state=seed)

    trajectories = sim.generate(n_trajectories=N, trajectory_length=T)

    return trajectories


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Tiger Simulators Test")
    print("=" * 70)

    print("\n1. Original Tiger POMDP (static, noisy observations)")
    print("-" * 70)
    data_pomdp = simu_tiger(N=5, T=20, seed=42, obs_def="alt")
    print(f"✓ Generated {len(data_pomdp)} POMDP trajectories")
    print(f"  Shape: {data_pomdp[0][0].shape}, {data_pomdp[0][1].shape}")

    print("\n2. Original Tiger MDP (static, perfect observations)")
    print("-" * 70)
    data_mdp = simu_tiger(N=5, T=20, seed=42, obs_def="null")
    print(f"✓ Generated {len(data_mdp)} MDP trajectories")
    print(f"  Shape: {data_mdp[0][0].shape}, {data_mdp[0][1].shape}")

    print("\n3. Moving Tiger k=4 MDP (k-cycle, perfect observations)")
    print("-" * 70)
    sim = MovingTigerKthOrder(k=4, random_state=42)
    info = sim.get_pattern_info()
    print(f"✓ Created Moving Tiger with k={info['k']}")
    print(f"  Pattern: {info['pattern_str']}")
    print(f"  Observation accuracy: {info['obs_accuracy']} (perfect!)")

    data_k4 = sim.generate(n_trajectories=5, trajectory_length=20)
    print(f"✓ Generated {len(data_k4)} k=4 trajectories")
    print(f"  Shape: {data_k4[0][0].shape}, {data_k4[0][1].shape}")

    # Show first trajectory pattern
    (obs, acts), states = sim.generate_trajectory(12)
    print(f"\n  First 12 states: {states}")
    print(f"  First cycle (0-3): {states[0:4]}")
    print(f"  Second cycle (4-7): {states[4:8]}")
    print(f"  Third cycle (8-11): {states[8:12]}")

    if states[0:4] == states[4:8]:
        print(f"  ✓ Pattern repeats correctly!")

    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print("✓ Original Tiger POMDP: Use simu_tiger(obs_def='alt')")
    print("✓ Original Tiger MDP: Use simu_tiger(obs_def='null')")
    print("✓ Moving Tiger k=4 MDP: Use MovingTigerKthOrder(k=4)")
    print("\nAll simulators ready for Markov property testing!")
    print("=" * 70)
