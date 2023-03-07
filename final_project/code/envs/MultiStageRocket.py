# Environment-related packages
from gym import Env
from gym.spaces import Box
from gym.utils import seeding

# Mathematics packages
import numpy as np
from scipy.integrate import solve_ivp

# Support functions
from dataclasses import dataclass

@dataclass
class Dynamics:
    mu: float = 1000.0                      # gravitational parameter
    ve: float = 1000.0                      # effective exhaust velocity
    mr: float = 100.0                       # rocket mass
    mf: float = 1000000.0                   # initial fuel mass
    update: float = 0.01                    # update rate

@dataclass
class Bounds:
    # State bounds
    r: list[float]      # Radial distance from orbital center
    theta: list[float]  # True anomaly
    v_r: list[float]    # Radial velocity
    v_t: list[float]    # Tangential velocity
    mf: list[float]     # Remaining amount of fuel mass, lower bound = 0
    rt: list[float]     # Remaining amount of time-to-go

    # Action bounds
    dV: list[float]     # Delta V (dV)
    phi: list[float]    # Direction of dV

@dataclass 
class Seeds:
    state: int = 1
    action: int = 2

@dataclass
class Target:
    r: float = 0.0
    theta: float = 0.0
    v_r: float = 0.0
    v_t: float = 0.0

@dataclass
class Reward:
    problem_type: str = "min_time"
    tgt_pos: float = -0.50
    tgt_theta: float = -0.50
    tgt_vel_rad: float = -0.001
    tgt_vel_tng: float = -0.001
    pls_budget: float = -0.01
    min_fuel: float = 0.025
    min_time: float = -0.01

@dataclass
class Termination:
    dist: float = 1
    angle: float = 0.01

class Pulser(Env):
    metadata = {"render_modes":["human", "anim"]}

    def __init__(self, render_mode=None, seed=1):
        # Environment seeding
        self.np_random, _ = seeding.np_random(seed)

        # Dynamics
        self.model_params = Dynamics

        # Rewards
        self.reward_params = Reward

        # Termination criteria
        self.term = Termination

        # State-action bounds
        self.bound_params = Bounds( [10.0, 100000.0], [0, 2*np.pi], [-np.Inf, np.Inf], [-np.Inf, np.Inf],
                                   [0, np.Inf], [0, np.Inf], [0, np.Inf], [0, 2*np.pi] )
        dV_lim = float(self.model_params.ve * np.log((self.model_params.mr+self.model_params.mf)/self.model_params.mr))
        self.sampling_bound_params = Bounds( [100.0, 1000.0], [0.0, np.pi], [-10.0, 10.0], [0.0, 100.0],
                                            [0.0, self.model_params.mf], [0.0, 20.0], [0.0, 500.0], [0.0, 2*np.pi] )
        
        # Define the observation and action space
        obs_lower_bound = np.array([self.bound_params.r[0], self.bound_params.theta[0], self.bound_params.v_r[0],
                       self.bound_params.v_t[0], 0.0, 0.0, 0.0, 0.0, self.bound_params.mf[0],
                       self.bound_params.rt[0]])
        obs_upper_bound = np.array([self.bound_params.r[1], self.bound_params.theta[1], self.bound_params.v_r[1],
                       self.bound_params.v_t[1], 0.0, 0.0, 0.0, 0.0, self.bound_params.mf[1],
                       self.bound_params.rt[1]])
        self.observation_space = Box(obs_lower_bound, obs_upper_bound, dtype=float) # type: ignore
        
        # Define the action space
        act_lower_bound = np.array([self.sampling_bound_params.dV[0], self.sampling_bound_params.phi[0]])
        act_upper_bound = np.array([self.sampling_bound_params.dV[1], self.sampling_bound_params.phi[1]])
        self.action_space = Box(act_lower_bound, act_upper_bound, dtype=float) # type: ignore

        # Target position
        self.target = Target

        # Observation
        self.observation = self.reset()

        # Rendering options
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def reset(self):
        # Set the initial state to a circular orbit
        r0 = np.random.uniform(self.sampling_bound_params.r[0], self.sampling_bound_params.r[1])
        theta0 = 0.0
        v_r0 = 0.0
        v_t0 = np.sqrt(self.model_params.mu / r0)
        mf0 = self.sampling_bound_params.mf[1]
        rt0 = 0.0

        # Target
        self.target = Target(
            np.random.uniform(self.sampling_bound_params.r[0], self.sampling_bound_params.r[1]),
            np.pi/4 + np.random.uniform(self.sampling_bound_params.theta[0], self.sampling_bound_params.theta[1]),
            np.random.uniform(self.sampling_bound_params.v_r[0],self.sampling_bound_params.v_r[1]),
            np.random.uniform(self.sampling_bound_params.v_t[0],self.sampling_bound_params.v_t[1])
        )

        # Observation based on state and target
        observation = np.array([r0, theta0, v_r0, v_t0, self.target.r - r0, self.target.theta - theta0, self.target.v_r - v_r0, self.target.v_t - v_t0, mf0, rt0])
        self.observation = observation

        return observation
    
    def reward(self, action):
        # Initialize
        opt_reward = 0.0

        # Reach the desired point
        tgt_reward_params = np.array([self.reward_params.tgt_pos, self.reward_params.tgt_theta, self.reward_params.tgt_vel_rad, self.reward_params.tgt_vel_tng])
        tgt_reward = tgt_reward_params * np.abs(self.observation[4:8]) * self.model_params.update

        if self.reward_params.problem_type=="min_fuel": opt_reward = self.reward_params.min_fuel * self.observation[8] * self.model_params.update
        if self.reward_params.problem_type=="min_time": opt_reward = self.reward_params.min_time * self.observation[9] * self.model_params.update

        # Penalty proportional to dV
        budget_reward = self.reward_params.pls_budget * action[0]

        # Encourage 

        # Total reward
        total_reward = np.sum(tgt_reward) + budget_reward + opt_reward
        return (total_reward, tgt_reward, opt_reward, budget_reward)
    
    def step(self, action):
        # Calculate the reward at the current step
        total_reward, tgt_reward, opt_reward, budget_reward = self.reward(action)

        # Forward propagate the dynamics
        def dynamics(action):
            # Calculate acceleration and final mass based on desired deltaV
            def apply_thrust(dT, dV):
                """
                apply_thrust
                Determine the amount of thrust to apply based on the Tsiolkovsky rocket equation. If
                there isn't enough fuel to make the last pulse, use the remaining fuel
                
                Input:
                    - dT:   Duration of time thrust pulse is on for (constant thrust)
                    - dV:   The change in velocity magnitude for a single instant in time
                """
                # Invert Tsiolkovsky equation
                total_mass = self.observation[8] + self.model_params.mr
                mfinal = total_mass * np.exp(-dV / self.model_params.ve)

                # Check to see if there is enough fuel
                if (mfinal < self.model_params.mr):
                    dV = self.model_params.ve * np.log(total_mass / self.model_params.mr)
                    mfinal = self.model_params.mr
                
                return dV/dT, mfinal-self.model_params.mr
            accel, mf = apply_thrust( self.model_params.update, action[0] )
            phi = action[1]

            # Observations
            y0 = self.observation[0:4]
            tf = self.model_params.update

            def ODE(t, y):
                # Dynamics
                dydt    = np.zeros(y.shape)
                dydt[0] = y[2]
                dydt[1] = y[3] / y[0]
                dydt[2] = y[3]**2 / y[0] - self.model_params.mu / y[0]**2 + accel * np.sin(phi)
                dydt[3] = y[2]*y[3] / y[0] + accel * np.cos(phi)
                return dydt
            
            # Run ODE solver to propagate ode forward
            res = solve_ivp(ODE, (0,tf), y0, 'Radau', rtol = 1e-6)

            # Wrap the angle to 2*pi
            while res.y[1,-1] > 2*np.pi: res.y[1,-1] -= 2*np.pi
            while res.y[1,-1] < 0: res.y[1,-1] += 2*np.pi
            
            # Target
            target = np.array([self.target.r, self.target.theta, self.target.v_r, self.target.v_t])

            # Observation
            observation = np.array(res.y[0:4,-1])
            observation = np.append(observation, res.y[0:4,-1] - target)
            observation = np.append(observation, mf)
            observation = np.append(observation, self.observation[9] + self.model_params.update)
            return observation
        observation = dynamics(action)
        self.observation = observation

        # Determine termination criteria
        done = False
        if ((observation[0] <= self.bound_params.r[0]) or (observation[0] >= self.bound_params.r[1])): 
            done = True
        
        # Within a bubble of the goal OR within a range of the true anomaly
        distance = np.sqrt(observation[0]**2 + self.target.r**2 - 2*observation[0]*self.target.r*np.cos(observation[1]-self.target.theta))
        if ((distance <= self.term.dist) or (np.abs(observation[1]-self.target.theta) <= self.term.angle)):
            done = True

        info = [tgt_reward, opt_reward, budget_reward]

        return observation, total_reward, done, info

    def render(self, mode='none'):
        return