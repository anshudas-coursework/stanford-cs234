# Environment-related packages
from gym import Env
from gym.spaces import Box
from gym.utils import seeding

# Mathematics packages
import numpy as np
from scipy.integrate import solve_ivp

# Support functions
from dataclasses import dataclass

# Rendering
import pygame

@dataclass
class Dynamics:
    update: float = 1 # ode solving update rate
    max_time: int = 100 # horizon time
    mu: float = 1e6 # gravitational parameter
    mr: float = 1.0 # rocket mass
    mf: float = 10.0 # initial fuel mass
    dv_max: float = 25.0*update # maximum instantaneous change in velocity
    ve: float = dv_max/np.log((mf+mr)/mr) # effective exhaust velocity
    fuel_pct: float = 0.05 # percentage of fuel used at each timestep

@dataclass
class Bounds:
    # State bounds
    r: list[float] # radial distance from orbital center
    theta: list[float] # true anomaly
    v_r: list[float] # radial velocity
    w: list[float] # angular velocity
    mf: list[float] # remaining amount of fuel mass, lower bound = 0.0
    rt: list[float] # remaining amount of time-to-go

    # Action bounds
    dV: list[float] # commanded change in velocity
    phi: list[float] # commanded direction of dV

@dataclass 
class Seeds:
    state: int = 1
    action: int = 2

@dataclass
class Target:
    r: float = 0.0
    theta: float = 0.0
    v_r: float = 0.0
    w: float = 0.0

@dataclass
class Reward:
    dist_reward: float = 10.0
    dist_scaling: float = 8.0

    dv_reward: float = 0.1
    dv_scaling: float = 8.0

    phi_reward: float = 0.01
    phi_scaling: float = 8.0

    opt_type: str = "min_fuel" # "min_time" or "min_fuel"
    min_fuel_reward: float = 0.05
    min_time_reward: float = 0.05
    opt_scaling: float = 8.0

@dataclass
class Termination:
    dist: float = 1
    angle: float = 5*np.pi/180

@dataclass
class Information:
    orbit_type: str = "circular" # {circular, elliptical, parabolic, hyperbolic}

@dataclass
class Animation:
    tgt_radius: float = 50.0

class Pulser(Env):
    metadata = {"render_modes": ["human", "rgb_array", "none"], "render_fps": 15}

    def __init__(self, render_mode=None, seed=1, opt_type='min_time'):
        # Environment seeding
        self.np_random, _ = seeding.np_random(seed)

        # Dynamics
        self.model_params = Dynamics

        # Rewards
        self.reward_params = Reward(opt_type=opt_type)

        # Termination criteria
        self.term = Termination

        # State-action bounds
        self.bound_params = Bounds(
            [1e2, 2e3],
            [0.0, 2*np.pi],
            [-np.Inf, np.Inf],
            [0.0, np.Inf],
            [0.0, np.Inf],
            [0.0, np.Inf],
            [0.0, np.Inf],
            [0.0, 2*np.pi]
        )
        self.sampling_bound_params = Bounds( 
            [5e2, 1e3],
            [(120-30)*np.pi/180, (120+30)*np.pi/180],
            [-10.0, 10.0],
            [np.sqrt(self.model_params.mu/1e3), np.sqrt(self.model_params.mu/1e2)],
            [0.0, self.model_params.mf],
            [0.0, 0.0],
            [0.0, self.model_params.dv_max],
            [0.0, 2*np.pi]
        )
        
        # Define the observation and action space
        obs_lower_bound = np.array([
            self.bound_params.r[0], 
            self.bound_params.theta[0],
            self.bound_params.v_r[0],
            self.bound_params.w[0],
            -np.Inf,
            -np.Inf,
            -np.Inf,
            -np.Inf,
            self.bound_params.mf[0],
            self.bound_params.rt[0]
        ])
        obs_upper_bound = np.array([
            self.bound_params.r[1],
            self.bound_params.theta[1],
            self.bound_params.v_r[1],
            self.bound_params.w[1],
            np.Inf,
            np.Inf,
            np.Inf,
            np.Inf,
            self.bound_params.mf[1],
            self.bound_params.rt[1]
        ])
        self.observation_space = Box(obs_lower_bound, obs_upper_bound, dtype=float) # type: ignore
        
        # Define the action space
        act_lower_bound = np.array([
            self.sampling_bound_params.dV[0],
            self.sampling_bound_params.phi[0]
        ])
        act_upper_bound = np.array([
            self.sampling_bound_params.dV[1],
            self.sampling_bound_params.phi[1]
        ])
        self.action_space = Box(act_lower_bound, act_upper_bound, dtype=float) # type: ignore
        
        # Rendering options
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.window_size = 500
        self.clock = None

        # Core information for logging
        self.target = Target
        self.observation = np.zeros(10)
        self.action = np.array([0.0, 0.0])
        self.info = Information

        # Buffer for plotting
        self.pos = np.array([[]])

    def reset(self):
        # Set the initial state to a circular orbit
        r0 = self.sampling_bound_params.r[1]/2
        theta0 = 0.0
        v_r0 = 0.0
        w = np.sqrt(self.model_params.mu / (r0**3))
        mf0 = self.sampling_bound_params.mf[1]
        rt0 = 0.0

        # Target
        self.target = Target(
            np.random.uniform(self.sampling_bound_params.r[0], self.sampling_bound_params.r[1]),
            np.random.uniform(self.sampling_bound_params.theta[0], self.sampling_bound_params.theta[1]),
            np.random.uniform(self.sampling_bound_params.v_r[0],self.sampling_bound_params.v_r[1]),
            np.random.uniform(self.sampling_bound_params.w[0],self.sampling_bound_params.w[1])
        )

        # Observation based on state and target
        observation = np.array([r0, theta0, v_r0, w, self.target.r - r0, self.target.theta - theta0, self.target.v_r - v_r0, self.target.w - w, mf0, rt0])
        self.observation = observation

        self.pos = np.array([[observation[0]*np.cos(observation[1]), observation[0]*np.sin(observation[1])]])

        # Render
        if self.render_mode=='human':
            self.render()
        
        return observation
    
    def path_reward(self):
        
        # Penalty proportional to dV
        dv_reward = 0.0
        if self.action[0]!= 0.0:
            dv_reward = self.reward_params.dv_reward * (1 - self.action[0]/self.model_params.dv_max)**self.reward_params.dv_scaling

        # Incentivize the thruster to move in the direction of the rocket
        angle_diff = self.action[1] - self.observation[1]
        angle_diff = np.abs((angle_diff + np.pi) % 2*np.pi - np.pi)
        phi_reward = self.reward_params.phi_reward * (1 - angle_diff/2*np.pi)**2

        # Total reward
        total_reward = dv_reward + phi_reward
        return total_reward * self.model_params.update
    
    def terminal_reward(self):
        reward = 0.0

        # Proportional to how close you are to hitting the target
        distance = np.sqrt(self.observation[0]**2 + self.target.r**2 - 2*self.observation[0]*self.target.r*np.cos(self.observation[1]-self.target.theta))
        reward += self.reward_params.dist_reward * (1 - distance/self.bound_params.r[1]/2)**self.reward_params.dist_scaling

        # Proportional to how much fuel is remaining in the rocket (scaled with max fuel)
        if self.reward_params.opt_type=="min_fuel":
            reward += self.reward_params.min_fuel_reward * (self.observation[8]/self.model_params.mf)**self.reward_params.opt_scaling

        # Proportional to how much time has passed in the simulation (scaled with max time allowed to simulation)
        elif self.reward_params.opt_type=="min_time":
            reward += self.reward_params.min_time_reward * (1 - self.observation[9]/self.model_params.max_time)**self.reward_params.opt_scaling

        return reward
    
    def step(self, action):

        # Forward propagate the dynamics
        def dynamics(action):

            def apply_thrust(dV):
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
                
                return dV, mfinal-self.model_params.mr
            
            dV, mf = apply_thrust( action[0] )
            action[0] = dV
            self.action = action

            # Observations
            y0 = self.observation[0:4]
            tf = self.model_params.update

            def ODE(t, y):
                # Dynamics
                dydt    = np.zeros(y.shape)
                dydt[0] = y[2]
                dydt[1] = y[3]
                dydt[2] = y[0]*y[3]**2 - self.model_params.mu/y[0]**2 + action[0]/tf*np.sin(action[1]-y[1])
                dydt[3] = (action[0]/tf*np.cos(action[1]-y[1]) - 2*y[2]*y[3]) / y[0]
                return dydt
            
            # Crossing the target angle
            def crossed_target_angle(t, y): return y[1] - self.target.theta

            # Hit the planet or the boundary
            def hit_planet(t, y): return y[0] - self.bound_params.r[0]
            def out_of_range(t, y): return y[0] - self.bound_params.r[1]
            
            # Run ODE solver to propagate ode forward
            res = solve_ivp(ODE, (0,tf), y0, 'Radau', events=(crossed_target_angle, hit_planet, out_of_range), rtol = 1e-9)

            # Wrap the angle to 2*pi
            while res.y[1,-1] > 2*np.pi: res.y[1,-1] -= 2*np.pi
            while res.y[1,-1] < 0: res.y[1,-1] += 2*np.pi
            
            # Target
            target = np.array([self.target.r, self.target.theta, self.target.v_r, self.target.w])

            # Observation
            observation = np.array(res.y[0:4,-1])
            observation = np.append(observation, res.y[0:4,-1] - target)
            observation = np.append(observation, mf)
            observation = np.append(observation, self.observation[9] + self.model_params.update)

            done = False
            # Check if there was an angle crossing
            if any(res.t_events[0]):
                done = True
                observation[1] = res.y_events[0][0][1]
            elif any(res.t_events[1]):
                done = True
                observation[0] = res.y_events[1][0][0]
            elif any(res.t_events[2]):
                done = True
                observation[0] = res.y_events[2][0][0]

            return observation, done
        
        # Accrued rewards at every timestep
        reward = self.path_reward()

        # Update observation based on dynamics
        observation, done = dynamics(action)
        self.observation = observation
        self.action = action
        self.pos = np.append( self.pos, [[observation[0]*np.cos(observation[1]), observation[0]*np.sin(observation[1])]], axis=0 )

        # Determine termination criteria
        if done: reward += self.terminal_reward()

        # Update the information
        sp_orb_energy = 0.5*(self.observation[2]**2 + (self.observation[0]*self.observation[3])**2) - self.model_params.mu/self.observation[0]
        sp_ang_momentum = self.observation[0]**2 * self.observation[3]
        eccentricity = np.sqrt(1 + 2*sp_orb_energy*sp_ang_momentum**2/self.model_params.mu**2)
        if eccentricity==0: otype = "circular"
        elif eccentricity>0 and eccentricity<1: otype = "ellipical"
        elif eccentricity==1: otype = "parabolic"
        elif eccentricity: otype = "hyperbolic"
        info = Information(
            orbit_type=otype
        )
        self.info = info

        # Render
        if self.render_mode=='human':
            self.render()

        return observation, reward, done, info

    def render(self):
        pygame.init()
        pygame.display.init()

        self.window = pygame.display.set_mode( ((3/2)*self.window_size, self.window_size) )
        self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        informatics = pygame.Surface((self.window_size/2, self.window_size))
        canvas.fill((255,255,255))
        informatics.fill((255,255,255))

        # Amount of time passed
        font = pygame.font.Font('freesansbold.ttf', 32)
        text = font.render(str(self.observation[9]), True, (255,255,255))
        

        # Center
        cx, cy = self.window_size/2, self.window_size/2
        
        # Scale
        scale = self.window_size/2/self.bound_params.r[1]

        # Central body
        pygame.draw.circle( canvas, (0,255,0), (cx,cy), self.bound_params.r[0]*scale )

        # Target
        target_x = self.target.r*np.cos(self.target.theta)*scale
        target_y = -self.target.r*np.sin(self.target.theta)*scale
        pygame.draw.circle( canvas, (255,0,0), (cx+target_x,cy+target_y), Animation.tgt_radius*scale )

        # Agent
        agent_x = self.observation[0]*np.cos(self.observation[1])*scale
        agent_y = -self.observation[0]*np.sin(self.observation[1])*scale
        pygame.draw.circle( canvas, (0,0,0), (cx+agent_x,cy+agent_y), 20*scale )

        # Agent's path
        for i in range(np.size(self.pos,axis=0)):
            cs = i/np.size(self.pos,axis=0) * 255
            pygame.draw.circle( canvas, (cs,cs,cs), (cx+self.pos[-i,0]*scale,cy-self.pos[-i,1]*scale), 10*scale )

        # Draw the bound of the system
        pygame.draw.circle( canvas, (0,0,0), (cx,cy), self.bound_params.r[1]*scale, width=1 )

        # Fuel bar
        fuel_bar_length = 200
        fuel_pct = self.observation[8]/self.model_params.mf
        pygame.draw.rect( informatics, (0,0,0), pygame.Rect(10, 10, fuel_bar_length, 30), 1 )
        pygame.draw.rect( informatics, (255*(1-fuel_pct),255*fuel_pct,0), pygame.Rect(10, 10, fuel_bar_length*fuel_pct, 30) )

        # The following line copies our drawings from `canvas` to the visible window
        self.window.blit(canvas, (0,0))
        self.window.blit(informatics, (self.window_size,0))
        #self.window.blit(text, (0,0))
        
        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        self.clock.tick(self.metadata["render_fps"])
        return
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()