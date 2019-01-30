import numpy as np
from physics_sim import PhysicsSim

def get_abs_reward(time, distance_x, distance_y, distance_z, velocity_z, euler_velocities, euler_accels):

    # the less is the distance to target - the better, (z distance is more important)
    # also there is some penalty for euler velocities and euler accelerations
    # the longer it flies, the better (otherwise it crashes too fast to the boundaries)

    reward = - 3.5 * distance_z \
              - 5 * abs(velocity_z)
#             0.2 * time
#             - 0.3 * distance_x - 0.3 * distance_y 
#            - 5 * abs(velocity_z) \
#            - 1.5 * euler_velocities - 0.5 * euler_accels
    
    return reward


class TaskImprovedReward():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3
        #self.action_repeat = 1

        self.state_size = self.action_repeat * 6
        #self.action_low = 0
        self.action_low = 450
        self.action_high = 900
        self.action_size = 4
        
        self.min_reward = float("inf")
        self.max_reward = float("-inf")
        
        self.success = False
        
        self.step_num = 0

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        
        self.min_reward_boundary = get_abs_reward(0, 300, 300, 300, 50, 50, 10)
        self.max_reward_boundary = get_abs_reward(5, 0, 0, 0, 0, 0, 0)
        
        print("min_reward_boundary={}, max_reward_boundary={}".format(self.min_reward_boundary, self.max_reward_boundary))
        

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()

        #distance = np.linalg.norm(self.sim.pose[:3] - self.target_pos)
        distance_x = abs(self.sim.pose[0] - self.target_pos[0])
        distance_y = abs(self.sim.pose[1] - self.target_pos[1])
        distance_z = abs(self.sim.pose[2] - self.target_pos[2])
        
        euler_velocities = np.max(self.sim.angular_v)
        euler_accels = np.max(self.sim.angular_accels)
        time = self.sim.time
        
        velocity_x = self.sim.v[0]
        velocity_y = self.sim.v[1]
        velocity_z = self.sim.v[2]
        
        #min_reward_boundary = -500
        #max_reward_boundary = 10

        reward = 0
        if self.sim.done and self.sim.time < self.sim.runtime and not self.success:
            reward_norm = -100000
            print(" collision! ")
#        elif distance_x < 5 and distance_y < 5 and distance_z < 5 \
#                and velocity_x < 2 and velocity_y < 2 and velocity_z < 6:
        elif distance_z < 5 and abs(velocity_z) < 10:
            reward_norm = 100000
            self.sim.done = True
            self.success = True
            print(" success! ")
        else:
#            reward = 10.0 + 0.2 * time - 0.3 * distance_x - 0.3 * distance_y \
#                    - 3.5 * abs(velocity_z) \
#                    - 3.5 * distance_z - 1.5 * euler_velocities - 0.5 * euler_accels

            reward = get_abs_reward(time, distance_x, distance_y, distance_z, velocity_z, euler_velocities, euler_accels)
            
            #normalize reward
            self.min_reward = min(reward, self.min_reward)
            self.max_reward = max(reward, self.max_reward)
            
            #reward_norm = np.tanh(reward)
            # just in case we made an error in estimating the min/max boundaries
            reward = np.clip(reward, self.min_reward_boundary, self.max_reward_boundary)
            
            # normalizing to [-1; 1]
            reward_norm = ((reward - self.min_reward_boundary)/(self.max_reward_boundary - self.min_reward_boundary) - 0.5 )*2
        
        #print("step_num={}, reward={}, reward_norm={}".format(self.step_num, reward, reward_norm))
        
        return reward_norm
    
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            self.step_num += 1
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            #cur_reward = self.get_reward() 
            #reward += cur_reward
            pose_all.append(self.sim.pose)
            #if cur_reward == 10:
            #    break
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        self.step_num = 0
        self.success = False
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state