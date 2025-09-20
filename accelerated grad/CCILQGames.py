import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.linalg import block_diag

from solve_lq_problem import solve_lq_game
from Diff_robot_uncertainty import UnicycleRobotUncertain
from Costs import ProximityCost, OverallCost, ReferenceCost, WallCost, ProximityCostUncertainLinear , ProximityCostUncertainQuad, InputCost, SpeedCost
from MultiAgentDynamics import MultiAgentDynamics


class CCILQGame():

    def __init__(self, config):
        self.x0s = config['x0s'] 
        self.initial_vels = config['initial_vels']
        self.append_velocities()
        self.xrefs = config['xrefs']
        self.prob = config['prob']
        self.num_agents = config['num_agents']
        self.sigmas = config['sigmas']
        self.agent_list = self.initialize_agents(self.x0s, self.xrefs, self.sigmas, self.num_agents)
        self.TOL_CC_ERROR = config['TOL_CC_ERROR']
        self.phi = config['phi']
        self.ref_cost_threshold = config['ref_cost_threshold']
        self.dt = config['dt']
        self.HORIZON = config['HORIZON']
        self.TIMESTEPS = int(self.HORIZON/self.dt)
        self.mp_dynamics = self.initialize_dynamics()
        self.costs = self.mp_dynamics.define_costs_lists(uncertainty=True)
        self.initial_mu = config['initial_mu']
        self.mu = self.get_mu()
        self.d_safe = config['d_safe']

    def append_velocities(self):
        for i in range(len(self.x0s)):
            self.x0s[i].append(self.initial_vels[i])

    def initialize_agents(self, x0s, xrefs, sigmas, num_agents):
        agent_list = []
        for i in range(num_agents):
            agent_list.append(UnicycleRobotUncertain(x0s[i], xrefs[i]))
            agent_list[i].set_uncertainty_params(sigmas[i])
        return agent_list
        
    def initialize_dynamics(self):
        mp_dynamics = MultiAgentDynamics(self.agent_list,self.dt, self.HORIZON, self.ref_cost_threshold, self.prob)
        return mp_dynamics
    
    def get_mu(self):
        mu = np.array([[1.0]*(self.num_agents)]*(self.num_agents))*self.initial_mu
        return mu

    def solve(self, plot = True):

        x_traj = [[] for _ in range(self.mp_dynamics.num_agents)]
        y_traj = [[] for _ in range(self.mp_dynamics.num_agents)]
        headings = [[] for _ in range(self.mp_dynamics.num_agents)]

        x_traj_real = [[] for _ in range(self.mp_dynamics.num_agents)]
        y_traj_real =  [[] for _ in range(self.mp_dynamics.num_agents)]
        headings_real = [[] for _ in range(self.mp_dynamics.num_agents)]

        vr = [[] for _ in range(self.mp_dynamics.num_agents)]
        vl = [[] for _ in range(self.mp_dynamics.num_agents)]

        ls = []
        Qs = []



        Rs = self.mp_dynamics.get_control_cost_matrix()

        total_time_steps = 0
        flag = 0

        last_points = None
        current_points = None

        u1 = [[0]*self.mp_dynamics.TIMESTEPS for agent in self.mp_dynamics.agent_list]
        u2 = [[0]*self.mp_dynamics.TIMESTEPS for agent in self.mp_dynamics.agent_list]
        xs = [[0]*self.mp_dynamics.TIMESTEPS for agent in self.mp_dynamics.agent_list]




        prev_control_inputs = np.zeros((self.mp_dynamics.num_agents, self.mp_dynamics.TIMESTEPS, 2))
        control_inputs = np.zeros((self.mp_dynamics.num_agents, self.mp_dynamics.TIMESTEPS, 2))
        total_costs = []

        
        Gs = np.empty((self.mp_dynamics.num_agents, self.mp_dynamics.TIMESTEPS, self.mp_dynamics.num_agents-1, 12), dtype=object)
        qs = np.empty((self.mp_dynamics.num_agents, self.mp_dynamics.TIMESTEPS, self.mp_dynamics.num_agents-1), dtype=object)
        rhos = np.empty((self.mp_dynamics.num_agents, self.mp_dynamics.TIMESTEPS, self.mp_dynamics.num_agents-1), dtype=object)

        # make Gs all zeros
        for i in range(self.mp_dynamics.num_agents):
            for t in range(self.mp_dynamics.TIMESTEPS):
                for j in range(self.mp_dynamics.num_agents-1):
                    Gs[i][t][j] = np.zeros((1, 12))
                    qs[i][t][j] = np.zeros(1)
                    rhos[i][t][j] = 0.0


        identity_size = 4

        # Create a block diagonal matrix of size 4 * num_agents by 4 * TIMESTEPS 

        sigmas_block_diag = block_diag(*[np.zeros((identity_size, identity_size))for _ in range(self.mp_dynamics.num_agents)])

        sigmas = np.array([sigmas_block_diag for _ in range(self.mp_dynamics.TIMESTEPS)])

        # define sigmas based on the sigma values of the robots only for the first state
        for i in range(self.mp_dynamics.num_agents):
            sigmas[0][i*identity_size:(i+1)*identity_size, i*identity_size:(i+1)*identity_size] = np.diag([sigma for sigma in self.mp_dynamics.agent_list[i].uncertainty_params])


        prox_cost_list = [[] for _ in range(len(self.mp_dynamics.agent_list))]
        for i in range(len(self.mp_dynamics.agent_list)):
            for j in range(len(self.mp_dynamics.agent_list)):
                if i != j:
                    prox_cost_list[i].append(ProximityCost(self.d_safe, i, j, 340.0))


        for i in range(self.mp_dynamics.num_agents):
            for t in range(self.mp_dynamics.TIMESTEPS):
                u1[i][t] = 0.0
                u2[i][t] = 0.0


        xs = self.mp_dynamics.integrate_dynamics_for_initial_mp(u1, u2, self.dt, True)
        current_points = xs
        last_points = xs

        Acs, Bcs, As, Bs = self.mp_dynamics.get_linearized_dynamics_for_initial_state(xs,u1,u2)

        for ii in range(self.mp_dynamics.TIMESTEPS - 1):
            sigmas[ii + 1] = Acs[ii] @ sigmas[ii] @ Acs[ii].T

        Gs, qs, rhos = self.mp_dynamics.get_Gs(xs, prox_cost_list, sigmas)

        for i in range(self.mp_dynamics.num_agents):
            for t in range(self.mp_dynamics.TIMESTEPS):
                prev_control_inputs[i][t] = [u1[i][t], u2[i][t]]

        lambdas = np.zeros((self.mp_dynamics.num_agents, self.mp_dynamics.num_agents))
        lambda_next = np.zeros((self.mp_dynamics.num_agents, self.mp_dynamics.num_agents))
        lambda_hats = np.zeros((self.mp_dynamics.num_agents, self.mp_dynamics.num_agents))
        Is = np.zeros((self.mp_dynamics.num_agents, self.mp_dynamics.num_agents-1))
        Is = self.mu.copy()

        # initialize the Ps with shape 3,50,2,12

        Ps = np.zeros((self.mp_dynamics.num_agents, self.mp_dynamics.TIMESTEPS, 2, self.mp_dynamics.num_agents*4))

        # initialize the alphas with shape 3,50,2
        alphas = np.zeros((self.mp_dynamics.num_agents, self.mp_dynamics.TIMESTEPS, 2))


        max_error = 100

        total_costs = []
        total_ref_costs = []
        total_prox_costs = []
        total_wall_costs = []
        total_input_costs = []


        plt.ion()
        fig, ax = plt.subplots()
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.grid(True)
        colors = ['ro', 'go', 'bo', 'co', 'mo', 'yo']

        t = 1
        total_iter = 1
        outer_iter = 1

        try:
            while(max_error > self.TOL_CC_ERROR):
                # define errors list as agent * agent-1 as list
                errors = [[[] for _ in range(self.mp_dynamics.num_agents)] for _ in range(self.mp_dynamics.num_agents)]
                if Gs[0][0][0] is not None:
                    for i, robot in enumerate(self.mp_dynamics.agent_list):
                        for j in range(self.mp_dynamics.TIMESTEPS):
                            for k in range(self.mp_dynamics.num_agents):
                                concatenated_states = np.concatenate([state[j] for state in xs])
                                if k != self.mp_dynamics.num_agents - 1:
                                    error = (Gs[i][j][k] @concatenated_states + qs[i][j][k] + rhos[i][j][k])
                                else:
                                    error = self.costs[i][0].subsystem_cost_functions[-1].evaluate(concatenated_states, control_inputs[i][j], lam = 1, flag = False)
                                errors[i][k].append(error)
                    max_errors = np.float32(np.max(np.array(errors), 2))
                    max_error = np.max(max_errors)
                print("Constraint Violation: ", max_error)
                total_iter += 1

                t_next = (1+np.sqrt(1+4*t**2))/2
                print("t: ", t_next)


                for i in range(self.mp_dynamics.num_agents):
                    for j in range(self.mp_dynamics.num_agents):
                        if j != self.mp_dynamics.num_agents - 1:
                            lambda_next[i][j] = max(0, lambda_hats[i][j] + (2/np.sqrt(outer_iter)) * (self.prob - max_errors[i][j]))
                            lambda_hats[i][j] = lambda_next[i][j] + ((t-1)/t_next)*(lambda_next[i][j] - lambdas[i][j])
                            lambdas[i][j] = lambda_next[i][j]
                        else:
                            lambda_next[i][j] = max(0, lambda_hats[i][j] + (2/np.sqrt(outer_iter)) * (max_errors[i][j]))
                            lambda_hats[i][j] = lambda_next[i][j] + ((t-1)/t_next)*(lambda_next[i][j] - lambdas[i][j])
                            lambdas[i][j] = lambda_next[i][j]
                            
                        Is[i][j] = 0 if (self.prob - max_error < 0.0)&(lambdas[i][j] == 0) else 0.005

                print("lamdas: ", lambdas)

                t = t_next

                for i in range(self.mp_dynamics.num_agents):
                    for j in range(self.mp_dynamics.num_agents):
                        self.mu[i][j] *= self.phi

                flag = 0
                # total_time_steps = 0
                # print("New Mu Values: ", self.mu)

                outer_iter += 1
                print(f'Outer Iteration {outer_iter}')

                while (flag == 0):

                    start = time.time()
                    errors = [[[] for _ in range(self.mp_dynamics.num_agents)] for _ in range(self.mp_dynamics.num_agents)]
                    if Gs[0][0][0] is not None:
                        for i, robot in enumerate(self.mp_dynamics.agent_list):
                            for j in range(self.mp_dynamics.TIMESTEPS):
                                for k in range(self.mp_dynamics.num_agents):
                                    concatenated_states = np.concatenate([state[j] for state in xs])
                                    if k != self.mp_dynamics.num_agents - 1:
                                        error = (Gs[i][j][k]@concatenated_states + qs[i][j][k] + rhos[i][j][k])
                                    else:
                                        error = self.costs[i][0].subsystem_cost_functions[-1].evaluate(concatenated_states, control_inputs[i][j], lam = 1, flag = False)
                                    errors[i][k].append(error)
                        max_errors = np.float32(np.max(np.array(errors), 2))
                        max_error = np.max(max_errors)
                        
                    # print('Max Error:', max_error)
                    
                    # integrate the dynamics
                    
                    xs, control_inputs = self.mp_dynamics.compute_op_point(Ps, alphas, current_points, prev_control_inputs, 0.02 , False)

                    ax.clear()
                    ax.grid(True)
                    ax.set_xlim(-4, 4)
                    ax.set_ylim(-4, 4)

                    # get the first elements of xs

                    for i in range(self.mp_dynamics.num_agents):
                        ax.plot([x[0] for x in xs[i]], [x[1] for x in xs[i]], colors[i], label=f'Robot {i}', markersize=5)

                    plt.pause(0.01)
                    time.sleep(0.01)
                    plt.show()

                    u1 = control_inputs[:,:,0]
                    u2 = control_inputs[:,:,1]

                    xs_real = self.mp_dynamics.integrate_dynamics_for_initial_mp(u1, u2, self.dt, True)

                    last_points = current_points
                    current_points = xs

                    # get the linearized dynamics
                    Acs, Bcs, As, Bs = self.mp_dynamics.get_linearized_dynamics_for_initial_state(xs,u1,u2)

                    for ii in range(self.mp_dynamics.TIMESTEPS-1):
                        sigmas[ii + 1] = Acs[ii] @ sigmas[ii] @ Acs[ii].T 

                    # get the linearized constraint matrices
                    Gs, qs, rhos = self.mp_dynamics.get_Gs(xs, prox_cost_list, sigmas)

                    Qs = [[] for _ in range(self.mp_dynamics.num_agents)]
                    ls = [[] for _ in range(self.mp_dynamics.num_agents)]
                    Rs = [[[] for _ in range(self.mp_dynamics.num_agents)] for _ in range(self.mp_dynamics.num_agents)]

                    # Iterate over timesteps
                    total_costs.append([])
                    total_ref_costs.append([])
                    total_prox_costs.append([])
                    total_wall_costs.append([])
                    total_input_costs.append([])

                    for ii in range(self.mp_dynamics.TIMESTEPS):
                        concatenated_states = np.concatenate([state[ii] for state in xs])
                        hessian_list = []
                        gradient_list = []
                        for i, robot in enumerate(self.mp_dynamics.agent_list):
                            # Calculate the Hessian and gradient for each constraint
                            hessian_list = []
                            gradient_list = []
                            for j in range(self.mp_dynamics.num_agents-1):
                                gradient_x_0 = self.costs[i][0].gradient_x(concatenated_states, control_inputs[i][ii], Gs[i][ii][j], qs[i][ii][j], rhos[i][ii][j], lambdas[i][j], Is[i][j], timestep=ii)
                                hessian_x_0 = self.costs[i][0].hessian_manual(concatenated_states, control_inputs[i][ii], Gs[i][ii][j], qs[i][ii][j], rhos[i][ii][j], lambdas[i][j], Is[i][j], timestep=ii)
                                hessian_list.append(hessian_x_0)
                                gradient_list.append(gradient_x_0)
                            
                            hessian_u = self.costs[i][0].hessian_u(concatenated_states, control_inputs[i][ii])

                            # Add up the Hessian matrices element-wise
                            hessian_x_sum = sum(hessian_list)
                            gradient_x_sum = sum(gradient_list)

                            # Append the summed Hessian matrix to Qs
                            Qs[i].append(hessian_x_sum)

                            # Append gradients, costs, etc. as before
                            ls[i].append(gradient_x_sum)
                            Rs[i][i].append(hessian_u)
                            total_costs[total_time_steps].append(self.costs[i][0].evaluate(concatenated_states, control_inputs[i][ii], Gs[i][ii][0], qs[i][ii][0], rhos[i][ii][0], lambdas[i][0], Is[i][0]))
                            for cost in self.costs[i][0].subsystem_cost_functions:
                                if isinstance(cost, SpeedCost):
                                    total_costs[total_time_steps].append(cost.evaluate(concatenated_states, control_inputs[i][ii],lambdas[i][k]))
                                if isinstance(cost, ReferenceCost):
                                    total_ref_costs[total_time_steps].append(cost.evaluate(concatenated_states, control_inputs[i][ii]))
                                if isinstance(cost, WallCost):
                                    total_wall_costs[total_time_steps].append(cost.evaluate(concatenated_states, control_inputs[i][ii]))
                                if isinstance(cost, InputCost):
                                    total_input_costs[total_time_steps].append(cost.evaluate(concatenated_states, control_inputs[i][ii]))
                                if isinstance(cost, ProximityCostUncertainLinear):
                                    for k in range(len(Gs[i][ii])):
                                        total_prox_costs[total_time_steps].append(cost.evaluate(concatenated_states, Gs[i][ii][k], qs[i][ii][k], rhos[i][ii][k], lambdas[i][k]))
                                if isinstance(cost, ProximityCostUncertainQuad):
                                    for k in range(len(Gs[i][ii])):
                                        total_prox_costs[total_time_steps].append(cost.evaluate(concatenated_states, Gs[i][ii][k], qs[i][ii][k], rhos[i][ii][k], Is[i][k]))
                            

                    # sum the costs 
                    for i in range(self.mp_dynamics.num_agents):
                        for j in range(self.mp_dynamics.num_agents):
                            if i != j:
                                Rs[i][j] = [np.zeros((2, 2)) for _ in range(self.mp_dynamics.TIMESTEPS)]       

                    total_costs[total_time_steps] = sum(total_costs[total_time_steps])
                    total_prox_costs[total_time_steps] = sum(total_prox_costs[total_time_steps])
                    total_ref_costs[total_time_steps] = sum(total_ref_costs[total_time_steps])
                    total_input_costs[total_time_steps] = sum(total_input_costs[total_time_steps])
                    total_wall_costs[total_time_steps] = sum(total_wall_costs[total_time_steps])


                    Ps, alphas = solve_lq_game(As, Bs, Qs, ls, Rs)

                    prev_control_inputs = control_inputs
                    
                    
                    if total_time_steps > 0:
                        flag = self.mp_dynamics.check_convergence(current_points, last_points)                

                    total_time_steps += 1
                    # print the iteration with text
                    # print(f'Iteration {total_time_steps}')
                    
                    end = time.time()
                    #print(f'Time: {end - start}')

        except KeyboardInterrupt:
            for ii in range(self.mp_dynamics.TIMESTEPS):
                for i, agent in enumerate(self.mp_dynamics.agent_list):
                    x_traj[i].append(xs[i][ii][0])
                    y_traj[i].append(xs[i][ii][1])
                    headings[i].append(xs[i][ii][2])

                    x_traj_real[i].append(xs_real[i][ii][0])
                    y_traj_real[i].append(xs_real[i][ii][1])
                    headings_real[i].append(xs_real[i][ii][2])

            vr, vl = self.mp_dynamics.compute_wheel_speeds(u1, u2)
            for ii in range(len(total_costs)):
                if type(total_costs[ii]) is list: 
                    total_costs[ii] = sum(total_costs[ii])
                    total_prox_costs[ii] = sum(total_prox_costs[ii])
                    total_ref_costs[ii] = sum(total_ref_costs[ii])
                    total_input_costs[ii] = sum(total_input_costs[ii])
                    total_wall_costs[ii] = sum(total_wall_costs[ii])

        plt.ioff()
        plt.close()

        for ii in range(self.mp_dynamics.TIMESTEPS):
            for i, agent in enumerate(self.mp_dynamics.agent_list):
                x_traj[i].append(xs[i][ii][0])
                y_traj[i].append(xs[i][ii][1])
                headings[i].append(xs[i][ii][2])

                x_traj_real[i].append(xs_real[i][ii][0])
                y_traj_real[i].append(xs_real[i][ii][1])
                headings_real[i].append(xs_real[i][ii][2])

        vr, vl = self.mp_dynamics.compute_wheel_speeds(u1, u2)

            
        # plot costs
        plt.figure()
        # plt.plot(total_costs)
        # plt.plot(total_prox_costs)
        plt.plot(total_ref_costs + total_input_costs + total_prox_costs)
        print("total cost is:", (total_ref_costs +  total_input_costs + total_prox_costs)[-1])
        # plt.plot(total_input_costs)
        # plt.plot(total_wall_costs)

        plt.legend(['Total Cost'])
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Costs over Iterations')
        plt.show()

        
        if plot:
            plt.ion()
            fig, ax = plt.subplots()
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            ax.grid(True)
            colors = ['ro', 'go', 'bo', 'co', 'mo', 'yo']
            
            for kk in range(self.mp_dynamics.TIMESTEPS):    
                ax.clear()
                ax.grid(True)
                ax.set_xlim(-4, 4)
                ax.set_ylim(-4, 4)

                for i in range(self.mp_dynamics.num_agents):
                    ax.plot(x_traj[i][kk], y_traj[i][kk], colors[i], label=f'Robot {i}', markersize=15)
                    ax.arrow(x_traj[i][kk], y_traj[i][kk], 0.2 * np.cos(headings[i][kk]), 0.2 * np.sin(headings[i][kk]), head_width=0.02)

                plt.pause(0.01)
                time.sleep(0.01)
                plt.show()
                
            plt.ioff()


            plt.figure()
            for i in range(self.mp_dynamics.num_agents):
                plt.plot(x_traj[i], y_traj[i], colors[i],  label=f'Robot {i}', linestyle='solid', markersize=3)
                plt.plot(x_traj_real[i], y_traj_real[i], colors[i],  label=f'Robot {i} Real', linestyle='dashed', markersize=3)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('State Trajectories')
            plt.legend()
            plt.show()


        # # plot wheel speeds in subplots
        # plt.figure()
        # plt.subplot(2,1,1)
        # for i in range(self.mp_dynamics.num_agents):
        #     plt.plot(vr[i], label=f'Robot {i}')
        # plt.xlabel('Time Step')
        # plt.ylabel('Right Wheel Speed')
        # plt.title('Right Wheel Speeds')
        # plt.legend()

        # plt.subplots_adjust(hspace=0.5)

        '''plt.subplot(2,1,2)
        for i in range(self.mp_dynamics.num_agents):
            plt.plot(vl[i], label=f'Robot {i}')
        plt.xlabel('Time Step')
        plt.ylabel('Left Wheel Speed')
        plt.title('Left Wheel Speeds')
        plt.legend()
        plt.show()'''

        # create a list without the velocities and with the shape num of agents x TIMESTEPS x 3 
        xs_wo_vel = np.array(xs)[:,:,0:3]

        time_list = np.arange(0, self.HORIZON + self.dt, self.dt)
       

        return xs_wo_vel, time_list, xs, xs_real

def main(initial_positions, reference_positions):
    # Define the configuration dictionary
    sigma_value = 0.005
    num_elements = 4
    num_agents = 3

    total_violations = 0

    for i in range(1):

        config = {
            'dt': 0.6,
            'HORIZON': 30,
            'initial_vels': [0.0, 0.0, 0.0],
            'x0s': initial_positions,
            'xrefs': reference_positions,
            'sigmas': [[sigma_value] * num_elements for _ in range(num_agents)],
            'num_agents': 3,
            'ref_cost_threshold': 25,
            'prob': 0.99,
            'TOL_CC_ERROR': 0.00001,
            'initial_mu': 0.005,
            'phi':2,
            'd_safe': 0.5
            }

        # Create an instance of the CCILQGame class
        game = CCILQGame(config)
        # Solve the game
        traj, times, xs, xs_real = game.solve(plot = True)
        
        violation = False
        for i in range(len(xs)):
            for j in range(len(xs[i])):
                if np.abs(xs[i][j][3]) > 0.6:
                    violation = True
                    print(f'Robot {i} has speed {xs[i][j][3]} at time {times[j]}')
                for k in range(len(xs)):
                    if k != i:
                        distance = np.sqrt((xs_real[i][j][0] - xs_real[k][j][0])**2 + (xs_real[i][j][1] - xs_real[k][j][1])**2)
                        if distance < config['d_safe']:
                            violation = True
                            print(f'Robots {i} and {k} are too close ({distance}) at time {times[j]}')

        if not violation:
            print('No speed or distance violation detected')
        else:
            total_violations += 1

    print(f'Violation count: {total_violations}')


def on_click(event):
    # Ensure click is within the plot area
    if event.inaxes is not None:
        # Determine where to add the point (initial or reference or heading)
        if len(initial_positions) < num_agents:
            # Record initial positions for robots
            initial_positions.append([event.xdata, event.ydata])
            # Plot the initial position (robots)
            plt.scatter(event.xdata, event.ydata, color='blue', label='Initial Positions' if len(initial_positions) == 1 else "")
            plt.text(event.xdata + 0.1, event.ydata, f'R{len(initial_positions)}', fontsize=12, color='blue')
            plt.draw()  # Redraw the plot with the new point
        elif len(reference_positions) < num_agents:
            # Record reference positions for robots
            reference_positions.append([event.xdata, event.ydata])
            # Plot the reference position (targets)
            plt.scatter(event.xdata, event.ydata, color='red', label='Reference Positions' if len(reference_positions) == 1 else "")
            plt.text(event.xdata + 0.1, event.ydata, f'T{len(reference_positions)}', fontsize=12, color='red')
            plt.draw()  # Redraw the plot with the new point
        elif len(initial_headings) < num_agents:
            # Record initial heading for robots by determining the direction
            heading_x = event.xdata - initial_positions[len(initial_headings)][0]
            heading_y = event.ydata - initial_positions[len(initial_headings)][1]
            heading_angle = np.arctan2(heading_y, heading_x)  # Calculate the angle (theta_0)
            initial_headings.append(heading_angle)
            # Plot the initial heading (direction) of the robot using an arrow
            plt.arrow(initial_positions[len(initial_headings) - 1][0], initial_positions[len(initial_headings) - 1][1], 
                      0.5 * np.cos(heading_angle), 0.5 * np.sin(heading_angle), 
                      head_width=0.1, head_length=0.1, fc='blue', ec='blue')
            plt.text(initial_positions[len(initial_headings) - 1][0] + 0.1, 
                     initial_positions[len(initial_headings) - 1][1] + 0.1, f'Heading {len(initial_headings)}', 
                     fontsize=12, color='blue')
            plt.draw()  # Redraw the plot with the heading arrow
        elif len(reference_headings) < num_agents:
            # Record reference heading for robots by determining the direction
            heading_x = event.xdata - reference_positions[len(reference_headings)][0]
            heading_y = event.ydata - reference_positions[len(reference_headings)][1]
            heading_angle = np.arctan2(heading_y, heading_x)  # Calculate the angle (theta_ref)
            reference_headings.append(heading_angle)
            # Plot the reference heading (direction) of the robot using an arrow
            plt.arrow(reference_positions[len(reference_headings) - 1][0], reference_positions[len(reference_headings) - 1][1], 
                      0.5 * np.cos(heading_angle), 0.5 * np.sin(heading_angle), 
                      head_width=0.1, head_length=0.1, fc='red', ec='red')
            plt.text(reference_positions[len(reference_headings) - 1][0] + 0.1, 
                     reference_positions[len(reference_headings) - 1][1] + 0.1, f'Ref Heading {len(reference_headings)}', 
                     fontsize=12, color='red')
            plt.draw()  # Redraw the plot with the heading arrow

        # If all positions and headings are selected, stop taking input
        if len(initial_positions) == num_agents and len(reference_positions) == num_agents and len(initial_headings) == num_agents and len(reference_headings) == num_agents:
            print("Initial Positions, Reference Positions, Initial Headings and Reference Headings selected.")
            plt.legend(loc='upper left')


if __name__ == "__main__":

    # initial_positions = []
    # reference_positions = []
    # initial_headings = []
    # reference_headings = []
    # num_agents = 3

    # # Set up the plot
    # fig, ax = plt.subplots(figsize=(8, 6))
    # ax.set_xlim(-4, 4)
    # ax.set_ylim(-4, 4)
    # ax.set_xlabel('X Position')
    # ax.set_ylabel('Y Position')
    # ax.set_title('Click to Set Initial and Reference Positions')

    # # Add the event listener for mouse clicks
    # fig.canvas.mpl_connect('button_press_event', on_click)

    # # Show the plot and wait for clicks
    # plt.grid(True)
    # plt.show()

    # # merge init positions and headings
    # for i in range(len(initial_positions)):
    #     initial_positions[i].append(initial_headings[i])
    
    # for i in range(len(reference_positions)):
    #     reference_positions[i].append(reference_headings[i])
    #     reference_positions[i].append(0.0)


    # print("Initial Positions (x0s) and Headings:", initial_positions)
    # print("Reference Positions (xrefs) and Headings:", reference_positions)


    first_init = [[-2, 2, 0], [-2, 3, 0], [-2, 0, 0]]
    first_reference = [[3, 3, 0,0], [3, 0, 0,0], [3, -2, 0, 0]]

    second_init = [[-1, 3.5, -np.pi/2], [2, 3, -np.pi], [-2, 1, 0]]
    second_reference = [[-1, -2, -np.pi/2,0], [-3, 2, -np.pi,0], [2, 3, 0, 0]]

    third_init = [[3, 0, -np.pi], [0, 3, -np.pi/2], [0, -3, np.pi/2]]
    third_reference = [[-3, 0, -np.pi,0], [-1, -3, -np.pi/2,0], [-1, 3, np.pi/2, 0]]

    fourth_init = [[2, 1, -np.pi], [1, 3, -np.pi/2], [1, -3, np.pi/2]]
    fourth_reference = [[-3, 0, -np.pi,0], [-1, -3, -np.pi/2,0], [1, 3, np.pi/2, 0]]

    main(second_init, second_reference)