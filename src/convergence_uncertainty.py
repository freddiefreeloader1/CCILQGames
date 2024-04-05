import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.linalg import block_diag

from solve_lq_problem import solve_lq_game
from Diff_robot_uncertainty import UnicycleRobotUncertain
from Costs import ProximityCost, OverallCost, ReferenceCost, WallCost, ProximityCostUncertainLinear , ProximityCostUncertainQuad, InputCost
from MultiAgentDynamics import MultiAgentDynamics

dt = 0.2
HORIZON = 10
TIMESTEPS = int(HORIZON / dt)
scenerio = "overtaking"

if scenerio == "intersection":   # introduce ref cost after 20th timestep
    x0_1 = [-2.0, -2.0, 0.0, 1.0]
    x0_2 = [-2.0, 2.0, 0.0, 1.0]
    x0_3 = [0.0, 4.0, -np.pi/2, 3.0]
    x0_4 = [-2.0, 3.0, 0.0, 0.0]
    x0_5 = [0.0, 1.0, 0.0, 0.0]
    x0_6 = [1.0, 0.0, 0.0, 0.0]


    x_ref_1 = np.array([3, -2, 0, 0])
    x_ref_2 = np.array([3, 2, 0, 0])
    x_ref_3 = np.array([0, -3, 0, 0])
    x_ref_4 = np.array([2, 0, 0, 0])
    x_ref_5 = np.array([-2, 0, 0, 0])
    x_ref_6 = np.array([0, -1, 0, 0])

    ref_cost_threshold = 20

if scenerio == "overtaking":  # introduce ref cost after 35th timestep
    x0_1 = [-3.0, -1.0, 0, 0.0]
    x0_2 = [-3.1, 1.0, 0, 0.0]
    x0_3 = [-3.0, 0.0, 0, 0.0]
    x0_4 = [-2.0, 3.0, 0.0, 0.0]
    x0_5 = [0.0, 1.0, 0.0, 0.0]
    x0_6 = [1.0, 0.0, 0.0, 0.0]


    x_ref_1 = np.array([2, 1, 0, 0])
    x_ref_2 = np.array([2, -1, 0, 0])
    x_ref_3 = np.array([3, 0, 0, 0])
    x_ref_4 = np.array([2, 0, 0, 0])
    x_ref_5 = np.array([-2, 0, 0, 0])
    x_ref_6 = np.array([0, -1, 0, 0])

    ref_cost_threshold = 35

if scenerio == "line":   # introduce ref cost after 20th timestep
    x0_1 = [-1.0, -1.0, 0, 0]
    x0_2 = [-3.1, 2.0, 0, 0]
    x0_3 = [3.0, -1.0, 0, 0]
    x0_4 = [-2.0, 3.0, 0.0, 0.0]
    x0_5 = [0.0, 1.0, 0.0, 0.0]
    x0_6 = [1.0, 0.0, 0.0, 0.0]


    x_ref_1 = np.array([0, 3, 0, 0])
    x_ref_2 = np.array([0, -3, 0, 0])
    x_ref_3 = np.array([0, 0, 0, 0])
    x_ref_4 = np.array([2, 0, 0, 0])
    x_ref_5 = np.array([-2, 0, 0, 0])
    x_ref_6 = np.array([0, -1, 0, 0])

    ref_cost_threshold = 20

if scenerio == "arrow":
    x0_1 = [-1.0, -1.0, 0, 0]
    x0_2 = [-3.1, 2.0, 0, 0]
    x0_3 = [3.0, -1.0, 0, 0]
    x0_4 = [-2.0, 3.0, 0.0, 0.0]
    x0_5 = [0.0, 1.0, 0.0, 0.0]
    x0_6 = [1.0, 0.0, 0.0, 0.0]


    x_ref_1 = np.array([0, 3, 0, 0])
    x_ref_2 = np.array([0, 1, 0, 0])
    x_ref_3 = np.array([0, -1, 0, 0])
    x_ref_4 = np.array([0, -3, 0, 0])
    x_ref_5 = np.array([2, 1, 0, 0])
    x_ref_6 = np.array([-2, 1, 0, 0])

    ref_cost_threshold = 20


sigma1 = [0.1, 0.1, 0.1, 0.1]
sigma2 = [0.1, 0.1, 0.1, 0.1]
sigma3 = [0.1, 0.1, 0.1, 0.1]
sigma4 = [0.1, 0.1, 0.1, 0.1]
sigma5 = [0.1, 0.1, 0.1, 0.1]
sigma6 = [0.1, 0.1, 0.1, 0.1]


robot1 = UnicycleRobotUncertain(x0_1, x_ref_1, dt)
robot1.set_uncertainty_params(sigma1)
robot2 = UnicycleRobotUncertain(x0_2, x_ref_2, dt)
robot2.set_uncertainty_params(sigma2)
robot3 = UnicycleRobotUncertain(x0_3, x_ref_3, dt)
robot3.set_uncertainty_params(sigma3)


robot4 = UnicycleRobotUncertain(x0_4, x_ref_4, dt)
robot4.set_uncertainty_params(sigma4)
robot5 = UnicycleRobotUncertain(x0_5, x_ref_5, dt)
robot5.set_uncertainty_params(sigma5)
robot6 = UnicycleRobotUncertain(x0_6, x_ref_6, dt)
robot6.set_uncertainty_params(sigma6)


prob = 0.95
# mp_dynamics = MultiAgentDynamics([robot1, robot2, robot3, robot4, robot5, robot6], dt, HORIZON)
mp_dynamics = MultiAgentDynamics([robot1, robot2, robot3], dt, HORIZON, ref_cost_threshold, prob)


costs = mp_dynamics.define_costs_lists(uncertainty=True)

x_traj = [[] for _ in range(mp_dynamics.num_agents)]
y_traj = [[] for _ in range(mp_dynamics.num_agents)]
headings = [[] for _ in range(mp_dynamics.num_agents)]
vr = [[] for _ in range(mp_dynamics.num_agents)]
vl = [[] for _ in range(mp_dynamics.num_agents)]

ls = []
Qs = []



Rs = mp_dynamics.get_control_cost_matrix()

total_time_steps = 0
flag = 0

last_points = None
current_points = None

u1 = [[0]*mp_dynamics.TIMESTEPS for agent in mp_dynamics.agent_list]
u2 = [[0]*mp_dynamics.TIMESTEPS for agent in mp_dynamics.agent_list]
xs = [[0]*mp_dynamics.TIMESTEPS for agent in mp_dynamics.agent_list]




prev_control_inputs = np.zeros((mp_dynamics.num_agents, mp_dynamics.TIMESTEPS, 2))
control_inputs = np.zeros((mp_dynamics.num_agents, mp_dynamics.TIMESTEPS, 2))
total_costs = []

mu = np.array([[1]*(mp_dynamics.num_agents-1)]*mp_dynamics.num_agents)*0.005
phi = 2

Gs = np.empty((mp_dynamics.num_agents, mp_dynamics.TIMESTEPS, mp_dynamics.num_agents-1, 12), dtype=object)
qs = np.empty((mp_dynamics.num_agents, mp_dynamics.TIMESTEPS, mp_dynamics.num_agents-1), dtype=object)
rhos = np.empty((mp_dynamics.num_agents, mp_dynamics.TIMESTEPS, mp_dynamics.num_agents-1), dtype=object)

# make Gs all zeros
for i in range(mp_dynamics.num_agents):
    for t in range(mp_dynamics.TIMESTEPS):
        for j in range(mp_dynamics.num_agents-1):
            Gs[i][t][j] = np.zeros((1, 12))
            qs[i][t][j] = np.zeros(1)
            rhos[i][t][j] = 0.0


identity_size = 4

# Create a block diagonal matrix of size 4 * num_agents by 4 * TIMESTEPS 

sigmas_block_diag = block_diag(*[np.zeros((identity_size, identity_size))for _ in range(mp_dynamics.num_agents)])

sigmas = np.array([sigmas_block_diag for _ in range(mp_dynamics.TIMESTEPS)])

# define sigmas based on the sigma values of the robots only for the first state
for i in range(mp_dynamics.num_agents):
    sigmas[0][i*identity_size:(i+1)*identity_size, i*identity_size:(i+1)*identity_size] = np.diag([sigma for sigma in mp_dynamics.agent_list[i].uncertainty_params])


prox_cost_list = [[] for _ in range(len(mp_dynamics.agent_list))]
for i in range(len(mp_dynamics.agent_list)):
    for j in range(len(mp_dynamics.agent_list)):
        if i != j:
            prox_cost_list[i].append(ProximityCost(1.0, i, j, 200.0))


for i in range(mp_dynamics.num_agents):
    for t in range(mp_dynamics.TIMESTEPS):
        u1[i][t] = 0.0
        u2[i][t] = 0.0


xs = mp_dynamics.integrate_dynamics_for_initial_mp(u1, u2, mp_dynamics.dt, True)
current_points = xs
last_points = xs

Acs, Bcs, As, Bs = mp_dynamics.get_linearized_dynamics_for_initial_state(xs,u1,u2)

for ii in range(mp_dynamics.TIMESTEPS - 1):
    sigmas[ii + 1] = Acs[ii] @ sigmas[ii] @ Acs[ii].T

Gs, qs, rhos = mp_dynamics.get_Gs(xs, prox_cost_list, sigmas)

for i in range(mp_dynamics.num_agents):
    for t in range(mp_dynamics.TIMESTEPS):
        prev_control_inputs[i][t] = [u1[i][t], u2[i][t]]

lambdas = np.zeros((mp_dynamics.num_agents, mp_dynamics.num_agents-1))
Is = np.zeros((mp_dynamics.num_agents, mp_dynamics.num_agents-1))
Is = mu

# initialize the Ps with shape 3,50,2,12

Ps = np.zeros((mp_dynamics.num_agents, mp_dynamics.TIMESTEPS, 2, mp_dynamics.num_agents*4))

# initialize the alphas with shape 3,50,2
alphas = np.zeros((mp_dynamics.num_agents, mp_dynamics.TIMESTEPS, 2))


TOL_CC_ERROR = 0.08
max_error = 100

total_costs = []
total_ref_costs = []
total_prox_costs = []
total_wall_costs = []
total_input_costs = []


plt.ion()
fig, ax = plt.subplots()
ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-1.6, 1.6)
ax.grid(True)
colors = ['ro', 'go', 'bo', 'co', 'mo', 'yo']

try:
    while(max_error > TOL_CC_ERROR):
        # define errors list as agent * agent-1 as list
        errors = [[[] for _ in range(mp_dynamics.num_agents-1)] for _ in range(mp_dynamics.num_agents)]
        if Gs[0][0][0] is not None:
            for i, robot in enumerate(mp_dynamics.agent_list):
                for j in range(mp_dynamics.TIMESTEPS):
                    for k in range(mp_dynamics.num_agents-1):
                        concatenated_states = np.concatenate([state[j] for state in xs])
                        error = (Gs[i][j][k]@concatenated_states + qs[i][j][k] + rhos[i][j][k])
                        errors[i][k].append(error)
            max_errors = np.float32(np.max(np.array(errors), 2))
            max_error = np.max(max_errors)
        print(max_error)
        for i in range(mp_dynamics.num_agents):
            for j in range(mp_dynamics.num_agents-1):
                lambdas[i][j] = max(0,lambdas[i][j] + mu[i][j] * np.abs((prob - max_errors[i][j])))
                Is[i][j] = 0 if (prob - max_error < 0.0)&(lambdas[i][j] == 0) else mu[i][j]

        for i in range(mp_dynamics.num_agents):
            for j in range(mp_dynamics.num_agents-1):
                mu[i][j] *= phi
        flag = 0
        # total_time_steps = 0
        print("New Mu Values: ", mu)
        while (flag == 0):

            start = time.time()
            errors = [[[] for _ in range(mp_dynamics.num_agents-1)] for _ in range(mp_dynamics.num_agents)]
            if Gs[0][0][0] is not None:
                for i, robot in enumerate(mp_dynamics.agent_list):
                    for j in range(mp_dynamics.TIMESTEPS):
                        for k in range(mp_dynamics.num_agents-1):
                            concatenated_states = np.concatenate([state[j] for state in xs])
                            error = (Gs[i][j][k]@concatenated_states + qs[i][j][k] + rhos[i][j][k])
                            errors[i][k].append(error)
                max_errors = np.float32(np.max(np.array(errors), 2))
                max_error = np.max(max_errors)
            print('Max Error:', max_error)
            
            # integrate the dynamics
            
            xs, control_inputs = mp_dynamics.compute_op_point(Ps, alphas, current_points, prev_control_inputs, 0.02 , False)

            ax.clear()
            ax.grid(True)
            ax.set_xlim(-3.5, 3.5)
            ax.set_ylim(-1.6, 1.6)

            # get the first elements of xs

            for i in range(mp_dynamics.num_agents):
                ax.plot([x[0] for x in xs[i]], [x[1] for x in xs[i]], colors[i], label=f'Robot {i}', markersize=5)

            plt.pause(0.01)
            time.sleep(0.01)
            plt.show()

            u1 = control_inputs[:,:,0]
            u2 = control_inputs[:,:,1]

            xs_real = mp_dynamics.integrate_dynamics_for_initial_mp(u1, u2, mp_dynamics.dt, True)

            last_points = current_points
            current_points = xs

            # get the linearized dynamics
            Acs, Bcs, As, Bs = mp_dynamics.get_linearized_dynamics_for_initial_state(xs,u1,u2)

            for ii in range(mp_dynamics.TIMESTEPS-1):
                sigmas[ii + 1] = Acs[ii] @ sigmas[ii] @ Acs[ii].T 

            # get the linearized constraint matrices
            Gs, qs, rhos = mp_dynamics.get_Gs(xs, prox_cost_list, sigmas)

            Qs = [[] for _ in range(mp_dynamics.num_agents)]
            ls = [[] for _ in range(mp_dynamics.num_agents)]
            Rs = [[[] for _ in range(mp_dynamics.num_agents)] for _ in range(mp_dynamics.num_agents)]

            # Iterate over timesteps
            total_costs.append([])
            total_ref_costs.append([])
            total_prox_costs.append([])
            total_wall_costs.append([])
            total_input_costs.append([])

            for ii in range(mp_dynamics.TIMESTEPS):
                concatenated_states = np.concatenate([state[ii] for state in xs])
                hessian_list = []
                gradient_list = []
                for i, robot in enumerate(mp_dynamics.agent_list):
                    # Calculate the Hessian and gradient for each constraint
                    hessian_list = []
                    gradient_list = []
                    for j in range(mp_dynamics.num_agents-1):
                        gradient_x_0 = costs[i][0].gradient_x(concatenated_states, control_inputs[i][ii], Gs[i][ii][j], qs[i][ii][j], rhos[i][ii][j], lambdas[i][j], Is[i][j], timestep=ii)
                        hessian_x_0 = costs[i][0].hessian_manual(concatenated_states, control_inputs[i][ii], Gs[i][ii][j], qs[i][ii][j], rhos[i][ii][j], lambdas[i][j], Is[i][j], timestep=ii)
                        hessian_list.append(hessian_x_0)
                        gradient_list.append(gradient_x_0)
                    
                    hessian_u = costs[i][0].hessian_u(concatenated_states, control_inputs[i][ii])

                    # Add up the Hessian matrices element-wise
                    hessian_x_sum = sum(hessian_list)
                    gradient_x_sum = sum(gradient_list)

                    # Append the summed Hessian matrix to Qs
                    Qs[i].append(hessian_x_sum)

                    # Append gradients, costs, etc. as before
                    ls[i].append(gradient_x_sum)
                    Rs[i][i].append(hessian_u)
                    total_costs[total_time_steps].append(costs[i][0].evaluate(concatenated_states, control_inputs[i][ii], Gs[i][ii][0], qs[i][ii][0], rhos[i][ii][0], lambdas[i][0], Is[i][0]))
                    for cost in costs[i][0].subsystem_cost_functions:
                        if isinstance(cost, ReferenceCost):
                            total_ref_costs[total_time_steps].append(cost.evaluate(concatenated_states, control_inputs[i][ii]))
                        if isinstance(cost, WallCost):
                            total_wall_costs[total_time_steps].append(cost.evaluate(concatenated_states, control_inputs[i][ii]))
                        if isinstance(cost, InputCost):
                            total_input_costs[total_time_steps].append(cost.evaluate(concatenated_states, control_inputs[i][ii]))
                        if isinstance(cost, ProximityCostUncertainLinear):
                            total_prox_costs[total_time_steps].append(cost.evaluate(concatenated_states, Gs[i][ii][0], qs[i][ii][0], rhos[i][ii][0], lambdas[i][0]))
                        if isinstance(cost, ProximityCostUncertainQuad):
                            total_prox_costs[total_time_steps].append(cost.evaluate(concatenated_states, Gs[i][ii][1], qs[i][ii][1], rhos[i][ii][1], Is[i][1]))
                    

            # sum the costs 
            for i in range(mp_dynamics.num_agents):
                for j in range(mp_dynamics.num_agents):
                    if i != j:
                        Rs[i][j] = [np.zeros((2, 2)) for _ in range(mp_dynamics.TIMESTEPS)]       

            total_costs[total_time_steps] = sum(total_costs[total_time_steps])
            total_prox_costs[total_time_steps] = sum(total_prox_costs[total_time_steps])
            total_ref_costs[total_time_steps] = sum(total_ref_costs[total_time_steps])
            total_input_costs[total_time_steps] = sum(total_input_costs[total_time_steps])
            total_wall_costs[total_time_steps] = sum(total_wall_costs[total_time_steps])


            Ps, alphas = solve_lq_game(As, Bs, Qs, ls, Rs)

            prev_control_inputs = control_inputs
            
            
            if total_time_steps > 0:
                flag = mp_dynamics.check_convergence(current_points, last_points)                

            total_time_steps += 1
            # print the iteration with text
            print(f'Iteration {total_time_steps}')
            
            end = time.time()
            print(f'Time: {end - start}')

except KeyboardInterrupt:
    for ii in range(mp_dynamics.TIMESTEPS):
        for i, agent in enumerate(mp_dynamics.agent_list):
            x_traj[i].append(xs_real[i][ii][0])
            y_traj[i].append(xs_real[i][ii][1])
            headings[i].append(xs_real[i][ii][2])
    vr, vl = mp_dynamics.compute_wheel_speeds(u1, u2)
    for ii in range(len(total_costs)):
        if type(total_costs[ii]) is list: 
            total_costs[ii] = sum(total_costs[ii])
            total_prox_costs[ii] = sum(total_prox_costs[ii])
            total_ref_costs[ii] = sum(total_ref_costs[ii])
            total_input_costs[ii] = sum(total_input_costs[ii])
            total_wall_costs[ii] = sum(total_wall_costs[ii])

plt.ioff()
plt.close()

for ii in range(mp_dynamics.TIMESTEPS):
    for i, agent in enumerate(mp_dynamics.agent_list):
        x_traj[i].append(xs_real[i][ii][0])
        y_traj[i].append(xs_real[i][ii][1])
        headings[i].append(xs_real[i][ii][2])

vr, vl = mp_dynamics.compute_wheel_speeds(u1, u2)

    
# plot costs
plt.figure()
plt.plot(total_costs)
plt.plot(total_prox_costs)
plt.plot(total_ref_costs)
plt.plot(total_input_costs)
plt.plot(total_wall_costs)

plt.legend(['Total Cost','Proximity Cost', 'Reference Cost', 'Input Cost', 'Wall Cost'])
plt.xlabel('Time Step')
plt.ylabel('Cost')
plt.title('Costs over Time')
plt.show()


plt.ion()
fig, ax = plt.subplots()
ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-1.6, 1.6)
ax.grid(True)
colors = ['ro', 'go', 'bo', 'co', 'mo', 'yo']

for kk in range(mp_dynamics.TIMESTEPS):    
    ax.clear()
    ax.grid(True)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-1.6, 1.6)

    for i in range(mp_dynamics.num_agents):
        ax.plot(x_traj[i][kk], y_traj[i][kk], colors[i], label=f'Robot {i}', markersize=15)
        ax.arrow(x_traj[i][kk], y_traj[i][kk], 0.3 * np.cos(headings[i][kk]), 0.3 * np.sin(headings[i][kk]), head_width=0.05)

    plt.pause(0.01)
    time.sleep(0.01)
    plt.show()
    
plt.ioff()


plt.figure()
for i in range(mp_dynamics.num_agents):
    plt.plot(x_traj[i], y_traj[i], colors[i],  label=f'Robot {i}')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('State Trajectories')
plt.legend()
plt.show()


# plot wheel speeds in subplots
plt.figure()
plt.subplot(2,1,1)
for i in range(mp_dynamics.num_agents):
    plt.plot(vr[i], label=f'Robot {i}')
plt.xlabel('Time Step')
plt.ylabel('Right Wheel Speed')
plt.title('Right Wheel Speeds')
plt.legend()

plt.subplots_adjust(hspace=0.5)

plt.subplot(2,1,2)
for i in range(mp_dynamics.num_agents):
    plt.plot(vl[i], label=f'Robot {i}')
plt.xlabel('Time Step')
plt.ylabel('Left Wheel Speed')
plt.title('Left Wheel Speeds')
plt.legend()
plt.show()
