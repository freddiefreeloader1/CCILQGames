from scipy.linalg import block_diag

from Costs import ProximityCost, OverallCost, ReferenceCost, WallCost, InputCost, ProximityCostUncertainLinear, ProximityCostUncertainQuad, SpeedCost

import numpy as np
from scipy.special import erfinv
from Diff_robot_uncertainty import UnicycleRobotUncertain

class MultiAgentDynamics():
    def __init__(self, agent_list, dt, HORIZON=3.0, ref_cost_threshold = 20, prob = 0.70):
        self.agent_list = agent_list
        self.dt = dt
        self.num_agents = len(agent_list)
        self.x0_mp = np.concatenate([agent.x0 for agent in agent_list])
        self.xref_mp = np.concatenate([agent.xref for agent in agent_list])
        self.TIMESTEPS = int(HORIZON/dt)
        self.prob = prob
        self.ref_cost_threshold = ref_cost_threshold


    def get_linearized_dynamics_for_initial_state(self, x_states, u1, u2):
        A_traj_mp = []
        As = []
        Bs = []
        A_traj_c = []
        B_cs = []
        A_cs = []
        for i, agent in enumerate(self.agent_list):
            A_c, B_c, A_traj, B_traj = agent.linearize_dynamics_along_trajectory_for_states(x_states[i], u1[i], u2[i], self.dt)
            A_traj_mp.append(A_traj)
            A_traj_c.append(A_c)
            if i == 0:
                B_traj = [np.concatenate((B, np.zeros((4 * (self.num_agents -1), 2))), axis=0) for B in B_traj]
                B_c = [np.concatenate((B, np.zeros((4 * (self.num_agents -1), 2))), axis=0) for B in B_c]
            else:
                B_traj = [np.concatenate((np.zeros((4 * i, 2)), B, np.zeros((4 * (self.num_agents - i - 1), 2))), axis=0) for B in B_traj]   
                B_c = [np.concatenate((np.zeros((4 * i, 2)), B, np.zeros((4 * (self.num_agents - i - 1), 2))), axis=0) for B in B_c]

            Bs.append(B_traj)
            B_cs.append(B_c)

        As = [block_diag(*A_list) for A_list in zip(*A_traj_mp)]
        A_cs = [block_diag(*A_list) for A_list in zip(*A_traj_c)]

        return A_cs, B_cs, As, Bs

    def define_costs_lists(self, uncertainty = False):
        ref_cost_list = [[] for _ in range(len(self.agent_list))]
        prox_cost_list = [[] for _ in range(len(self.agent_list))]
        wall_cost_list = [[] for _ in range(len(self.agent_list))]
        input_cost_list = [[] for _ in range(len(self.agent_list))]
        overall_cost_list = [[] for _ in range(len(self.agent_list))]
        speed_cost_list = [[] for _ in range(len(self.agent_list))]

        for i, agent in enumerate(self.agent_list):
            ref_cost_list[i].append(ReferenceCost(i, self.xref_mp, [40,40,1,10]))
            input_cost_list[i].append(InputCost(i, 600.0, 600.0))
            
            if True:
                speed_cost_list[i].append(SpeedCost(i, 10))
                

        if uncertainty == False:
            for i in range(len(self.agent_list)):
                for j in range(len(self.agent_list)):
                    if i != j:
                        prox_cost_list[i].append(ProximityCost(0.5, i, j, 500.0))
        else:
            for i in range(len(self.agent_list)):
                for j in range(len(self.agent_list)-1):
                    prox_cost_list[i].append(ProximityCostUncertainLinear(1.0))
                    prox_cost_list[i].append(ProximityCostUncertainQuad(0.0))
                    
        for i in range(len(self.agent_list)):
            wall_cost_list[i].append(WallCost(i, 0.04))

        for i in range(len(self.agent_list)):
            # add the reference cost and the proximity cost to the overall cost list
            cost_list = ref_cost_list[i] + prox_cost_list[i] + wall_cost_list[i] + input_cost_list[i] + speed_cost_list[i]
            overall_cost_list[i].append(OverallCost(cost_list, self.ref_cost_threshold))
        return overall_cost_list
   

    def integrate_dynamics_for_initial_mp(self, u1, u2, dt, uncertainty = False):
        xs = [[agent.x0] for agent in self.agent_list]
        for i, agent in enumerate(self.agent_list):
            xs[i] = xs[i] + (agent.integrate_dynamics_for_initial_state(agent.x0, u1[i], u2[i], dt, self.TIMESTEPS, uncertainty))
        return xs


    def get_control_cost_matrix(self):
        R_eye = np.array([[1, 0],[0, 25]])
        R_zeros = np.zeros((2, 2))

        # Initialize R_matrices and Z_matrices lists
        R_matrices = [R_eye.copy() for _ in range(self.TIMESTEPS)]
        Z_matrices = [R_zeros.copy() for _ in range(self.TIMESTEPS)]

        # Initialize Rs list based on the number of robots
        Rs = []
        for i in range(self.num_agents):
            R_terms = [R_matrices.copy() if i == j else Z_matrices.copy() for j in range(self.num_agents)]
            Rs.append(R_terms)

        return Rs

    def check_convergence(self, current_points, last_points):

        if last_points is None:
            return 0
        for i in range(len(current_points)):
            for j in range(len(current_points[i])):
                for k in range(len(current_points[i][j])):
                    if np.abs(np.array(current_points[i][j][k]) - np.array(last_points[i][j][k])) > 0.1:
                        return 0
        return 1

    def get_Gs(self, xs, prox_cost_list, sigmas):
        Gs = np.empty((self.num_agents, self.TIMESTEPS, self.num_agents-1), dtype=object)
        qs = np.empty((self.num_agents, self.TIMESTEPS, self.num_agents-1), dtype=object)
        rhos = np.empty((self.num_agents, self.TIMESTEPS, self.num_agents-1), dtype=object)

        xs_concatenated = [np.concatenate([state[t] for state in xs]) for t in range(self.TIMESTEPS)]
        for i in range(self.num_agents):
            for t in range(self.TIMESTEPS):
                for j in range(self.num_agents-1):
                    Gs[i][t][j] = prox_cost_list[i][j].gradient_x(xs_concatenated[t], [0]*self.num_agents*4)
                    qs[i][t][j] = prox_cost_list[i][j].evaluate(xs_concatenated[t], [0]*self.num_agents*4) - Gs[i][t][j] @ xs_concatenated[t]
                    rhos[i][t][j] = np.sqrt(2*(Gs[i][t][j]@sigmas[t])@np.array(Gs[i][t][j]).T)*erfinv(2*self.prob - 1)
        return Gs, qs, rhos 

    def compute_op_point(self, Ps, alphas, current_x, u_prev, zeta = 0.02, uncertainty = False):
        u_next = np.zeros((self.num_agents, self.TIMESTEPS, 2))
        xs = np.zeros((self.num_agents, self.TIMESTEPS, 4))
        # make the first state of the robots the same as the current state
        for i, agent in enumerate(self.agent_list):
            xs[i][0] = agent.x0

        zeta = self.line_search(Ps, alphas, current_x, u_prev)

        if current_x is not None:
            for i, agent in enumerate(self.agent_list):
                for ii in range(self.TIMESTEPS - 1):
                    # concatenate the states of all the robots
                    concatenated_states = np.concatenate([state[ii] for state in xs])
                    concatenated_states_current = np.concatenate([state[ii] for state in current_x])
                    u_next[i][ii] = u_prev[i][ii] - zeta*alphas[i][ii] - Ps[i][ii] @ (concatenated_states - concatenated_states_current)
                    [u1, u2] = u_next[i][ii]
                    xs[i][ii+1] = agent.runge_kutta_4_integration(xs[i][ii], u1, u2, self.dt, uncertainty)
        else:
            for i, agent in enumerate(self.agent_list):
                for ii in range(self.TIMESTEPS):
                    u_next[i][ii] = u_prev[i][ii] - zeta*alphas[i][ii]
                    [u1, u2] = u_next[i][ii]
                    xs[i][ii] = agent.runge_kutta_4_integration(xs[i][ii], u1, u2, self.dt, uncertainty)
        return xs, u_next
      

    def line_search(self, Ps, alphas, current_x, u_prev):
        zeta = 1.0
        while True:
            xs, u_next = self.compute_op_point_imposter(Ps, alphas, current_x, u_prev, zeta)
            if (np.linalg.norm(xs-current_x) < 50):
                break
            zeta = zeta/2

        print(f"zeta: {zeta} \n")
        return zeta

    def compute_op_point_imposter(self, Ps, alphas, current_x, u_prev, zeta = 0.02, uncertainty = False):
        u_next = np.zeros((self.num_agents, self.TIMESTEPS, 2))
        xs = np.zeros((self.num_agents, self.TIMESTEPS, 4))
        # make the first state of the robots the same as the current state
        for i, agent in enumerate(self.agent_list):
            xs[i][0] = agent.x0
            
        if current_x is not None:
            for i, agent in enumerate(self.agent_list):
                for ii in range(self.TIMESTEPS - 1):
                    # concatenate the states of all the robots
                    concatenated_states = np.concatenate([state[ii] for state in xs])
                    concatenated_states_current = np.concatenate([state[ii] for state in current_x])
                    u_next[i][ii] = u_prev[i][ii] - zeta*alphas[i][ii] - Ps[i][ii] @ (concatenated_states - concatenated_states_current)
                    [u1, u2] = u_next[i][ii]
                    xs[i][ii+1] = agent.runge_kutta_4_integration(xs[i][ii], u1, u2, self.dt, uncertainty)
        else:
            for i, agent in enumerate(self.agent_list):
                for ii in range(self.TIMESTEPS):
                    u_next[i][ii] = u_prev[i][ii] - zeta*alphas[i][ii]
                    [u1, u2] = u_next[i][ii]
                    xs[i][ii] = agent.runge_kutta_4_integration(xs[i][ii], u1, u2, self.dt, uncertainty)
        return xs, u_next


    def compute_wheel_speeds(self, w, v):
        vr = np.zeros((self.num_agents, self.TIMESTEPS))
        vl = np.zeros((self.num_agents, self.TIMESTEPS))
        for i, agent in enumerate(self.agent_list):
            for ii in range(self.TIMESTEPS):
                vr[i][ii] = v[i][ii] + w[i][ii]*agent.WHEEL_BASE/2
                vl[i][ii] = v[i][ii] - w[i][ii]*agent.WHEEL_BASE/2
        return vr , vl