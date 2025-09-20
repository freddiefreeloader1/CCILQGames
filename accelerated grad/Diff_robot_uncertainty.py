import torch
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class UnicycleRobotUncertain:

    def __init__(self, x0, xref, dt=0.1, WHEEL_BASE=0.1):
        self.x0 = x0
        self.state = torch.tensor([x0[0], x0[1], x0[2], x0[3]], requires_grad=True)  # (x, y, theta, v)
        self.xref = xref
        self.dt = dt
        self.uncertainty_params = [0.1, 0.1, 0.1, 0.1] 
        self.WHEEL_BASE = WHEEL_BASE

    def set_uncertainty_params(self, sigmas):
        self.uncertainty_params = sigmas

    def dynamics(self, u1, u2):

        x, y, theta, v = self.state

        if uncertainty:
            x_uncertainty = np.random.normal(0.0, self.uncertainty_params[0])
            y_uncertainty = np.random.normal(0.0, self.uncertainty_params[1])
            theta_uncertainty = np.random.normal(0.0, self.uncertainty_params[2])
            v_uncertainty = np.random.normal(0.0, self.uncertainty_params[3])
        else:
            x_uncertainty = 0
            y_uncertainty = 0
            theta_uncertainty = 0
            v_uncertainty = 0

        x_dot = v * np.cos(theta) + x_uncertainty
        y_dot = v * np.sin(theta) + y
        theta_dot = np.tensor(u1) + theta_uncertainty
        v_dot = np.tensor(u2) +  v_uncertainty

        return torch.stack([x_dot, y_dot, theta_dot, v_dot])

    def dynamics_for_given_state(self, state, u1, u2, uncertainty=False):
        if uncertainty:
            x_uncertainty = np.random.normal(0.0, self.uncertainty_params[0])
            y_uncertainty = np.random.normal(0.0, self.uncertainty_params[1])
            theta_uncertainty = np.random.normal(0.0, self.uncertainty_params[2])
            v_uncertainty = np.random.normal(0.0, self.uncertainty_params[3])
        else:
            x_uncertainty = 0
            y_uncertainty = 0
            theta_uncertainty = 0
            v_uncertainty = 0

        x, y, theta, v  = state

        x_dot = v * np.cos(theta) + x_uncertainty
        y_dot = v * np.sin(theta) + y_uncertainty
        theta_dot = u1 + theta_uncertainty
        v_dot = u2 + v_uncertainty

        return [x_dot, y_dot, theta_dot, v_dot]

    def integrate_dynamics_clone(self, u1, u2, dt):
        x_dot = self.dynamics(u1, u2)
        updated_state = self.state + self.dt * x_dot.detach().clone()
        return updated_state.data  

    def integrate_dynamics_for_given_state(self, state, u1, u2, dt, uncertainty=False):
        x_dot = self.dynamics_for_given_state(state, u1, u2, uncertainty)
        updated_state = [self.dt*i for i in x_dot] 
        updated_state= [i + j for i, j in zip(state, updated_state)]
        return updated_state

    def runge_kutta_4_integration(self,state, u1, u2, dt, uncertainty):
        # Runge-Kutta 4 integration method

        k1 = np.array(self.dynamics_for_given_state(state, u1, u2, uncertainty = False))
        k2 = np.array(self.dynamics_for_given_state(state + 0.5 * dt * k1, u1, u2, uncertainty = False))
        k3 = np.array(self.dynamics_for_given_state(state + 0.5 * dt * k2, u1, u2, uncertainty = False))
        k4 = np.array(self.dynamics_for_given_state(state + dt * k3, u1, u2, uncertainty = False))

        updated_state = [dt / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) + state[i] for i in range(4)]
        return updated_state

    def integrate_dynamics_for_initial_state(self, state, u1s, u2s, dt, TIMESTEP, uncertainty=False):
        states = []
        for i in range(TIMESTEP-1):
            if uncertainty:
                state = self.runge_kutta_4_integration(state, u1s[i], u2s[i], dt, uncertainty) + np.random.normal(0.0, self.uncertainty_params) 
            else:
                state = self.runge_kutta_4_integration(state, u1s[i], u2s[i], dt, uncertainty)
            states.append(state)
        return states

    def integrate_dynamics(self, u1, u2, dt):
        # Integrate forward in time using Euler method
        x_dot = self.dynamics(u1, u2)
        updated_state = self.state + self.dt * x_dot.detach().clone()
        self.state.data =  updated_state.data  # Update state without creating a view

    def linearize_autograd(self, x_torch, u_torch):
        
        updated_state = self.integrate_dynamics_clone(u_torch[0], u_torch[1], self.dt)

        A = np.array([[0, 0, updated_state[3].detach().numpy() * -torch.sin(updated_state[2]).item(), torch.cos(updated_state[2]).item()], 
                     [0, 0, updated_state[3].item() * torch.cos(updated_state[2]).item(), torch.sin(updated_state[2]).item()],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]])

        B = np.array([[0, 0],
                     [0, 0],
                     [1, 0],
                     [0, 1]])
        return A, B

    def linearize(self, x, u):
        
        A = np.array([[0, 0, x[3] * -np.sin(x[2]), np.cos(x[2])], 
                    [0, 0, x[3] * np.cos(x[2]), np.sin(x[2])],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]])

        B = np.array([[0, 0],
                    [0, 0],
                    [1, 0],
                    [0, 1]])
        return A, B
    
    def linearize_discrete(self, A, B, dt):

        A_d = scipy.linalg.expm(A * dt)
        B_d = np.linalg.pinv(A) @ (scipy.linalg.expm(A * dt) - np.eye(4)) @ B
        # make the values of A_d and B_d to be 0 if they are very close to 0
        A_d[np.abs(A_d) < 1e-10] = 0
        B_d[np.abs(B_d) < 1e-10] = 0

        return A_d, B_d
        
    def linearize_dynamics_along_trajectory(self, u1_traj, u2_traj, dt):
        # Linearize dynamics along the trajectory
        num_steps = len(u2_traj)

        A_list = []
        B_list = []
        A_d_list = []
        B_d_list = []

        for t in range(num_steps):
            # Integrate forward in time
            updated_state = self.integrate_dynamics_clone(u1_traj[t], u2_traj[t], self.dt)

            # Linearize at the current state and control
            x_torch = updated_state.clone().detach().requires_grad_(True)
            u_torch = torch.tensor([u1_traj[t], u2_traj[t]], requires_grad=True)

            A, B = self.linearize_autograd(x_torch, u_torch)
            A_d, B_d = self.linearize_discrete(A, B, dt)
            A_d_list.append(A_d)
            B_d_list.append(B_d)
            A_list.append(A)
            B_list.append(B)
          
        return np.array(A_list), np.array(B_list), np.array(A_d_list), np.array(B_d_list)

    def linearize_dynamics_along_trajectory_for_states(self,states, u1_traj, u2_traj, dt):
        # Linearize dynamics along the trajectory
        num_steps = len(u2_traj)

        A_list = []
        B_list = []
        A_d_list = []
        B_d_list = []

        for t in range(num_steps):
            u = [u1_traj[t], u2_traj[t]]
            A, B = self.linearize(states[t], u)
            A_d, B_d = self.linearize_discrete(A, B, dt)
            A_d_list.append(A_d)
            B_d_list.append(B_d)
            A_list.append(A)
            B_list.append(B)
          
        return np.array(A_list), np.array(B_list), np.array(A_d_list), np.array(B_d_list)

