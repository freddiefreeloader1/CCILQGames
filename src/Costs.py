import autograd.numpy as np
from scipy.optimize import approx_fprime
from autograd import elementwise_grad as egrad
import time

class ProximityCost:
    def __init__(self, d_threshold=0.5, idx1 = 0, idx2 = 0, weight = 1.0):
        self.d_threshold = d_threshold 
        self.idx1 = idx1
        self.idx2 = idx2
        self.weight = weight

    def evaluate(self, x, u):
        dist = np.sqrt((x[4*self.idx1] - x[4*self.idx2])**2 + (x[4*self.idx1 + 1] - x[4*self.idx2 + 1])**2)
        return  0.0 if dist > self.d_threshold else self.weight * (self.d_threshold - dist)**2

    def gradient_x(self, x, u):
        dist = np.sqrt((x[4*self.idx1] - x[4*self.idx2])**2 + (x[4*self.idx1 + 1] - x[4*self.idx2 + 1])**2)
        if dist > self.d_threshold:
            return [0.0 for _ in range(len(x))]
        # denom = -self.weight/(2*np.sqrt((x[4*self.idx1] - x[4*self.idx2])**2 + (x[4*self.idx1 + 1] - x[4*self.idx2 + 1])**2) + 1e-6)
        denom = - self.weight * 2 * (self.d_threshold - dist)
        grad_x = [0.0 for _ in range(len(x))] 
        grad_x[4*self.idx1] = 2*(x[4*self.idx1] - x[4*self.idx2])*denom
        grad_x[4*self.idx1 + 1] = 2*(x[4*self.idx1 + 1] - x[4*self.idx2 + 1])*denom
        grad_x[4*self.idx2] = -2*(x[4*self.idx1] - x[4*self.idx2])*denom
        grad_x[4*self.idx2 + 1] = -2*(x[4*self.idx1 + 1] - x[4*self.idx2 + 1])*denom
        return grad_x
    

    def hessian_x(self, x, u):
        dist = np.sqrt((x[4*self.idx1] - x[4*self.idx2])**2 + (x[4*self.idx1 + 1] - x[4*self.idx2 + 1])**2)
        if dist > self.d_threshold:
            # If the distance is greater than the threshold, the Hessian is zero
            return np.zeros((len(x), len(x)))
        
        # Calculate the denominator
        denom = -2 * self.weight * (self.d_threshold - dist)
        
        # Calculate the Hessian
        H = np.zeros((len(x), len(x)))
        H[4*self.idx1, 4*self.idx1] = 2 * denom
        H[4*self.idx1 + 1, 4*self.idx1 + 1] = 2 * denom
        H[4*self.idx2, 4*self.idx2] = 2 * denom
        H[4*self.idx2 + 1, 4*self.idx2 + 1] = 2 * denom
        H[4*self.idx1, 4*self.idx2] = -2 * denom
        H[4*self.idx2, 4*self.idx1] = -2 * denom
        H[4*self.idx1 + 1, 4*self.idx2 + 1] = -2 * denom
        H[4*self.idx2 + 1, 4*self.idx1 + 1] = -2 * denom
        
        return H

    def gradient_u(self, x, u):
        return [0.0 for _ in range(len(u))]
    
        

class ProximityCostUncertainLinear:
    def __init__(self, weight = 1.0):
        self.weight = weight

    def evaluate(self, x, G, q, rho, lam):
        cost = self.weight*lam*(G @ x  + q + rho)
        return  cost
        

    def gradient_x(self, x, G, q, rho, lam):
        grad_x = self.weight*np.array(G)*lam
        return grad_x

    def gradient_u(self, x, u):
        return [0.0 for _ in range(len(u))]

    def hessian_x(self, x, G, q, rho, lam):
        return np.zeros((len(x), len(x)))

class ProximityCostUncertainQuad:
    def __init__(self, weight = 1.0):
        self.weight = weight

    def evaluate(self, x, G, q, rho, I):
        x = np.array(x)
        G = np.array(G)
        q = np.array(q)
        rho = np.array(rho)
        
        cost = self.weight*(I/2)*((G@x  + q + rho))**2
        return  cost

    def gradient_x(self, x, G, q, rho, I):
        cost = self.weight*(I/2)*(G @ x  + q + rho)
        grad_x = 2 * cost* np.array(G)
        return grad_x

    def gradient_u(self, x, u):
        return [0.0 for _ in range(len(u))]

    def hessian_x(self, x, G, q, rho, I):
        return self.weight*I*np.array(G).T @ np.array(G)

class ReferenceCost:
    def __init__(self, idx = 0, x_ref = np.array([0, 0, 0, 0, 0, 0, 0, 0]), weight = [1, 1, 1, 1]):
        self.idx = idx
        self.x_ref = x_ref
        self.weight = weight

    def evaluate(self, x, u):
        dist = np.sqrt(
        self.weight[0]*(x[4*self.idx] - self.x_ref[4*self.idx])**2 + 
        self.weight[1]*(x[4*self.idx + 1] - self.x_ref[4*self.idx + 1])**2 + 
        self.weight[2]*(x[4*self.idx + 2] - self.x_ref[4*self.idx + 2])**2 + 
        self.weight[3]*(x[4*self.idx + 3] - self.x_ref[4*self.idx + 3])**2)**2
        return dist
    
    def gradient_x(self, x, u):
        '''denom = self.weight/(2*np.sqrt((x[4*self.idx] - self.x_ref[4*self.idx])**2 + 
        (x[4*self.idx + 1] - self.x_ref[4*self.idx + 1])**2 + 
        (x[4*self.idx + 2] - self.x_ref[4*self.idx + 2])**2 + 
        (x[4*self.idx + 3] - self.x_ref[4*self.idx + 3])**2))'''

        grad_x = [0.0 for _ in range(len(x))] 
        grad_x[4*self.idx] = 2*self.weight[0]*(x[4*self.idx] - self.x_ref[4*self.idx])
        grad_x[4*self.idx + 1] = 2*self.weight[1]*(x[4*self.idx + 1] - self.x_ref[4*self.idx + 1])
        grad_x[4*self.idx + 2] = 2*self.weight[2]*(x[4*self.idx + 2] - self.x_ref[4*self.idx + 2])
        grad_x[4*self.idx + 3] = 2*self.weight[3]*(x[4*self.idx + 3] - self.x_ref[4*self.idx + 3])
        return grad_x

    def hessian_x(self, x, u):
        H = np.zeros((len(x), len(x)))
        for i in range(4):
            H[4*self.idx + i, 4*self.idx + i] = 2 * self.weight[i]
        return H

    def gradient_u(self, x, u):
        return [0.0 for _ in range(len(u))]


class InputCost:
    def __init__(self, idx, weight1=1.0, weight2=1.0):
        self.weight1 = weight1
        self.weight2 = weight2
        self.idx = idx
        
    def evaluate(self, x, u):
        return self.weight1*u[0]**2 + self.weight2*u[1]**2

    def gradient_x(self, x, u):
        return [0.0 for _ in range(len(x))]
    
    def gradient_u(self, x, u):
        grad_u = [0.0 for _ in range(len(u))]
        grad_u[0] = 2*self.weight1*u[0]
        grad_u[1] = 2*self.weight2*u[1]
        return grad_u
        
    def hessian_x(self, x, u):
        return np.zeros((len(x), len(x)))



class WallCost:
    def __init__(self, idx, weight=1.0):
        self.idx = idx
        self.weight = weight

    def evaluate(self, x, u):
        x_robot = x[4 * self.idx]
        y_robot = x[4 * self.idx + 1]


        side_length = 7.0
        x_center = 0.0
        y_center = 0.0

        dx = max(0, abs(x_robot - x_center) - 0.5 * side_length)
        dy = max(0, abs(y_robot - y_center) - 0.5 * side_length)

        dist_penalty = np.sqrt(dx**2 + dy**2)**2

        return self.weight * dist_penalty

    def gradient_x(self, x, u):
        x_robot = x[4 * self.idx]
        y_robot = x[4 * self.idx + 1]

      
        side_length = 7.0
        x_center = 0.0
        y_center = 0.0

        dx = max(0, abs(x_robot - x_center) - 0.5 * side_length)
        dy = max(0, abs(y_robot - y_center) - 0.5 * side_length)

        dist_penalty = np.sqrt(dx**2 + dy**2)
        grad_x = np.zeros(len(x))
        if dx > 0:
            if x_robot > x_center:
                grad_x[4 * self.idx] = self.weight*dx*2
            else:
                grad_x[4 * self.idx] = -self.weight*dx*2
        if dy > 0:
            if y_robot > y_center:
                grad_x[4 * self.idx + 1] = self.weight*dy*2
            else:
                grad_x[4 * self.idx + 1] = -self.weight*dy*2
        return grad_x

    def gradient_u(self, x, u):
        return [0.0 for _ in range(len(u))]

    def hessian_x(self, x, u):
        x_robot = x[4 * self.idx]
        y_robot = x[4 * self.idx + 1]

        side_length = 7.0
        x_center = 0.0
        y_center = 0.0

        dx = max(0, abs(x_robot - x_center) - 0.5 * side_length)
        dy = max(0, abs(y_robot - y_center) - 0.5 * side_length)

        hessian_x = np.zeros((len(x), len(x)))
        if dx > 0:
            hessian_x[4 * self.idx, 4 * self.idx] = 2 * self.weight
        if dy > 0:
            hessian_x[4 * self.idx + 1, 4 * self.idx + 1] = 2 * self.weight
        return hessian_x

class SpeedCost:
    def __init__(self, idx, weight=1.0):
        self.weight = weight
        self.idx = idx

    def evaluate(self, x, u):
        return self.weight*((x[4*self.idx+3])**2)

    def gradient_x(self, x, u):
        grad_x = [0.0 for _ in range(len(x))]
        grad_x[4*self.idx+3] = 2*self.weight*(x[4*self.idx+3])
        return grad_x
    
    def gradient_u(self, x, u):
        return [0.0 for _ in range(len(u))]

    def hessian_x(self, x, u):
        hessian_x = np.zeros((len(x), len(x)))
        hessian_x[4*self.idx + 3, 4*self.idx + 3] = 2 * self.weight
        return hessian_x    

class TrialCost:
    def __init__(self, d_threshold=0.5):
        self.d_threshold = 0.5
    def evaluate(self, x, u):
        dist = x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2
        return dist

class OverallCost:
    def __init__(self, subsystem_cost_functions, ref_cost_threshold = 20):
        self.subsystem_cost_functions = subsystem_cost_functions
        self.ref_cost_threshold = ref_cost_threshold

    def evaluate(self, x, u, G = 1, q = 1, rho = 1, lam = 1, I = 1):
        total_cost = 0.0
        for subsystem_cost in self.subsystem_cost_functions:
            if isinstance(subsystem_cost, ProximityCostUncertainLinear):
                total_cost += subsystem_cost.evaluate(x, G , q, rho, lam)
            elif isinstance(subsystem_cost, ProximityCostUncertainQuad):
                total_cost += subsystem_cost.evaluate(x, G , q, rho, I)
            else:
                total_cost += subsystem_cost.evaluate(x, u)
            # print(subsystem_cost.evaluate(x, u))
        return total_cost
    
    def evaluate_grad(self, x, u, G = 1, q = 1, rho = 1, lam = 1, I = 1, timestep = 0):
        total_cost = np.zeros(len(x))
        for subsystem_cost in self.subsystem_cost_functions:
            if isinstance(subsystem_cost, ProximityCostUncertainLinear):
                total_cost += np.array(subsystem_cost.gradient_x(x, G, q, rho, lam))
            elif isinstance(subsystem_cost, ProximityCostUncertainQuad):
                total_cost += np.array(subsystem_cost.gradient_x(x, G, q, rho, I))
            else:
                if isinstance(subsystem_cost, ReferenceCost):
                    if timestep > self.ref_cost_threshold:
                        total_cost += np.array(subsystem_cost.gradient_x(x, u))
                else:
                    if timestep > 0:
                        total_cost += np.array(subsystem_cost.gradient_x(x, u))

        return total_cost

    def gradient_x(self, x, u, G = 1, q = 1, rho = 1, lam = 1, I = 1, timestep = 0):
        # grad_x = approx_fprime(x, lambda x: self.evaluate(x, u), epsilon=1e-6)
        grad_x = np.zeros(len(x))
        for subsystem_cost in self.subsystem_cost_functions:
            if isinstance(subsystem_cost, ProximityCostUncertainLinear):
                grad_x += np.array(subsystem_cost.gradient_x(x, G, q, rho, lam))
            elif isinstance(subsystem_cost, ProximityCostUncertainQuad):
                grad_x += np.array(subsystem_cost.gradient_x(x, G, q, rho, I))
            else:
                if isinstance(subsystem_cost, ReferenceCost):
                    if timestep > self.ref_cost_threshold:
                        grad_x += np.array(subsystem_cost.gradient_x(x, u))
                else:
                     if timestep > 0:
                        grad_x += np.array(subsystem_cost.gradient_x(x, u))

        return grad_x

    def gradient_u(self, x, u):
        # grad_u = approx_fprime(u, lambda u: self.evaluate(x, u), epsilon=1e-6)
        grad_u = np.zeros(len(u))
        for subsystem_cost in self.subsystem_cost_functions:
            grad_u += np.array(subsystem_cost.gradient_u(x, u))
        return grad_u

    def hessian_x(self, x, u, G = 1, q = 1, rho = 1, lam = 1, I = 1, timestep = 0):
        hessian_x = approx_fprime(x, lambda x: self.evaluate_grad(x, u, G, q, rho, I, timestep=timestep), epsilon=1e-6)
        return hessian_x

    def hessian_x_2(self, x, u):
        hessian_x = approx_fprime(x, lambda x: self.gradient_x(x, u), epsilon=1e-6)
        return hessian_x

    def hessian_u(self, x, u):
        hessian_u = approx_fprime(u, lambda u: self.gradient_u(x, u), epsilon=1e-6)
        return hessian_u

    def hessian_manual(self, x, u, G = 1, q = 1, rho = 1, lam = 1, I = 1, timestep = 0):
        hessian_x = np.zeros((len(x), len(x)))
        for subsystem_cost in self.subsystem_cost_functions:
            if isinstance(subsystem_cost, ProximityCostUncertainLinear):
                hessian_x += subsystem_cost.hessian_x(x, G , q, rho, lam)
            elif isinstance(subsystem_cost, ProximityCostUncertainQuad):
                hessian_x += subsystem_cost.hessian_x(x, G , q, rho, I)
            else:
                hessian_x += subsystem_cost.hessian_x(x, u)
        return hessian_x

def trial():

    overall_cost = OverallCost([ProximityCost(idx1 = 0, idx2 = 1)])
    prox_cost = ProximityCost(idx1 = 0, idx2 = 1)

    x_example = np.array([1, 1.1, 1.0, 1.0, 1.1, 1.0, 1.0, 1.0, 1.1, 1.0, 1.0, 1.0])
    u_example = np.array([1, 1])

    hessian_x = overall_cost.hessian_x(x_example, u_example)
    hessian_x = np.round(hessian_x, 1)
    hessian_x_2 = prox_cost.hessian_x(x_example, u_example)
    hessian_x_2 = np.round(hessian_x_2, 1)

    print(hessian_x)
    print("--------------------------------------------")
    print(hessian_x_2)

# trial()

