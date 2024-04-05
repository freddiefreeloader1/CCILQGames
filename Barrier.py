import numpy as np
from scipy.optimize import minimize

class OptimalControlBarrierMethod:
    def __init__(self, A, B, Q, R, x0, x_constraint, max_iter=100, tol=1e-6):
        self.A = A  # State transition matrix
        self.B = B  # Control input matrix
        self.Q = Q  # State cost matrix
        self.R = R  # Control cost matrix
        self.x0 = x0  # Initial state
        self.x_constraint = x_constraint  # State constraint
        self.max_iter = max_iter  # Maximum number of iterations
        self.tol = tol  # Tolerance for convergence

    def barrier_function(self, x):
        return -np.dot(x.T, np.dot(self.Q, x))

    def barrier_constraint(self, x):
        return x - self.x_constraint

    def objective_function(self, u):
        x = np.zeros((self.A.shape[0], self.max_iter + 1))
        x[:, 0] = self.x0
        cost = 0

        for t in range(self.max_iter):
            cost += np.dot(x[:, t].T, np.dot(self.Q, x[:, t])) + np.dot(u[:, t].T, np.dot(self.R, u[:, t]))

            x[:, t + 1] = np.dot(self.A, x[:, t]) + np.dot(self.B, u[:, t])

            # Add barrier function term to the cost
            cost += self.barrier_function(x[:, t + 1])

            # Check if state constraint is violated
            constraint_violation = self.barrier_constraint(x[:, t + 1])
            if np.any(constraint_violation > 0):
                # Add large penalty for violating constraint
                cost += 1e6 * np.sum(constraint_violation ** 2)

        return cost

    def optimize(self):
        # Initial guess for control inputs
        u0 = np.zeros((self.B.shape[1], self.max_iter))

        # Bounds for control inputs
        bounds = [(-1, 1) for _ in range(self.B.shape[1] * self.max_iter)]

        # Equality constraint: state dynamics
        cons = ({'type': 'eq', 'fun': lambda u: np.dot(self.A, self.x0).flatten() + np.dot(self.B, u.reshape(2, -1)).flatten() - self.x0},
                {'type': 'eq', 'fun': lambda u: np.dot(self.A, np.dot(self.A, self.x0).flatten()) +
                                                np.dot(self.B, u.reshape(-1, self.B.shape[1])).flatten() - np.dot(self.A, self.x0).flatten()})




        # Minimize the objective function subject to constraints
        res = minimize(self.objective_function, u0.flatten(), method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': self.max_iter, 'ftol': self.tol})

        if res.success:
            u_opt = res.x.reshape((self.B.shape[1], self.max_iter))
            return u_opt
        else:
            print("Optimization failed.")
            return None

# Example usage
if __name__ == "__main__":
    # Define system matrices
    A = np.array([[1, 1], [0, 1]])  # State transition matrix
    B = np.array([[0], [1]])  # Control input matrix
    Q = np.eye(2)  # State cost matrix
    R = np.eye(1)  # Control cost matrix

    # Initial state
    x0 = np.array([0, 0])

    # State constraint
    x_constraint = np.array([2, 2])

    # Create optimal control object
    oc = OptimalControlBarrierMethod(A, B, Q, R, x0, x_constraint)

    # Solve optimal control problem using barrier method
    u_opt = oc.optimize()

    if u_opt is not None:
        print("Optimal control sequence:\n", u_opt)
    else:
        print("Optimization failed.")
