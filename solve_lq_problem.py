import numpy as np
from collections import deque

def solve_lq_game(As, Bs, Qs, ls, Rs):

    horizon = len(As) - 1
    num_players = len(Bs)
    x_dim = As[0].shape[0]
    u_dims = np.empty(num_players, dtype=int)
    for i, Bi in enumerate(Bs):
        u_dims[i] = Bi[0].shape[1]


    Zs = [[Qis[-1]] for Qis in Qs]
    zetas = [[lis[-1]] for lis in ls]
    Fs = deque()
    Ps = [deque() for ii in range(num_players)]
    betas = deque()
    alphas = [[] for _ in range(num_players)]

    for k in range(horizon, -1, -1):
        A = As[k]
        B = [Bi[k] for Bi in Bs]
        Q = [Qis[k] for Qis in Qs]
        l = [lis[k] for lis in ls]
        R = [[Rij[k] for Rij in Ris] for Ris in Rs]

        Z = [Zi[0] for Zi in Zs]
        zeta = [zetai[0] for zetai in zetas]

        S_rows = []
        for ii in range(num_players):
            Zi = Z[ii]  
            S_rows.append(np.concatenate([
                R[ii][ii] + B[ii].T @ Zi @ B[ii] if jj == ii else B[ii].T @ Zi @ B[jj]
                for jj in range(num_players)
            ], axis=1))

        S = np.concatenate(S_rows, axis=0)
        Y = np.concatenate([B[ii].T @ Z[ii] @ A for ii in range(num_players)], axis=0)

        P, _, _, _ = np.linalg.lstsq(a=S, b=Y, rcond=None)
        P_split = np.split(P, np.cumsum(u_dims[:-1]), axis=0)

        for ii in range(num_players):
            Ps[ii].appendleft(P_split[ii])

        F = A - sum([B[ii] @ P_split[ii] for ii in range(num_players)])
        Fs.appendleft(F)

        for ii in range(num_players):
            Zs[ii].insert(0, F.T @ Z[ii] @ F + Q[ii] + sum([
                P_split[jj].T @ R[ii][jj] @ P_split[jj]
                for jj in range(num_players)
            ]))

        Y = np.concatenate([B[ii].T @ zeta[ii] for ii in range(num_players)], axis=0)
        alpha, _, _, _ = np.linalg.lstsq(a=S, b=Y, rcond=None)
        alpha_split = np.split(alpha, np.cumsum(u_dims[:-1]), axis=0)
        for ii in range(num_players):
            alphas[ii].insert(0, alpha_split[ii])

        beta = -sum([B[ii] @ alpha_split[ii] for ii in range(num_players)])
        betas.appendleft(beta)

        for ii in range(num_players):
            zetas[ii].insert(0, F.T @ (zeta[ii] + Z[ii] @ beta) + l[ii] + sum([
                P_split[jj].T @ R[ii][jj] @ alpha_split[jj]
                for jj in range(num_players)
            ]))

    return [list(Pis) for Pis in Ps], [list(alphais) for alphais in alphas]
