import numpy as np
from sympy.physics.quantum.spin import Rotation


l_max = 8
n_coeffs = int(0.5 * (l_max + 1) * (l_max + 2))

b = 8  # rotation bandwidth

alphas = np.array([i * np.pi / b for i in range(2 * b + 1)])
betas = np.array([np.pi * (2 * i + 1) / (4 * b) for i in range(2 * b + 1)])
gammas = np.array([i * np.pi / b for i in range(2 * b + 1)])


def T_matrix(l):
    T = np.zeros((2 * l + 1, 2 * l + 1), dtype=np.complex64)
    for i, m in enumerate(range(-l, l + 1)):
        for j, n in enumerate(range(-l, l + 1)):
            if m == n == 0:
                T[i, j] = 1
            if m > 0 and n == m:
                T[i, j] = (-1) ** m / np.sqrt(2)
            if m > 0 and n == -m:
                T[i, j] = 1 / np.sqrt(2)
            if m < 0 and n == m:
                T[i, j] = 1j / np.sqrt(2)
            if m < 0 and n == -m:
                T[i, j] = -1j * (-1) ** m / np.sqrt(2)
    return T


def d_matrix(l, beta):
    d = np.zeros((2 * l + 1, 2 * l + 1), dtype=complex)
    for i, k in enumerate(range(-l, l + 1)):
        for j, m in enumerate(range(-l, l + 1)):
            d[i, j] = complex(Rotation.d(l, k, m, beta).doit())
    return d


Rs = np.zeros((len(alphas), len(betas), len(gammas), n_coeffs, n_coeffs))
for l in range(0, l_max + 1, 2):
    T = T_matrix(l)
    for b, beta in enumerate(betas):
        d = d_matrix(l, beta)
        for a, alpha in enumerate(alphas):
            for c, gamma in enumerate(gammas):
                print(
                    f"l = {l}, "
                    f"alpha = {np.round(alpha, 4)}, "
                    f"beta = {np.round(beta, 4)}, "
                    f"gamma = {np.round(gamma, 4)}",
                    end="\r",
                )
                D = np.zeros((2 * l + 1, 2 * l + 1), dtype=complex)
                for i, k in enumerate(range(-l, l + 1)):
                    for j, m in enumerate(range(-l, l + 1)):
                        D[i, j] = np.exp(-1j * k * alpha) * np.exp(-1j * m * gamma)
                D *= d
                U = (T.conj() @ D @ T.T).real
                Rs[
                    a,
                    b,
                    c,
                    int(0.5 * (l - 1) * l) : int(0.5 * (l - 1) * l) + 2 * l + 1,
                    int(0.5 * (l - 1) * l) : int(0.5 * (l - 1) * l) + 2 * l + 1,
                ] = U

np.save("Rs.npy", Rs)
