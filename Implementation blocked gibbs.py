import numpy as np
from scipy.stats import norm, gamma, t
from numpy.linalg import inv
import time

def X_eta_tX(eta, X, X_transpose): #Produit de la matrice X et de sa transposée, avec le facteur d'échelle eta
    n = X.shape[0]
    p = X.shape[1]
    X_eta_tX_matrix = np.zeros((p, p))
    for j in range(p):
        X_eta_tX_matrix[j, j] = eta[j] * np.sum(np.square(X[:, j]))
    return X_eta_tX_matrix

def log_ratio(x, eta, X_eta_tX_matrix, y, a0, b0, xi_interval):# Calcul le logarithme de deux probabilités (préciser lesquelles)
    n = y.size
    p = X_eta_tX_matrix.shape[0]
    d_matrix = X_eta_tX_matrix + np.identity(p) * x
    d_matrix_inv = inv(d_matrix)
    ssr = np.sum(np.square(y - np.dot(X_eta_tX_matrix @ d_matrix_inv, y))) + np.sum(np.square(d_matrix_inv))
    log_ratio = -0.5 * (n * np.log(ssr) - a0 * np.log(x) - b0 / x - np.log(eta))
    if xi_interval is not None:
        if xi_interval[0] > x or xi_interval[1] < x:
            log_ratio = -np.inf
    return log_ratio

def log_ratio_approx(x, eta, X, X_transpose, y, a0, b0, active_set, xi_interval): #approximation du logarithme 
    n = y.size
    p = X.shape[1]
    d_matrix = np.identity(p)
    d_matrix_inv = inv(d_matrix)
    ssr = np.sum(np.square(y - np.dot(X_transpose @ d_matrix_inv, y))) + np.sum(np.square(d_matrix_inv))
    log_ratio = -0.5 * (n * np.log(ssr) - a0 * np.log(x) - b0 / x - np.log(eta))
    if xi_interval is not None:
        if xi_interval[0] > x or xi_interval[1] < x:
            log_ratio = -np.inf
    return log_ratio

def crn_max_xi_coupling(current_xi_1, eta_1, current_xi_2, eta_2, X, X_transpose, y, a0, b0, std_MH, approximate_algo_delta=0, epsilon_xi=0, fixed=False, xi_interval=None):
    n = X.shape[0]
    p = X.shape[1]

    if fixed: # Les valeurs xi sont fixes et ne doivent pas être mises à jour
        min_xi_1 = current_xi_1
        min_xi_2 = current_xi_2
        active_set_1 = ((min_xi_1 * eta_1)**(-1) > approximate_algo_delta)
        active_set_2 = ((min_xi_2 * eta_2)**(-1) > approximate_algo_delta)

        if sum(active_set_1) > n:
            X_eta_tX_matrix_1 = X_eta_tX(eta_1[active_set_1], X[:, active_set_1], X_transpose[active_set_1])
            log_ratio_current_ssr_matrixinv_1 = log_ratio(current_xi_1, eta_1[active_set_1], X_eta_tX_matrix_1, y, a0, b0, xi_interval)
        else:
            log_ratio_current_ssr_matrixinv_1 = log_ratio_approx(current_xi_1, eta_1, X, X_transpose, y, a0, b0, active_set_1, xi_interval)

        if sum(active_set_2) > n:
            X_eta_tX_matrix_2 = X_eta_tX(eta_2[active_set_2], X[:, active_set_2], X_transpose[active_set_2])
            log_ratio_current_ssr_matrixinv_2 = log_ratio(current_xi_2, eta_2[active_set_2], X_eta_tX_matrix_2, y, a0, b0, xi_interval)
        else:
            log_ratio_current_ssr_matrixinv_2 = log_ratio_approx(current_xi_2, eta_2, X, X_transpose, y, a0, b0, active_set_2, xi_interval)
    else:
        standard_normal = np.random.normal(0, 1, 1)
        log_proposed_xi_1 = standard_normal * np.sqrt(std_MH) + np.log(current_xi_1)

        relative_error_delta = np.abs(np.log(current_xi_1) - np.log(current_xi_2))

        if 0 < relative_error_delta and relative_error_delta < epsilon_xi:
            if np.log(norm.pdf(log_proposed_xi_1, loc=np.log(current_xi_1), scale=std_MH)) + np.log(np.random.uniform(0, 1, 1)) < np.log(norm.pdf(log_proposed_xi_1, loc=np.log(current_xi_2), scale=std_MH)):
                log_proposed_xi_2 = log_proposed_xi_1
            else:
                reject = True
                y_proposal = np.nan
                attempts = 0
                while reject:
                    attempts += 1
                    y_proposal = np.random.normal(np.log(current_xi_2), std_MH)
                    reject = np.log(norm.pdf(y_proposal, np.log(current_xi_2), std_MH)) + np.log(np.random.uniform(0, 1, 1)) < np.log(norm.pdf(y_proposal, np.log(current_xi_1), std_MH))
                log_proposed_xi_2 = y_proposal
        else:
            log_proposed_xi_2 = standard_normal * np.sqrt(std_MH) + np.log(current_xi_2)

        proposed_xi_1 = np.exp(log_proposed_xi_1)
        proposed_xi_2 = np.exp(log_proposed_xi_2)

        min_xi_1 = min(current_xi_1, proposed_xi_1)
        min_xi_2 = min(current_xi_2, proposed_xi_2)
        active_set_1 = ((min_xi_1 * eta_1)**(-1) > approximate_algo_delta)
        active_set_2 = ((min_xi_2 * eta_2)**(-1) > approximate_algo_delta)

        if sum(active_set_1) > n:
            X_eta_tX_matrix_1 = X_eta_tX(eta_1[active_set_1], X[:, active_set_1], X_transpose[active_set_1])
            log_ratio_current_ssr_matrixinv_1 = log_ratio(current_xi_1, eta_1[active_set_1], X_eta_tX_matrix_1, y, a0, b0, xi_interval)
            log_ratio_proposed_ssr_matrixinv_1 = log_ratio(proposed_xi_1, eta_1[active_set_1], X_eta_tX_matrix_1, y, a0, b0, xi_interval)
        else:
            log_ratio_current_ssr_matrixinv_1 = log_ratio_approx(current_xi_1, eta_1, X, X_transpose, y, a0, b0, active_set_1, xi_interval)
            log_ratio_proposed_ssr_matrixinv_1 = log_ratio_approx(proposed_xi_1, eta_1, X, X_transpose, y, a0, b0, active_set_1, xi_interval)

        if sum(active_set_2) > n:
            X_eta_tX_matrix_2 = X_eta_tX(eta_2[active_set_2], X[:, active_set_2], X_transpose[active_set_2])
            log_ratio_current_ssr_matrixinv_2 = log_ratio(current_xi_2, eta_2[active_set_2], X_eta_tX_matrix_2, y, a0, b0, xi_interval)
            log_ratio_proposed_ssr_matrixinv_2 = log_ratio(proposed_xi_2, eta_2[active_set_2], X_eta_tX_matrix_2, y, a0, b0, xi_interval)
        else:
            log_ratio_current_ssr_matrixinv_2 = log_ratio_approx(current_xi_2, eta_2, X, X_transpose, y, a0, b0, active_set_2, xi_interval)
            log_ratio_proposed_ssr_matrixinv_2 = log_ratio_approx(proposed_xi_2, eta_2, X, X_transpose, y, a0, b0, active_set_2, xi_interval)

        log_u = np.log(np.random.uniform(0, 1, 1))

        if log_u < (log_ratio_proposed_ssr_matrixinv_1 - log_ratio_current_ssr_matrixinv_1) + (log_ratio_proposed_ssr_matrixinv_2 - log_ratio_current_ssr_matrixinv_2):
            min_xi_1 = proposed_xi_1
            min_xi_2 = proposed_xi_2

    return min_xi_1, min_xi_2


# Example
n = 100
p = 10
X = np.random.normal(0, 1, (n, p))
y = np.random.normal(0, 1, n)
X_transpose = X.T
eta = np.random.gamma(1, 1, p)

current_xi_1 = 1
eta_1 = np.random.gamma(1, 1, p)
current_xi_2 = 1
eta_2 = np.random.gamma(1, 1, p)
a0 = 1
b0 = 1
std_MH = 0.1
approximate_algo_delta = 0
epsilon_xi = 0
fixed = False
xi_interval = None

start_time = time.time()
min_xi_1, min_xi_2 = crn_max_xi_coupling(current_xi_1, eta_1, current_xi_2, eta_2, X, X_transpose, y, a0, b0, std_MH, approximate_algo_delta, epsilon_xi, fixed, xi_interval)
end_time = time.time()

print("min_xi_1:", min_xi_1)
print("min_xi_2:", min_xi_2)
print("Execution time:", end_time - start_time, "seconds")
