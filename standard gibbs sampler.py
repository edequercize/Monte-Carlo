import numpy as np
import pandas as pd
from scipy.stats import invgamma, multivariate_normal, t, gamma
from numpy.linalg import cholesky

np.random.seed(200)

# Nombre d'observations
nombre_observations = 50

# Matrice de modèle simulée
X = np.column_stack([np.ones(nombre_observations), np.random.normal(0, 1, nombre_observations),
                     np.random.normal(5, 10, nombre_observations), np.random.normal(100, 10, nombre_observations)])

# Vrais coefficients beta
vrais_coefficients_beta = np.array([1000, 50, -50, 10])

# Vraie valeur de phi
vraie_phi = 10000
matrice_identite = np.eye(nombre_observations)  # Matrice identité utilisée pour la matrice de covariance

# Simuler la variable dépendante pour la régression
y = multivariate_normal.rvs(mean=np.dot(X, vrais_coefficients_beta),
                            cov=vraie_phi * matrice_identite)

# Valeurs initiales
n, p = X.shape
beta_init = np.zeros(p)
eta = np.zeros(p)
zeta = 0
sigma_sq = 1
num_iterations = 1000
nu = 2

# Fonction pour calculer mu_j
def compute_mu_j(beta, X, y, eta, sigma_sq, j):
    eta_j = eta[j]
    n = X.shape[0]
    mu_j = np.sum(X[:, j] * (y - np.dot(X, beta) + X[:, j] * beta[j])) / (X[:, j].dot(X[:, j]) / eta_j + 1 / sigma_sq)
    return mu_j

# Fonction pour calculer sigma_j^2
def compute_sigma_j_sq(X, eta, sigma_sq, j):
    eta_j = eta[j]
    sigma_j_sq = 1 / (X[:, j].dot(X[:, j]) / eta_j + 1 / sigma_sq)
    return sigma_j_sq

def target_density_eta(eta, beta, sigma_sq, zeta, nu):
    """
    Calcul de la densité a posteriori conditionnelle de eta_t+1.

    Arguments :
    eta : Valeurs échantillonnées de eta_t+1.
    beta : Vecteur de coefficients beta_t.
    sigma_sq : Variance sigma_t^2.
    zeta : Valeur zeta_t.
    nu : Paramètre nu.

    Returns :
    posterior_density : Densité a posteriori conditionnelle de eta_t+1.
    """
    p = len(eta)
    posterior_density = 1

    for j in range(p):
        m_tj = zeta * beta[j]**2 / (2 * sigma_sq)
        # Terme exponentiel
        exp_term = np.exp(-m_tj * eta[j])
        # Terme de normalisation
        normalization_term = eta[j]**((1 - nu) / 2) * (1 + nu * eta[j])**(nu + 1)
        # Mise à jour de la densité a posteriori conditionnelle
        posterior_density *= exp_term / normalization_term

    return posterior_density

def slice_sampling_eta(num_features, beta, sigma_sq, zeta, nu, initial_value_eta, num_samples, step_size=1.0):
    samples_eta = [initial_value_eta]

    for _ in range(num_samples):
        current_value_eta = samples_eta[-1]
        # Étape de "slice"
        height_eta = np.random.uniform(0, target_density_eta(current_value_eta, beta, sigma_sq, zeta, nu))
        # Étape de réduction de la tranche
        left_eta = current_value_eta - np.random.exponential(scale=step_size)
        right_eta = left_eta + step_size
        while target_density_eta(left_eta, beta, sigma_sq, zeta, nu) < height_eta:
            left_eta -= step_size
        while target_density_eta(right_eta, beta, sigma_sq, zeta, nu) < height_eta:
            right_eta += step_size
        # Étape d'échantillonnage
        new_value_eta = np.random.uniform(left_eta, right_eta)
        samples_eta.append(new_value_eta)

    return samples_eta[1:]  # On retire la valeur initiale


# Fonction pour l'échantillonnage de Gibbs
def gibbs_sampling(X, y, beta_init, eta_init, zeta_init, sigma_sq, num_iterations, nu):
    num_features = X.shape[1]
    zeta = zeta_init
    eta = eta_init
    # Boucle sur le nombre d'itérations
    for t in range(num_iterations):
        beta_new = beta_init
        eta = slice_sampling_eta(num_features, beta_new, sigma_sq, zeta, nu, eta, 1)
        zeta = sample_zeta(zeta, eta, sigma_sq, sigma_mrth=0.8)

        # Boucle sur chaque coordonnée beta_j
        for j in range(num_features):
            mu_j = compute_mu_j(beta_new, X, y, eta, sigma_sq, j)
            sigma_j_sq = compute_sigma_j_sq(X, eta, sigma_sq, j)
            beta_new[j] = np.random.normal(mu_j, np.sqrt(sigma_j_sq))

        # Mettre à jour les coordonnées beta pour l'itération suivante
        beta_current = beta_new

    return beta_current

# Valeurs initiales
n, p = X.shape
beta_init = np.zeros(p)
eta = np.zeros(p)
zeta = 0
sigma_sq = 1
num_iterations = 1000
nu = 2

print(gibbs_sampling(X, y, beta_init, eta, sigma_sq, num_iterations, nu))
