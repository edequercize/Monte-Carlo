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
beta_init = np.ones(p)
eta = np.ones(p)
zeta = 1
sigma_sq = 1
num_iterations = 1000
nu = 2

# Fonction pour calculer mu_j
def compute_mu_j(beta, X, y, eta, sigma_sq, j):
    eta_j = eta[j]
    n = X.shape[0]
    mu_j = np.sum(X[:, j] * (y - np.dot(X, beta) + X[:, j] * beta[j])) / (X[:, j].dot(X[:, j]) / eta_j + 1 / sigma_sq)
    return mu_j

def compute_sigma_j_sq(X, eta, sigma_sq, j):
    sigma_j_sq = 1 / (np.sum(X[:, j]**2) / eta[j] + 1 / sigma_sq)
    assert sigma_j_sq.shape == (), f"sigma_j_sq should be a scalar, but its shape is {sigma_j_sq.shape}"
    return sigma_j_sq
    print(sigma_j_sq)

# Fonction pour évaluer la fonction de densité de la distribution conditionnelle de chaque composante de 𝜼_t
def cond_density_eta(eta_j, beta, sigma2, mu, X, y):
    # Calculer la matrice de covariance Sigma
    Sigma = np.linalg.inv(np.dot(X.T, X) + mu * np.eye(X.shape[1]))
    # Calculer le vecteur de moyenne m
    m = np.dot(Sigma, np.dot(X.T, y))
    # Calculer la valeur de la fonction de densité de la distribution conditionnelle de eta_j
    density = np.exp(-mu * eta_j**2 / (2 * sigma2)) * np.exp(-np.dot(beta - m, np.dot(Sigma, beta - m)) / (2 * sigma2))
    return density

# Fonction pour implémenter le slice sampling pour chaque composante de 𝜼_t
def slice_sampling_eta(beta, sigma2, mu, X, y, n_samples):
    # Initialiser la matrice des échantillons de 𝜼_t
    eta_samples = np.zeros((n_samples, X.shape[1]))
    # Initialiser la valeur initiale de 𝜼_t
    eta = np.random.normal(0, 1, X.shape[1])
    # Itération du slice sampling pour chaque composante de 𝜼_t
    for j in range(X.shape[1]):
        # Itération du slice sampling pour chaque échantillon de 𝜼_t,j
        for i in range(n_samples):
            # Évaluer la fonction de densité de la distribution conditionnelle de eta_j à la valeur actuelle de eta_j
            density = cond_density_eta(eta[j], beta, sigma2, mu, X, y)
            # Générer une valeur aléatoire uniforme entre 0 et la valeur de la fonction de densité
            u = np.random.uniform(0, density)
            # Trouver l'intervalle horizontal qui contient u dans le graphique de la fonction de densité
            # Utiliser la recherche par dichotomie pour trouver l'intervalle
            lower_bound = eta[j] - 1
            upper_bound = eta[j] + 1
            while True:
                if cond_density_eta(lower_bound, beta, sigma2, mu, X, y) < u:
                    lower_bound = (lower_bound + eta[j]) / 2
                elif cond_density_eta(upper_bound, beta, sigma2, mu, X, y) < u:
                    upper_bound = (upper_bound + eta[j]) / 2
                else:
                    break
            # Générer une valeur aléatoire uniforme dans l'intervalle horizontal
            eta_new = np.random.uniform(lower_bound, upper_bound)
            # Accepter la nouvelle valeur avec probabilité 1
            eta[j] = eta_new
            # Enregistrer la valeur actuelle de eta_j dans la matrice des échantillons
            eta_samples[i, j] = eta[j]
    return eta_samples

# Définir la fonction de proposition
def prop_log_zeta(log_zeta_actuel, sigma_prop):
    return np.random.normal(log_zeta_actuel, sigma_prop)

# Définir la fonction pour mettre à jour zeta
def metropolis_hastings_zeta(y, X, omega, zeta_actuel, sigma_prop, a_prior, b_prior, n_iter):
    # Initialiser la chaîne de Markov pour zeta
    zeta_chain = np.zeros(n_iter)
    zeta_chain[0] = zeta_actuel
    # Boucle pour les itérations du Gibbs sampling
    for i in range(1, n_iter):
        # Générer une nouvelle valeur proposée pour log(zeta)
        log_zeta_prop = prop_log_zeta(np.log(zeta_actuel), sigma_prop)
        # Calculer la vraisemblance marginale de y donné omega et la nouvelle valeur proposée de zeta
        M = np.eye(len(y)) + 1/zeta_prop * X @ np.diag(1/omega) @ X.T
        L_prop = np.linalg.det(M)**(-len(y)/2) * np.exp(-1/2 * y.T @ np.linalg.inv(M) @ y)
        # Calculer la vraisemblance marginale de y donné omega et la valeur actuelle de zeta
        M_actuel = np.eye(len(y)) + 1/zeta_actuel * X @ np.diag(1/omega) @ X.T
        L_actuel = np.linalg.det(M_actuel)**(-len(y)/2) * np.exp(-1/2 * y.T @ np.linalg.inv(M_actuel) @ y)
        # Calculer le rapport d'acceptation
        alpha = min(1, L_prop * np.exp(-a_prior * log_zeta_prop - b_prior * np.exp(-log_zeta_prop)) / L_actuel * np.exp(-a_prior * np.log(zeta_actuel) - b_prior * zeta_actuel))
        # Accepter ou rejeter la nouvelle valeur proposée
        if np.random.rand() < alpha:
            zeta_actuel = np.exp(log_zeta_prop)
        # Enregistrer la valeur actuelle de zeta dans la chaîne de Markov
        zeta_chain[i] = zeta_actuel
    # Retourner la chaîne de Markov pour zeta
    return zeta_chain

# Fonction pour l'échantillonnage de Gibbs coordonnée par coordonnée
def gibbs_sampling_coord(X, y, beta_init, eta_init, zeta_init, sigma_sq, num_iterations, nu):
    num_features = X.shape[1]
    zeta = zeta_init
    eta = eta_init
    beta = beta_init
    # Boucle sur le nombre d'itérations
    for t in range(num_iterations):
        # Mettre à jour chaque coordonnée de beta
        for j in range(num_features):
            mu_j = compute_mu_j(beta, X, y, eta, sigma_sq, j)
            sigma_j_sq = compute_sigma_j_sq(X, eta, sigma_sq, j)
            print(sigma_j_sq, j)
            beta[j] = np.random.normal(mu_j, np.sqrt(max(sigma_j_sq, 0)))
        # Mettre à jour chaque coordonnée de eta
        eta = slice_sampling_eta(beta, sigma_sq, zeta, X, y, 1)
        # Mettre à jour zeta
        zeta = metropolis_hastings_zeta(y, X, eta, zeta, 0.5, 0, 0, 1)[0]
    return beta, eta, zeta



# Échantillonnage de Gibbs coordonnée par coordonnée
beta_samples, eta_samples, zeta_samples = gibbs_sampling_coord(X, y, beta_init, eta, zeta, sigma_sq, num_iterations, nu)

# Afficher les résultats
print("Estimation de beta :", np.mean(beta_samples, axis=0))
print("Estimation de eta :", np.mean(eta_samples, axis=0))
print("Estimation de zeta :", np.mean(zeta_samples))
