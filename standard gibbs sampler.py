import numpy as np
import pandas as pd
from scipy.stats import invgamma, multivariate_normal, t, gamma
from numpy.linalg import cholesky

np.random.seed(200)

nombre_observations = 10000

# Matrice de modèle simulée
X = np.column_stack([np.random.normal(0, 1, nombre_observations), np.random.normal(0, 1, nombre_observations)])

# Vrais coefficients beta
vrais_coefficients_beta = np.array([1000, 50])

# Vraie valeur de phi
vraie_phi = 10000
matrice_identite = np.eye(nombre_observations)  # Matrice identité utilisée pour la matrice de covariance

# Simuler la variable dépendante pour la régression
y = multivariate_normal.rvs(mean=np.dot(X, vrais_coefficients_beta),
                            cov=vraie_phi * matrice_identite)

# Valeurs initiales
n, p = X.shape
beta_init = np.ones(p)
eta_init = np.ones(p)
eta = np.ones(p)
zeta = 1
zeta_init = 1
sigma_sq = 3
num_iterations = 10000
nu = 2
mu=2
print(n,p)

# Fonction pour calculer mu_j
def compute_mu_j(beta, X, y, eta, sigma_sq, j):
    eta_j = eta[j]
    mu_j = np.sum(X[:, j] * (y - np.dot(X, beta) + X[:, j] * beta[j])) / (X[:, j].dot(X[:, j]) / eta_j + 1 / sigma_sq)
    return mu_j

def compute_sigma_j_sq(X, eta, sigma_sq, j):
    sigma_j_sq = 1 / (np.sum(X[:, j]**2) / eta[j] + 1 / sigma_sq)
    assert sigma_j_sq.shape == (), f"sigma_j_sq should be a scalar, but its shape is {sigma_j_sq.shape}"
    return sigma_j_sq
    print(sigma_j_sq)

# Fonction pour évaluer la fonction de densité de la distribution conditionnelle de chaque composante de 𝜼_t
def cond_density_eta(eta_j, beta, sigma_sq, mu, X, y):
    # Calculer la matrice de covariance Sigma
    Sigma = np.linalg.inv(np.dot(X.T, X) + mu * np.eye(X.shape[1]))
    # Calculer le vecteur de moyenne m
    m = np.dot(Sigma, np.dot(X.T, y))
    # Calculer la valeur de la fonction de densité de la distribution conditionnelle de eta_j
    density = np.exp(-mu * eta_j**2 / (2 * sigma_sq)) * np.exp(-np.dot(beta - m, np.dot(Sigma, beta - m)) / (2 * sigma_sq))
    return density
print(cond_density_eta(eta_j, beta_init, sigma_sq, mu, X, y))

# Fonction pour implémenter le slice sampling pour chaque composante de 𝜼_t
def slice_sampling_eta(beta, sigma_sq, mu, X, y, n_samples):
    # Initialiser la matrice des échantillons de 𝜼_t
    eta_samples = np.zeros((n_samples, X.shape[1]))
    # Initialiser la valeur initiale de 𝜼_t
    eta = np.random.normal(0, 1, X.shape[1])
    # Itération du slice sampling pour chaque composante de 𝜼_t
    for j in range(X.shape[1]):
        # Itération du slice sampling pour chaque échantillon de 𝜼_t,j
        for i in range(n_samples):
            # Évaluer la fonction de densité de la distribution conditionnelle de eta_j à la valeur actuelle de eta_j
            density = cond_density_eta(eta[j], beta, sigma_sq, mu, X, y)
            # Générer une valeur aléatoire uniforme entre 0 et la valeur de la fonction de densité
            u = np.random.uniform(0, density)
            # Trouver l'intervalle horizontal qui contient u dans le graphique de la fonction de densité
            # Utiliser la recherche par dichotomie pour trouver l'intervalle
            lower_bound = eta[j] - 1
            upper_bound = eta[j] + 1
            while True:
                if cond_density_eta(lower_bound, beta, sigma_sq, mu, X, y) < u:
                    lower_bound = (lower_bound + eta[j]) / 2
                elif cond_density_eta(upper_bound, beta, sigma_sq, mu, X, y) < u:
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

print(slice_sampling_eta(beta_init, sigma_sq, mu, X, y, n_samples=10))

def calculate_likelihood(y, zeta, eta, beta, sigma_sq):   
    """
    Calcul de la vraisemblance de y sachant zeta et eta.
    
    Arguments :
    y : Vecteur des observations.
    X : Matrice des prédicteurs.
    beta: Échantillon de beta.
    sigma_sq : Échantillon de sigma_sq.
    zeta : Valeur de zeta_t+1.
    eta : Vecteur des valeurs eta_t+1.

    Returns :
    log_likelihood : Log de la vraisemblance de y sachant zeta et eta.
    """

    # On calcule d'abord M_zeta_eta : 
    M_zeta_eta = np.eye(n) + 1/zeta * X@np.diag(1/eta)@X.T

    log_likelihood = -1/2 * np.log(np.linalg.det(M_zeta_eta)) - (1+n)/2 * np.log(1 + y.T@M_zeta_eta@y)
    
    return log_likelihood

beta = beta_init
print(calculate_likelihood(y, zeta, eta, beta, sigma_sq))


def sample_zeta(zeta_previous, eta_sampled, beta, sigma, sigma_mrth=0.8): 
    """
    Échantillonne la valeur de zeta_t+1 conditionnellement à zeta_previous et eta_t+1.

    Arguments :
    zeta_previous : Valeur zeta_t.
   eta_tplus1 : Valeur échantillonnée de eta_t+1.
    y : Vecteur de données.
    X : Matrice de design.
    beta : Vecteur de coefficients beta_t.
    sigma_sq : Variance sigma_t^2.
    sigma_mrth : Écart-type de la proposition normale = 0.8 selon l'article.

    Returns :
    zeta_sampled : Valeur échantillonnée de zeta_t+1.
    """

    # Proposition d'un nouvel échantillon de log(zeta_t+1)
    log_zeta_proposed = np.random.normal(np.log(zeta_previous), sigma_mrth)
    
    
    # Calcul des termes de probabilité a priori
    prior_current = -0.5 * zeta_previous ** 2
    prior_proposed = -0.5 * np.exp(2 * log_zeta_proposed)

    # On calcule la vraisemblance conditionnelle des données
    log_likelihood = (calculate_likelihood(y, np.exp(log_zeta_proposed), eta_sampled, beta, sigma)) # cf fonction calculate_likelihood au dessus
    
    # Log-probabilité du log-posterior pour les valeurs actuelles et proposées
    log_posterior_current = log_likelihood + prior_current
    log_posterior_proposed = log_likelihood + prior_proposed
    
    # Calcul du ratio de probabilité
    acceptance_ratio = np.exp(log_posterior_proposed - log_posterior_current)
    
    
    # Acceptation ou rejet de la proposition
    if np.random.uniform(0, 1) < acceptance_ratio:
        zeta_sampled = np.exp(log_zeta_proposed)
    else:
        zeta_sampled = zeta_previous

    return zeta_sampled

# Fonction pour l'échantillonnage de Gibbs coordonnée par coordonnée
def gibbs_sampling_coord(X, y, beta_init, eta_init, zeta_init, sigma_sq, num_iterations, nu):
    zeta = zeta_init
    eta = eta_init
    beta = beta_init
    betas = [[] for _ in range(num_iterations)] 
    # Boucle sur le nombre d'itérations
    for t in range(num_iterations):
        # Mettre à jour chaque coordonnée de beta
        for j in range(p):
            mu_j = compute_mu_j(beta, X, y, eta, sigma_sq, j)
            sigma_j_sq = compute_sigma_j_sq(X, eta, sigma_sq, j)
            beta[j] = np.random.normal(mu_j, sigma_j_sq)
            betas[t].append(beta[j])
        # Mettre à jour chaque coordonnée de eta
        slice_sampling_eta(beta, sigma_sq, mu, X, y, n_samples=10)
        # Mettre à jour zeta
        zeta = sample_zeta(zeta, eta, beta, sigma_sq, sigma_mrth=0.8)
    return betas, eta, zeta

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

betas = np.array(beta_samples)
print( "betas:", betas)
samples = betas
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

hist, xedges, yedges = np.histogram2d(samples[:, 0], samples[:, 1], bins=50)
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = np.zeros_like(xpos)

dx = dy = 0.5 * np.ones_like(zpos)
dz = hist.ravel()

ax.scatter(xpos, ypos, dz, c='b', marker='o')

ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Frequency')
ax.set_title('Blocked Gibbs Sampling (dim = 2)')
path = "/home/onyxia/work/Monte-Carlo/"
plt.savefig(path + 'test bgs.png')
plt.show()
    




