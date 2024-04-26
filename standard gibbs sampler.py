import numpy as np
import time

def half_t_sample(nu):
    u = np.random.uniform(size=1)
    eta = 1 / (u**((2 - nu) / 2) * (1 + nu * u)**((nu + 1) / 2))
    return eta
#La fonction half_t_sample(nu) retourne un échantillon tiré de cette distribution half-t avec le paramètre de degré de liberté nu. 
#Ce paramètre contrôle à quel point la distribution est "épaisse" autour de zéro. 
# Plus nu est grand, plus la queue de la distribution devient lourde, ce qui signifie qu'elle assigne plus de probabilité aux valeurs éloignées de zéro.

# Initialisation des paramètres
def initialisation(p, a0, b0, nu=1):
    sigma_sq = 1 / np.random.gamma(a0, 1/b0)
    beta = np.zeros(p)
    eta = np.zeros(p)
    zeta = (abs(np.random.standard_cauchy()))**(-2)
    for j in range(p):
        eta[j] = half_t_sample(nu)
        beta[j] = np.random.normal(0, sigma_sq / (eta[j] * zeta))
    return beta.tolist(), sigma_sq, zeta, eta.tolist()
initialisation(2,1,1)

# Mise à jour séquentielle de zeta grâce à une distribution de Cauchy
def maj_eta_j(beta, sigma_sq, zeta, nu, j):
    m_tj = zeta * (beta[j]**2) / (2 * sigma_sq)
    u = np.random.uniform(size=1)
    eta_t = (np.exp(-m_tj * u)) / (u**((1 - nu) / 2) * (1 + nu * u)**((nu + 1) / 2))
    return eta_t
    print(eta_t)
maj_eta_j([1],1,1,1,0)

def maj_zeta(beta, zeta, eta, nu, sigma_mrth):
    p = len(beta)
    for j in range(p):
        # Proposer une nouvelle valeur pour log(zeta[j]) avec une proposition normale
        log_zeta_prop = np.log(zeta[j]) + np.random.normal(scale=sigma_mrth)
        zeta_prop = np.exp(log_zeta_prop)
        # Calculer la densité de probabilité de la nouvelle proposition
        density_prop = np.prod(np.sqrt((1 + nu * eta[j]**2) / zeta_prop) * np.exp(-(beta[j]**2) / (2 * zeta_prop * (1 + nu * eta[j]**2))))
        # Calculer la densité de probabilité de l'état actuel
        density_current = np.prod(np.sqrt((1 + nu * eta[j]**2) / zeta[j]) * np.exp(-(beta[j]**2) / (2 * zeta[j] * (1 + nu * eta[j]**2))))
        # Calculer le ratio de probabilité de transition
        ratio = density_prop / density_current
        # Accepter ou rejeter la proposition
        if np.random.uniform() < min(1, ratio):
            zeta[j] = zeta_prop
    return zeta
maj_zeta([1], [1], [1], 1, 0.2)

# Mise à jour séquentielle de beta
def maj_beta_j(X, y, sigma_sq, zeta, eta, j):
    n, p = X.shape
    XtX = np.dot(X.T, X)
    eta = np.array(eta)
    XtX_inv = np.linalg.inv(XtX + np.diag(1.0 / eta))
    beta = np.dot(XtX_inv, np.dot(X.T, y)) / sigma_sq
    return beta[j]

X = np.array([[1, 2], [3, 4]])
maj_beta_j(X, 1, 1, [1], [1], 0)




