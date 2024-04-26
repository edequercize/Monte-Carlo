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
