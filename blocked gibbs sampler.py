import numpy as np
import time

# Initialisation des parametres
def initialisation(p,a0, b0,nu=1):
    sigma_sq = 1 / np.random.gamma(a0, 1/b0)
    beta = np.zeros(p)
    eta = np.zeros(p)
    zeta = (abs(np.random.standard_cauchy()))**(-2)
    for j in range(p):
        eta[j] = half_t_sample(nu)
        beta[j] = np.random.normal(0,sigma_sq/(eta[j]*zeta))
    return beta, sigma_sq, zeta, eta

def half_t_sample(nu):
    u = np.random.uniform(size=1)
    eta = 1 / (u**((2-nu)/2) * (1 + nu*u)**((nu+1)/2))
    return eta

# Mise à jour de eta grâce à la formule cf (4) : loi de cauchy 
#"We sample 𝜂t+1|𝛽t,𝜎2t ,𝜉t component-wise independently using slice sampling"
def maj_eta(beta,sigma_sq,zeta,nu):
    p=len(beta)
    m_t=[]
    eta_t=[]
    for j in range (p):
        m_tj= zeta * (beta[j]**2) /(2* sigma_sq)
        m_t.append(m_tj)
    for j in range (p):
        u = np.random.uniform(size=1)
        eta_t.append( (np.exp(-m_t[j]*u)) / (u**((1-nu)/2) * (1 + nu*u)**((nu+1)/2)))
    return eta_t

# Mise à jour de zeta
#"sample 𝜉t+1|𝜂t+1 using Metropolis–Rosenbluth–Teller–Hastings, with a Normal proposal on the logarithm of 𝜉 with standard deviation 𝜎MRTH"

#ca code pas ce qu'ils ont dit je crois
'''def maj_zeta(beta, zeta, eta, nu):
    p = len(beta)
    for j in range(p):
        eta_j_sq = np.sum((beta[j] / (zeta * (1 + nu * eta[j] ** 2))) ** 2)
        eta[j] = np.sqrt(np.random.noncentral_chisquare(nu + 1, eta_j_sq))
    return eta'''
import numpy as np

def maj_zeta(beta, zeta, eta, nu, sigma_mrth):
    #σ MRTH represents the standard deviation of the normal proposal distribution used for the Metropolis step
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
    
'''En résumé, l'algorithme Metropolis-Hastings génère une séquence de propositions d'états candidats pour les paramètres à échantillonner, 
en les acceptant ou en les rejetant selon un ratio calculé à partir des densités de probabilité des états candidats et actuels. 
Cela permet d'explorer progressivement l'espace des paramètres et de générer des échantillons à partir de la distribution cible.'''

# Mise à jour de beta
def maj_beta(X, y, sigma_sq, zeta, eta):
    n, p = X.shape
    XtX = np.dot(X.T, X)
    XtX_inv = np.linalg.inv(XtX + np.diag(1.0 / eta))  
    beta = np.dot(XtX_inv, np.dot(X.T, y)) / sigma_sq
    return beta

# Mise à jour de sigma
def maj_sigma_sq(X, y, beta, zeta, eta,a0,b0):
    n, p = X.shape
    M= np.eye(n)+ np.dot(np.dot(np.dot(np.linalg.inv(zeta), X), np.diag(np.linalg.inv(eta))), np.transpose(X))
    a=(a0+n)/2
    b=((np.dot(np.dot(np.transpose(y),np.linalg.inv(M) ),y))+b0)/2
    return 1 / np.random.gamma(a, 1/b)

# Step 6: Blocked Gibbs Sampler
def blocked_gibbs_sampler(X, y, num_iterations,a0, b0,nu=1):
    n, p = X.shape
    beta, sigma_sq, zeta, eta = initialisation(p,a0, b0,nu)
    samples = []
    for _ in range(num_iterations):
        beta = maj_beta(X, y, sigma_sq, zeta, eta)
        sigma_sq = maj_sigma_sq(X, y, beta, zeta, eta,a0,b0)
        zeta = maj_zeta(beta)
        eta = maj_eta(beta, zeta, eta, nu)
        samples.append((beta, sigma_sq, zeta, eta))
    return samples

#Example 
n = 100
p = 10
X = np.random.normal(0, 1, (n, p))
y = np.random.normal(0, 1, n)
X_transpose = X.T
a0 = 1
b0 = 1
nu=1
num_iterations=1000
start_time = time.time()
end_time = time.time()
print(blocked_gibbs_sampler(X, y, num_iterations,a0, b0,nu))
print(end_time - start_time)

