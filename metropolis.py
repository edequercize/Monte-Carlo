import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def target_distribution(x):
    return np.exp(-0.5 * (x[0]**2 + x[1]**2)) / (2 * np.pi)  # Distribution cible non normalisée pour cet exemple 2D

def theta_t(theta, s):
    n = len(theta)
    theta_sampled = np.random.normal(theta, s, size=n)
    return theta_sampled

def random_walk_metropolis(X, num_samples, theta0, s):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    hist, xedges, yedges = np.histogram2d(X[:, 0], X[:, 1], bins=50)
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    Y = hist.ravel()
    X = [[xpos[i], ypos[i]] for i in range(len(xpos))]

    # Créer des caractéristiques polynomiales
    degree = 10  # Degré du polynôme
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    # Ajuster un modèle de régression linéaire aux caractéristiques polynomiales
    model = LinearRegression()
    model.fit(X_poly, Y)


    def estimated_target_distribution(x, m=model):
        return m.predict(poly.fit_transform([x]))

    samples = [theta0]
    position = theta0

    for i in range(num_samples):
        proposition = theta_t(position, s)
        prob_d_acceptation = min(1, estimated_target_distribution(proposition) / estimated_target_distribution(position))

        if np.random.rand() < prob_d_acceptation:
            position = proposition

        samples.append(position)

    return np.array(samples[1:])

# Example d'usage
X = [np.random.normal(0, 1, size=2) for _ in range(1000)]
X = np.array(X)
num_samples = 1000000
initial_state = np.array([0, 0])
proposal_std = 1

samples = random_walk_metropolis(X, num_samples, initial_state, proposal_std)

# 3D scatter plot
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
ax.set_title('Metropolis-Hastings Sampling (Multidimensional)')
path = "/home/onyxia/work/Monte-Carlo/"
plt.savefig(path + 'test.png')
plt.show()
