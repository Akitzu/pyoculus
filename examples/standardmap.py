import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import jacfwd

def standardmap(xy, K):
    ynew = xy[1] - K*jnp.sin(2*jnp.pi*xy[0])/(2*jnp.pi)
    xnew = jnp.mod(xy[0] + ynew, 1)
    return jnp.array([xnew, ynew])

def reversedmap(xy, K):
    xold = jnp.mod(xy[0] - xy[1], 1)
    yold = xy[1] + K*jnp.sin(2*jnp.pi*xold)/(2*jnp.pi)
    return jnp.array([xold, yold])

from jax import jit
jitedmap = jit(standardmap, static_argnums=1)
jitedreversed = jit(reversedmap, static_argnums=1)

# def standardmap(x, y, K=1.2):
#     ynew = np.mod(y + K*np.sin(x), 2*np.pi)
#     xnew = np.mod(x + ynew, 2*np.pi)
#     return xnew, ynew
dstandardmap = jacfwd(standardmap)
jiteddmap = jit(dstandardmap, static_argnums=1)
xi = np.linspace(0., 1, 30)
yi = np.linspace(0.5, 1.5, 10)
# xi = np.linspace(0., 2*np.pi, 10)
Xi = np.meshgrid(xi, yi)
Xi = np.vstack((Xi[0].flatten(), Xi[1].flatten())).T
# Xi = np.vstack((np.linspace(0., 1, 10), np.linspace(0., 1, 10))).T

nev = 100
Ev = np.empty(((nev,)+Xi.shape))
Ev[0] = Xi
Ev.shape
K = 0.97
for i in range(1, Ev.shape[0]):
    evolved = np.array([jitedmap(Ev[i-1,j,:], K) for j in range(Ev.shape[1])])
    Ev[i,:,:] = evolved
fig, ax = plt.subplots()
for i in range(len(Ev[0,:,0])):
    ax.scatter(Ev[:, i, 0], Ev[:, i, 1], s=0.1, alpha=1, zorder = 10, c='black')
# ax.axis('off')
from scipy.optimize import root

def fixedpoint(xy, m=1):
    xyev = xy
    while m > 0:
        xyev = jitedmap(xyev, K)
        m -= 1
    return xy -xyev

sol = root(fixedpoint, [0.5, 0.5])
sol.x
jacobian = jiteddmap(jnp.array([0.5, 1.]), K)

fig