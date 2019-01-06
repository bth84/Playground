import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import seaborn as sns
import pymc3 as pm
import theano.tensor as tt


#Difference between two similar, but yet different distributions (pdf)
x_axis = np.arange(-10,10,0.001)
#μ = 0, σ = 2

dist_a = stats.norm.pdf(x_axis, 0, 2)
dist_b = stats.norm.pdf(x_axis, 1, 2)

plt.plot(x_axis, dist_a)
plt.plot(x_axis, dist_b)
plt.fill_between(x_axis, dist_a, dist_b, where=dist_b > dist_a, facecolor='green', interpolate = True)
plt.fill_between(x_axis, dist_a, dist_b, where=dist_b < dist_a, facecolor='blue', interpolate = True)
plt.show()


#computing first KL Differences
actual = np.array([.4, .6])   #actual distribution
model1 = np.array([.2, .8])   #arbitrary different distribution
model2 = np.array([.35, .65]) #another arbitray different distribution

kl1 = (model1 * np.log(model1/actual)).sum()
print('Model 1: ', kl1 )
kl2 = (model2 * np.log(model2/actual)).sum()
print('Model 2: ', kl2)






### Dice, Polls & Dirichlet

y = np.asarray([20,  21, 17, 19, 17, 28])
k = len(y)
p = 1/k
n = y.sum()

with pm.Model() as dice_model:
    # initializes the Dirichlet distribution with a uniform prior:
    a = np.ones(k)

    theta = pm.Dirichlet("theta", a=a)

    # Since theta[5] will hold the posterior probability of rolling a 6
    # we'll compare this to the reference value p = 1/6
    six_bias = pm.Deterministic("six_bias", theta[k - 1] - p)

    results = pm.Multinomial("results", n=n, p=theta, observed=y)


pm.model_to_graphviz(dice_model)
plt.show()

with dice_model:
    dice_trace = pm.sample(1000)

with dice_model:
    pm.traceplot(dice_trace, combined=True, lines={"theta": p})
    plt.show()
