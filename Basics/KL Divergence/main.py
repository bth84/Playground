import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


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

# actual = np.array([.4, .6])   #actual distribution
# model1 = np.array([.2, .8])   #arbitrary different distribution
# model2 = np.array([.35, .65]) #another arbitray different distribution