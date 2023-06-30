import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from scipy.optimize import root


pixel_size_wide = 0.00525
pixel_num = 2500
ccd_length = pixel_size_wide * pixel_num
tar_resolation = 0.01
tar_mea_range = 10
pixel_num_resolation_min = 2
pixel_num_resolation_max = 3
theta_min = pixel_size_wide * pixel_num_resolation_min / tar_resolation
theta_max = pixel_size_wide * pixel_num_resolation_max / tar_resolation
alpha = 30/360*2*np.pi

def f(x):
    return theta_min*np.cos(alpha)/np.square(np.sin(alpha)) - np.cos(x)/(1-np.square(np.cos(x)))

beta = root(f,30/360*2*np.pi).x
print(theta_min)
print(beta)
a_0=60
b_0=a_0*np.tan(alpha)/np.tan(beta)
print(b_0)

x=np.linspace(-10,20,1000)
# y=1*x*np.sin(np.pi/4)/(1*np.sin(np.pi/4)-x*np.sin(np.pi/2))
y=x*b_0*np.sin(alpha)/(a_0*np.sin(beta)-x*np.sin(alpha+beta))
theta=a_0*b_0*np.sin(alpha)*np.sin(beta)/np.square(a_0*np.sin(beta)-x*np.sin(alpha+beta))
focal_length = a_0*b_0/(a_0+b_0)
print(focal_length)
plt.subplot(2,1,1)
plt.plot(x,y)
plt.subplot(2,1,2)
plt.plot(x,theta)

plt.show()