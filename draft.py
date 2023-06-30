import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button

alpha = 40
beta = 40
a_0=10
b_0=10

x=np.linspace(-10,10,1000)
# y=1*x*np.sin(np.pi/4)/(1*np.sin(np.pi/4)-x*np.sin(np.pi/2))
y=x*10*np.sin(np.pi/4)/(10*np.cos(np.pi/4)+x)
plt.plot(x,y)

plt.show()