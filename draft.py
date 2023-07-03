import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from scipy.optimize import root

pixel_num_resolation_min =2
pixel_num_resolation_max = 3
ax_position_height = 0.4

interval = 50  # ms, time between animation frames
fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 12))

ax1.set_position([0.1, 0.75, 0.8, 0.2])
ax1.set_title("x_y")
ax2.set_position([0.1, 0.5, 0.8, 0.2])
ax2.set_title("x_theta")

ax_a_0 = plt.axes([0.15, ax_position_height, 0.65, 0.03]) #a0长度用于设计机械尺寸
ax_position_height -= 0.05

ax_alpha = plt.axes([0.15, ax_position_height, 0.65, 0.03]) #alpha角度用于设计机械尺寸
ax_position_height -= 0.05

ax_tar_mea_range = plt.axes([0.15, ax_position_height, 0.65, 0.03]) #测量范围，甲方要求
ax_position_height -= 0.05

ax_min_pixel_num_resolation = plt.axes([0.15, ax_position_height, 0.65, 0.03]) #每分辨率最小像素数，算法要求
ax_position_height -= 0.05

ax_CCD_L_pixel_num = plt.axes([0.15, ax_position_height, 0.65, 0.03]) #像素密度，甲方要求成本控制 (个像素/mm)
ax_position_height -= 0.05

ax_tar_resolation = plt.axes([0.15,ax_position_height, 0.65, 0.03]) #目标分辨率，甲方要求
ax_position_height -= 0.05


sli_a_0 = Slider(ax_a_0, "a_0", 10, 100, valinit=60)
sli_alpha = Slider(ax_alpha, "alpha", 10, 40, valinit=30)
sli_tar_mea_range = Slider(ax_tar_mea_range, "tar_mea_range", 10, 100, valinit=1)
sli_min_pixel_num_resolation = Slider(ax_min_pixel_num_resolation, "min_pixel_num_resolation", 2, 10, valinit=2)
sli_CCD_L_pixel_num = Slider(ax_CCD_L_pixel_num, "pixel_num_CCD_L", 150, 500, valinit=1/0.00525)
sli_tar_resolation = Slider(ax_tar_resolation, "tar_resolation", 0.001, 0.01, valinit=0.01)

def update(val):
    a_0 = sli_a_0.val
    alpha = sli_alpha.val / 360 * 2 * np.pi
    tar_mea_range = sli_tar_mea_range.val
    min_pixel_num_resolation = sli_min_pixel_num_resolation.val
    CCD_L_pixel_num = sli_CCD_L_pixel_num.val
    tar_resolation = sli_tar_resolation.val


sli_a_0.on_changed(update)
sli_alpha.on_changed(update)
sli_tar_mea_range.on_changed(update)
sli_min_pixel_num_resolation.on_changed(update)
sli_CCD_L_pixel_num.on_changed(update)
sli_tar_resolation.on_changed(update)

def calc_theta_min(ccd_l_pixel_num, min_pixel_num_resolation, tar_resolation):
    return 1/ccd_l_pixel_num * min_pixel_num_resolation / tar_resolation



def f_beta(x, theta_min, alpha):
    
    return theta_min * np.cos(alpha) / np.square(np.sin(alpha)) - np.cos(x) / (1 - np.square(np.cos(x)))

def calc_beta(theta_min, alpha):
    return root(f_beta, 30 / 360 * 2 * np.pi, args=(theta_min, alpha)).x

# beta= root(f_beta, 30 / 360 * 2 * np.pi).x
# beta = calc_beta(alpha)
# print("beta: ", beta)
# a_0 = 60

def calc_b_0(a_0, alpha, beta):
    return a_0 * np.tan(alpha) / np.tan(beta)

def calc_focal_length(a_0, b_0):
    return a_0 * b_0 / (a_0 + b_0)

# b_0 = a_0 * np.tan(alpha) / np.tan(beta)
# x = np.linspace(-2, 20, 1000)
# y=1*x*np.sin(np.pi/4)/(1*np.sin(np.pi/4)-x*np.sin(np.pi/2))

def update_x_y(x,a_0,b_0,alpha,beta):
    y = x * b_0 * np.sin(alpha) / (a_0 * np.sin(beta) - x * np.sin(alpha + beta))
    ax1.clear()
    ax1.plot(x, y)
    ax1.set_title("x_y")
    
def update_x_theta(x,a_0,b_0,alpha,beta):
    theta = a_0* b_0* np.sin(alpha)* np.sin(beta)/ np.square(a_0 * np.sin(beta) - x * np.sin(alpha + beta))
    ax2.clear()
    ax2.plot(x, theta)
    ax2.set_title("x_theta")


# y = x * b_0 * np.sin(alpha) / (a_0 * np.sin(beta) - x * np.sin(alpha + beta))
# theta = a_0* b_0* np.sin(alpha)* np.sin(beta)/ np.square(a_0 * np.sin(beta) - x * np.sin(alpha + beta))
# focal_length = a_0 * b_0 / (a_0 + b_0)

def animate(frame):
    a_0 = sli_a_0.val
    alpha = sli_alpha.val / 360 * 2 * np.pi
    beta = calc_beta(calc_theta_min(sli_CCD_L_pixel_num.val, sli_min_pixel_num_resolation.val, sli_tar_resolation.val), alpha)
    b_0 = calc_b_0(sli_a_0.val, alpha, beta)
    x = np.linspace(-0.2*sli_tar_mea_range.val, 1.2*sli_tar_mea_range.val, 1000)
    
    update_x_y(x,a_0,b_0,alpha,beta)
    update_x_theta(x,a_0,b_0,alpha,beta)
    fig.canvas.draw_idle()
    
ani = animation.FuncAnimation(fig, animate, interval)


plt.show()
