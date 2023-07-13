import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from scipy.optimize import root

pixel_num_resolation_min = 2
pixel_num_resolation_max = 3
ax_position_height = 0.45

# 定义各坐标轴
interval = 50  # ms, time between animation frames
fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 12))

ax1.set_position([0.1, 0.75, 0.8, 0.2])
ax2.set_position([0.1, 0.5, 0.8, 0.2])

ax_text = plt.axes([0.15, ax_position_height, 0.65, 0.03])
ax_text.axis("off")
ax_position_height -= 0.05

ax_a_0 = plt.axes([0.15, ax_position_height, 0.65, 0.03])  # a0长度用于设计机械尺寸
ax_position_height -= 0.05

ax_alpha = plt.axes([0.15, ax_position_height, 0.65, 0.03])  # alpha角度用于设计机械尺寸
ax_position_height -= 0.05

ax_gamma = plt.axes([0.15, ax_position_height, 0.65, 0.03])  # gamma角度用于设计机械尺寸
ax_position_height -= 0.05

ax_tar_mea_range = plt.axes([0.15, ax_position_height, 0.65, 0.03])  # 测量范围，甲方要求
ax_position_height -= 0.05

ax_min_pixel_num_resolation = plt.axes(
    [0.15, ax_position_height, 0.65, 0.03]
)  # 每分辨率最小像素数，算法要求
ax_position_height -= 0.05

ax_pixel_num_CCD_L = plt.axes(
    [0.15, ax_position_height, 0.65, 0.03]
)  # 像素密度，甲方要求成本控制 (个像素/mm)
ax_position_height -= 0.05

ax_tar_resolation = plt.axes([0.15, ax_position_height, 0.65, 0.03])  # 目标分辨率，甲方要求
ax_position_height -= 0.05

# 创建滑动条
sli_a_0 = Slider(ax_a_0, "a_0", 20, 300, valinit=40)
sli_alpha = Slider(ax_alpha, "alpha", 30, 40, valinit=35)
sli_tar_mea_range = Slider(
    ax_tar_mea_range, "tar_mea_range", 1, 100, valinit=6, valstep=0.5
)
sli_min_pixel_num_resolation = Slider(
    ax_min_pixel_num_resolation, "min_pixel_num_resolation", 2, 10, valinit=3, valstep=1
)
sli_pixel_num_CCD_L = Slider(
    ax_pixel_num_CCD_L, "pixel_num_CCD_L", 150, 500, valinit=1 / 0.00525
)
sli_tar_resolation = Slider(
    ax_tar_resolation, "tar_resolation", 0.001, 0.01, valinit=0.01
)


# 计算gamma
def f_pregamma(x):
    return sli_tar_mea_range.val / np.sin(x) - sli_a_0.val / np.sin(
        (180 - sli_alpha.val) / 180 * np.pi - x
    )


def calc_pregamma():
    return root(f_pregamma, np.pi / 16).x


# pregamma = calc_pregamma()
def calc_gamma():
    return np.pi / 2 - calc_pregamma() / 2


gamma = calc_gamma().item()
sli_gamma = Slider(ax_gamma, "gamma", 0, 90, valinit=68.1, valstep=0.01)


# 进度条更新和resset'按钮
def update(val):
    a_0 = sli_a_0.val
    alpha = sli_alpha.val / 360 * 2 * np.pi
    gamma = sli_gamma.val / 360 * 2 * np.pi
    tar_mea_range = sli_tar_mea_range.val
    min_pixel_num_resolation = sli_min_pixel_num_resolation.val
    CCD_L_pixel_num = sli_pixel_num_CCD_L.val
    tar_resolation = sli_tar_resolation.val


sli_a_0.on_changed(update)
sli_alpha.on_changed(update)
sli_gamma.on_changed(update)
sli_tar_mea_range.on_changed(update)
sli_min_pixel_num_resolation.on_changed(update)
sli_pixel_num_CCD_L.on_changed(update)
sli_tar_resolation.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
reset_button = Button(resetax, "Reset", color="yellow", hovercolor="0.975")


def reset(event):
    sli_a_0.reset()
    sli_alpha.reset()
    sli_pixel_num_CCD_L.reset()
    sli_min_pixel_num_resolation.reset()
    sli_tar_mea_range.reset()
    sli_tar_resolation.reset()
    sli_gamma.reset()


reset_button.on_clicked(reset)


# 计算theta_min
def calc_theta_min(ccd_l_pixel_num, min_pixel_num_resolation, tar_resolation):
    return 1 / ccd_l_pixel_num * min_pixel_num_resolation / tar_resolation


# 计算beta
def f_beta(x, theta_min, alpha, gamma):
    return theta_min * np.square(np.sin(x)) * np.sin(gamma - alpha) - np.square(
        np.sin(alpha)
    ) * np.sin(gamma + x)


def calc_beta(theta_min, alpha, gamma):
    return root(f_beta, 40 / 360 * 2 * np.pi, args=(theta_min, alpha, gamma)).x


# beta= root(f_beta, 30 / 360 * 2 * np.pi).x
# beta = calc_beta(alpha)
# print("beta: ", beta)
# a_0 = 60


# 计算b_0
def calc_b_0(a_0, alpha, beta, gamma):
    return (
        a_0
        * np.sin(alpha)
        * np.sin(beta + gamma)
        / np.sin(gamma - alpha)
        / np.sin(beta)
    )


# 计算焦距
def calc_focal_length(a_0, b_0):
    return a_0 * b_0 / (a_0 + b_0)


# b_0 = a_0 * np.tan(alpha) / np.tan(beta)
# x = np.linspace(-2, 20, 1000)
# y=1*x*np.sin(np.pi/4)/(1*np.sin(np.pi/4)-x*np.sin(np.pi/2))


# 更新x,y
def update_x_y(x, a_0, b_0, alpha, beta):
    y = x * b_0 * np.sin(alpha) / (a_0 * np.sin(beta) - x * np.sin(alpha + beta))
    xy_start_point = (0, 0)
    xy_end_point = (
        sli_tar_mea_range.val,
        (
            sli_tar_mea_range.val
            * b_0
            * np.sin(alpha)
            / (a_0 * np.sin(beta) - sli_tar_mea_range.val * np.sin(alpha + beta))
        ).item(),
    )
    ax1.clear()
    ax1.set_title("x_y")
    ax1.set_xlim(-0.2 * sli_tar_mea_range.val, 1.2 * sli_tar_mea_range.val)
    ax1.set_ylim(
        -0.2
        * sli_tar_mea_range.val
        * b_0
        * np.sin(alpha)
        / (a_0 * np.sin(beta) - -0.2 * sli_tar_mea_range.val * np.sin(alpha + beta)),
        1.2
        * sli_tar_mea_range.val
        * b_0
        * np.sin(alpha)
        / (a_0 * np.sin(beta) - 1.2 * sli_tar_mea_range.val * np.sin(alpha + beta)),
    )
    ax1.plot(x, y)
    # 起始点横坐标
    ax1.plot([xy_start_point[0], xy_start_point[0]], [y[0], xy_start_point[1]], "r--")
    ax1.text(xy_start_point[0], y[0], format(xy_start_point[0], ".2f"), color="r")
    # 起始点纵坐标
    ax1.plot(
        [x[0], xy_start_point[0]],
        [xy_start_point[1], xy_start_point[1]],
        "r--",
    )
    ax1.text(x[0], xy_start_point[1], format(xy_start_point[1], ".2f"), color="r")
    # 终点横坐标
    ax1.plot([xy_end_point[0], xy_end_point[0]], [y[0], xy_end_point[1]], "r--")
    ax1.text(xy_end_point[0], y[0], format(xy_end_point[0], ".2f"), color="r")
    # 终点纵坐标
    ax1.plot([x[0], xy_end_point[0]], [xy_end_point[1], xy_end_point[1]], "r--")
    ax1.text(x[0], xy_end_point[1], format(xy_end_point[1], ".2f"), color="r")


# 更新x,theta
def update_x_theta(x, a_0, b_0, alpha, beta):
    theta = (
        a_0
        * b_0
        * np.sin(alpha)
        * np.sin(beta)
        / np.square(a_0 * np.sin(beta) - x * np.sin(alpha + beta))
    )
    xtheta_start_point = (
        0,
        (
            a_0 * b_0 * np.sin(alpha) * np.sin(beta) / np.square(a_0 * np.sin(beta))
        ).item(),
    )
    xtheta_end_point = (
        sli_tar_mea_range.val,
        (
            a_0
            * b_0
            * np.sin(alpha)
            * np.sin(beta)
            / np.square(
                a_0 * np.sin(beta) - sli_tar_mea_range.val * np.sin(alpha + beta)
            )
        ).item(),
    )
    ax2.clear()
    ax2.set_title("x_theta")
    ax2.set_xlim(-0.2 * sli_tar_mea_range.val, 1.2 * sli_tar_mea_range.val)
    ax2.set_ylim(
        a_0
        * b_0
        * np.sin(alpha)
        * np.sin(beta)
        / np.square(
            a_0 * np.sin(beta) + 0.2 * sli_tar_mea_range.val * np.sin(alpha + beta)
        ),
        a_0
        * b_0
        * np.sin(alpha)
        * np.sin(beta)
        / np.square(
            a_0 * np.sin(beta) - 1.2 * sli_tar_mea_range.val * np.sin(alpha + beta)
        ),
    )
    ax2.plot(x, theta)
    # 起始点横坐标
    ax2.plot(
        [xtheta_start_point[0], xtheta_start_point[0]],
        [theta[0], xtheta_start_point[1]],
        "r--",
    )
    ax2.text(
        xtheta_start_point[0], theta[0], format(xtheta_start_point[0], ".2f"), color="r"
    )
    # 起始点纵坐标
    ax2.plot(
        [x[0], xtheta_start_point[0]],
        [xtheta_start_point[1], xtheta_start_point[1]],
        "r--",
    )
    ax2.text(
        x[0], xtheta_start_point[1], format(xtheta_start_point[1], ".2f"), color="r"
    )
    # 终点横坐标
    ax2.plot(
        [xtheta_end_point[0], xtheta_end_point[0]],
        [theta[0], xtheta_end_point[1]],
        "r--",
    )
    ax2.text(
        xtheta_end_point[0], theta[0], format(xtheta_end_point[0], ".2f"), color="r"
    )
    ax2.plot(
        [x[0], xtheta_end_point[0]], [xtheta_end_point[1], xtheta_end_point[1]], "r--"
    )
    ax2.text(x[0], xtheta_end_point[1], format(xtheta_end_point[1], ".2f"), color="r")


# y = x * b_0 * np.sin(alpha) / (a_0 * np.sin(beta) - x * np.sin(alpha + beta))
# theta = a_0* b_0* np.sin(alpha)* np.sin(beta)/ np.square(a_0 * np.sin(beta) - x * np.sin(alpha + beta))
# focal_length = a_0 * b_0 / (a_0 + b_0)


# 更新文本
def update_text(a_0, b_0, alpha, beta, focal_length):
    text_num = 5 - 1

    pixel_num = (
        sli_tar_mea_range.val
        * b_0
        * np.sin(alpha)
        / (a_0 * np.sin(beta) - sli_tar_mea_range.val * np.sin(alpha + beta))
        * sli_pixel_num_CCD_L.val
    )

    CCD_L = (
        sli_tar_mea_range.val
        * b_0
        * np.sin(alpha)
        / (a_0 * np.sin(beta) - sli_tar_mea_range.val * np.sin(alpha + beta))
    )

    ax_text.clear()
    ax_text.axis("off")
    ax_text.text(0 / text_num, 0, "pixel_num = {:.2f}".format(pixel_num))
    ax_text.text(1 / text_num, 0, "CCD_L = {:.2f}".format(CCD_L))
    ax_text.text(2 / text_num, 0, "b_0 = {:.2f}".format(b_0))
    ax_text.text(3 / text_num, 0, "beta = {:.2f}".format(beta / 2 / np.pi * 360))
    ax_text.text(4 / text_num, 0, "focal_length = {:.2f}".format(focal_length))


def animate(frame):
    a_0 = sli_a_0.val
    alpha = sli_alpha.val / 360 * 2 * np.pi
    gamma = sli_gamma.val / 360 * 2 * np.pi

    beta = calc_beta(
        calc_theta_min(
            sli_pixel_num_CCD_L.val,
            sli_min_pixel_num_resolation.val,
            sli_tar_resolation.val,
        ),
        alpha,
        gamma,
    ).item()

    b_0 = calc_b_0(sli_a_0.val, alpha, beta, gamma).item()
    # print(b_0)
    x = np.linspace(-0.2 * sli_tar_mea_range.val, 1.2 * sli_tar_mea_range.val, 1000)
    focal_length = calc_focal_length(a_0, b_0)

    update_x_y(x, a_0, b_0, alpha, beta)
    update_x_theta(x, a_0, b_0, alpha, beta)
    update_text(a_0, b_0, alpha, beta, focal_length)
    fig.canvas.draw_idle()


ani = animation.FuncAnimation(fig, animate, interval)

plt.show()
