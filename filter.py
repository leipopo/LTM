import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib.animation as animation
from matplotlib.widgets import Slider
from scipy import ndimage

with open("data.txt", "r") as file:
    data = file.read().strip()

# 将数据以空格分割为各个十六进制数
datalist = data.split(" ")
result_raw = []
result_mcuiir = []
result_zeropoints = []
# 拼接处理后的数据
for i in range(0, len(datalist)):
    if datalist[i] == "0A" and datalist[i + 1] == "0D" and datalist[i + 2] == "FF":
        i += 3
        while i + 4 < len(datalist) and (
            datalist[i + 2] != "0A"
            or datalist[i + 3] != "0D"
            or datalist[i + 4] != "00"
        ):
            value = datalist[i + 1] + datalist[i]
            result_mcuiir.append(int(value, 16))
            i += 2
        else:
            if len(result_mcuiir) > 2000:
                value = datalist[i + 1] + datalist[i]
                result_mcuiir.append(int(value, 16))
                break
            else:
                result_mcuiir.clear()
                continue


for i in range(0, len(datalist)):
    if datalist[i] == "0A" and datalist[i + 1] == "0D" and datalist[i + 2] == "00":
        i += 3
        while i + 4 < len(datalist) and (
            datalist[i + 2] != "0A"
            or datalist[i + 3] != "0D"
            or datalist[i + 4] != "0F"
        ):
            value = datalist[i + 1] + datalist[i]
            result_raw.append(int(value, 16))
            i += 2
        else:
            if len(result_raw) > 2000:
                value = datalist[i + 1] + datalist[i]
                result_raw.append(int(value, 16))
                break
            else:
                result_raw.clear()
                continue
    else:
        continue


for i in range(0, len(datalist)):
    if datalist[i] == "0A" and datalist[i + 1] == "0D" and datalist[i + 2] == "0F":
        i += 3
        while i + 4 < len(datalist) and (
            datalist[i + 2] != "0A"
            or datalist[i + 3] != "0D"
            or datalist[i + 4] != "FF"
        ):
            value = datalist[i + 1] + datalist[i]
            result_zeropoints.append(int(value, 16))
            i += 2
        else:
            if len(result_zeropoints) > 150:
                value = datalist[i + 1] + datalist[i]
                result_zeropoints.append(int(value, 16))
                break
            else:
                result_zeropoints.clear()
                continue
    else:
        continue
# 输出处理后的结果
# print(result)
# print(len(result_raw))

fig, (ax_orig, ax_mcuiir, ax_iir, ax_KF) = plt.subplots(4, figsize=(12, 12))

ax_orig.set_position([0.05, 0.65, 0.4, 0.3])
ax_mcuiir.set_position([0.55, 0.65, 0.4, 0.3])
ax_iir.set_position([0.05, 0.3, 0.4, 0.3])
ax_KF.set_position([0.55, 0.3, 0.4, 0.3])


# ax_highpass = plt.axes([0.1, 0.2, 0.8, 0.03])
ax_lowpass = plt.axes([0.1, 0.2, 0.8, 0.03])
ax_KF_Q = plt.axes([0.1, 0.15, 0.8, 0.03])
ax_KF_R = plt.axes([0.1, 0.1, 0.8, 0.03])
ax_text = plt.axes([0.1, 0.05, 0.8, 0.03])

sli_KF_Q = Slider(ax_KF_Q, "KF_Q", 0.00001, 2, valinit=0.25, valstep=0.00005)
sli_KF_R = Slider(ax_KF_R, "KF_R", 0.00001, 20, valinit=9, valstep=0.00005)
sli_lowpass = Slider(ax_lowpass, "lowpass", 0.0001, 0.02, valinit=0.01, valstep=0.00005)
# sli_highpass = Slider(
#     ax_highpass, "highpass", 0.0001, 0.02, valinit=0.01, valstep=0.00005
# )


def update(val):
    KF_Q = sli_KF_Q.val
    KF_P = sli_KF_R.val
    highpass = sli_lowpass.val


sli_KF_Q.on_changed(update)
sli_KF_R.on_changed(update)
sli_lowpass.on_changed(update)
# sli_lowpass.on_changed(update)


class KF_1thorder:
    def __init__(self, Q, R, P, A, H):
        self.x = 0  # 状态估计
        self.P = P  # 状态估计误差协方差
        self.Q = Q  # 状态噪声协方差
        self.R = R  # 观测噪声协方差
        self.K = 0  # 卡尔曼增益
        self.A = A  # 状态转移矩阵
        self.H = H  # 观测矩阵
        self.z = 0  # 观测值

    def predict(self):
        self.x = (1 - self.K * self.H) * self.x + self.K * self.z  # 状态估计
        self.A = 1 - self.K * self.H  # 状态转移矩阵更新
        self.P = self.A * self.P * self.A + self.Q  # 先验估计误差协方差更新

    def correct(self):
        self.K = self.P * self.H / (self.H * self.P * self.H + self.R)  # 卡尔曼增益
        self.x = self.x + self.K * (self.z - self.H * self.x)  # 状态估计
        self.P = (1 - self.K * self.H) * self.P  # 状态估计误差协方差更新


def kalmanfilter_1th(kf, z):
    kf.z = z
    kf.predict()
    kf.correct()
    return kf.x


def find_zero_points(data):
    diff_data = np.diff(data)
    zeropoints = []
    for i in range(0, len(diff_data) - 1):
        if diff_data[i] * diff_data[i + 1] < 0:
            if len(zeropoints) == 0:
                zeropoints.append([i, data[i]])
            else:
                if i - zeropoints[-1][0] > 50:
                    zeropoints.append([i, data[i]])
        else:
            continue
    # zeropoints.remove(zeropoints[0])
    # zeropoints.remove(zeropoints[-1])
    # print(zeropoints)
    return zeropoints


def find_min_zero_point(zerodatas):
    minpoint = zerodatas[0]
    for i in range(0, len(zerodatas) - 1):
        if zerodatas[i][1] < minpoint[1]:
            minpoint = zerodatas[i]
        else:
            continue
    return minpoint


def plot_2minpoint_zeropoints_on_ax(ax, zeropoints):
    for i in range(0, len(zeropoints) - 1):
        ax.scatter(zeropoints[i][0], zeropoints[i][1], c="r", s=5)
    minpoint = find_min_zero_point(zeropoints)
    ax.scatter(minpoint[0], minpoint[1], c="b", s=20)
    zeropoints.remove(minpoint)
    minpoint = find_min_zero_point(zeropoints)
    ax.scatter(minpoint[0], minpoint[1], c="b", s=20)


kf = KF_1thorder(1, 1, 1, 1, 1)


def animate(i):
    ax_orig.clear()
    ax_mcuiir.clear()
    ax_iir.clear()
    ax_KF.clear()
    ax_text.clear()

    lowpass = sli_lowpass.val
    # highpass = sli_highpass.val
    # if lowpass >= highpass:
    #     highpass = lowpass + 0.0001
    KF_Q = sli_KF_Q.val
    KF_R = sli_KF_R.val

    # Original signal
    ax_orig.plot(result_raw)
    ax_orig.set_title("Original signal")

    ax_mcuiir.plot(result_mcuiir)
    ax_mcuiir.set_title("MCU IIR signal")

    # # IIR signal
    b, a = signal.iirfilter(2, lowpass, btype="lowpass")
    print("b: " + str(b[0]) + ", " + str(b[1]) + ", " + str(b[2]))
    print("a: " + str(a[0]) + ", " + str(a[1]) + ", " + str(a[2]))
    iir_result = signal.filtfilt(b, a, result_raw)
    # iir_result = result_raw
    ax_iir.plot(iir_result)
    ax_iir.set_title("IIR signal")

    # KF signal
    result_kf = []
    kf.Q = KF_Q
    kf.R = KF_R
    for i in range(0, len(result_zeropoints)):
        result_kf.append(kalmanfilter_1th(kf, result_zeropoints[i]))

    ax_KF.plot(result_zeropoints)
    ax_KF.plot(result_kf)
    ax_KF.set_title("KF signal")

    # Convolved signal
    # step_signal = np.ones(steplen)
    # for i in range(0, len(step_signal)):
    #     step_signal[i] = 1 / steplen
    # convolved_data = np.convolve(result_raw, step_signal, mode="full")
    # ax_convol.plot(convolved_data)
    # ax_convol.set_title("Convolved signal")

    # # Gauss signal
    # gauss_signal = ndimage.gaussian_filter1d(result_raw, gauss_std)
    # ax_gauss.plot(gauss_signal)
    # ax_gauss.set_title("Gauss signal")

    # plot points
    # plot_2minpoint_zeropoints_on_ax(ax_iir, find_zero_points(iir_result))
    # plot_2minpoint_zeropoints_on_ax(ax_convol, find_zero_points(convolved_data))
    # plot_2minpoint_zeropoints_on_ax(ax_gauss, find_zero_points(gauss_signal))

    ax_text.axis("off")
    ax_text.text(0.1, 0.1, "b: " + str(b))
    ax_text.text(0.6, 0.1, "a: " + str(a))


ani = animation.FuncAnimation(fig, animate, 50)

plt.show()
