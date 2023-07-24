import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib.animation as animation
from matplotlib.widgets import Slider

with open("data.txt", "r") as file:
    data = file.read().strip()

# 将数据以空格分割为各个十六进制数
list = data.split(" ")
result = []
# 拼接处理后的数据
for i in range(0, len(list)):
    if list[i] == "0A" and list[i + 1] == "0D" and list[i + 2] == "00":
        i += 3
        while (
            list[i + 2] != "0A"
            or list[i + 3] != "0D"
            or list[i + 4] != "00"
            and i < len(list) - 5
        ):
            value = list[i + 1] + list[i]
            result.append(int(value, 16))
            i += 2
        else:
            if len(result) > 2000:
                value = list[i + 1] + list[i]
                result.append(int(value, 16))
                break
            else:
                result.clear()
                continue
    else:
        continue
# 输出处理后的结果
# print(result)
print(len(result))

fig, (ax_orig, ax_IIR, ax_convol, ax_diff) = plt.subplots(4, figsize=(12, 12))

ax_orig.set_position([0.05, 0.65, 0.4, 0.3])
ax_IIR.set_position([0.55, 0.65, 0.4, 0.3])
ax_convol.set_position([0.05, 0.3, 0.4, 0.3])
ax_diff.set_position([0.55, 0.3, 0.4, 0.3])

# ax_diff_sli = plt.axes([0.1, 0.05, 0.8, 0.03])
ax_steplen = plt.axes([0.1, 0.1, 0.8, 0.03])
# ax_lowpass = plt.axes([0.1, 0.15, 0.8, 0.03])
ax_highpass = plt.axes([0.1, 0.2, 0.8, 0.03])

# sli_diff = Slider(ax_diff_sli, "diff", 1, 3, valinit=1, valstep=1)
sli_steplen = Slider(ax_steplen, "steplen", 1, 10000, valinit=300, valstep=1)
# sli_lowpass = Slider(
#     ax_lowpass, "lowpass", 0.0001, 0.9999, valinit=0.0001, valstep=0.00005
# )
sli_highpass = Slider(
    ax_highpass, "highpass", 0.0001, 0.9999, valinit=0.005, valstep=0.00005
)


def update(val):
    # diff = sli_diff.val
    steplen = sli_steplen.val
    # lowpass = sli_lowpass.val
    highpass = sli_highpass.val


# sli_diff.on_changed(update)
sli_steplen.on_changed(update)
sli_highpass.on_changed(update)
# sli_lowpass.on_changed(update)


def animate(i):
    ax_orig.clear()
    ax_convol.clear()
    ax_IIR.clear()
    ax_diff.clear()

    # lowpass = sli_lowpass.val
    highpass = sli_highpass.val
    # if lowpass >= highpass:
    #     highpass = lowpass + 0.0001
    steplen = sli_steplen.val
    # diff = sli_diff.val

    ax_orig.plot(result)
    ax_orig.set_title("Original signal")

    b, a = signal.iirfilter(2, highpass, btype="lowpass")
    IIR_result = signal.filtfilt(b, a, result)
    ax_IIR.plot(IIR_result)
    ax_IIR.set_title("IIR signal")

    step_signal = np.ones(steplen)
    for i in range(0, len(step_signal)):
        step_signal[i] = 1 / steplen
    convolved_data = np.convolve(IIR_result, step_signal, mode="full")
    ax_convol.plot(convolved_data)
    ax_convol.set_title("Convolved signal")

    diff_data = np.diff(convolved_data)
    zeropoint = []
    for i in range(0, len(diff_data)):
        if np.abs(diff_data[i] - 0) < 0.1:
            if len(zeropoint) == 0:
                zeropoint.append(i)
            elif np.abs(i - zeropoint[0]) > 10:
                zeropoint.append(i)
            if len(zeropoint) > 1:
                break
            else:
                continue
        else:
            continue

    ax_diff.plot(diff_data)
    ax_diff.set_title("Diff signal")
    zeropoint_conv = (zeropoint[1], convolved_data[zeropoint[1]])
    zeropoint_diff = (zeropoint[1], diff_data[zeropoint[1]])
    ax_convol.plot(zeropoint_conv[0], zeropoint_conv[1], "ro")
    ax_convol.text(
        zeropoint_conv[0],
        zeropoint_conv[1],
        str(zeropoint_conv[0]) + "," + str(zeropoint_conv[1]),
    )
    ax_diff.plot(zeropoint_diff[0], zeropoint_diff[1], "ro")
    ax_diff.text(
        zeropoint_diff[0],
        zeropoint_diff[1],
        str(zeropoint_diff[0]) + "," + str(zeropoint_diff[1]),
    )


ani = animation.FuncAnimation(fig, animate, 50)

plt.show()
