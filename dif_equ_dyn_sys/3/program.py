import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--parameters", action = "store_true",
                    help = "Name of parameters file.\n\nYAML file with entries: rainy, nice, cloudy, steps.",)
args = parser.parse_args()
if not args.parameters:
    file_name = "params"
else:
    file_name = args.parameters
with open(file_name + ".yaml", "r") as file:
        params = yaml.safe_load(file)


# [rainy, nice, cloudy] x [[rainy], [nice], [cloudy]]
A = np.array([[0.50, 0.50, 0.25],
              [0.25, 0.00, 0.25],
              [0.25, 0.50, 0.50]])

x = np.array([[params["rainy"]],
              [params["nice"]],
              [params["cloudy"]]])

weather = np.zeros((3, params["steps"]))
weather[:, 0] = x[:, 0]
for i in range(1, params["steps"]):
    x = A@x
    weather[:, i] = x[:, 0]

probabilities = np.linalg.eig(A)[1][:, 0]
probabilities /= np.sum(probabilities)

print("Probabilities via iteration\t:\t{}".format(weather[:, -1]))
print("Probabilities via eigenvector\t:\t{}".format(probabilities))


fig = plt.figure(figsize = (8, 4), tight_layout = True)
gs = fig.add_gridspec(3, 2)

fig.add_subplot(111, frameon = False)
plt.tick_params(labelcolor = "none", which = "both",
                top = False, bottom = False,
                left = False, right = False)
plt.xlabel("Time (days)")
plt.ylabel("Probability")

titles = ["Rainy", "Nice", "Cloudy"]
for i in range(3):
    ax = fig.add_subplot(gs[i, 0])
    ax.plot(weather[i, :])
    ax.set_title(titles[i])
    ax.label_outer()

ax = fig.add_subplot(gs[:, 1])
ax.plot(weather.T)
ax.grid(True)
ax.legend(titles)

plt.show()