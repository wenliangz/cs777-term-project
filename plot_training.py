# ## Plot training result

import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

rcParams.update({'figure.autolayout': True})

epochs = []
costs = []
with open('docs/training_printout.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith('Epoch'):
            epoch = int(line.split(':')[0].split(' ')[1].strip())
            epochs.append(epoch)
        if 'Cost' in line:
            cost_line = line.split(', ')[2]
            cost = float(cost_line.split('=')[1].split(',')[0].strip())
            costs.append(cost)
        else:
            continue
# print(epochs)
# print(costs)


epochs_toplot = epochs[1:]
costs_toplot = costs[1:]

fig, ax = plt.subplots()

ax.plot(epochs_toplot, costs_toplot)
ax.set_xlabel('epochs')
ax.set_ylabel('costs')
plt.title('Logistic Regression \n using Gradient Descent with Bold Driver')
ax.ticklabel_format(useOffset=False)
plt.savefig('trainingcurve.jpg')
plt.show()