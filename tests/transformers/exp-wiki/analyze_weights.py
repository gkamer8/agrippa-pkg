import matplotlib.pyplot as plt
import os
import pickle

weights_dict = {}
with open(os.path.join('model', 'weights_9.5.pkl'), 'rb') as fhand:
    weights_dict = pickle.load(fhand)

weights_list = []
i = 0
for key in weights_dict:
    i += 1
    x = list(weights_dict[key].flatten())
    weights_list.extend(x)
    if i > 10:
        break

print(f"Weights N: {len(weights_list)}")

plt.hist(weights_list)
plt.show()