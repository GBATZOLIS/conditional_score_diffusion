import matplotlib.pyplot as plt
import pickle
import numpy as np

filepath = '/Users/gbatz97/Desktop/score-based-modelling/projects/scoreVAE/experiments/pretrained/cifar10/encoder_only/VAE/corrected_only_encoder_VAE_KLweight_0.01/inspection/contribution.pkl'
with open(filepath, 'rb') as f:
    contribution = pickle.load(f)

print(contribution.keys())

plt.figure()
plt.title('mean')
for member in contribution.keys():
    times = sorted(list(contribution[member].keys()))
    plt.plot(times, [np.mean(contribution[member][t]) for t in times], label=member)
    below = np.array([np.mean(contribution[member][t]) for t in times]) - np.array([3*np.std(contribution[member][t]) for t in times])
    above = np.array([np.mean(contribution[member][t]) for t in times]) + np.array([3*np.std(contribution[member][t]) for t in times])
    plt.fill_between(times, below, above, alpha=0.5)


plt.legend()
plt.show()

