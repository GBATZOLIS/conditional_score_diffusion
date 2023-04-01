import matplotlib.pyplot as plt
import pickle
import numpy as np
import sde_lib
import torch

filepath = '/Users/gbatz97/Desktop/score-based-modelling/projects/scoreVAE/experiments/pretrained/cifar10/encoder_only/VAE/corrected_only_encoder_VAE_KLweight_0.01/inspection/contribution.pkl'
with open(filepath, 'rb') as f:
    contribution = pickle.load(f)

print(contribution.keys())

eps=1e-3
sde = sde_lib.VPSDE()

plt.figure()
plt.title('mean')
colors = {'encoder':'red', 'auxiliary':'blue', 'pretrained':'green'}
store_bps = {}
for member in contribution.keys():
    sorted_times = sorted(list(contribution[member].keys()))
    sorted_snrs = sde.snr(torch.tensor(np.array(sorted_times))).numpy().tolist()
    
    plt.plot(sorted_times, [np.mean(contribution[member][t]) for t in sorted_times], c=colors[member])

    '''
    list_of_arrays = []
    for t in sorted_times:
        list_of_arrays.append(contribution[member][t])
    
    store_bps[member] = plt.boxplot(list_of_arrays, notch=True, labels=[round(x, 1) for x in sorted_snrs], patch_artist=True)
    
    for i in range(len(list_of_arrays)):
        store_bps[member]['boxes'][i].set_facecolor(colors[member])
        store_bps[member]['boxes'][i].set_color(colors[member])
    '''

#plt.plot(sorted_times, [np.mean(contribution[member][t]) for t in sorted_times], label=colors[member])

'''
bp_list = []
name_list = []
for member in contribution.keys():
    num_times = len(contribution[member].keys())
    bp_list.extend([store_bps[member]["boxes"][x] for x in range(num_times)])
    name_list.extend([member for _ in range(num_times)])

plt.legend(bp_list, name_list, loc='upper right')
'''

    #plt.plot(times, [np.mean(contribution[member][t]) for t in times], label=member)
    #below = np.array([np.mean(contribution[member][t]) for t in times]) - np.array([3*np.std(contribution[member][t]) for t in times])
    #above = np.array([np.mean(contribution[member][t]) for t in times]) + np.array([3*np.std(contribution[member][t]) for t in times])
    #plt.fill_between(times, below, above, alpha=0.5)

plt.grid()
plt.show()

