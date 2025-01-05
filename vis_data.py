import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

env = "OfflineCarRun-v0"

for env in ["OfflineCarRun-v0", "OfflineBallCircle-v0"]:

    with open(f'offline_dataset/{env}/dataset.pkl', 'rb') as file:
        data = pickle.load(file)

    reward_l1_norms = [np.sum(np.abs(item)) for item in data['observations']]
    action_l1_norms = [np.sum(np.abs(item)) for item in data['actions']]

    plt.figure(figsize=(10, 6))
    sns.histplot(reward_l1_norms, bins=10)
    plt.xlabel('L1 Norm of observations')
    plt.ylabel('Frequency')
    plt.title('Distribution of L1 Norm of observations')
    plt.savefig(f'data_vis/{env}_observations.png')

    bins = 50
    reward_bins = np.linspace(min(reward_l1_norms), max(reward_l1_norms), bins)
    action_bins = np.linspace(min(action_l1_norms), max(action_l1_norms), bins)

    heatmap_data, _, _ = np.histogram2d(reward_l1_norms, action_l1_norms, bins=[reward_bins, action_bins])

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, xticklabels=np.round(action_bins, 2), yticklabels=np.round(reward_bins, 2),
                cmap="YlGnBu")

    plt.xlabel('L1 Norm of action')
    plt.ylabel('L1 Norm of observations')
    plt.title('Heatmap of L1 Norm of observations and action')
    plt.savefig(f'data_vis/{env}_observations_and_action.png')
