import torch
import numpy as np
import pickle
from lightning_data_modules import HaarDecomposedDataset, ImageDatasets, PairedDataset, SyntheticDataset, SyntheticPairedDataset, Synthetic1DConditionalDataset, SyntheticTimeSeries, SRDataset, SRFLOWDataset, KSphereDataset, MammothDataset, LineDataset, GanDataset #needed for datamodule registration
from lightning_data_modules.utils import create_lightning_datamodule
from configs.utils import read_config


def create_dataset(train_dataloader):        
    print(f'------ Creating dataset --------')
    X=[]
    Y=[]
    for _, item in enumerate(train_dataloader):
        if isinstance(item, list):
            labels = True
            x , y = item
        else:
            x = item
        X.append(x.view(x.shape[0],-1))
        if labels:
            Y.append(y.numpy())
    data_np = torch.cat(X, dim=0).numpy()
    if labels:
        Y = np.concatenate(Y, axis=0)
    data_np.reshape(data_np.shape[0],-1).shape
    print(f'------ Dataset created --------')
    if labels:
        return data_np.astype(np.float64), Y
    else:
        return data_np.astype(np.float64)



from sklearn.manifold import Isomap
from sklearn.ensemble import RandomForestClassifier

config = read_config('configs/mnist/unconditional_jan.py')
#config = read_config('configs/ksphere/N_1/uniform.py')
DataModule = create_lightning_datamodule(config)
DataModule.setup()
train_dataloader = DataModule.train_dataloader()
test_dataloader = DataModule.test_dataloader()


X, Y = create_dataset(train_dataloader)
X_test, Y_test = create_dataset(test_dataloader)


values = []
clf_scores = []
ks = list(range(1,11,1)) + list(range(11, 200, 10))
N = 1000

for k in ks:
    print(f'k = {k}')
    embedding = Isomap(n_components=k)
    Z = embedding.fit_transform(X[:N])
    error = embedding.reconstruction_error()
    print(f'Reconstruction error: {error}')
    values.append(error)

    clf = RandomForestClassifier(random_state=0)
    clf.fit(Z, Y[:N])
    Z_test = embedding.transform(X_test)
    score = clf.score(Z_test, Y_test)
    print(f'Clf score: {score}')
    clf_scores.append(score)


from matplotlib import pyplot as plt
plt.figure(figsize=(10,10))
plt.plot(ks, values)
plt.savefig('isomap/reconstruction_error.png', dpi=300, facecolor='white')
with open('isomap/reconstruction_error.pkl', 'wb') as f:
    pickle.dump(values, f)
plt.figure(figsize=(10,10))
plt.plot(ks, clf_scores)
plt.savefig('isomap/clf_scores.png', dpi=300, facecolor='white')
with open('isomap/clf_scores.pkl', 'wb') as f:
    pickle.dump(clf_scores, f)