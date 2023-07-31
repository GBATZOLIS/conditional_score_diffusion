import sys
import os
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import run_lib
import pickle
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--balance_classes', default=True, type=int)


def main(args):
    print('loading data from pickle')
    with open('tmp/y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)
    with open('tmp/y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)
    with open('tmp/Z_train.pkl', 'rb') as f:
        Z_train = pickle.load(f)
    with open('tmp/Z_test.pkl', 'rb') as f:
        Z_test = pickle.load(f)

    # balance classes
    if args.balance_classes:
        print('balancing classes')
        # balance train set
        # get the number of samples in each class
        n_samples = np.bincount(y_train)
        # get the number of samples to remove from each class
        n_remove = (n_samples - n_samples.min())
        # get the indices to remove from each class
        idx_remove = np.hstack([np.random.choice(np.where(y_train == i)[0], n, replace=False) for i, n in enumerate(n_remove)])
        # remove the samples
        y_train = np.delete(y_train, idx_remove, axis=0)
        Z_train = np.delete(Z_train, idx_remove, axis=0)

        # balance test set
        # get the number of samples in each class
        n_samples = np.bincount(y_test)
        # get the number of samples to remove from each class
        n_remove = (n_samples - n_samples.min())
        # get the indices to remove from each class
        idx_remove = np.hstack([np.random.choice(np.where(y_test == i)[0], n, replace=False) for i, n in enumerate(n_remove)])
        # remove the samples
        y_test = np.delete(y_test, idx_remove, axis=0)
        Z_test = np.delete(Z_test, idx_remove, axis=0)


    from sklearn.linear_model import LogisticRegression
    # fit a logistic model
    print('fitting logistic regression on cartesian coordinates')
    cls = LogisticRegression(penalty=None, max_iter=1000)
    cls.fit(Z_train.reshape(len(Z_train), -1), y_train)

    # accuracy on train
    print('accuracy on train')
    y_pred = cls.predict(Z_train.reshape(len(Z_train), -1))
    print((y_pred == y_train).mean())

    # accuracy on test
    print('accuracy on test')
    y_pred = cls.predict(Z_test.reshape(len(Z_test), -1))
    print((y_pred == y_test).mean())

    # pickle the classifier into tmp
    print('pickling classifier')
    with open('tmp/cls.pkl', 'wb') as f:
        pickle.dump(cls, f)

    # Spherical coordinates
    from semantic_manipulation.utils import cartesian_to_spherical, spherical_to_cartesian
    Z_train_spherical = cartesian_to_spherical(Z_train)
    Z_test_spherical = cartesian_to_spherical(Z_test)

    # fit a logistic model
    print('fitting logistic regression on spherical coordinates')
    cls = LogisticRegression(penalty=None, max_iter=1000)
    cls.fit(Z_train_spherical.reshape(len(Z_train_spherical), -1), y_train)

    # accuracy on train
    print('accuracy on train')
    y_pred = cls.predict(Z_train_spherical.reshape(len(Z_train_spherical), -1))
    print((y_pred == y_train).mean())

    # accuracy on test
    print('accuracy on test')
    y_pred = cls.predict(Z_test_spherical.reshape(len(Z_test_spherical), -1))
    print((y_pred == y_test).mean())

    # pickle the classifier into tmp
    with open('tmp/cls_spherical.pkl', 'wb') as f:
        pickle.dump(cls, f)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)