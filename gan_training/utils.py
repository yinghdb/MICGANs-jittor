import jittor as jt
import numpy as np

import os

def compute_purity(y_pred, y_true):
        """
        Calculate the purity, a measurement of quality for the clustering 
        results.
        
        Each cluster is assigned to the class which is most frequent in the 
        cluster.  Using these classes, the percent accuracy is then calculated.
        
        Returns:
          A number between 0 and 1.  Poor clusterings have a purity close to 0 
          while a perfect clustering has a purity of 1.

        """

        # get the set of unique cluster ids
        clusters = set(y_pred)

        # find out what class is most frequent in each cluster
        cluster_classes = {}
        correct = 0
        for cluster in clusters:
            # get the indices of rows in this cluster
            indices = np.where(y_pred == cluster)[0]

            cluster_labels = y_true[indices]
            majority_label = np.argmax(np.bincount(cluster_labels))
            correct += np.sum(cluster_labels == majority_label)
        
        return float(correct) / len(y_pred)

def get_nsamples(data_loader, N):
    x = []
    y = []
    n = 0
    for x_next, y_next, _ in data_loader:
        x.append(x_next)
        y.append(y_next)
        n += x_next.size(0)
        if n > N:
            break
    x = jt.contrib.concat(x, dim=0)[:N]
    y = jt.contrib.concat(y, dim=0)[:N]
    return x, y
