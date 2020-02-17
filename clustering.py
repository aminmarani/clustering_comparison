import pandas as pd
import csv
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import



#reading datasets and putting them into numpy arrays
cho = np.loadtxt(open('cho.txt',newline=''),delimiter='\t')
cho = np.loadtxt(open('iyer.txt',newline=''),delimiter='\t')


#make a PCA object
pca = PCA(n_components=2)
#find PCA coeff.
cho_pca = pca.fit_transform(cho[:,2:])

plt.scatter(x=cho_pca[:,0],y=cho_pca[:,1],c=cho[:,1])
plt.show()

