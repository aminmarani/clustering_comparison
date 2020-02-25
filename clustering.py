import pandas as pd
import csv
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import sys
import time





def visualize(data,clusters):
	#make a PCA object
	pca = PCA(n_components=2)
	#find PCA coeff.
	data_pca = pca.fit_transform(data)


	plt.figure('PCA 2D visualization')
	plt.scatter(x=data_pca[:,0],y=data_pca[:,1],c=clusters)
	plt.show()

def Kmeans(data,K,eps=0.001):
	#select centers randomly
	centers = np.zeros(shape=(K,len(data[0])))#initiate centers
	centers = data[np.random.permutation(len(data))[0:K],:]
	old_centers = np.zeros(shape=(K,len(data[0])))#initiate centers
	chnages = 1
	clusters = np.zeros(shape=(len(data),1))#initiate cluster labels

	#body of Kmeans
	while(chnages >= eps):
		#computing centers - data distance and assign clusters
		for i in range(len(data)):
			#computing distance
			dist = compute_dist(data[i,:],centers)
			#assign clusters
			clusters[i] = np.argmin(dist)
		#storing current centers
		old_centers[:,:] = centers[:,:]

		#update means and making new centers
		for i in range(0,K):
			centers[i,:] = np.mean(data[np.argwhere(clusters==i)[:,0],:],axis=0)

		#compute differences between new and old centers
		chnages = 0
		for i in range(0,K):
			chnages += np.sqrt( sum( np.power(old_centers[i,:]-centers[i,:],2) ))

	#visualize changes
	visualize(data,clusters[:,0])
	return(clusters)
		


def compute_dist(data,centers):
	dist = []
	for c in centers:
		dist.append( np.sqrt( sum( np.power(data-c,2) )) )
	return dist



#####main part
data_file = sys.argv[1]
#reading datasets and putting them into numpy arrays
data = np.loadtxt(open(data_file,newline=''),delimiter='\t')
visualize(data[:,2:],data[:,1])

#call K-means
C1 = Kmeans(data[:,2:],7)



#later think of ways to solve outliers
