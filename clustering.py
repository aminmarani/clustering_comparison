import pandas as pd
import csv
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import sys
import time
import math





def visualize(data,clusters,title=""):
	#make a PCA object
	pca = PCA(n_components=2)
	#find PCA coeff.
	data_pca = pca.fit_transform(data)


	plt.figure('PCA 2D visualization')
	plt.title(title)
	plt.scatter(x=data_pca[:,0],y=data_pca[:,1],c=clusters)
	plt.show()

def Kmeans(data,K,eps=0.001):
	#select centers randomly
	centers = np.zeros(shape=(K,len(data[0])))#initiate centers
	centers = data[np.random.permutation(len(data))[0:K],:]
	old_centers = np.zeros(shape=(K,len(data[0])))#initiate centers
	chnages = 1
	clusters = np.zeros(shape=(len(data)))#initiate cluster labels

	#body of Kmeans
	while(chnages >= eps):
		#computing centers - data distance and assign clusters
		for i in range(len(data)):
			#computing distance
			dist = compute_dist(data[i,:],centers)
			#assign clusters
			clusters[i] = np.argmin(dist)  #clusters start from 1
		#storing current centers
		old_centers[:,:] = centers[:,:]

		#update means and making new centers
		for i in range(0,K):
			centers[i,:] = np.mean(data[np.argwhere(clusters==i),:],axis=0)

		#compute differences between new and old centers
		chnages = 0
		for i in range(0,K):
			chnages += np.sqrt( sum( np.power(old_centers[i,:]-centers[i,:],2) ))

	#visualize changes
	visualize(data,clusters,'Kmeans results')
	return(clusters)
		


def compute_dist(data,centers):
	dist = []
	for c in centers:
		dist.append( np.sqrt( sum( np.power(data-c,2) )) )
	return dist

def compute_external(clusters,classes):
	#compute entropy for each class
	entropy = 0
	purity = 0
	for i in list(set(clusters)):
		epr = 0
		pur = 0
		s = 0
		for j in list(set(classes)):
			if(j==-1): #don't consider outliers
				continue
			indcls = [x for x in range(len(classes)) if classes[x]==j] #find number of classess i in data
			indclu = [x for x in range(len(clusters)) if clusters[x]==i]#find number of cluster j in data
			t = len([x for x in indclu if x in indcls])/len(indcls) #number of items with class j in cluster i / all items of class j
			if(t != 0):
				epr += t*math.log2(t) #entropy for each cluster
				s += len([x for x in indclu if x in indcls]) #number of element for each cluster
				if len([x for x in indclu if x in indcls]) > pur:
					pur = len([x for x in indclu if x in indcls])
		entropy += (s/len(classes))*(-1*epr)
		purity += (s/len(classes))*(pur)
	return entropy,purity





#####main part
data_file = sys.argv[1]
#reading datasets and putting them into numpy arrays
data = np.loadtxt(open(data_file,newline=''),delimiter='\t')
visualize(data[:,2:],data[:,1],'raw data')

#call K-means
C1 = Kmeans(data[:,2:],7)
en,pr = compute_external(C1.tolist(),data[:,1])
print('Entropy of K-means: ',en)
print('Purity of K-means: ',pr)



#later think of ways to solve outliers
