import pandas as pd
import csv
import numpy as np
from numpy import linalg
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import sys
import time
import math
from sklearn.cluster import KMeans, SpectralClustering






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

	return(clusters,centers)


		
def spectral_clustering(data,K=5,sigma=1,weighted=True):
	#here K is a parameter to KNN
	W = np.ndarray(shape=(len(data),len(data)))
	D = np.zeros(shape=(len(data),len(data)))
	#compute distance of each item to other items
	for i in range(len(data)):
		t = np.asarray(compute_dist(data[i,:],data))
		t = np.exp((-1*t)/(2*sigma**2))

		#select top-K neighbours for adjancy-similarity matrix
		t[t<(sorted(t)[-K-1])] = 0
		W[i,:] = t
		D[i,i] = sum(t)
	L = D - W

	v,E = linalg.eig(L)
	#print(sorted(v))
	kmeans_K = len(data) - np.argmax(np.sort_complex(v)[1:]-np.sort_complex(v)[0:-1])#select k for kmeans as biggest difference
	print('chosen K for spectral clustering : ',kmeans_K)

	#find top-k eigenvalues and feed those corresponding eigen-vectors to K-means
	ind = np.argsort(v)[-K:]
	edata = E[:,ind]
	clusters,centers = Kmeans(edata,kmeans_K)
	return clusters,centers



def compute_dist(data,centers):
	dist = []
	for c in centers:
		dist.append( np.sqrt( sum( np.power(data-c,2) )) )
	return dist

def compute_external(clusters,classes):
	#compute entropy for each class
	entropy = 0
	purity = 0
	Hyc = 0 #P(Class|Cluster)
	Hy = 0 #P(class)
	Hc = 0 ##P(cluster)
	for i in list(set(clusters)):
		epr = 0
		pur = 0
		s = 0
		for j in list(set(classes)):
			if(j==-1): #don't consider outliers
				continue
			indcls = [x for x in range(len(classes)) if classes[x]==j] #find number of classess j in data
			indclu = [x for x in range(len(clusters)) if clusters[x]==i]#find number of cluster i in data
			t = len([x for x in indcls if x in indclu])/len(indclu) #number of items with class j in cluster i / all items of class j
			if(t != 0):
				epr += t*math.log2(t) #entropy for each cluster
				s += len([x for x in indclu if x in indcls]) #number of element for each cluster
				pur = max(t,pur)
		Hyc += (-1*len(indclu)/len(clusters) ) * epr
		entropy += (len(indclu)/len(clusters))*(-1*epr)
		purity += (len(indclu)/len(clusters))*(pur)
		Hc += (-1)* (len(indclu)/len(clusters))*math.log2(len(indclu)/len(clusters))
	#computing Hy
	Hy = 0
	for j in list(set(classes)):
		if(j!=-1):
			indcls = [x for x in range(len(classes)) if classes[x]==j] #find number of classess i in data
			Hy += -1 * (len(indcls)/len(classes)) * math.log2((len(indcls)/len(classes)))

	NMI = (2* (Hy-Hyc) )/(Hy+Hc)
	return entropy,purity,NMI

def compute_internal(clusters,centers,data,no_center=False):
	#those clustering models like spectral that does not have centers, should be skipped 
	dist = 0
	bic = 0
	if(not no_center):
		for i in range(len(data)):
			dist += np.sqrt( sum( np.power(data[i,:]-centers[int(clusters[i]),:],2) ))
		bic = dist + math.log(len(data)) * len(centers) * len(centers[0])

	#CH index	
	ch = calinski_harabasz_score(data,clusters)

	#Davies-Boulding index
	BD = 0
	if(not no_center):
		for c in list(set(clusters)):
			maxd = 0
			c = int(c)
			distc = sum(compute_dist(centers[c,:],data[clusters==c,:]))
			for j in list(set(clusters)):
				j = int(j)
				if(c==j): #do not check the same cluster
					continue
				distj = sum(compute_dist(centers[j,:],data[clusters==j,:]))
				dist_cj = sum(compute_dist(centers[c,:],[centers[j,:]]))
				if (distc+distj)/dist_cj > maxd:
					maxd = (distc+distj)/dist_cj
			BD += maxd 
		BD = BD/len(list(set(clusters)))

	#silhouette index
	SH = [0]*len(data)
	for i in range(len(data)):
		cl = int(clusters[i])#cluster of current item
		#if there is only one data in this cluster, S-index = 0
		if len(np.argwhere(clusters==cl)) < 2:
			SH[i] = 0
			continue
		#inter cluster distance
		a = sum(compute_dist(data[i,:],data[clusters==cl,:]))/(len(np.argwhere(clusters==cl))-1)
		b = []
		for c in list(set(clusters)):
			c = int(c)
			#do not consider similar cluster
			if(c == cl): continue
			b.append(sum(compute_dist(data[i,:],data[clusters==c,:]))/(len(np.argwhere(clusters==c))))
		b = min(b)
		SH[i] = (b-a)/max(b,a)
	SH = np.mean(SH)


	return bic,ch,BD,SH


#test part
L = np.zeros(shape=(10,10))
L[0,:] = [5,-1,-1,0,0,-1,0,0,-1,-1]
L[1,:] = [-1,2,-1,0,0,0,0,0,0,0]
L[2,:] = [-1,-1,2,0,0,0,0,0,0,0]
L[3,:] = [0,0,0,2,-1,-1,0,0,0,0]
L[4,:] = [0,0,0,-1,2,-1,0,0,0,0]
L[5,:] = [-1,0,0,-1,-1,5,-1,-1,0,0]
L[6,:] = [0,0,0,0,0,-1,2,-1,0,0]
L[7,:] = [0,0,0,0,0,-1,-1,2,0,0]
L[8,:] = [-1,0,0,0,0,0,0,0,2,-1]
L[9,:] = [-1,0,0,0,0,0,0,0,-1,2]

K=4
v,Le = linalg.eig(L)
ind = np.argsort(v)[-K:]
ind = np.argsort(v)[1:K]
Le = Le[:,ind]
CL,cen = Kmeans(Le,K=K)
CL = KMeans(n_clusters=K).fit(Le)

#CL,cen = spectral_clustering(L,K=5)
print(CL.labels_);exit()
print(CL); exit()


#####main part
data_file = sys.argv[1]
if(len(sys.argv)==3):
	arg = sys.argv[2]
else:
	arg = 'single_run'

#reading datasets and putting them into numpy arrays
data = np.loadtxt(open(data_file,newline=''),delimiter='\t')


if(arg == 'single_run'):
	
	visualize(data[:,2:],data[:,1],'raw data')

	#call K-means
	C1,cen1 = Kmeans(data[:,2:],5,eps=0.00001)
	#visualize changes
	visualize(data[:,2:],C1,'Kmeans results')		
	en,pr,nmi = compute_external(C1.tolist(),data[:,1])
	bic,ch,BD,sh = compute_internal(C1,cen1,data[:,2:])

	print('##### running Kmeans ######')
	print('Entropy of K-means: ',en)
	print('Purity of K-means: ',pr)
	print('Normalized Mutual Information of K-means: ',nmi)
	print('BIC of K-means: ',bic)
	print('Calinski Harasbaz index of K-means: ',ch)
	print('Davies-Boulding index of K-means: ',BD)
	print('silhouette index of K-means: ',sh)

	print('##### running ScikitLearn Kmeans ######')
	res = KMeans(n_clusters=5).fit(data[:,2:])
	visualize(data[:,2:],C1,'Scikit Learn Kmeans results')		
	en,pr,nmi = compute_external(res.labels_,data[:,1])
	bic,ch,BD,sh = compute_internal(res.labels_,res.cluster_centers_,data[:,2:])

	print('##### running Kmeans ######')
	print('Entropy of K-means: ',en)
	print('Purity of K-means: ',pr)
	print('Normalized Mutual Information of K-means: ',nmi)
	print('BIC of K-means: ',bic)
	print('Calinski Harasbaz index of K-means: ',ch)
	print('Davies-Boulding index of K-means: ',BD)
	print('silhouette index of K-means: ',sh)




	#call spectral_clustering
	C2,cen2 = spectral_clustering(data[:,2:],5)
	#cen2 = np.real(cen2)
	visualize(data[:,2:],C2,'spectral_clustering results')		
	en,pr,nmi = compute_external(C2.tolist(),data[:,1])
	bic,ch,BD,sh = compute_internal(C2,cen2,data[:,2:],no_center = True)

	print('##### running spectral_clustering ######')
	print('Entropy of spectral clustering: ',en)
	print('Purity of spectral clustering: ',pr)
	print('Normalized Mutual Information of spectral clustering: ',nmi)
	print('Calinski Harasbaz index of spectral clustering: ',ch)
	print('silhouette index of spectral clustering: ',sh)



	res = SpectralClustering(n_clusters=5).fit(data[:,2:])
	visualize(data[:,2:],res.labels_,'ScikitLearn spectral_clustering results')		
	en,pr,nmi = compute_external(res.labels_,data[:,1])
	bic,ch,BD,sh = compute_internal(res.labels_,cen2,data[:,2:],no_center = True)

	print('##### running Scikit Learn spectral_clustering ######')
	print('Entropy of spectral clustering: ',en)
	print('Purity of spectral clustering: ',pr)
	print('Normalized Mutual Information of spectral clustering: ',nmi)
	print('Calinski Harasbaz index of spectral clustering: ',ch)
	print('silhouette index of spectral clustering: ',sh)




	#internal index for actual labels
	print('##### internal index for actual classes ######')
	bic,ch,BD,sh = compute_internal(data[:,1],cen2,data[:,2:],no_center = True)
	print('Calinski Harasbaz index of actual labels: ',ch)
	print('silhouette index of actual labels: ',sh)

if(arg == 'compare'):
	metrics = np.zeros(shape=(19,4))

	for K in range(2,21):
		#call K-means
		C1,cen1 = Kmeans(data[:,2:],K,eps=0.00001)
		#visualize changes
		en,pr,nmi = compute_external(C1.tolist(),data[:,1])
		bic,ch,BD,sh = compute_internal(C1,cen1,data[:,2:])
		metrics[K-2,0] = K; metrics[K-2,1] = en; metrics[K-2,2] = pr; metrics[K-2,3] = sh
		
	plt.figure('Kmeans entropy')
	plt.xlabel('K#'); plt.ylabel('Entropy')
	plt.plot(metrics[:,0],metrics[:,1])
	plt.show()

	plt.figure('Kmeans Purity')
	plt.xlabel('K#'); plt.ylabel('Purity')
	plt.plot(metrics[:,0],metrics[:,2])
	plt.show()

	plt.figure('Kmeans silhouette')
	plt.xlabel('K#'); plt.ylabel('silhouette')
	plt.plot(metrics[:,0],metrics[:,3])
	plt.show()


	for K in range(2,21):
		#call K-means
		C2,cen2 = spectral_clustering(data[:,2:],K)	
		en,pr,nmi = compute_external(C2.tolist(),data[:,1])
		bic,ch,BD,sh = compute_internal(C2,cen2,data[:,2:],no_center = True)
		metrics[K-2,0] = K; metrics[K-2,1] = en; metrics[K-2,2] = pr; metrics[K-2,3] = sh
		
	plt.figure('Spectral Clustering entropy')
	plt.xlabel('K#'); plt.ylabel('Entropy')
	plt.plot(metrics[:,0],metrics[:,1])
	plt.show()

	plt.figure('Spectral Clustering Purity')
	plt.xlabel('K#'); plt.ylabel('Purity')
	plt.plot(metrics[:,0],metrics[:,2])
	plt.show()

	plt.figure('Spectral Clustering silhouette')
	plt.xlabel('K#'); plt.ylabel('silhouette')
	plt.plot(metrics[:,0],metrics[:,3])
	plt.show()

