# Script to obtain importance of filters using the proposed operato norm pruning method, entry-wise l_1 norm based scores and geometric median based scores.

#%% Import modules
import numpy as np    
import os
from scipy.stats.mstats import gmean
from scipy.spatial import distance

os.chdir('~/importance_scores/VGG16_MNIST/')

#%%  Proposed pruning framework
def operator_norm_pruning(W):
	C_M=[]	
	mean_vec=[]
	for i in range(np.shape(W)[1]):
		A=W[:,i,:].T
		A_mean=np.mean(A,0)
		e=np.tile(A_mean,(np.shape(A)[0],1))
		A_centred=A-e
		mean_vec.append(A_mean)
		u,q,v=np.linalg.svd(A_centred)
		u1=np.reshape(u[:,0],(np.shape(A)[0],1))
		v1=np.reshape(v[0,:],(np.shape(A)[1],1))
		c_1=np.matmul(u1,v1.T)
		c_1_norm=c_1[0,:]/np.linalg.norm(c_1[0,:])
		C_M.append(c_1_norm)
	Score=[]
	for i in range(np.shape(W)[2]):
		Score.append(np.trace((np.matmul((W[:,:,i]-np.array(mean_vec).T).T,np.array(C_M).T))))
	Mse_score=(np.array(Score))**2
	Mse_score_norm=Mse_score/np.max(Mse_score)
	return Mse_score_norm

#%% entry-wise l_1 norm based scores
def L1_Imp_index(W):
	Score=[]
	for i in range(np.shape(W)[2]):
		Score.append(np.sum(np.abs(W[:,:,i])))
	return Score/np.max(Score)

#%% Geometric median based scores
def GM_Imp_index(W):
	G_GM=gmean(np.abs(W.flatten()))
	Diff=[]
	for i in range(np.shape(W)[2]):
		F_GM=gmean(np.abs(W[:,:,i]).flatten())
		Diff.append((G_GM-F_GM)**2)
	return Diff/np.max(Diff)	
   
#%% load weights from the unpruned network (we have used numpy format to save  and load the pre-trained weights)

W_init = list(np.load('/~/VGG_MNIST/VGG_MNIST_baseline_200/best_weights_numpy.npy', allow_pickle=True))#list(np.load('/home/arshdeep/Pruning/SPL/VGG_pruned_Model/VGG-CIFAR100_Pruning/data/VGG_weights100.npy',allow_pickle=True))
    
#%% Obtaining layer-wise importance scores of CNN filters
indexes=[0,6,12,18,24,30,36,42,48,54,60,66,72] # indexes of convolution layers in W_init for VGG-16
L=[1,2,3,4,5,6,7,8,9,10,11,12,13]  #%% convolutional layer number (for VGG-16, it is from 1 to 13)


for j in range(len(L)):
	print(j)
	W_2D=W_init[indexes[j]]
	W=np.reshape(W_2D,(9,np.shape(W_2D)[2],np.shape(W_2D)[3]))
	print(np.shape(W),'layer  :','  ',L[j])
	print(np.shape(W),'shape of weights')
	score_norm_m1 = operator_norm_pruning((W)
#	Score_L1=CVPR_L1_Imp_index(W)  #l_1 entry wise norm based important scores
# 	Score_GM=CVPR_GM_Imp_index(W)  #Geomettric median based important scores
 	file_name='sim_index'+str(L[j])+'.npy'
 	np.save(file_name,np.argsort(score_norm_m1)) # save sorted arguments from low to high importance.




    
