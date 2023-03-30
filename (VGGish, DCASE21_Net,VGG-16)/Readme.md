

# Python scripts 
The repository contains following scripts,

(a) Fine_tuning_pruned_network.py: To obtain a pruned network, given a pruning ratio and important indexes per convolutional layer and fine-tuning the same pruned network.[ DCASE21_pruning_finetunig.py,VGGish_pruning_finetunig.py,  VGG16_MNIST_finetunig.py ] 

(b) Unpruned model testing:   unpruned_xxxx_testing.py [Please upload a pruned model from the links below to evalute pruned model) 


# Folders


(a) "importance score":  Sorted indexes per convolutional layer as obtained using the proposed pruning, l_1-norm based pruning and geometrical median based pruning methods for networks, (i) VGGish_Net, (ii) DCASE21_Net and (iii) VGG-16 network.


 Links: Datasets (numpy format features), unpruned models and pruned model obtained using the proposed method,
Link:  https://doi.org/10.5281/zenodo.7119930

The above link contains three folders corresponding to the following networks,
(i) DCASE21_Net:  Pruned model , unpruned model and the dataset (features stored in numpy array)

(ii) VGGish_Net: Pruned model, unpruned model and the dataset (features stored in numpy array)

(iii) VGG16_MNIST: Pruned and unpruned models.
