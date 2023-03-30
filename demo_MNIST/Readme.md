This is a demonstration of performance (accuracy, paramters) given by the unpruned network(VGG-16 with 15M paramters) and the pruned network (pruned VGG-16 with 0.18M parameters) obtained using the Operator norm based pruning for MNIST testing dataset

# Before running the demonstration,
 
 (a) Please download the zip folder "MNIST_test_data_with_groundtruth.zip" containing testing datast in the format "SrNo_class_(True_class).txt", 
 Here SrNo is the index of the testing file and it varies from 0 to (10000-1). True class is a groundtruth label.
 
 (b) Download pruned model and unpruned model corresponding to VGG-16 network from the given Link:  https://doi.org/10.5281/zenodo.7119930  
 
 (Folder: VGG16_MNIST) 
 
 
 (c) Set unpruned/pruned model path in the demo_MNIST.py script.
 
 (d) change the current directory wherever the (a) folder is downloaded


# How to run demo? Follow the steps given below,

1. Select a CNN Model , (i) Unpruned model or (ii) Pruned model
2. Paste file name from (a) folder (say 0_class_7.txt) and enter "upload & prediction" button.
3. The model size, predicted category, Top-3 predictions with probabilities will be shown.
4. To test other model as selected in (1), select other model and press again "upload & prediction" button.
5. To clear prediction, press "clear prediction" button.
