# GlaucomaRecognition
This is the code of Glaucoma Grading from Multi-Modality imAges. Task 1 is glaucoma grading, task 2 is macular fovea localization, and task 3 is optic disc/cup segmentation.  
Framework: PaddlePaddle (It is a Torch-like DL framework built by Baidu)  
GPU: Tesla V100  
Mainly achieved by Jupter Notebook in Python, transforms.py and functional.py are written for data augmentation and image processing.    

## Task 1
Glaucoma Grading (Classification, 3 classed): non, early, mid-advanced  
Backbone: ResNet101  
Cross Entropy Loss  
Metric: kappa  

## Task 2
Locate the macular fovea by (x,y)  
Loss = 0.5MSE Loss + 0.5Euclidean Distance   

## Task 3
Segment optic disc/cup from fundus images.  
UNet, G Residual Network.  
Cross Entropy Loss   
Metric: Dice  

