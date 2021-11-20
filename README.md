# GlaucomaRecognition
This is the code of Glaucoma Grading from Multi-Modality imAges. Task 1 is glaucoma grading, task 2 is macular fovea localization, and task 3 is optic disc/cup segmentation.

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
Cross Entropy Loss 
Metric: Dice

