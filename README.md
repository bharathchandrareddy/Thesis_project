# Thesis_project


# INTRO
My thesis is focused on improving the generalization ability of the existing benchmarks using simple and unique augmentation method. This research work includes analysing the benchmark systems for any undetected biases or scope for improvement. We observed that benchmark systems are not accurately detecting the edges and corners of the buildings which leads to less score in minor and major damage. 

To improve the edge detection capability of the model we included a simple fusion-based augmentation which is initially proved to be effective in satellite image super resolution. The rest of the work is followed by implementing fusion based augmentation for building damage detection and evaluating its effectiveness in new unseen disaster event dataset by using different domain adaptation methods.

This repository contains code for experiments for implementing the fusion-based augmentation and testing it on unseen locations using domain adaptation methods using the top-3 winning solutions for xView2 "xView2: Assess Building Damage" challenge.

The code for winning solutions are provided in:
  first place solution: https://github.com/DIUx-xView/xView2_first_place
  second place solution: https://github.com/DIUx-xView/xView2_second_place
  third place solution: https://github.com/DIUx-xView/xView2_third_place/tree/master
  
Detailed description for running those solutions are provided in thier respective repositories. Make sure to follow the same environment and requirements as their solutions.
# Creating test set
To Analyse the benchmark solutions i've created 17 new datasets which consists of images from all the locations, unlike test set of competition which consists of images only from few locations
- taken out 10% of the data from train and tier3 sets of xbd and created a seperate folder. Make sure to download these folders to get accurate results as in thesis link: https://drive.google.com/file/d/1HaoGiyhs3VooryxA9CtZ_BKc_tNVa_94/view?usp=sharing
- Since i've used data from train and tier-3 sets of xbd for testing the model, you have to make sure to delete those images before training the model. To delete the test images from train and tier-3 set run the following file ./delete_test_samples.py
- be sure to modify the arguments marked as ####To be changed#### in the ./delete_test_samples.py files.
# Analysing benchmark solutions on In-Domain dataset
since the model weights are provided, you can use them to analyse the model on provided test set to check its generalizability.


# How to use
# Adding Fusion augmentation to benchmark codes
add the code from augmentation folder into augmentation pipeline of the benchmark solutions. To make it simple, you can just go into the benchmark code augmentation file and replace it with the following code to run the fusion augmentation
* Add the file contents of first_place_augmentations.py file into utils.py in first place winning solution
* add the file contents of second_place_augmentations.py file into augs.py in second place winning solution
* add the file contents of third_place_augmentations.py file into augmentations.py file in third place solution
I've run the top-3 winning solutions by adding proposed methodology and the final evaulation proved that the fusion based augmentation method helped the model to generalize better in minor and major damage classes by improving F1 score by 5-7%.

Augmented Model weights:
the following folder has the model weights of xview2 solution after adding augmentation
- link: https://drive.google.com/drive/folders/1UI3n5hnqkR4RJByXghwweFEP-x4TrK0Q?usp=sharing
