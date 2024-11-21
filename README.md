# Thesis_project
This repo has all the files related to my thesis

#INTRO
My thesis is focused on improving the generalization ability of the existing benchmarks using simple and unique augmentation method. This research work includes analysing the benchmark systems for any undetected biases or scope for improvement. We observed that benchmark systems are not accurately detecting the edges and corners of the buildings which leads to less score in minor and major damage. To improve the edge detection capability of the model we included a simple fusion-based augmentation which is initially proved to be effective in satellite image super resolution. The rest of the work is followed by implementing fusion based augmentation for building damage detection and evaluating its effectiveness in new unseen disaster event dataset by using different domain adaptation methods.

This repository contains code for experiments for implementing the fusion-based augmentation and testing it on unseen locations using domain adaptation methods using the top-3 winning solutions for xView2 "xView2: Assess Building Damage" challenge.

The code for winning solutions are provided in:
  first place solution: https://github.com/DIUx-xView/xView2_first_place
  second place solution: https://github.com/DIUx-xView/xView2_second_place
  third place solution: https://github.com/DIUx-xView/xView2_third_place/tree/master
Detailed description for running those solutions are provided in thier respective repositories. Make sure to follow the same environment and requirements as their solutions.
