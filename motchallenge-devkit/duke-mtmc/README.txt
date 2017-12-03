DukeMTMC Tracking Challenge
==================================
---- http://motchallenge.net -----
http://vision.cs.duke.edu/DukeMTMC
----------------------------------

Version 1.00

This development kit provides scripts to evaluate tracking results for the DukeMTMC Challenge.
It extends the motchallenge devkit v1.1 with the DukeMTMC identity measures of performance.

Please report bugs to ristani@cs.duke.edu or anton.milan@adelaide.edu.au


Requirements
============
- MATLAB (tested on Windows only with MATLAB R2016b)
  

Usage
=====

You can run demoDukeMTMCEvaluate and it will compute the baseline results on the set trainval_mini.

This is what you should expect from the script:

-------Results-------
Test set: trainval_mini
Single-all
 IDF1  IDP  IDR| Rcll  Prcn   FAR  GT| MT  PT  ML    FP|   FN  IDs   FM  MOTA|  MOTP MOTAL 
 75.0 85.5 66.8| 74.7  95.6  0.09 1489|912 481  96 40726|301360  378 1277  71.3|  76.7  71.3 
Multi-cam	IDF1: 54.98	IDP: 62.67	IDR: 48.97

Cam_1
 IDF1  IDP  IDR| Rcll  Prcn   FAR  GT| MT  PT  ML    FP|   FN  IDs   FM  MOTA|  MOTP MOTAL 
 62.2 91.0 47.3| 49.3  94.9  0.09 274| 27 216  31  5280|101222   36   97  46.6|  77.8  46.6 
Cam_2
 IDF1  IDP  IDR| Rcll  Prcn   FAR  GT| MT  PT  ML    FP|   FN  IDs   FM  MOTA|  MOTP MOTAL 
 81.4 87.5 76.1| 83.7  96.3  0.12 272|202  56  14  7465|37324   94  281  80.4|  78.9  80.5 
Cam_3
 IDF1  IDP  IDR| Rcll  Prcn   FAR  GT| MT  PT  ML    FP|   FN  IDs   FM  MOTA|  MOTP MOTAL 
 81.4 93.1 72.4| 75.5  97.1  0.03 127| 82  32  13  1598|17211   25   67  73.2|  76.3  73.2 
Cam_4
 IDF1  IDP  IDR| Rcll  Prcn   FAR  GT| MT  PT  ML    FP|   FN  IDs   FM  MOTA|  MOTP MOTAL 
 78.6 84.1 73.7| 81.6  93.2  0.10 106| 77  24   5  5609|17373   25   77  75.6|  76.5  75.7 
Cam_5
 IDF1  IDP  IDR| Rcll  Prcn   FAR  GT| MT  PT  ML    FP|   FN  IDs   FM  MOTA|  MOTP MOTAL 
 83.9 88.4 79.9| 88.1  97.4  0.04 144|124  14   6  2495|12667   37  112  85.8|  78.0  85.8 
Cam_6
 IDF1  IDP  IDR| Rcll  Prcn   FAR  GT| MT  PT  ML    FP|   FN  IDs   FM  MOTA|  MOTP MOTAL 
 71.1 74.9 67.7| 85.9  95.0  0.17 215|181  28   6 10377|32706   98  261  81.4|  76.8  81.4 
Cam_7
 IDF1  IDP  IDR| Rcll  Prcn   FAR  GT| MT  PT  ML    FP|   FN  IDs   FM  MOTA|  MOTP MOTAL 
 85.7 90.2 81.5| 86.5  95.7  0.05 110| 92  16   2  2781| 9662   21  183  82.6|  70.0  82.6 
Cam_8
 IDF1  IDP  IDR| Rcll  Prcn   FAR  GT| MT  PT  ML    FP|   FN  IDs   FM  MOTA|  MOTP MOTAL 
 68.9 88.3 56.5| 61.2  95.8  0.09 241|127  95  19  5121|73195   42  199  58.5|  75.2  58.5 


Details
=======
The evaluation script accepts 4 arguments:

results = evaluateDukeMTMC(trackerOutput, iou_threshold, world, testSet);

- trackerOutput in format [cam, id, frame, left, top, width, height, worldX, worldY]
- intersection-over-union threshold (default = 0.5)
- world or image plane evaluation (default = false)
- name of testSet (default = 'trainval_mini')

The results will be computed for each individual camera (Cam_1-8), for all single cameras 
(aggreage result Single-all) and for the multi-camera scenario (Multi-cam).


Version history
===============

1.01 - Mar 25, 2017
  - performance enhacements

1.00 - Nov 28, 2016
  - initial release