Dual Attention Graph Convolutional Network
=============

About
-----

PyTorch implementation of DAGCN (Dual Attention Graph Convolutional Networks).

Requirements: python 2.7 or python 3.6; pytorch >= 0.4.0


Installation
------------

This implementation is based on Hanjun Dai's structure2vec graph backend. Under the "lib/" directory, type

    make -j4

to compile the necessary c++ files.

Or type 

    ./run_DAGCN.sh DATANAME FOLD

to run on dataset = DATANAME using fold number = FOLD (1-10, corresponds to which fold to use as test data in the cross-validation experiments).

If you set FOLD = 0, e.g., typing "./run_DAGCN.sh DD 0", then it will run 10-fold cross validation on DD and report the average accuracy.

Alternatively, type

    ./run_DAGCN.sh DATANAME 1 200

to use the last 200 graphs in the dataset as testing graphs. The fold number 1 will be ignored.

Check "run_DAGCN.sh" for more options.

Datasets
--------

Default graph datasets are stored in "data/DSName/DSName.txt". Check the "data/README.md" for the format. 

Reference
---------

If you find the code useful, please cite our paper:

  @article{chen2019dagcn,\
  title={DAGCN: Dual Attention Graph Convolutional Networks},\
  author={Chen, Fengwen and Pan, Shirui and Jiang, Jing and Huo, Huan and Long, Guodong},\
  journal={arXiv preprint arXiv:1904.02278},\
  year={2019}\
}

Fengwen Chen, \
FENGWEN.CHEN@student.uts.edu.au, \
22 Aug, 2019
