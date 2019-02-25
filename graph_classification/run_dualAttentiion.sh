#!/bin/bash

# input arguments
DATA="${1-DD}"  # MUTAG, ENZYMES, NCI1, NCI109, DD, PTC, PROTEINS, COLLAB, IMDBBINARY, IMDBMULTI
fold=${2-1}  # which fold as testing data
#test_number=${3-0}  # if specified, use the last test_number graphs as test data
GPU=${3-0} # gpu number

# output setting
#name="ENZYMES_trail1.txt"
#logDir="./log/ENZYMES_trail1.txt"
#logDes="10 cross, both latents 64, 0.0001 epoch 500 1b3d3k "

# general settings
gm=attention  # model
gpu_or_cpu=0
CONV_SIZE=64
FP_LEN=0  # final dense layer's input dimension, decided by data
n_hidden=64  # final dense layer's hidden size
bsize=50  # batch size

max_block=1
dropout=0.3
max_k=3

reg=0
multi_head=1

# dataset-specific settings
case ${DATA} in
NCI1)
  name="NCI1_trail6.pk"
  logDir="./log/NCI1_trail6.txt"

  max_block=1
  dropout=0.5
  max_k=5

  reg=2
  multi_head=16

  num_epochs=500
  learning_rate=0.0001
;;
MUTAG)
  name="MUTAG_Trail11.pk"
  logDir="./log/MUTAG_Trail11.txt"

  max_block=1
  dropout=0.5
  max_k=2

  reg=2

  num_epochs=500
  learning_rate=0.0001
  ;;
ENZYMES)
  name="ENZYMES_Trail15.pk"
  logDir="./log/ENZYMES_Trail15.txt"

  max_block=1
  dropout=0.3
  max_k=5

  reg=2
  multi_head=16

  num_epochs=500
  learning_rate=0.0001
  ;;
NCI109)
  name="NCI109_trail5.pk"
  logDir="./log/NCI109_trail5.txt"

  max_block=1
  dropout=0.5
  max_k=5

  reg=2
  multi_head=16

  num_epochs=500
  learning_rate=0.0001
  ;;
DD)
  name="DD_Trail23.pk"
  logDir="./log/DD_Trail23.txt"

  max_block=10
  dropout=0.5
  max_k=10

  multi_head=32
  reg=2

  num_epochs=500
  learning_rate=0.0001
  ;;
PTC)
  name="PTC_trail5.pk"
  logDir="./log/PTC_trail5.txt"

  max_block=1
  dropout=0.5
  max_k=3

  reg=2

  num_epochs=500
  learning_rate=0.00001
  ;;
PROTEINS)
  name="PROTEINS_trail20.pk"
  logDir="./log/PROTEINS_trail20.txt"

  CONV_SIZE=64
  n_hidden=64  # final dense layer's hidden size

  max_block=10
  dropout=0.5
  max_k=5

  reg=2
  multi_head=32

  num_epochs=500
  learning_rate=0.0001
  ;;
COLLAB)
  name="COLLAB_trail4.pk"
  logDir="./log/COLLAB_trail4.txt"

  max_block=5
  dropout=0.5
  max_k=5

  reg=2
  multi_head=16

  num_epochs=500
  learning_rate=0.001
  ;;
IMDBBINARY)
  name="IMDBBINARY_trail11.pk"
  logDir="./log/IMDBBINARY_trail11.txt"

  max_block=5
  dropout=0.5
  max_k=5

  reg=2
  multi_head=32

  num_epochs=500
  learning_rate=0.0001
  ;;
IMDBMULTI)
  name="IMDBMULTI_trail18.pk"
  logDir="./log/IMDBMULTI_trail18.txt"

  max_block=20
  dropout=0.5
  max_k=5

  reg=2
  multi_head=16

  num_epochs=500
  learning_rate=0.0001
  ;;
*)
  num_epochs=500
  learning_rate=0.0001
  ;;
esac

if [ ${fold} == 0 ]; then
  rm $name
  echo "Running 10-fold cross validation"
  start=`date +%s`
  for i in $(seq 1 10)
  do
    CUDA_VISIBLE_DEVICES=${GPU} python main.py \
        -seed 1 \
        -data $DATA \
        -fold $i \
        -learning_rate $learning_rate \
        -num_epochs $num_epochs \
        -hidden $n_hidden \
        -latent_dim $CONV_SIZE \
        -out_dim $FP_LEN \
        -batch_size $bsize \
        -gm $gm \
        -mode $GPU \
        -max_block $max_block \
        -dropout $dropout \
        -max_k $max_k \
        -name $name \
        -logDir $logDir \
        -multi_h_emb_weight=$multi_head \
        -reg $reg
  done
  stop=`date +%s`
  echo "End of cross-validation"
  echo "The total running time is $[stop - start] seconds."
#  echo "The accuracy results for ${DATA} are as follows:"
#  cat $name
#  echo "Average accuracy is"
#  cat $name | awk '{ sum += $1; n++ } END { if (n > 0) print sum / n; }'
else
  CUDA_VISIBLE_DEVICES=${GPU} python main.py \
      -seed 1 \
      -data $DATA \
      -fold $fold \
      -learning_rate $learning_rate \
      -num_epochs $num_epochs \
      -hidden $n_hidden \
      -latent_dim $CONV_SIZE \
      -out_dim $FP_LEN \
      -batch_size $bsize \
      -gm $gm \
      -mode $GPU \
      -max_block $max_block \
      -dropout $dropout \
      -max_k $max_k \
      -name $name \
      -logDir $logDir \
      -reg $reg \
      -multi_h_emb_weight=$multi_head \
      -test_number ${test_number}
fi