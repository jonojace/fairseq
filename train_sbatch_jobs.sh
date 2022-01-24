####################################### 18th aug ######################################
#
#MODEL_NAME=model_ALL_SimpleLSTMEnc_lr0.001
#DATA=/home/s1785140/data/ljspeech_wav2vec2_reps/wav2vec2-large-960h/layer-15/word_level/
#../sbatch.sh 2080 fairseq-train $DATA \
#    --tensorboard-logdir tb_logs/$MODEL_NAME \
#    --task learn_lexicon \
#    --arch lexicon_learner \
#    --criterion lexicon_learner \
#    --optimizer adam \
#    --batch-size 64 \
#    --min-train-examples-per-wordtype 10 \
#    --max-train-examples-per-wordtype 10 \
#    --valid-seen-wordtypes 100 \
#    --valid-unseen-wordtypes 100 \
#    --valid-examples-per-wordtype 10 \
#    --valid-subset valid-seen,valid-unseen \
#    --save-dir checkpoints/$MODEL_NAME \
#    --save-interval 1 --max-epoch 1000 \
#    --lr 0.001
#
#MODEL_NAME=model_ALL_SimpleLSTMEnc_lr0.01
#DATA=/home/s1785140/data/ljspeech_wav2vec2_reps/wav2vec2-large-960h/layer-15/word_level/
#./sbatch.sh 2080 fairseq-train $DATA \
#    --tensorboard-logdir tb_logs/$MODEL_NAME \
#    --task learn_lexicon \
#    --arch lexicon_learner \
#    --criterion lexicon_learner \
#    --optimizer adam \
#    --batch-size 64 \
#    --min-train-examples-per-wordtype 10 \
#    --max-train-examples-per-wordtype 10 \
#    --valid-seen-wordtypes 100 \
#    --valid-unseen-wordtypes 100 \
#    --valid-examples-per-wordtype 10 \
#    --valid-subset valid-seen,valid-unseen \
#    --save-dir checkpoints/$MODEL_NAME \
#    --save-interval 1 --max-epoch 1000 \
#    --lr 0.01
#
#MODEL_NAME=model_ALL_SimpleLSTMEnc_lr0.1
#DATA=/home/s1785140/data/ljspeech_wav2vec2_reps/wav2vec2-large-960h/layer-15/word_level/
#./sbatch.sh 2080 fairseq-train $DATA \
#    --tensorboard-logdir tb_logs/$MODEL_NAME \
#    --task learn_lexicon \
#    --arch lexicon_learner \
#    --criterion lexicon_learner \
#    --optimizer adam \
#    --batch-size 64 \
#    --min-train-examples-per-wordtype 10 \
#    --max-train-examples-per-wordtype 10 \
#    --valid-seen-wordtypes 100 \
#    --valid-unseen-wordtypes 100 \
#    --valid-examples-per-wordtype 10 \
#    --valid-subset valid-seen,valid-unseen \
#    --save-dir checkpoints/$MODEL_NAME \
#    --save-interval 1 --max-epoch 1000 \
#    --lr 0.1
#
#MODEL_NAME=model_ALL_SimpleLSTMEnc_lr0.0001
#DATA=/home/s1785140/data/ljspeech_wav2vec2_reps/wav2vec2-large-960h/layer-15/word_level/
#./sbatch.sh 2080 fairseq-train $DATA \
#    --tensorboard-logdir tb_logs/$MODEL_NAME \
#    --task learn_lexicon \
#    --arch lexicon_learner \
#    --criterion lexicon_learner \
#    --optimizer adam \
#    --batch-size 64 \
#    --min-train-examples-per-wordtype 10 \
#    --max-train-examples-per-wordtype 10 \
#    --valid-seen-wordtypes 100 \
#    --valid-unseen-wordtypes 100 \
#    --valid-examples-per-wordtype 10 \
#    --valid-subset valid-seen,valid-unseen \
#    --save-dir checkpoints/$MODEL_NAME \
#    --save-interval 1 --max-epoch 1000 \
#    --lr 0.0001
#
####################################### 19th aug ######################################
#
#MODEL_NAME=model_ALL_SimpleLSTMEnc_lr0.001_dropout0.1
#DATA=/home/s1785140/data/ljspeech_wav2vec2_reps/wav2vec2-large-960h/layer-15/word_level/
#./sbatch.sh 2080 fairseq-train $DATA \
#    --tensorboard-logdir tb_logs/$MODEL_NAME \
#    --task learn_lexicon \
#    --arch lexicon_learner \
#    --criterion lexicon_learner \
#    --optimizer adam \
#    --batch-size 64 \
#    --min-train-examples-per-wordtype 10 \
#    --max-train-examples-per-wordtype 10 \
#    --valid-seen-wordtypes 100 \
#    --valid-unseen-wordtypes 100 \
#    --valid-examples-per-wordtype 10 \
#    --valid-subset valid-seen,valid-unseen \
#    --save-dir checkpoints/$MODEL_NAME \
#    --save-interval 1 --max-epoch 1000 \
#    --lr 0.001 \
#    --enc-dropout-in 0.1 --enc-dropout-out 0.1
#
#MODEL_NAME=model_ALL_SimpleLSTMEnc_lr0.001_dropout0.2
#DATA=/home/s1785140/data/ljspeech_wav2vec2_reps/wav2vec2-large-960h/layer-15/word_level/
#./sbatch.sh 2080 fairseq-train $DATA \
#    --tensorboard-logdir tb_logs/$MODEL_NAME \
#    --task learn_lexicon \
#    --arch lexicon_learner \
#    --criterion lexicon_learner \
#    --optimizer adam \
#    --batch-size 64 \
#    --min-train-examples-per-wordtype 10 \
#    --max-train-examples-per-wordtype 10 \
#    --valid-seen-wordtypes 100 \
#    --valid-unseen-wordtypes 100 \
#    --valid-examples-per-wordtype 10 \
#    --valid-subset valid-seen,valid-unseen \
#    --save-dir checkpoints/$MODEL_NAME \
#    --save-interval 1 --max-epoch 1000 \
#    --lr 0.001 \
#    --enc-dropout-in 0.2 --enc-dropout-out 0.2
#
#MODEL_NAME=model_ALL_SimpleLSTMEnc_lr0.001_dropout0.3
#DATA=/home/s1785140/data/ljspeech_wav2vec2_reps/wav2vec2-large-960h/layer-15/word_level/
#./sbatch.sh 2080 fairseq-train $DATA \
#    --tensorboard-logdir tb_logs/$MODEL_NAME \
#    --task learn_lexicon \
#    --arch lexicon_learner \
#    --criterion lexicon_learner \
#    --optimizer adam \
#    --batch-size 64 \
#    --min-train-examples-per-wordtype 10 \
#    --max-train-examples-per-wordtype 10 \
#    --valid-seen-wordtypes 100 \
#    --valid-unseen-wordtypes 100 \
#    --valid-examples-per-wordtype 10 \
#    --valid-subset valid-seen,valid-unseen \
#    --save-dir checkpoints/$MODEL_NAME \
#    --save-interval 1 --max-epoch 1000 \
#    --lr 0.001 \
#    --enc-dropout-in 0.3 --enc-dropout-out 0.3
#
#MODEL_NAME=model_ALL_SimpleLSTMEnc_lr0.001_dropout0.4
#DATA=/home/s1785140/data/ljspeech_wav2vec2_reps/wav2vec2-large-960h/layer-15/word_level/
#./sbatch.sh 2080 fairseq-train $DATA \
#    --tensorboard-logdir tb_logs/$MODEL_NAME \
#    --task learn_lexicon \
#    --arch lexicon_learner \
#    --criterion lexicon_learner \
#    --optimizer adam \
#    --batch-size 64 \
#    --min-train-examples-per-wordtype 10 \
#    --max-train-examples-per-wordtype 10 \
#    --valid-seen-wordtypes 100 \
#    --valid-unseen-wordtypes 100 \
#    --valid-examples-per-wordtype 10 \
#    --valid-subset valid-seen,valid-unseen \
#    --save-dir checkpoints/$MODEL_NAME \
#    --save-interval 1 --max-epoch 1000 \
#    --lr 0.001 \
#    --enc-dropout-in 0.4 --enc-dropout-out 0.4
#
#MODEL_NAME=model_ALL_SimpleLSTMEnc_lr0.001_dropout0.5
#DATA=/home/s1785140/data/ljspeech_wav2vec2_reps/wav2vec2-large-960h/layer-15/word_level/
#./sbatch.sh 2080 fairseq-train $DATA \
#    --tensorboard-logdir tb_logs/$MODEL_NAME \
#    --task learn_lexicon \
#    --arch lexicon_learner \
#    --criterion lexicon_learner \
#    --optimizer adam \
#    --batch-size 64 \
#    --min-train-examples-per-wordtype 10 \
#    --max-train-examples-per-wordtype 10 \
#    --valid-seen-wordtypes 100 \
#    --valid-unseen-wordtypes 100 \
#    --valid-examples-per-wordtype 10 \
#    --valid-subset valid-seen,valid-unseen \
#    --save-dir checkpoints/$MODEL_NAME \
#    --save-interval 1 --max-epoch 1000 \
#    --lr 0.001 \
#    --enc-dropout-in 0.5 --enc-dropout-out 0.5
#
#MODEL_NAME=model_ALL_SimpleLSTMEnc_lr0.001_dropout0.5
#DATA=/home/s1785140/data/ljspeech_wav2vec2_reps/wav2vec2-large-960h/layer-15/word_level/
#fairseq-train $DATA \
#    --tensorboard-logdir tb_logs/$MODEL_NAME \
#    --task learn_lexicon \
#    --arch lexicon_learner \
#    --criterion lexicon_learner \
#    --optimizer adam \
#    --batch-size 64 \
#    --min-train-examples-per-wordtype 10 \
#    --max-train-examples-per-wordtype 10 \
#    --valid-seen-wordtypes 100 \
#    --valid-unseen-wordtypes 100 \
#    --valid-examples-per-wordtype 10 \
#    --valid-subset valid-seen,valid-unseen \
#    --save-dir checkpoints/$MODEL_NAME \
#    --save-interval 1 --max-epoch 1000 \
#    --lr 0.001 \
#    --enc-dropout-in 0.5 --enc-dropout-out 0.5
#
#
## debug 0 train loss and valid loss error
#MODEL_NAME=model_ALL_SimpleLSTMEnc_DEBUG_TRAIN_VALID_LOSS_lr0.001
#DATA=/home/s1785140/data/ljspeech_wav2vec2_reps/wav2vec2-large-960h/layer-15/word_level/
#./sbatch.sh 2080 fairseq-train $DATA \
#    --tensorboard-logdir tb_logs/$MODEL_NAME \
#    --task learn_lexicon \
#    --arch lexicon_learner \
#    --criterion lexicon_learner \
#    --optimizer adam \
#    --batch-size 64 \
#    --min-train-examples-per-wordtype 10 \
#    --max-train-examples-per-wordtype 10 \
#    --valid-seen-wordtypes 100 \
#    --valid-unseen-wordtypes 100 \
#    --valid-examples-per-wordtype 10 \
#    --valid-subset valid-seen,valid-unseen \
#    --save-dir checkpoints/$MODEL_NAME \
#    --save-interval 1 --max-epoch 1000 \
#    --lr 0.001 \
#    --enc-dropout-in 0.25 --enc-dropout-out 0.25
#
#MODEL_NAME=model_ALL_SimpleLSTMEnc_DEBUG_TRAIN_VALID_LOSS_lr0.0001
#DATA=/home/s1785140/data/ljspeech_wav2vec2_reps/wav2vec2-large-960h/layer-15/word_level/
#./sbatch.sh 2080 fairseq-train $DATA \
#    --tensorboard-logdir tb_logs/$MODEL_NAME \
#    --task learn_lexicon \
#    --arch lexicon_learner \
#    --criterion lexicon_learner \
#    --optimizer adam \
#    --batch-size 64 \
#    --min-train-examples-per-wordtype 10 \
#    --max-train-examples-per-wordtype 10 \
#    --valid-seen-wordtypes 100 \
#    --valid-unseen-wordtypes 100 \
#    --valid-examples-per-wordtype 10 \
#    --valid-subset valid-seen,valid-unseen \
#    --save-dir checkpoints/$MODEL_NAME \
#    --save-interval 1 --max-epoch 1000 \
#    --lr 0.0001 \
#    --enc-dropout-in 0.25 --enc-dropout-out 0.25
#
#
#MODEL_NAME=model_ALL_SimpleLSTMEnc_DEBUG_TRAIN_VALID_LOSS_lr0.0001
#DATA=/home/s1785140/data/ljspeech_wav2vec2_reps/wav2vec2-large-960h/layer-15/word_level/
#fairseq-train $DATA \
#    --tensorboard-logdir tb_logs/$MODEL_NAME \
#    --task learn_lexicon \
#    --arch lexicon_learner \
#    --criterion lexicon_learner \
#    --optimizer adam \
#    --batch-size 64 \
#    --min-train-examples-per-wordtype 10 \
#    --max-train-examples-per-wordtype 10 \
#    --valid-seen-wordtypes 100 \
#    --valid-unseen-wordtypes 100 \
#    --valid-examples-per-wordtype 10 \
#    --valid-subset valid-seen,valid-unseen \
#    --save-dir checkpoints/$MODEL_NAME \
#    --save-interval 1 --max-epoch 1000 \
#    --lr 0.0001 \
#    --enc-dropout-in 0.25 --enc-dropout-out 0.25
#
#
#MODEL_NAME=test_tblogging2
#DATA=/home/s1785140/data/ljspeech_wav2vec2_reps/wav2vec2-large-960h/layer-15/word_level/
#./sbatch.sh 2080 fairseq-train $DATA \
#    --tensorboard-logdir tb_logs/$MODEL_NAME \
#    --task learn_lexicon \
#    --arch lexicon_learner \
#    --criterion lexicon_learner \
#    --optimizer adam \
#    --batch-size 64 \
#    --min-train-examples-per-wordtype 10 \
#    --max-train-examples-per-wordtype 10 \
#    --valid-seen-wordtypes 100 \
#    --valid-unseen-wordtypes 100 \
#    --valid-examples-per-wordtype 10 \
#    --valid-subset valid-seen,valid-unseen \
#    --save-dir checkpoints/$MODEL_NAME \
#    --save-interval 1 --max-epoch 500 \
#    --lr 0.0001 \
#    --enc-dropout-in 0.50 --enc-dropout-out 0.50
#
#
#
#    ##################################################################
#
#
#MODEL_NAME=model_ALL_SimpleLSTMEnc_1hidlayer_dropout0.25
#DATA=/home/s1785140/data/ljspeech_wav2vec2_reps/wav2vec2-large-960h/layer-15/word_level/
#./sbatch.sh 2080 fairseq-train $DATA \
#    --tensorboard-logdir tb_logs/$MODEL_NAME \
#    --task learn_lexicon \
#    --arch lexicon_learner \
#    --criterion lexicon_learner \
#    --optimizer adam \
#    --batch-size 64 \
#    --min-train-examples-per-wordtype 10 \
#    --max-train-examples-per-wordtype 10 \
#    --valid-seen-wordtypes 50 \
#    --valid-unseen-wordtypes 50 \
#    --valid-examples-per-wordtype 10 \
#    --valid-subset valid-seen,valid-unseen \
#    --save-dir checkpoints/$MODEL_NAME \
#    --save-interval 1 --max-epoch 500 \
#    --lr 0.0001 \
#    --enc-dropout-in 0.25 \
#    --enc-dropout-out 0.25 \
#    --enc-num-layers 1
#
#MODEL_NAME=model_ALL_SimpleLSTMEnc_2hidlayer_dropout0.25
#DATA=/home/s1785140/data/ljspeech_wav2vec2_reps/wav2vec2-large-960h/layer-15/word_level/
#./sbatch.sh 2080 fairseq-train $DATA \
#    --tensorboard-logdir tb_logs/$MODEL_NAME \
#    --task learn_lexicon \
#    --arch lexicon_learner \
#    --criterion lexicon_learner \
#    --optimizer adam \
#    --batch-size 64 \
#    --min-train-examples-per-wordtype 10 \
#    --max-train-examples-per-wordtype 10 \
#    --valid-seen-wordtypes 50 \
#    --valid-unseen-wordtypes 50 \
#    --valid-examples-per-wordtype 10 \
#    --valid-subset valid-seen,valid-unseen \
#    --save-dir checkpoints/$MODEL_NAME \
#    --save-interval 1 --max-epoch 500 \
#    --lr 0.0001 \
#    --enc-dropout-in 0.25 \
#    --enc-dropout-out 0.25 \
#    --enc-num-layers 2
#
#MODEL_NAME=model_ALL_SimpleLSTMEnc_3hidlayer_dropout0.25
#DATA=/home/s1785140/data/ljspeech_wav2vec2_reps/wav2vec2-large-960h/layer-15/word_level/
#./sbatch.sh 2080 fairseq-train $DATA \
#    --tensorboard-logdir tb_logs/$MODEL_NAME \
#    --task learn_lexicon \
#    --arch lexicon_learner \
#    --criterion lexicon_learner \
#    --optimizer adam \
#    --batch-size 64 \
#    --min-train-examples-per-wordtype 10 \
#    --max-train-examples-per-wordtype 10 \
#    --valid-seen-wordtypes 50 \
#    --valid-unseen-wordtypes 50 \
#    --valid-examples-per-wordtype 10 \
#    --valid-subset valid-seen,valid-unseen \
#    --save-dir checkpoints/$MODEL_NAME \
#    --save-interval 1 --max-epoch 500 \
#    --lr 0.0001 \
#    --enc-dropout-in 0.25 \
#    --enc-dropout-out 0.25 \
#    --enc-num-layers 3
#
#
#MODEL_NAME=model_ALL_SimpleLSTMEnc_1hidlayer_dropout0.5
#DATA=/home/s1785140/data/ljspeech_wav2vec2_reps/wav2vec2-large-960h/layer-15/word_level/
#./sbatch.sh 2080 fairseq-train $DATA \
#    --tensorboard-logdir tb_logs/$MODEL_NAME \
#    --task learn_lexicon \
#    --arch lexicon_learner \
#    --criterion lexicon_learner \
#    --optimizer adam \
#    --batch-size 64 \
#    --min-train-examples-per-wordtype 10 \
#    --max-train-examples-per-wordtype 10 \
#    --valid-seen-wordtypes 50 \
#    --valid-unseen-wordtypes 50 \
#    --valid-examples-per-wordtype 10 \
#    --valid-subset valid-seen,valid-unseen \
#    --save-dir checkpoints/$MODEL_NAME \
#    --save-interval 1 --max-epoch 500 \
#    --lr 0.0001 \
#    --enc-dropout-in 0.5 \
#    --enc-dropout-out 0.5 \
#    --enc-num-layers 1
#
#MODEL_NAME=model_ALL_SimpleLSTMEnc_2hidlayer_dropout0.5
#DATA=/home/s1785140/data/ljspeech_wav2vec2_reps/wav2vec2-large-960h/layer-15/word_level/
#./sbatch.sh 2080 fairseq-train $DATA \
#    --tensorboard-logdir tb_logs/$MODEL_NAME \
#    --task learn_lexicon \
#    --arch lexicon_learner \
#    --criterion lexicon_learner \
#    --optimizer adam \
#    --batch-size 64 \
#    --min-train-examples-per-wordtype 10 \
#    --max-train-examples-per-wordtype 10 \
#    --valid-seen-wordtypes 50 \
#    --valid-unseen-wordtypes 50 \
#    --valid-examples-per-wordtype 10 \
#    --valid-subset valid-seen,valid-unseen \
#    --save-dir checkpoints/$MODEL_NAME \
#    --save-interval 1 --max-epoch 500 \
#    --lr 0.0001 \
#    --enc-dropout-in 0.5 \
#    --enc-dropout-out 0.5 \
#    --enc-num-layers 2
#
#MODEL_NAME=model_ALL_SimpleLSTMEnc_3hidlayer_dropout0.5
#DATA=/home/s1785140/data/ljspeech_wav2vec2_reps/wav2vec2-large-960h/layer-15/word_level/
#./sbatch.sh 2080 fairseq-train $DATA \
#    --tensorboard-logdir tb_logs/$MODEL_NAME \
#    --task learn_lexicon \
#    --arch lexicon_learner \
#    --criterion lexicon_learner \
#    --optimizer adam \
#    --batch-size 64 \
#    --min-train-examples-per-wordtype 10 \
#    --max-train-examples-per-wordtype 10 \
#    --valid-seen-wordtypes 50 \
#    --valid-unseen-wordtypes 50 \
#    --valid-examples-per-wordtype 10 \
#    --valid-subset valid-seen,valid-unseen \
#    --save-dir checkpoints/$MODEL_NAME \
#    --save-interval 1 --max-epoch 500 \
#    --lr 0.0001 \
#    --enc-dropout-in 0.5 \
#    --enc-dropout-out 0.5 \
#    --enc-num-layers 3
#
#
## run some tests of your first batch of vanilla pytorch models
#DATA=/home/s1785140/data/ljspeech_hubert_reps/hubert-base/layer-6/word_level_without_padding_idx_offset
#RUN_ID=run4
#NHEADS=2
#LAYERS=4
#FFDIM=2048
#DROPOUT=0.1
#for x in 1 2 4 6 8
#do
##  LAYERS=$x
#  NHEADS=$x
#  MODEL_NAME=ALL_hubert_transformer_${RUN_ID}_${LAYERS}layers_${NHEADS}heads_${FFDIM}ffdim_${DROPOUT}dropout
#  cd ~/fairseq || exit
#  ./sbatch.sh fairseq-train $DATA \
#      --tensorboard-logdir tb_logs/$MODEL_NAME \
#      --task learn_lexicon_discrete_inputs \
#      --arch lexicon_learner_seq2seq \
#      --transformer-nheads $NHEADS \
#      --transformer-enc-layers $LAYERS \
#      --transformer-dec-layers $LAYERS \
#      --transformer-dim-feedforward $FFDIM \
#      --transformer-dropout $DROPOUT \
#      --criterion lexicon_learner \
#      --optimizer adam \
#      --batch-size 64 \
#      --padding-index-offset 1 \
#      --min-train-examples-per-wordtype 5 \
#      --max-train-examples-per-wordtype 25 \
#      --valid-seen-wordtypes 100 \
#      --valid-unseen-wordtypes 100 \
#      --valid-examples-per-wordtype 10 \
#      --valid-subset valid-seen,valid-unseen \
#      --save-dir checkpoints/$MODEL_NAME \
#      --save-interval 1 --max-epoch 500 \
#      --lr 0.001 \
#      --cache-all-data
#done
#
#### Run some tests for implementation of soft-dtw loss fn that replaces lstm summariser
#DATA=/home/s1785140/data/ljspeech_hubert_reps/hubert-base/layer-6/word_level_without_padding_idx_offset
#RUN_ID=run5
#NHEADS=2
#LAYERS=4
#FFDIM=2048
#DROPOUT=0.1
#MARGIN=1.0
##for x in 0.1 0.5 1.0 2.0 4.0 6.0
#for x in 1 2 4 6 8
#do
##  LAYERS=$x
##  NHEADS=$x
##  MARGIN=$x
#  MODEL_NAME=ALL_hubert_transformer_${RUN_ID}_${LAYERS}layers_${NHEADS}heads_${FFDIM}ffdim_${DROPOUT}dropout_${MARGIN}margin_wPositional_normAll
#  cd ~/fairseq || exit
#  ./sbatch.sh fairseq-train $DATA \
#      --tensorboard-logdir tb_logs/$MODEL_NAME \
#      --task learn_lexicon_discrete_inputs \
#      --arch lexicon_learner_seq2seq \
#      --transformer-nheads $NHEADS \
#      --transformer-enc-layers $LAYERS \
#      --transformer-dec-layers $LAYERS \
#      --transformer-dim-feedforward $FFDIM \
#      --transformer-dropout $DROPOUT \
#      --criterion lexicon_learner \
#      --triplet-loss-margin 1.0 \
#      --optimizer adam \
#      --batch-size 64 \
#      --padding-index-offset 1 \
#      --min-train-examples-per-wordtype 5 \
#      --max-train-examples-per-wordtype 25 \
#      --valid-seen-wordtypes 100 \
#      --valid-unseen-wordtypes 100 \
#      --valid-examples-per-wordtype 10 \
#      --valid-subset valid-seen,valid-unseen \
#      --save-dir checkpoints/$MODEL_NAME \
#      --save-interval 1 --max-epoch 500 \
#      --lr 0.001 \
#      --normalize-src \
#      --normalize-tgt \
#      --normalize-out \
#      --cache-all-data
#done
#
#### Run Transformer enc/dec + LSTM Summariser jobs to see if loss reduces better than your soft-dtw loss implementation
#DATA=/home/s1785140/data/ljspeech_hubert_reps/hubert-base/layer-6/word_level_without_padding_idx_offset
#MODEL_NAME=ALL_hubert_transformer_summariser
#RUN_ID=debug2
#B_SIZE=64
#NHEADS=2
#LAYERS=2
#FFDIM=2048
#DROPOUT=0.1
#MARGIN=1.0
##for x in 0.1 0.5 1.0 2.0 4.0 6.0 # margin
##for x in 1 2 4 6 8 # layers
##for x in 1 2 4
#for x in 4 16 64 128 320 640 1280 # batch size
#do
##  LAYERS=$x
##  NHEADS=$x
##  MARGIN=$x
#  B_SIZE=$x
#  EXP_NAME=${MODEL_NAME}_${RUN_ID}_${LAYERS}layers_${NHEADS}heads_${FFDIM}ffdim_${DROPOUT}dropout_${MARGIN}margin
#  cd ~/fairseq || exit
#  ./sbatch.sh fairseq-train $DATA \
#      --tensorboard-logdir tb_logs/$EXP_NAME \
#      --task learn_lexicon_discrete_inputs \
#      --arch lexicon_learner_seq2seq \
#      --sequence-loss-method summariser \
#      --transformer-nheads $NHEADS \
#      --transformer-enc-layers $LAYERS \
#      --transformer-dec-layers $LAYERS \
#      --transformer-dim-feedforward $FFDIM \
#      --transformer-dropout $DROPOUT \
#      --criterion lexicon_learner \
#      --triplet-loss-margin 1.0 \
#      --optimizer adam \
#      --batch-size $B_SIZE \
#      --padding-index-offset 1 \
#      --min-train-examples-per-wordtype 5 \
#      --max-train-examples-per-wordtype 25 \
#      --valid-seen-wordtypes 100 \
#      --valid-unseen-wordtypes 100 \
#      --valid-examples-per-wordtype 10 \
#      --valid-subset valid-seen,valid-unseen \
#      --save-dir checkpoints/$EXP_NAME \
#      --save-interval 1 --max-epoch 500 \
#      --lr 0.001 \
#      --normalize-src \
#      --normalize-tgt \
#      --normalize-out \
#      --cache-all-data
#done

###############################################################
# SAC 20th Jan 2022
# Train
# - baseline (with and without randomisation)
# - model with vctk external speechreps (with and without randomisation)
# Also see if halving tokens per update from 240000 to 120000 still produces a working model
###############################################################
source ~/activate_fairseq.sh

cd ~/fairseq

GPU_TYPE=""
#GPU_TYPE=2080

MAX_TOKENS=30000 # max num tokens per device, 30000 is default for transformer TTS
NUM_WORKERS=2 # dataloader workers per gpu
CLIP_NORM=0.01 # clip gradients during training to given value (default value is 5.0)
SAVE_INTERVAL_EPOCHS=10
VAL_INTERVAL_EPOCHS=20
FEATURE_MANIFEST_ROOT=/home/s1785140/data/LJSpeech-1.1/feature_manifest

#DATE=$(date +'%d_%m_%Y')
RUN_ID=run2

#############################
# 240000 tokens per batch
MAX_TOKENS=30000

NUM_GPUS=2
UPDATE_FREQ=4

#NUM_GPUS=3
#UPDATE_FREQ=3

#NUM_GPUS=4
#UPDATE_FREQ=2

MODEL_NAME=${RUN_ID}_SAC_baseline_normal_updatefreq
./sbatch.sh $NUM_GPUS $GPU_TYPE fairseq-train ${FEATURE_MANIFEST_ROOT} \
  --save-dir checkpoints/$MODEL_NAME --tensorboard-logdir tb_logs/$MODEL_NAME \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --num-workers $NUM_WORKERS --max-tokens $MAX_TOKENS --max-update 200000 \
  --save-interval $SAVE_INTERVAL_EPOCHS --validate-interval $VAL_INTERVAL_EPOCHS \
  --task speech_audio_corrector --criterion sac_tts --arch sac_transformer \
  --clip-norm $CLIP_NORM --n-frames-per-step 4 --bce-pos-weight 5.0 \
  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
  --encoder-normalize-before --decoder-normalize-before \
  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --new-logmelspec-dir None \
  --log-interval 1000 \
  --seed 1 --update-freq $UPDATE_FREQ --best-checkpoint-metric loss

#############################
# 120000 tokens per batch
MAX_TOKENS=30000

NUM_GPUS=2
UPDATE_FREQ=2

#NUM_GPUS=3
#UPDATE_FREQ=1

#NUM_GPUS=4
#UPDATE_FREQ=1

MODEL_NAME=${RUN_ID}_SAC_baseline_halved_updatefreq
./sbatch.sh $NUM_GPUS $GPU_TYPE fairseq-train ${FEATURE_MANIFEST_ROOT} \
  --save-dir checkpoints/$MODEL_NAME --tensorboard-logdir tb_logs/$MODEL_NAME \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --num-workers $NUM_WORKERS --max-tokens $MAX_TOKENS --max-update 200000 \
  --save-interval $SAVE_INTERVAL_EPOCHS --validate-interval $VAL_INTERVAL_EPOCHS \
  --task speech_audio_corrector --criterion sac_tts --arch sac_transformer \
  --clip-norm $CLIP_NORM --n-frames-per-step 4 --bce-pos-weight 5.0 \
  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
  --encoder-normalize-before --decoder-normalize-before \
  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --new-logmelspec-dir None \
  --log-interval 1000 \
  --seed 1 --update-freq $UPDATE_FREQ --best-checkpoint-metric loss

MODEL_NAME=${RUN_ID}_SAC_random_halved_updatefreq
./sbatch.sh $NUM_GPUS $GPU_TYPE fairseq-train ${FEATURE_MANIFEST_ROOT} \
  --save-dir checkpoints/$MODEL_NAME --tensorboard-logdir tb_logs/$MODEL_NAME \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --num-workers $NUM_WORKERS --max-tokens $MAX_TOKENS --max-update 200000 \
  --save-interval $SAVE_INTERVAL_EPOCHS --validate-interval $VAL_INTERVAL_EPOCHS \
  --task speech_audio_corrector --criterion sac_tts --arch sac_transformer \
  --clip-norm $CLIP_NORM --n-frames-per-step 4 --bce-pos-weight 5.0 \
  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
  --encoder-normalize-before --decoder-normalize-before \
  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --randomise-examples --randomise-examples-p 1.0 \
  --new-logmelspec-dir None \
  --log-interval 1000 \
  --seed 1 --update-freq $UPDATE_FREQ --best-checkpoint-metric loss

MODEL_NAME=${RUN_ID}_SAC_random_with_ext_speechreps_halved_updatefreq
./sbatch.sh $NUM_GPUS $GPU_TYPE fairseq-train ${FEATURE_MANIFEST_ROOT} \
  --save-dir checkpoints/$MODEL_NAME --tensorboard-logdir tb_logs/$MODEL_NAME \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --num-workers $NUM_WORKERS --max-tokens $MAX_TOKENS --max-update 200000 \
  --save-interval $SAVE_INTERVAL_EPOCHS --validate-interval $VAL_INTERVAL_EPOCHS \
  --task speech_audio_corrector --criterion sac_tts --arch sac_transformer \
  --clip-norm $CLIP_NORM --n-frames-per-step 4 --bce-pos-weight 5.0 \
  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
  --encoder-normalize-before --decoder-normalize-before \
  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --randomise-examples --randomise-examples-p 1.0 \
  --use-ext-word2speechreps-p 0.5 \
  --new-logmelspec-dir None \
  --log-interval 1000 \
  --seed 1 --update-freq $UPDATE_FREQ --best-checkpoint-metric loss

MODEL_NAME=${RUN_ID}_SAC_with_ext_speechreps_halved_updatefreq
./sbatch.sh $NUM_GPUS $GPU_TYPE fairseq-train ${FEATURE_MANIFEST_ROOT} \
  --save-dir checkpoints/$MODEL_NAME --tensorboard-logdir tb_logs/$MODEL_NAME \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --num-workers $NUM_WORKERS --max-tokens $MAX_TOKENS --max-update 200000 \
  --save-interval $SAVE_INTERVAL_EPOCHS --validate-interval $VAL_INTERVAL_EPOCHS \
  --task speech_audio_corrector --criterion sac_tts --arch sac_transformer \
  --clip-norm $CLIP_NORM --n-frames-per-step 4 --bce-pos-weight 5.0 \
  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
  --encoder-normalize-before --decoder-normalize-before \
  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --use-ext-word2speechreps-p 0.5 \
  --new-logmelspec-dir None \
  --log-interval 1000 \
  --seed 1 --update-freq $UPDATE_FREQ --best-checkpoint-metric loss

MODEL_NAME=${RUN_ID}_SAC_random0.5_halved_updatefreq
./sbatch.sh $NUM_GPUS $GPU_TYPE fairseq-train ${FEATURE_MANIFEST_ROOT} \
  --save-dir checkpoints/$MODEL_NAME --tensorboard-logdir tb_logs/$MODEL_NAME \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --num-workers $NUM_WORKERS --max-tokens $MAX_TOKENS --max-update 200000 \
  --save-interval $SAVE_INTERVAL_EPOCHS --validate-interval $VAL_INTERVAL_EPOCHS \
  --task speech_audio_corrector --criterion sac_tts --arch sac_transformer \
  --clip-norm $CLIP_NORM --n-frames-per-step 4 --bce-pos-weight 5.0 \
  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
  --encoder-normalize-before --decoder-normalize-before \
  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --randomise-examples --randomise-examples-p 0.5 \
  --new-logmelspec-dir None \
  --log-interval 1000 \
  --seed 1 --update-freq $UPDATE_FREQ --best-checkpoint-metric loss

MODEL_NAME=${RUN_ID}_SAC_random0.5_with_ext_speechreps_halved_updatefreq
./sbatch.sh $NUM_GPUS $GPU_TYPE fairseq-train ${FEATURE_MANIFEST_ROOT} \
  --save-dir checkpoints/$MODEL_NAME --tensorboard-logdir tb_logs/$MODEL_NAME \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --num-workers $NUM_WORKERS --max-tokens $MAX_TOKENS --max-update 200000 \
  --save-interval $SAVE_INTERVAL_EPOCHS --validate-interval $VAL_INTERVAL_EPOCHS \
  --task speech_audio_corrector --criterion sac_tts --arch sac_transformer \
  --clip-norm $CLIP_NORM --n-frames-per-step 4 --bce-pos-weight 5.0 \
  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
  --encoder-normalize-before --decoder-normalize-before \
  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --randomise-examples --randomise-examples-p 0.5 --use-ext-word2speechreps-p 0.5 \
  --new-logmelspec-dir None \
  --log-interval 1000 \
  --seed 1 --update-freq $UPDATE_FREQ --best-checkpoint-metric loss

MAX_TOKENS=25000
MODEL_NAME=${RUN_ID}_SAC_one_mask_token_per_grapheme_maxtokens${MAX_TOKENS}_halved_updatefreq
./sbatch.sh $NUM_GPUS $GPU_TYPE fairseq-train ${FEATURE_MANIFEST_ROOT} \
  --save-dir checkpoints/$MODEL_NAME --tensorboard-logdir tb_logs/$MODEL_NAME \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --num-workers $NUM_WORKERS --max-tokens $MAX_TOKENS --max-update 200000 \
  --save-interval $SAVE_INTERVAL_EPOCHS --validate-interval $VAL_INTERVAL_EPOCHS \
  --task speech_audio_corrector --criterion sac_tts --arch sac_transformer \
  --clip-norm $CLIP_NORM --n-frames-per-step 4 --bce-pos-weight 5.0 \
  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
  --encoder-normalize-before --decoder-normalize-before \
  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --new-logmelspec-dir None \
  --one-mask-tok-per-grapheme \
  --log-interval 1000 \
  --seed 1 --update-freq $UPDATE_FREQ --best-checkpoint-metric loss

