##################################################################################################
# Get suitable GPU node
NUM_GPUS=4
CPUS_PER_TASK=2
MEM=32000
EXCLUDE=arnold
#EXCLUDE=duflo,arnold
srun --part=ILCC_GPU,CDT_GPU --gres=gpu:$NUM_GPUS --cpus-per-task=$CPUS_PER_TASK --mem=$MEM --exclude=$EXCLUDE --pty bash
#srun --part=ILCC_GPU,CDT_GPU --gres=gpu:gtx2080ti:$NUM_GPUS --cpus-per-task=$CPUS_PER_TASK --mem=$MEM --exclude=$EXCLUDE --pty bash

##################################################################################################
# Set training params common to all jobs
cd ~
source activate_fairseq.sh
NUM_GPUS=4
NUM_WORKERS=2
UPDATE_FREQ=3
MAX_TOKENS=20000 # 30000 is default for transformer TTS
FEATURE_MANIFEST_ROOT=/home/s1785140/data/LJSpeech-1.1/feature_manifest

##################################################################################################
# Run training for different experiments

#mask speech timesteps from decoder
MODEL_NAME=test_sac_normal_masking2
fairseq-train ${FEATURE_MANIFEST_ROOT} \
  --save-dir checkpoints/$MODEL_NAME --tensorboard-logdir tb_logs/$MODEL_NAME \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --num-workers $NUM_WORKERS --max-tokens $MAX_TOKENS --max-update 200000 \
  --task speech_audio_corrector --criterion sac_tts --arch sac_transformer \
  --clip-norm 5.0 --n-frames-per-step 4 --bce-pos-weight 5.0 \
  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
  --encoder-normalize-before --decoder-normalize-before \
  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --mask-speech-timesteps \
  --seed 1 --update-freq $UPDATE_FREQ --best-checkpoint-metric loss

#no speech masking from decoder
MODEL_NAME=test_sac_normal_masking_maxtokens${MAX_TOKENS}_updatefreq${UPDATE_FREQ}_gpus${NUM_GPUS}_nospeechmasking
fairseq-train ${FEATURE_MANIFEST_ROOT} \
  --save-dir checkpoints/$MODEL_NAME --tensorboard-logdir tb_logs/$MODEL_NAME \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --num-workers $NUM_WORKERS --max-tokens $MAX_TOKENS --max-update 200000 \
  --task speech_audio_corrector --criterion sac_tts --arch sac_transformer \
  --clip-norm 5.0 --n-frames-per-step 4 --bce-pos-weight 5.0 \
  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
  --encoder-normalize-before --decoder-normalize-before \
  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --seed 1 --update-freq $UPDATE_FREQ --best-checkpoint-metric loss

# randomise word-level speech reps
MODEL_NAME=test_randomise_examples
fairseq-train ${FEATURE_MANIFEST_ROOT} \
  --save-dir checkpoints/$MODEL_NAME --tensorboard-logdir tb_logs/$MODEL_NAME \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --num-workers $NUM_WORKERS --max-tokens $MAX_TOKENS --max-update 200000 \
  --task speech_audio_corrector --criterion sac_tts --arch sac_transformer \
  --clip-norm 5.0 --n-frames-per-step 4 --bce-pos-weight 5.0 \
  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
  --encoder-normalize-before --decoder-normalize-before \
  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --seed 1 --update-freq $UPDATE_FREQ --best-checkpoint-metric loss \
  --randomise-examples

# no word-level alignment information
MODEL_NAME=test_no_word_pos_embeddings
fairseq-train ${FEATURE_MANIFEST_ROOT} \
  --save-dir checkpoints/$MODEL_NAME --tensorboard-logdir tb_logs/$MODEL_NAME \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --num-workers $NUM_WORKERS --max-tokens $MAX_TOKENS --update-freq $UPDATE_FREQ --max-update 200000 \
  --task speech_audio_corrector --criterion sac_tts --arch sac_transformer \
  --clip-norm 5.0 --n-frames-per-step 4 --bce-pos-weight 5.0 \
  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
  --encoder-normalize-before --decoder-normalize-before \
  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --seed 1 --best-checkpoint-metric loss \
  --no-word-pos

# INFERENCE - no word-level alignment information
# TEST SET OOVS (words in test set but not in train)
# generate entire test set (using random masking just like training)
cd ~/fairseq

MODEL=test_no_word_pos_embeddings
VOCODER=wav_22050hz_hifigan
SAVE_DIR=checkpoints/$MODEL
SPLIT=test
FEATURE_MANIFEST_ROOT=/home/s1785140/data/LJSpeech-1.1/feature_manifest
CHECKPOINT_NAME=1019
CHECKPOINT_PATH=${SAVE_DIR}/checkpoint${CHECKPOINT_NAME}.pt
OUT_DIR=inference/$MODEL/$CHECKPOINT_NAME

SPLIT=test
python -m examples.speech_audio_corrector.generate_waveform_sac ${FEATURE_MANIFEST_ROOT} \
  --config-yaml config.yaml --gen-subset ${SPLIT} --task speech_audio_corrector \
  --path ${CHECKPOINT_PATH} --max-tokens 50000 --spec-bwd-max-iter 32 \
  --results-path $OUT_DIR \
  --vocoder hifigan \
  --dump-waveforms
mkdir $OUT_DIR/$VOCODER/LJ_TEST_SET
mv $OUT_DIR/$VOCODER/*.wav $OUT_DIR/$VOCODER/LJ_TEST_SET

SPLIT=test
python -m examples.speech_audio_corrector.generate_waveform_sac ${FEATURE_MANIFEST_ROOT} \
  --config-yaml config.yaml --gen-subset ${SPLIT} --task speech_audio_corrector \
  --path ${CHECKPOINT_PATH} --max-tokens 50000 --spec-bwd-max-iter 32 \
  --results-path $OUT_DIR \
  --vocoder hifigan \
  --dump-waveforms \
  --batch-size 32 \
  --speechreps-add-mask-tokens \
  --txt-file examples/speech_audio_corrector/test_utts_test_set_oovs.txt \
  --add-count-to-filename
mkdir $OUT_DIR/$VOCODER/test_oovs
mv $OUT_DIR/$VOCODER/*.wav $OUT_DIR/$VOCODER/test_oovs


# 19 Jan 2022
# Examine whether refactoring changes to dataset loader broke anything
NUM_GPUS=4
CPUS_PER_TASK=2
MEM=32000
EXCLUDE=
#EXCLUDE=duflo,arnold
srun --part=ILCC_GPU,CDT_GPU --gres=gpu:$NUM_GPUS --cpus-per-task=$CPUS_PER_TASK --mem=$MEM --exclude=$EXCLUDE --pty bash

./interactive_node_1gpu.sh
source ~/activate_fairseq.sh

cd ~/fairseq
UPDATE_FREQ=1
NUM_WORKERS=1
MAX_TOKENS=30000 #30000 # 30000 is default for transformer TTS
CLIP_NORM=0.01 # clip gradients during training to given value (default value is 5.0)
SAVE_INTERVAL_EPOCHS=1
VAL_INTERVAL_EPOCHS=1
FEATURE_MANIFEST_ROOT=/home/s1785140/data/LJSpeech-1.1/feature_manifest

scratch_disk=$(bash examples/speech_audio_corrector/bash_scripts/check_scratchdisks.sh)
echo "scratch_disk is $scratch_disk"

MODEL_NAME=test_SAC_inference_during_training
fairseq-train ${FEATURE_MANIFEST_ROOT} \
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
  --use-ext-word2speechreps-p 0.5 \
  --new-logmelspec-dir ${scratch_disk} \
  --eval-inference \
  --seed 1 --update-freq $UPDATE_FREQ --best-checkpoint-metric loss
