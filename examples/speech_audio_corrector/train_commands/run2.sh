
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
MAX_UPDATES=1000000 # determines end of training
FEATURE_MANIFEST_ROOT=/home/s1785140/data/LJSpeech-1.1/feature_manifest

RUN_ID=run2

##############################
## 240000 tokens per batch
#MAX_TOKENS=30000
#
#NUM_GPUS=2
#UPDATE_FREQ=4
#
##NUM_GPUS=3
##UPDATE_FREQ=3
#
##NUM_GPUS=4
##UPDATE_FREQ=2
#
#MODEL_NAME=${RUN_ID}_SAC_baseline_normal_updatefreq
#./sbatch.sh $NUM_GPUS $GPU_TYPE fairseq-train ${FEATURE_MANIFEST_ROOT} \
#  --save-dir checkpoints/${MODEL_NAME} --tensorboard-logdir tb_logs/${MODEL_NAME} \
#  --config-yaml config.yaml --train-subset train --valid-subset dev \
#  --num-workers ${NUM_WORKERS} --max-tokens ${MAX_TOKENS} --max-update ${MAX_UPDATES} \
#  --save-interval ${SAVE_INTERVAL_EPOCHS} --validate-interval ${VAL_INTERVAL_EPOCHS} \
#  --task speech_audio_corrector --criterion sac_tts --arch sac_transformer \
#  --clip-norm ${CLIP_NORM} --n-frames-per-step 4 --bce-pos-weight 5.0 \
#  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
#  --encoder-normalize-before --decoder-normalize-before \
#  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#  --new-logmelspec-dir None \
#  --log-interval 1000 \
#  --seed 1 --update-freq ${UPDATE_FREQ} --best-checkpoint-metric loss
#
##############################
## 100000 tokens per batch
#MAX_TOKENS=25000
#
#NUM_GPUS=2
#UPDATE_FREQ=2
#
#MODEL_NAME=${RUN_ID}_SAC_one_mask_token_per_grapheme_maxtokens${MAX_TOKENS}_halved_updatefreq
#./sbatch.sh $NUM_GPUS $GPU_TYPE fairseq-train ${FEATURE_MANIFEST_ROOT} \
#  --save-dir checkpoints/${MODEL_NAME} --tensorboard-logdir tb_logs/${MODEL_NAME} \
#  --config-yaml config.yaml --train-subset train --valid-subset dev \
#  --num-workers ${NUM_WORKERS} --max-tokens ${MAX_TOKENS} --max-update ${MAX_UPDATES} \
#  --save-interval ${SAVE_INTERVAL_EPOCHS} --validate-interval ${VAL_INTERVAL_EPOCHS} \
#  --task speech_audio_corrector --criterion sac_tts --arch sac_transformer \
#  --clip-norm ${CLIP_NORM} --n-frames-per-step 4 --bce-pos-weight 5.0 \
#  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
#  --encoder-normalize-before --decoder-normalize-before \
#  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#  --new-logmelspec-dir None \
#  --mask-tok-per-word "many" \
#  --log-interval 1000 \
#  --seed 1 --update-freq ${UPDATE_FREQ} --best-checkpoint-metric loss
#
#




#############################
# 120000 tokens per batch
MAX_TOKENS=30000

NUM_GPUS=2
UPDATE_FREQ=2

#MODEL_NAME=${RUN_ID}_SAC_baseline_halved_updatefreq
#./sbatch.sh $NUM_GPUS $GPU_TYPE fairseq-train ${FEATURE_MANIFEST_ROOT} \
#  --save-dir checkpoints/${MODEL_NAME} --tensorboard-logdir tb_logs/${MODEL_NAME} \
#  --config-yaml config.yaml --train-subset train --valid-subset dev \
#  --num-workers ${NUM_WORKERS} --max-tokens ${MAX_TOKENS} --max-update ${MAX_UPDATES} \
#  --save-interval ${SAVE_INTERVAL_EPOCHS} --validate-interval ${VAL_INTERVAL_EPOCHS} \
#  --task speech_audio_corrector --criterion sac_tts --arch sac_transformer \
#  --clip-norm ${CLIP_NORM} --n-frames-per-step 4 --bce-pos-weight 5.0 \
#  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
#  --encoder-normalize-before --decoder-normalize-before \
#  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#  --new-logmelspec-dir None \
#  --log-interval 1000 \
#  --seed 1 --update-freq ${UPDATE_FREQ} --best-checkpoint-metric loss
#
#MODEL_NAME=${RUN_ID}_SAC_random_halved_updatefreq
#./sbatch.sh $NUM_GPUS $GPU_TYPE fairseq-train ${FEATURE_MANIFEST_ROOT} \
#  --save-dir checkpoints/${MODEL_NAME} --tensorboard-logdir tb_logs/${MODEL_NAME} \
#  --config-yaml config.yaml --train-subset train --valid-subset dev \
#  --num-workers ${NUM_WORKERS} --max-tokens ${MAX_TOKENS} --max-update ${MAX_UPDATES} \
#  --save-interval ${SAVE_INTERVAL_EPOCHS} --validate-interval ${VAL_INTERVAL_EPOCHS} \
#  --task speech_audio_corrector --criterion sac_tts --arch sac_transformer \
#  --clip-norm ${CLIP_NORM} --n-frames-per-step 4 --bce-pos-weight 5.0 \
#  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
#  --encoder-normalize-before --decoder-normalize-before \
#  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#  --randomise-examples-p 1.0 \
#  --new-logmelspec-dir None \
#  --log-interval 1000 \
#  --seed 1 --update-freq ${UPDATE_FREQ} --best-checkpoint-metric loss
#
#MODEL_NAME=${RUN_ID}_SAC_random_with_ext_speechreps_halved_updatefreq
#./sbatch.sh $NUM_GPUS $GPU_TYPE fairseq-train ${FEATURE_MANIFEST_ROOT} \
#  --save-dir checkpoints/${MODEL_NAME} --tensorboard-logdir tb_logs/${MODEL_NAME} \
#  --config-yaml config.yaml --train-subset train --valid-subset dev \
#  --num-workers ${NUM_WORKERS} --max-tokens ${MAX_TOKENS} --max-update ${MAX_UPDATES} \
#  --save-interval ${SAVE_INTERVAL_EPOCHS} --validate-interval ${VAL_INTERVAL_EPOCHS} \
#  --task speech_audio_corrector --criterion sac_tts --arch sac_transformer \
#  --clip-norm ${CLIP_NORM} --n-frames-per-step 4 --bce-pos-weight 5.0 \
#  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
#  --encoder-normalize-before --decoder-normalize-before \
#  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#  --randomise-examples-p 1.0 \
#  --use-ext-word2speechreps-p 0.5 \
#  --new-logmelspec-dir None \
#  --log-interval 1000 \
#  --seed 1 --update-freq ${UPDATE_FREQ} --best-checkpoint-metric loss
#
#MODEL_NAME=${RUN_ID}_SAC_with_ext_speechreps_halved_updatefreq
#./sbatch.sh $NUM_GPUS $GPU_TYPE fairseq-train ${FEATURE_MANIFEST_ROOT} \
#  --save-dir checkpoints/${MODEL_NAME} --tensorboard-logdir tb_logs/${MODEL_NAME} \
#  --config-yaml config.yaml --train-subset train --valid-subset dev \
#  --num-workers ${NUM_WORKERS} --max-tokens ${MAX_TOKENS} --max-update ${MAX_UPDATES} \
#  --save-interval ${SAVE_INTERVAL_EPOCHS} --validate-interval ${VAL_INTERVAL_EPOCHS} \
#  --task speech_audio_corrector --criterion sac_tts --arch sac_transformer \
#  --clip-norm ${CLIP_NORM} --n-frames-per-step 4 --bce-pos-weight 5.0 \
#  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
#  --encoder-normalize-before --decoder-normalize-before \
#  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#  --use-ext-word2speechreps-p 0.5 \
#  --new-logmelspec-dir None \
#  --log-interval 1000 \
#  --seed 1 --update-freq ${UPDATE_FREQ} --best-checkpoint-metric loss
#
#MODEL_NAME=${RUN_ID}_SAC_random0.5_halved_updatefreq
#./sbatch.sh $NUM_GPUS $GPU_TYPE fairseq-train ${FEATURE_MANIFEST_ROOT} \
#  --save-dir checkpoints/${MODEL_NAME} --tensorboard-logdir tb_logs/${MODEL_NAME} \
#  --config-yaml config.yaml --train-subset train --valid-subset dev \
#  --num-workers ${NUM_WORKERS} --max-tokens ${MAX_TOKENS} --max-update ${MAX_UPDATES} \
#  --save-interval ${SAVE_INTERVAL_EPOCHS} --validate-interval ${VAL_INTERVAL_EPOCHS} \
#  --task speech_audio_corrector --criterion sac_tts --arch sac_transformer \
#  --clip-norm ${CLIP_NORM} --n-frames-per-step 4 --bce-pos-weight 5.0 \
#  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
#  --encoder-normalize-before --decoder-normalize-before \
#  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#  --randomise-examples-p 0.5 \
#  --new-logmelspec-dir None \
#  --log-interval 1000 \
#  --seed 1 --update-freq ${UPDATE_FREQ} --best-checkpoint-metric loss
#
#MODEL_NAME=${RUN_ID}_SAC_random0.5_with_ext_speechreps_halved_updatefreq
#./sbatch.sh $NUM_GPUS $GPU_TYPE fairseq-train ${FEATURE_MANIFEST_ROOT} \
#  --save-dir checkpoints/${MODEL_NAME} --tensorboard-logdir tb_logs/${MODEL_NAME} \
#  --config-yaml config.yaml --train-subset train --valid-subset dev \
#  --num-workers ${NUM_WORKERS} --max-tokens ${MAX_TOKENS} --max-update ${MAX_UPDATES} \
#  --save-interval ${SAVE_INTERVAL_EPOCHS} --validate-interval ${VAL_INTERVAL_EPOCHS} \
#  --task speech_audio_corrector --criterion sac_tts --arch sac_transformer \
#  --clip-norm ${CLIP_NORM} --n-frames-per-step 4 --bce-pos-weight 5.0 \
#  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
#  --encoder-normalize-before --decoder-normalize-before \
#  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#  --randomise-examples-p 0.5 --use-ext-word2speechreps-p 0.5 \
#  --new-logmelspec-dir None \
#  --log-interval 1000 \
#  --seed 1 --update-freq ${UPDATE_FREQ} --best-checkpoint-metric loss

## no mask tokens per word
#MODEL_NAME=${RUN_ID}_SAC_zero_mask_toks_per_word_halved_updatefreq
#./sbatch.sh $NUM_GPUS $GPU_TYPE fairseq-train ${FEATURE_MANIFEST_ROOT} \
#  --save-dir checkpoints/${MODEL_NAME} --tensorboard-logdir tb_logs/${MODEL_NAME} \
#  --config-yaml config.yaml --train-subset train --valid-subset dev \
#  --num-workers ${NUM_WORKERS} --max-tokens ${MAX_TOKENS} --max-update ${MAX_UPDATES} \
#  --save-interval ${SAVE_INTERVAL_EPOCHS} --validate-interval ${VAL_INTERVAL_EPOCHS} \
#  --task speech_audio_corrector --criterion sac_tts --arch sac_transformer \
#  --clip-norm ${CLIP_NORM} --n-frames-per-step 4 --bce-pos-weight 5.0 \
#  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
#  --encoder-normalize-before --decoder-normalize-before \
#  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#  --new-logmelspec-dir None \
#  --log-interval 1000 \
#  --mask-tok-per-word "zero" \
#  --seed 1 --update-freq ${UPDATE_FREQ} --best-checkpoint-metric loss

## try replacing middle <eos> with <sos> token
#MODEL_NAME=${RUN_ID}_SAC_replace_middle_eos_with_sos_halved_updatefreq
#./sbatch.sh $NUM_GPUS $GPU_TYPE fairseq-train ${FEATURE_MANIFEST_ROOT} \
#  --save-dir checkpoints/${MODEL_NAME} --tensorboard-logdir tb_logs/${MODEL_NAME} \
#  --config-yaml config.yaml --train-subset train --valid-subset dev \
#  --num-workers ${NUM_WORKERS} --max-tokens ${MAX_TOKENS} --max-update ${MAX_UPDATES} \
#  --save-interval ${SAVE_INTERVAL_EPOCHS} --validate-interval ${VAL_INTERVAL_EPOCHS} \
#  --task speech_audio_corrector --criterion sac_tts --arch sac_transformer \
#  --clip-norm ${CLIP_NORM} --n-frames-per-step 4 --bce-pos-weight 5.0 \
#  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
#  --encoder-normalize-before --decoder-normalize-before \
#  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#  --new-logmelspec-dir None \
#  --log-interval 1000 \
#  --use-sos-symbol-in-middle \
#  --seed 1 --update-freq ${UPDATE_FREQ} --best-checkpoint-metric loss
#
## different learning rates sweep
#MODEL_NAME=${RUN_ID}_SAC_baseline_lr0.001_halved_updatefreq
#./sbatch.sh $NUM_GPUS $GPU_TYPE fairseq-train ${FEATURE_MANIFEST_ROOT} \
#  --save-dir checkpoints/${MODEL_NAME} --tensorboard-logdir tb_logs/${MODEL_NAME} \
#  --config-yaml config.yaml --train-subset train --valid-subset dev \
#  --num-workers ${NUM_WORKERS} --max-tokens ${MAX_TOKENS} --max-update ${MAX_UPDATES} \
#  --save-interval ${SAVE_INTERVAL_EPOCHS} --validate-interval ${VAL_INTERVAL_EPOCHS} \
#  --task speech_audio_corrector --criterion sac_tts --arch sac_transformer \
#  --clip-norm ${CLIP_NORM} --n-frames-per-step 4 --bce-pos-weight 5.0 \
#  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
#  --encoder-normalize-before --decoder-normalize-before \
#  --optimizer adam --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#  --new-logmelspec-dir None \
#  --log-interval 1000 \
#  --lr 0.001 \
#  --seed 1 --update-freq ${UPDATE_FREQ} --best-checkpoint-metric loss
#
#MODEL_NAME=${RUN_ID}_SAC_baseline_lr0.00001_halved_updatefreq
#./sbatch.sh $NUM_GPUS $GPU_TYPE fairseq-train ${FEATURE_MANIFEST_ROOT} \
#  --save-dir checkpoints/${MODEL_NAME} --tensorboard-logdir tb_logs/${MODEL_NAME} \
#  --config-yaml config.yaml --train-subset train --valid-subset dev \
#  --num-workers ${NUM_WORKERS} --max-tokens ${MAX_TOKENS} --max-update ${MAX_UPDATES} \
#  --save-interval ${SAVE_INTERVAL_EPOCHS} --validate-interval ${VAL_INTERVAL_EPOCHS} \
#  --task speech_audio_corrector --criterion sac_tts --arch sac_transformer \
#  --clip-norm ${CLIP_NORM} --n-frames-per-step 4 --bce-pos-weight 5.0 \
#  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
#  --encoder-normalize-before --decoder-normalize-before \
#  --optimizer adam --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#  --new-logmelspec-dir None \
#  --log-interval 1000 \
#  --lr 0.00001 \
#  --seed 1 --update-freq ${UPDATE_FREQ} --best-checkpoint-metric loss
#
#

#MAX_TOKENS=20000
#NUM_GPUS=2
#UPDATE_FREQ=2
#
## do not remove dup codes
#MODEL_NAME=${RUN_ID}_SAC_dont_remove_dup_codes_maxtoks${MAX_TOKENS}_halved_updatefreq
#./sbatch.sh $NUM_GPUS $GPU_TYPE fairseq-train ${FEATURE_MANIFEST_ROOT} \
#  --save-dir checkpoints/${MODEL_NAME} --tensorboard-logdir tb_logs/${MODEL_NAME} \
#  --config-yaml config.yaml --train-subset train --valid-subset dev \
#  --num-workers ${NUM_WORKERS} --max-tokens ${MAX_TOKENS} --max-update ${MAX_UPDATES} \
#  --save-interval ${SAVE_INTERVAL_EPOCHS} --validate-interval ${VAL_INTERVAL_EPOCHS} \
#  --task speech_audio_corrector --criterion sac_tts --arch sac_transformer \
#  --clip-norm ${CLIP_NORM} --n-frames-per-step 4 --bce-pos-weight 5.0 \
#  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
#  --encoder-normalize-before --decoder-normalize-before \
#  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#  --new-logmelspec-dir None \
#  --log-interval 1000 \
#  --remove-dup-codes-p 0.0 \
#  --seed 1 --update-freq ${UPDATE_FREQ} --best-checkpoint-metric loss

MAX_TOKENS=17500
NUM_GPUS=4
UPDATE_FREQ=2

# do not remove dup codes
MODEL_NAME=${RUN_ID}_SAC_dont_remove_dup_codes_maxtoks20000_4gpus_updfreq2
./sbatch.sh $NUM_GPUS $GPU_TYPE fairseq-train ${FEATURE_MANIFEST_ROOT} \
  --save-dir checkpoints/${MODEL_NAME} --tensorboard-logdir tb_logs/${MODEL_NAME} \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --num-workers ${NUM_WORKERS} --max-tokens ${MAX_TOKENS} --max-tokens-valid ${MAX_TOKENS} \
  --max-update ${MAX_UPDATES} \
  --save-interval ${SAVE_INTERVAL_EPOCHS} --validate-interval ${VAL_INTERVAL_EPOCHS} \
  --task speech_audio_corrector --criterion sac_tts --arch sac_transformer \
  --clip-norm ${CLIP_NORM} --n-frames-per-step 4 --bce-pos-weight 5.0 \
  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
  --encoder-normalize-before --decoder-normalize-before \
  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --new-logmelspec-dir None \
  --log-interval 1000 \
  --remove-dup-codes-p 0.0 \
  --seed 1 --update-freq ${UPDATE_FREQ} --best-checkpoint-metric loss
