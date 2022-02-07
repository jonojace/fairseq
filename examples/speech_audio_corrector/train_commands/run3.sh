
###############################################################
# SAC 4th feb 2022
# Find best settings for baseline model
# defaults are:
# do not remove dups
# zero mask tokens per word
# sos token in between graphemes and speech codes instead of eos token
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

RUN_ID=run3

# 80000 tokens per batch 20*2*2
MAX_TOKENS=20000
NUM_GPUS=2
UPDATE_FREQ=2

#MODEL_NAME=${RUN_ID}_SAC_baseline
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
## no mask tokens per word
#MODEL_NAME=${RUN_ID}_SAC_one_mask_tok_per_word
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
#  --mask-tok-per-word "one" \
#  --seed 1 --update-freq ${UPDATE_FREQ} --best-checkpoint-metric loss
#
#MODEL_NAME=${RUN_ID}_SAC_one_mask_tok_per_grapheme
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
#  --mask-tok-per-word "many" \
#  --seed 1 --update-freq ${UPDATE_FREQ} --best-checkpoint-metric loss
#
## try replacing middle <sos> with <eos> token
#MODEL_NAME=${RUN_ID}_SAC_eos_symbol_in_middle
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
#  --symbol-in-middle "eos" \
#  --seed 1 --update-freq ${UPDATE_FREQ} --best-checkpoint-metric loss
#
## remove dup codes
#MODEL_NAME=${RUN_ID}_SAC_remove_dup_codes
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
#  --remove-dup-codes-p 1.0 \
#  --seed 1 --update-freq ${UPDATE_FREQ} --best-checkpoint-metric loss
#
## try mixing in VCTK speech codes
#
#EXT_PROB=0.05
#
#MODEL_NAME=${RUN_ID}_SAC_ext_speechcodes_p_${EXT_PROB}
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
#  --use-ext-word2speechreps-p ${EXT_PROB} \
#  --seed 1 --update-freq ${UPDATE_FREQ} --best-checkpoint-metric loss
#
EXT_PROB=0.1

MODEL_NAME=${RUN_ID}_SAC_ext_speechcodes_p_${EXT_PROB}
./sbatch.sh $NUM_GPUS $GPU_TYPE fairseq-train ${FEATURE_MANIFEST_ROOT} \
  --save-dir checkpoints/${MODEL_NAME} --tensorboard-logdir tb_logs/${MODEL_NAME} \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --num-workers ${NUM_WORKERS} --max-tokens ${MAX_TOKENS} --max-update ${MAX_UPDATES} \
  --save-interval ${SAVE_INTERVAL_EPOCHS} --validate-interval ${VAL_INTERVAL_EPOCHS} \
  --task speech_audio_corrector --criterion sac_tts --arch sac_transformer \
  --clip-norm ${CLIP_NORM} --n-frames-per-step 4 --bce-pos-weight 5.0 \
  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
  --encoder-normalize-before --decoder-normalize-before \
  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --new-logmelspec-dir None \
  --log-interval 1000 \
  --use-ext-word2speechreps-p ${EXT_PROB} \
  --seed 1 --update-freq ${UPDATE_FREQ} --best-checkpoint-metric loss
#
#EXT_PROB=0.2
#
#MODEL_NAME=${RUN_ID}_SAC_ext_speechcodes_p_${EXT_PROB}
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
#  --use-ext-word2speechreps-p ${EXT_PROB} \
#  --seed 1 --update-freq ${UPDATE_FREQ} --best-checkpoint-metric loss
#
#EXT_PROB=0.3
#
#MODEL_NAME=${RUN_ID}_SAC_ext_speechcodes_p_${EXT_PROB}
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
#  --use-ext-word2speechreps-p ${EXT_PROB} \
#  --seed 1 --update-freq ${UPDATE_FREQ} --best-checkpoint-metric loss
#
#EXT_PROB=0.4
#
#MODEL_NAME=${RUN_ID}_SAC_ext_speechcodes_p_${EXT_PROB}
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
#  --use-ext-word2speechreps-p ${EXT_PROB} \
#  --seed 1 --update-freq ${UPDATE_FREQ} --best-checkpoint-metric loss
#
#EXT_PROB=0.5
#
#MODEL_NAME=${RUN_ID}_SAC_ext_speechcodes_p_${EXT_PROB}
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
#  --use-ext-word2speechreps-p ${EXT_PROB} \
#  --seed 1 --update-freq ${UPDATE_FREQ} --best-checkpoint-metric loss
