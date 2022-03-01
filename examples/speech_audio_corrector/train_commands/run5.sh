###############################################################
# 25th feb
# restarting jobs from run4 since installing apex broke previous checkpoints
###############################################################

###############################################################
# Setup Environment
#source ~/activate_fairseq.sh
#cd ~/fairseq

###############################################################
# Setup training hparams common to ALL jobs
NUM_WORKERS=2 # dataloader workers per gpu
CLIP_NORM=0.01 # clip gradients during training to given value (default value is 5.0)
SAVE_INTERVAL_EPOCHS=10
VAL_INTERVAL_EPOCHS=20
MAX_UPDATES=1000000 # determines end of training
FEATURE_MANIFEST_ROOT=/home/s1785140/data/LJSpeech-1.1/feature_manifest

###############################################################
# Setup Training batch size and gpus
GPU_TYPE=""
#GPU_TYPE=2080

# 80000 tokens per batch 20*2*2
NUM_GPUS=2
UPDATE_FREQ=2
MAX_TOKENS=20000

###############################################################
# Setup name of this run of jobs
RUN_ID=run5

# Train Vanilla TTS model
SLURMJOBNAME=vanilla_tts_transformer
MODEL_NAME=${RUN_ID}_${SLURMJOBNAME}
./sbatch.sh $SLURMJOBNAME $NUM_GPUS $GPU_TYPE fairseq-train ${FEATURE_MANIFEST_ROOT} \
  --save-dir checkpoints/${MODEL_NAME} --tensorboard-logdir tb_logs/${MODEL_NAME} \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --num-workers ${NUM_WORKERS} --max-tokens ${MAX_TOKENS} --max-update ${MAX_UPDATES} \
  --save-interval ${SAVE_INTERVAL_EPOCHS} --validate-interval ${VAL_INTERVAL_EPOCHS} \
  --task text_to_speech --criterion tacotron2 --arch tts_transformer \
  --clip-norm ${CLIP_NORM} --n-frames-per-step 4 --bce-pos-weight 5.0 \
  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
  --encoder-normalize-before --decoder-normalize-before \
  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --log-interval 1000 \
  --seed 1 --update-freq ${UPDATE_FREQ} --best-checkpoint-metric loss

SLURMJOBNAME=SAC_baseline
MODEL_NAME=${RUN_ID}_${SLURMJOBNAME}
./sbatch.sh $SLURMJOBNAME $NUM_GPUS $GPU_TYPE fairseq-train ${FEATURE_MANIFEST_ROOT} \
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
  --seed 1 --update-freq ${UPDATE_FREQ} --best-checkpoint-metric loss

# one mask token per word and do not mask encoder out speech timesteps
SLURMJOBNAME=one_mask_tok_per_word_AND_dont_mask_encoder_out_speech_timesteps
MODEL_NAME=${RUN_ID}_${SLURMJOBNAME}
./sbatch.sh $SLURMJOBNAME $NUM_GPUS $GPU_TYPE fairseq-train ${FEATURE_MANIFEST_ROOT} \
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
  --mask-tok-per-word "one" \
  --dont-mask-encoder-out-speech-timesteps \
  --seed 1 --update-freq ${UPDATE_FREQ} --best-checkpoint-metric loss

# km 50 100 200 to quantise LJspeech

SLURMJOBNAME=km50_quantized_ljspeech
MODEL_NAME=${RUN_ID}_${SLURMJOBNAME}
./sbatch.sh $SLURMJOBNAME $NUM_GPUS $GPU_TYPE fairseq-train ${FEATURE_MANIFEST_ROOT} \
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
  --quantized-speechreps-file "/home/s1785140/fairseq/examples/speech_audio_corrector/ljspeech_quantized_km50.txt" \
  --seed 1 --update-freq ${UPDATE_FREQ} --best-checkpoint-metric loss

SLURMJOBNAME=km200_quantized_ljspeech
MODEL_NAME=${RUN_ID}_${SLURMJOBNAME}
./sbatch.sh $SLURMJOBNAME $NUM_GPUS $GPU_TYPE fairseq-train ${FEATURE_MANIFEST_ROOT} \
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
  --quantized-speechreps-file "/home/s1785140/fairseq/examples/speech_audio_corrector/ljspeech_quantized_km200.txt" \
  --seed 1 --update-freq ${UPDATE_FREQ} --best-checkpoint-metric loss

#SLURMJOBNAME=baseline_using_new_km100_quantized_ljspeech
#MODEL_NAME=${RUN_ID}_${SLURMJOBNAME}
#./sbatch.sh $SLURMJOBNAME $NUM_GPUS $GPU_TYPE fairseq-train ${FEATURE_MANIFEST_ROOT} \
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
#  --quantized-speechreps-file "/home/s1785140/fairseq/examples/speech_audio_corrector/ljspeech_quantized_km100.txt" \
#  --seed 1 --update-freq ${UPDATE_FREQ} --best-checkpoint-metric loss
