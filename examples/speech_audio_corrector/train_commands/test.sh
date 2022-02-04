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

UPDATE_FREQ=2
RUN_ID=test

MODEL_NAME=${RUN_ID}_debug_SAC
fairseq-train ${FEATURE_MANIFEST_ROOT} \
  --save-dir checkpoints/${MODEL_NAME} --tensorboard-logdir tb_logs/${MODEL_NAME} \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --num-workers ${NUM_WORKERS} --max-tokens ${MAX_TOKENS} --max-update ${MAX_UPDATES} \
  --save-interval ${SAVE_INTERVAL_EPOCHS} --validate-interval ${VAL_INTERVAL_EPOCHS} \
  --task speech_audio_corrector --criterion sac_tts --arch sac_transformer \
  --clip-norm ${CLIP_NORM} --n-frames-per-step 4 --bce-pos-weight 5.0 \
  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
  --encoder-normalize-before --decoder-normalize-before \
  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --log-interval 1000 \
  --seed 1 --update-freq ${UPDATE_FREQ} --best-checkpoint-metric loss
