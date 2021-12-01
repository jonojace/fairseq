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
# Set training params
cd ~
source activate_fairseq.sh
NUM_GPUS=4
NUM_WORKERS=2

UPDATE_FREQ=3
MAX_TOKENS=20000 # 30000 is default for transformer TTS
MAX_SENTENCES=NULL # TODO use this to help control mem usage?
FEATURE_MANIFEST_ROOT=/home/s1785140/data/LJSpeech-1.1/feature_manifest

##################################################################################################
# Run training for different experiments
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
