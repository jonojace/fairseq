# Transformer Acoustic Corrector

# Install/setup conda env and fairseq

Install huggingface for getting CTC outputs

```bash
conda update -y -n base -c defaults conda
conda env remove -y --name huggingface
conda create -y -n huggingface python=3.8 # python must be <=3.8 for pytorch to work
conda activate huggingface
pip install --upgrade pip
conda install -y pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch # works with ILCC cluster 2080Ti GPU
pip install transformers datasets soundfile jupyterlab ipywidgets librosa


conda install ipython # ensures that jupyter can find env python packages
pip install jupyter # ensures that jupyter can find env python packages
#conda install -y -c conda-forge librosa
#pip install -r requirements.txt

# to make env visible to jupyter notebooks @ https://stackoverflow.com/questions/39604271/conda-environments-not-showing-up-in-jupyter-notebook
conda install ipykernel
conda install nb_conda_kernels # or conda install nb_conda
conda install ipywidgets
python -m ipykernel install --user --name fairseq --display-name "Python (fairseq)"
pip install jupyterlab
pip install torchdistill
```

# Setup TTS data 

(from https://github.com/pytorch/fairseq/blob/main/examples/speech_synthesis/docs/ljspeech_example.md)

```bash
cd /home/s1785140/data/LJSpeech-1.1/

mkdir audio_data
mkdir audio_manifest
mkdir feature_manifest

AUDIO_DATA_ROOT=/home/s1785140/data/LJSpeech-1.1/audio_data
AUDIO_MANIFEST_ROOT=/home/s1785140/data/LJSpeech-1.1/audio_manifest

python -m examples.speech_synthesis.preprocessing.get_ljspeech_audio_manifest \
  --output-data-root ${AUDIO_DATA_ROOT} \
  --output-manifest-root ${AUDIO_MANIFEST_ROOT}

FEATURE_MANIFEST_ROOT=/home/s1785140/data/LJSpeech-1.1/feature_manifest

python -m examples.speech_synthesis.preprocessing.get_feature_manifest \
  --audio-manifest-root ${AUDIO_MANIFEST_ROOT} \
  --output-root ${FEATURE_MANIFEST_ROOT}
  # --ipa-vocab --use-g2p # commented out as we want raw grapeheme inputs for TAC
```

# Vanilla TTS training command
```bash
NUM_GPUS=4
CPUS_PER_TASK=2
MEM=32000
EXCLUDE=arnold
#EXCLUDE=duflo,arnold
srun --part=ILCC_GPU,CDT_GPU --gres=gpu:$NUM_GPUS --cpus-per-task=$CPUS_PER_TASK --mem=$MEM --exclude=$EXCLUDE --pty bash
#srun --part=ILCC_GPU,CDT_GPU --gres=gpu:gtx2080ti:$NUM_GPUS --cpus-per-task=$CPUS_PER_TASK --mem=$MEM --exclude=$EXCLUDE --pty bash

cd ~
source activate_fairseq.sh
NUM_GPUS=4
NUM_WORKERS=2
UPDATE_FREQ=3
MAX_TOKENS=20000 # 30000 is default for transformer TTS
MAX_SENTENCES=NULL # TODO use this to help control mem usage?
FEATURE_MANIFEST_ROOT=/home/s1785140/data/LJSpeech-1.1/feature_manifest

MODEL_NAME=test_tts_maxtokens${MAX_TOKENS}_updatefreq${UPDATE_FREQ}_gpus${NUM_GPUS}
FEATURE_MANIFEST_ROOT=/home/s1785140/data/LJSpeech-1.1/feature_manifest
fairseq-train ${FEATURE_MANIFEST_ROOT} \
  --save-dir checkpoints/$MODEL_NAME --tensorboard-logdir tb_logs/$MODEL_NAME \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --num-workers $NUM_WORKERS --max-tokens $MAX_TOKENS --max-update 200000 \
  --task text_to_speech --criterion tacotron2 --arch tts_transformer \
  --clip-norm 5.0 --n-frames-per-step 4 --bce-pos-weight 5.0 \
  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
  --encoder-normalize-before --decoder-normalize-before \
  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --seed 1 --update-freq $UPDATE_FREQ --best-checkpoint-metric loss
```

# Choosing max tokens and update freq depending on number of GPUs

## default single gpu training for transformer TTS
default 30000 max tokens x 8 update frequency = 240000 tokens per update

## getting to new values for multigpu training

30000 max tokens per device results in OOM
want to reduce to 20000 per device

keep 240000 tokens per update as constant

4 gpus

240000 / (20000*4) = 3 

therefore set update freq to 3

Total tokens desired per update	240000
Num gpus	4

Calculate max tokens for different update freq:
max_tokens_per_gpu = total_tokens_per_update / (update_freq * num_gpus)
==============================
update_freq	max_tokens_per_gpu
==============================
2	        30000
3	        20000             # seems to work fine with 4 2080s/1080s
4	        15000
5	        12000
6	        10000
7	        8571.428571
8	        7500
9	        6666.666667
10	        6000
11	        5454.545455
12	        5000
13	        4615.384615
14	        4285.714286
15	        4000
16	        3750

# Training command (for debugging)
```bash
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
MODEL_NAME=test_sac_normal_masking_maxtokens${MAX_TOKENS}_updatefreq${UPDATE_FREQ}_gpus${NUM_GPUS}
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
  #--eval-inference --best-checkpoint-metric mcd_lossmetric mcd_lossv \
  --num-workers $NUM_WORKERS --max-tokens $MAX_TOKENS --max-update 200000 \
  --task speech_audio_corrector --criterion sac_tts --arch sac_transformer \
  --clip-norm 5.0 --n-frames-per-step 4 --bce-pos-weight 5.0 \
  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
  --encoder-normalize-before --decoder-normalize-before \
  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --seed 1 --update-freq $UPDATE_FREQ --best-checkpoint-metric loss  
  #--eval-inference --best-checkpoint-metric mcd_lossmetric mcd_loss
  
MODEL_NAME=test_sac_mask_all_speechreps
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
```

# Generate waveform

## Normal TTS speech synthesis 

Example follows https://github.com/pytorch/fairseq/blob/main/examples/speech_synthesis/docs/ljspeech_example.md#inference

### TTS Inference from vanilla transformer TTS model

```bash
cd ~/fairseq

MODEL=test_tts_maxtokens20000_updatefreq3_gpus4
SAVE_DIR=checkpoints/$MODEL
SPLIT=test
FEATURE_MANIFEST_ROOT=/home/s1785140/data/LJSpeech-1.1/feature_manifest
CHECKPOINT_NAME=avg_last_5
CHECKPOINT_PATH=${SAVE_DIR}/checkpoint_${CHECKPOINT_NAME}.pt
OUT_DIR=inference/$MODEL/$CHECKPOINT_NAME
python scripts/average_checkpoints.py --inputs ${SAVE_DIR} \
  --num-epoch-checkpoints 5 \
  --output ${CHECKPOINT_PATH}

python -m examples.speech_synthesis.generate_waveform ${FEATURE_MANIFEST_ROOT} \
  --config-yaml config.yaml --gen-subset ${SPLIT} --task speech_audio_corrector \
  --path ${CHECKPOINT_PATH} --max-tokens 50000 --spec-bwd-max-iter 32 \
  --results-path $OUT_DIR \
  --dump-waveforms
  
# copy wavs back to macbook
./copy_fairseq_samples_back_to_macbook.sh # run this script from macbook 
```

### TTS Inference from SAC model

```bash
cd ~/fairseq

MODEL=test_sac_normal_masking2
SAVE_DIR=checkpoints/$MODEL
SPLIT=test
FEATURE_MANIFEST_ROOT=/home/s1785140/data/LJSpeech-1.1/feature_manifest
CHECKPOINT_NAME=avg_last_5
CHECKPOINT_PATH=${SAVE_DIR}/checkpoint_${CHECKPOINT_NAME}.pt
OUT_DIR=inference/$MODEL/$CHECKPOINT_NAME
python scripts/average_checkpoints.py --inputs ${SAVE_DIR} \
  --num-epoch-checkpoints 5 \
  --output ${CHECKPOINT_PATH}

python -m examples.speech_audio_corrector.generate_waveform_sac ${FEATURE_MANIFEST_ROOT} \
  --config-yaml config.yaml --gen-subset ${SPLIT} --task speech_audio_corrector \
  --path ${CHECKPOINT_PATH} --max-tokens 50000 --spec-bwd-max-iter 32 \
  --results-path $OUT_DIR \
  --dump-waveforms
```

## Speech Audio Correction speech synthesis

# Setup Speech Reps data

1) obtain discretised speech reps
https://github.com/pytorch/fairseq/tree/master/examples/textless_nlp/gslm/speech2unit
either 
a) generate yourself:
```bash
# convert LJSpeech to 16khz 

# create manifests file (look at lj_speech_manifest.txt for an example)
# top line should be path to the folder containing the wav files

#download checkpoints
#hubert: https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt
#k-means clusters: https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km100/km.bin

# run speech reps model and quantisation
TYPE=hubert
KM_MODEL_PATH=../../fairseq/examples/textless_nlp/gslm/speech2unit/pretrained_models/hubert/km100/hubert_km100.bin
ACSTC_MODEL_PATH=../../fairseq/examples/textless_nlp/gslm/speech2unit/pretrained_models/hubert/hubert_base_ls960.pt
LAYER=6
MANIFEST=filelists/voice_conversion_test.txt
OUT_QUANTIZED_FILE=speech2unit_output/quantized/voice_conversion_test_quantized.txt
EXTENSION=".wav"

CUDA_VISIBLE_DEVICES=9 python ../../fairseq/examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py \
    --feature_type $TYPE \
    --kmeans_model_path $KM_MODEL_PATH \
    --acoustic_model_path $ACSTC_MODEL_PATH \
    --layer $LAYER \
    --manifest_path $MANIFEST \
    --out_quantized_file_path $OUT_QUANTIZED_FILE \
    --extension $EXTENSION
```
b) or use precomputed ones: fairseq/fairseq/models/lexicon_learner/lj_speech_quantized.txt

2) align speech reps at the word level using mfa 
run ipynb script

3) get lookup table (also look at fairseq/examples/lexicon_learner/get_hubert_lookup_table.py)
```python 
import joblib
kmeans_model_path = '../../fairseq/examples/textless_nlp/gslm/speech2unit/pretrained_models/hubert/km100/km.bin'
kmeans_model = joblib.load(open(kmeans_model_path, "rb")) # this is just a sklearn model
centroids = kmeans_model.cluster_centers_
```




# install reqs

# Train model



```bash
FEATURE_MANIFEST_ROOT=/home/s1785140/data/LJSpeech-1.1/feature_manifest
MODEL_NAME=test_tac
SAVE_DIR=checkpoints/$MODEL_NAME
fairseq-train ${FEATURE_MANIFEST_ROOT} --save-dir ${SAVE_DIR} \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --num-workers 4 --max-tokens 30000 --max-update 200000 \
  --task text_to_speech --criterion tacotron2 --arch tts_transformer \
  --clip-norm 5.0 --n-frames-per-step 4 --bce-pos-weight 5.0 \
  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
  --encoder-normalize-before --decoder-normalize-before \
  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --seed 1 --update-freq 8 --eval-inference --best-checkpoint-metric mcd_loss
```


```bash
DATA=/home/s1785140/data/ljspeech_wav2vec2_reps/wav2vec2-large-960h/layer-15/word_level/
fairseq-train $DATA \
    --task learn_lexicon \
    --arch lexicon_learner \
    --optimizer adam \
    --batch-size 4 \
    --num-wordtypes 100 \
    --max-examples-per-wordtype 100
```

Commands for fast debugging training of this model (hubert):

```bash
MODEL_NAME=test_hubert
DATA=/home/s1785140/data/ljspeech_hubert_reps/hubert-base/layer-6/word_level_without_padding_idx_offset
cd ~/fairseq
fairseq-train $DATA \
    --tensorboard-logdir tb_logs/$MODEL_NAME \
    --task learn_lexicon_discrete_inputs \
    --arch lexicon_learner_seq2seq \
    --criterion lexicon_learner \
    --sequence-loss-method summariser \
    --optimizer adam \
    --batch-size 2 \
    --padding-index-offset 1 \
    --max-train-wordtypes 10 \
    --min-train-examples-per-wordtype 2 \
    --max-train-examples-per-wordtype 2 \
    --valid-seen-wordtypes 5 \
    --valid-unseen-wordtypes 5 \
    --valid-examples-per-wordtype 2 \
    --valid-subset valid-seen,valid-unseen \
    --save-dir checkpoints/$MODEL_NAME \
    --save-interval 1 --max-epoch 2 \
    --lr 0.001 \
    --cache-all-data \
    --debug-only-include-words-beginning-with b \
    --normalize-out \
    --transformer-mask-outputs \
    --no-save
```

Commands for debugging training of this model (wav2vec2):

```bash
MODEL_NAME=debugging
DATA=/home/s1785140/data/ljspeech_wav2vec2_reps/wav2vec2-large-960h/layer-15/word_level/
fairseq-train $DATA \
    --tensorboard-logdir tb_logs/$MODEL_NAME \
    --task learn_lexicon \
    --arch lexicon_learner \
    --criterion lexicon_learner \
    --optimizer adam \
    --batch-size 64 \
    --min-train-examples-per-wordtype 10 \
    --max-train-examples-per-wordtype 10 \
    --valid-seen-wordtypes 100 \
    --valid-unseen-wordtypes 100 \
    --valid-examples-per-wordtype 10 \
    --valid-subset valid-seen,valid-unseen \
    --save-dir checkpoints/$MODEL_NAME \
    --save-interval 1 --max-epoch 2 \
    --lr 0.001 \
    --no-save
```

To submit as a slurm job, prepend the slurm script:

```bash
MODEL_NAME=test_model3
DATA=/home/s1785140/data/ljspeech_wav2vec2_reps/wav2vec2-large-960h/layer-15/word_level/
./sbatch.sh 2080 fairseq-train $DATA \
    --tensorboard-logdir tb_logs/$MODEL_NAME \
    --task learn_lexicon \
    --arch lexicon_learner \
    --criterion lexicon_learner \
    --optimizer adam \
    --batch-size 8 \
    --max-train-wordtypes 25 \
    --valid-seen-wordtypes 10 \
    --valid-unseen-wordtypes 10 \
    --max-train-examples-per-wordtype 25 \
    --valid-subset valid-seen,valid-unseen \
    --save-interval 1 --max-epoch 2 \
    --save-dir checkpoints/$MODEL_NAME \
    --no-save
```

(GET ME WORKING!) From config file and command line **(NOTE WE ARE USING fairseq-hydra-train now)**:

```bash
DATA=/home/s1785140/data/ljspeech_wav2vec2_reps/wav2vec2-large-960h/layer-15/word_level/
fairseq-hydra-train \
    --config-dir /home/s1785140/fairseq/examples/lexicon_learner/config \
    --config-name ALL_ljspeech \
    task.data=$DATA \
    dataset.batch_size=2 \
    --max-num-wordtypes 50 \
    --max-train-examples-per-wordtype 50 \
    --max-epoch 5 \
    --no-save
```



# Using tensorboard from local computer 

## 1) On local

https://stackoverflow.com/questions/38464559/how-to-locally-view-tensorboard-of-remote-server

```bash
ssh -NfL 1337:localhost:1337 username@remote_server_address

# i.e. 
ssh -NfL 1337:localhost:1337 s1785140@escience6.inf.ed.ac.uk
```

## 2) On server

ensure node has internet access (GPU nodes often do not)

```bash
source activate_fairseq.sh
tensorboard --logdir=tb_logs/ --port 1337 --bind_all
```


## Tips

If you can see tensorboard logs from another user, change the port number.

# Evaluate model


## Remote jupyter notebook development

Choose A or B:

### A: Using configured server (started yourself on a gpu node)

Instructions adapted from:
https://nero-docs.stanford.edu/jupyter-slurm.html

1. Start up jupyter server on remote GPU node. Make a note of the node you were assigned to ('duflo' in this example). 

```bash
ssh s1785140@escience6.inf.ed.ac.uk # replace s1785140 with your dice username
srun --part=ILCC_GPU,CDT_GPU --gres=gpu:gtx2080ti:1 --cpus-per-task=1 --mem=8000 --pty bash
conda activate word-templates
cd word-templates # (optional) go to project root to change the 'cwd' of jupyter
jupyter-lab --no-browser --ip=0.0.0.0 # or jupyter-lab --no-browser --ip=0.0.0.0
```

2. Find the notebook URL related to the node that you were assigned. For example in the example below the correct link is `http://duflo.inf.ed.ac.uk:8888/?token=95dd3ae95c8d91c466405cbcaf8114e944b85d731b481183`

```bash
[I 16:23:20.776 NotebookApp] Serving notebooks from local directory: /disk/nfs/ostrom/s1785140
[I 16:23:20.777 NotebookApp] Jupyter Notebook 6.4.0 is running at:
[I 16:23:20.777 NotebookApp] http://duflo.inf.ed.ac.uk:8888/?token=95dd3ae95c8d91c466405cbcaf8114e944b85d731b481183
[I 16:23:20.777 NotebookApp]  or http://127.0.0.1:8888/?token=95dd3ae95c8d91c466405cbcaf8114e944b85d731b481183
[I 16:23:20.777 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
```

3. Either click the link to directly develop in your browser or copy the link and paste into pycharm as a configured jupyter notebook server to develop there.

4. Test if GPU works in jupyter notebook. Enter in cell:

```python
import torch
torch.tensor([1.0, 2.0]).cuda()
```

If you want to make access to the notebook easier from the url without a token string, then set your jupyter notebook password

```bash
jupyter notebook password
```

### B: Using managed server (managed/started by pycharm) NOT SOLVED!!!
First need to setup a remote interpreter for jupyter to run in (https://www.jetbrains.com/help/pycharm/configuring-remote-interpreters-via-ssh.html)

https://www.jetbrains.com/help/pycharm/configuring-jupyter-notebook.html

NOT SOLVED... some info about how to possibly run pycharm using srun interactive gpu on slurm
https://intellij-support.jetbrains.com/hc/en-us/community/posts/115000489490-Running-remote-interpreter-on-a-cluster-with-srun
https://researchcomputing.princeton.edu/support/knowledge-base/pytorch#jupyter


## Remote debugging

https://www.jetbrains.com/help/pycharm/remote-debugging-with-product.html#remote-interpreter

