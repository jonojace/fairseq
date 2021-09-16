# Install/setup conda env and fairseq

```bash
conda update -y -n base -c defaults conda
conda env remove -y --name fairseq
conda create -y -n fairseq python=3.7 # python must be <=3.7 for tensorflow 1.15 to work
conda activate fairseq
conda install ipython # ensures that jupyter can find env python packages
pip install jupyter # ensures that jupyter can find env python packages
conda install -y pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch # works with ILCC cluster 2080Ti GPU
#conda install -y -c conda-forge librosa
#pip install -r requirements.txt
```

# Install fairseq
```bash
cd fairseq
pip install --editable ./
pip install pyarrow
```

# Setup data speech reps

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
DATA=/home/s1785140/data/ljspeech_wav2vec2_reps/wav2vec2-large-960h/layer-15/word_level/
fairseq-train $DATA \
    --task learn_lexicon \
    --arch lexicon_learner \
    --optimizer adam \
    --batch-size 4 \
    --num-wordtypes 100 \
    --max-examples-per-wordtype 100
```

Commands for debugging training of this model (hubert):

```bash
MODEL_NAME=test_hubert
DATA=/home/s1785140/data/ljspeech_hubert_reps/hubert-base/layer-6/word_level/
#DATA=/home/s1785140/data/ljspeech_hubert_reps/hubert-base/layer-6/word_level_with_padding_idx_offset/
fairseq-train $DATA \
    --tensorboard-logdir tb_logs/$MODEL_NAME \
    --task learn_lexicon_discrete_inputs \
    --arch lexicon_learner_seq2seq \
    --criterion lexicon_learner \
    --optimizer adam \
    --batch-size 64 \
    --max-train-wordtypes 100 \
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
tensorboard --logdir=tb_logs/ --port 1337
```


## Tips

If you can see tensorboard logs from another user, change the port number.

# Evaluate model
