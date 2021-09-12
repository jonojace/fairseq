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

Commands for debugging training of this model:

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
