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
