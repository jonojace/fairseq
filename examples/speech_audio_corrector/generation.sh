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
