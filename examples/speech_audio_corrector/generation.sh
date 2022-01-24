## INFERENCE - no word-level alignment information
## TEST SET OOVS (words in test set but not in train)
## generate entire test set (using random masking just like training)
#cd ~/fairseq
#
#MODEL=test_no_word_pos_embeddings
#SPLIT=test
#VOCODER=wav_22050hz_hifigan
#SAVE_DIR=checkpoints/$MODEL
#FEATURE_MANIFEST_ROOT=/home/s1785140/data/LJSpeech-1.1/feature_manifest
##CHECKPOINT_NAME=checkpoint_best
#CHECKPOINT_NAME=checkpoint_last
#CHECKPOINT_PATH=${SAVE_DIR}/checkpoint${CHECKPOINT_NAME}.pt
#OUT_DIR=inference/$MODEL/$CHECKPOINT_NAME
#
#SPLIT=test
#python -m examples.speech_audio_corrector.generate_waveform_sac ${FEATURE_MANIFEST_ROOT} \
#  --config-yaml config.yaml --gen-subset ${SPLIT} --task speech_audio_corrector \
#  --path ${CHECKPOINT_PATH} --max-tokens 50000 --spec-bwd-max-iter 32 \
#  --results-path $OUT_DIR \
#  --vocoder hifigan \
#  --dump-waveforms
#mkdir $OUT_DIR/$VOCODER/LJ_TEST_SET
#mv $OUT_DIR/$VOCODER/*.wav $OUT_DIR/$VOCODER/LJ_TEST_SET
#
#SPLIT=test
#python -m examples.speech_audio_corrector.generate_waveform_sac ${FEATURE_MANIFEST_ROOT} \
#  --config-yaml config.yaml --gen-subset ${SPLIT} --task speech_audio_corrector \
#  --path ${CHECKPOINT_PATH} --max-tokens 50000 --spec-bwd-max-iter 32 \
#  --results-path $OUT_DIR \
#  --vocoder hifigan \
#  --dump-waveforms \
#  --batch-size 32 \
#  --speechreps-add-mask-tokens \
#  --txt-file examples/speech_audio_corrector/test_utts_test_set_oovs.txt \
#  --add-count-to-filename
#mkdir $OUT_DIR/$VOCODER/test_oovs
#mv $OUT_DIR/$VOCODER/*.wav $OUT_DIR/$VOCODER/test_oovs

##########################################################################################
# 24th Jan 2022
# generate to investigate models ability to do multi-speaker corrections well
cd ~/fairseq

MODELS=(
#    run2_SAC_baseline_halved_updatefreq
    run2_SAC_baseline_normal_updatefreq
    run2_SAC_one_mask_token_per_grapheme_maxtokens25000_halved_updatefreq
    run2_SAC_random0.5_halved_updatefreq
    run2_SAC_random0.5_with_ext_speechreps_halved_updatefreq
    run2_SAC_random_halved_updatefreq
    run2_SAC_random_with_ext_speechreps_halved_updatefreq
    run2_SAC_with_ext_speechreps_halved_updatefreq
)
for MODEL in ${MODELS[*]}; do
    VOCODER=wav_22050hz_hifigan
    SPLIT=test
    FEATURE_MANIFEST_ROOT=/home/s1785140/data/LJSpeech-1.1/feature_manifest
    SAVE_DIR=checkpoints/$MODEL
    #CHECKPOINT_NUM=650
    #CHECKPOINT_NAME=epoch${CHECKPOINT_NUM}
    #CHECKPOINT_PATH=${SAVE_DIR}/checkpoint${CHECKPOINT_NUM}.pt
    CHECKPOINT_NAME=checkpoint_last
    CHECKPOINT_PATH=${SAVE_DIR}/${CHECKPOINT_NAME}.pt
    OUT_DIR=inference/$MODEL/${CHECKPOINT_NAME}
    TEXT_FILENAME=test_utts_vctk_dev_set_oovs # wordtypes that are in vctk devset but not in LJ Training data

    python -m examples.speech_audio_corrector.generate_waveform_sac ${FEATURE_MANIFEST_ROOT} \
      --config-yaml config.yaml --gen-subset ${SPLIT} --task speech_audio_corrector \
      --path ${CHECKPOINT_PATH} --max-tokens 50000 --spec-bwd-max-iter 32 \
      --results-path ${OUT_DIR} \
      --vocoder hifigan \
      --dump-waveforms \
      --batch-size 32 \
      --txt-file examples/speech_audio_corrector/${TEXT_FILENAME}.txt \
      --add-count-to-filename \
      --use-external-speechreps --use-ext-word2speechreps-p=1.0

    NEW_OUT_DIR=${OUT_DIR}/${VOCODER}/${TEXT_FILENAME}/
    mkdir -p ${NEW_OUT_DIR}
    mv ${OUT_DIR}/${VOCODER}/*.wav ${NEW_OUT_DIR}

#    mkdir: cannot create directory 'inference/run2_SAC_with_ext_speechreps_halved_updatefreq/checkpoint_last/wav_22050hz_hifigan/test_utts_vctk_dev_set_oovs': No such file or directory
#mv: cannot stat 'inference/run2_SAC_with_ext_speechreps_halved_updatefreq/checkpoint_last/wav_22050hz_hifigan/*.wav': No such file or directory
#ls inference/run2_SAC_with_ext_speechreps_halved_updatefreq/checkpoint_last/ext_speechreps/wav_22050hz_hifigan/

done


