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
# HANDLE IF JOBS WILL BE RUN INTERACTIVELY OR SENT AS SLURM JOBS
# interactive=true
# if interactive set sbatch slurm script

##########################################################################################
# 24th Jan 2022
# generate to investigate models ability to do multi-speaker corrections well
cd ~/fairseq

VOCODER=wav_22050hz_hifigan
SPLIT=test
FEATURE_MANIFEST_ROOT=/home/s1785140/data/LJSpeech-1.1/feature_manifest
TEXT_FILENAME=test_utts_vctk_dev_set_oovs # wordtypes that are in vctk devset but not in LJ Training data

MODELS=(
#    test_sac_normal_masking2
#    run2_SAC_baseline_halved_updatefreq
#    run2_SAC_baseline_normal_updatefreq
#    run2_SAC_one_mask_token_per_grapheme_maxtokens25000_halved_updatefreq
#    run2_SAC_random0.5_halved_updatefreq
#    run2_SAC_random0.5_with_ext_speechreps_halved_updatefreq
#    run2_SAC_random_halved_updatefreq
#    run2_SAC_random_with_ext_speechreps_halved_updatefreq
#    run2_SAC_with_ext_speechreps_halved_updatefreq
#    run2_SAC_baseline_lr0.001_halved_updatefreq
#    run2_SAC_baseline_lr0.00001_halved_updatefreq
#    run2_SAC_replace_middle_eos_with_sos_halved_updatefreq
#    run2_SAC_zero_mask_toks_per_word_halved_updatefreq
#    run2_SAC_dont_remove_dup_codes_maxtoks20000_4gpus_updfreq2
#    run2_SAC_dont_remove_dup_codes_maxtoks20000_halved_updatefreq
    run3_SAC_ext_speechcodes_p_0.3
    run3_SAC_ext_speechcodes_p_0.2
#    run3_SAC_remove_dup_codes
#    run3_SAC_eos_symbol_in_middle
    run3_SAC_ext_speechcodes_p_0.05
    run3_SAC_ext_speechcodes_p_0.1
    run3_SAC_baseline
#    run3_SAC_one_mask_tok_per_word
    run3_SAC_ext_speechcodes_p_0.4
#    run3_SAC_one_mask_tok_per_grapheme
    run3_SAC_ext_speechcodes_p_0.5
)

CHECKPOINTS=(
#    checkpoint3000
#    checkpoint3500
#    checkpoint4000
#    checkpoint4500
#    checkpoint5000
    checkpoint_last
)

for CHECKPOINT_NAME in ${CHECKPOINTS[*]};
do

    for MODEL in ${MODELS[*]};
    do
        ###############################################################################################################
        # make in paths and out paths
        SAVE_DIR=checkpoints/$MODEL # where checkpoints for this model are located
        CHECKPOINT_PATH=${SAVE_DIR}/${CHECKPOINT_NAME}.pt # to load checkpoint
        OUT_DIR=inference/${MODEL}/${CHECKPOINT_NAME} # to save generated samples

        ###############################################################################################################
        # LJ test set utts
        echo "Generating samples for checkpoint ${CHECKPOINT_PATH} using LJ ${SPLIT} set"
        python -m examples.speech_audio_corrector.generate_waveform_sac ${FEATURE_MANIFEST_ROOT} \
          --config-yaml config.yaml --gen-subset ${SPLIT} --task speech_audio_corrector \
          --path ${CHECKPOINT_PATH} --max-tokens 50000 --spec-bwd-max-iter 32 \
          --results-path ${OUT_DIR} \
          --vocoder hifigan \
          --dump-waveforms \
          --batch-size 32 \
          --add-count-to-filename \
          --randomise-examples-p 0.0 \
          --use-ext-word2speechreps-p 0.0 \
          --remove-dup-codes-p 0.0 \
          --mask-tok-per-word zero \
          --symbol-in-middle sos

        # move samples over to folder with more descriptive name
        NEW_OUT_DIR=${OUT_DIR}/${VOCODER}/LJ_TEST_SET/
        mkdir -p ${NEW_OUT_DIR}
        mv ${OUT_DIR}/${VOCODER}/*.wav ${NEW_OUT_DIR}

        ###############################################################################################################
        # test utts
        echo "Generating samples for checkpoint ${CHECKPOINT_PATH} using test utts file ${test_utts_vctk_dev_set_oovs}"
        python -m examples.speech_audio_corrector.generate_waveform_sac ${FEATURE_MANIFEST_ROOT} \
          --config-yaml config.yaml --gen-subset ${SPLIT} --task speech_audio_corrector \
          --path ${CHECKPOINT_PATH} --max-tokens 50000 --spec-bwd-max-iter 32 \
          --results-path ${OUT_DIR} \
          --vocoder hifigan \
          --dump-waveforms \
          --batch-size 32 \
          --txt-file examples/speech_audio_corrector/${TEXT_FILENAME}.txt \
          --add-count-to-filename \
          --randomise-examples-p 0.0 \
          --use-external-speechreps \
          --use-ext-word2speechreps-p 1.0 \
          --remove-dup-codes-p 0.0 \
          --mask-tok-per-word zero \
          --symbol-in-middle sos

        # move samples over to folder with more descriptive name
        NEW_OUT_DIR=${OUT_DIR}/${VOCODER}/${TEXT_FILENAME}/
        mkdir -p ${NEW_OUT_DIR}
        mv ${OUT_DIR}/${VOCODER}/*.wav ${NEW_OUT_DIR}
    done

done
