#!/usr/bin/env bash

# exit when any command fails
#set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

##########################################################################################
# generate to investigate models ability to do multi-speaker corrections well
cd ~/fairseq

VOCODER=wav_22050hz_hifigan
SPLIT=test
FEATURE_MANIFEST_ROOT=/home/s1785140/data/LJSpeech-1.1/feature_manifest
AUDIO_MANIFEST_ROOT=/home/s1785140/data/LJSpeech-1.1/audio_manifest # for eval
TEXT_FILENAME=test_utts_vctk_dev_set_oovs # wordtypes that are in vctk devset but not in LJ Training data

MAX_TOKENS=100000
MAX_SENTENCES=64

# Set default hparams for generation (the ones used for the baseline model)
DEFAULT_MID_SYMBOL=sos # put sos instead of eos token in between graphemes + speech sequences
DEFAULT_REM_DUP_CODE_P=0.0 # do not deduplicate speech codes
DEFAULT_MASK_TOK_PER_WORD=zero # no mask tokens in graphemes seq

WAV2VEC2_CHECKPOINT_PATH=/home/s1785140/fairseq/checkpoints/wav2vec/wav2vec_small_960h.pt
WAV2VEC2_DICT_DIR=/home/s1785140/fairseq/checkpoints/wav2vec

# list of model names and their hparams for generation
# MODEL REM_DUP_CODE_P MASK_TOK_PER_WORD MID_SYMBOL
# shellcheck disable=SC2054
SETUPS=(
#    test_sac_normal_masking2
#    RUN 2 ======================================================
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
#    RUN 3 ======================================================
    run3_SAC_baseline,${DEFAULT_REM_DUP_CODE_P},${DEFAULT_MASK_TOK_PER_WORD},${DEFAULT_MID_SYMBOL}
    run3_SAC_remove_dup_codes,1.0,${DEFAULT_MASK_TOK_PER_WORD},${DEFAULT_MID_SYMBOL}
    run3_SAC_eos_symbol_in_middle,${DEFAULT_REM_DUP_CODE_P},${DEFAULT_MASK_TOK_PER_WORD},eos
    run3_SAC_one_mask_tok_per_word,${DEFAULT_REM_DUP_CODE_P},one,${DEFAULT_MID_SYMBOL}
    run3_SAC_dont_mask_encoder_out_speech_timesteps,${DEFAULT_REM_DUP_CODE_P},${DEFAULT_MASK_TOK_PER_WORD},${DEFAULT_MID_SYMBOL}
#    run3_SAC_one_mask_tok_per_grapheme,${DEFAULT_REM_DUP_CODE_P},many,${DEFAULT_MID_SYMBOL}
#    run3_SAC_dropout_p_0.05,${DEFAULT_REM_DUP_CODE_P},${DEFAULT_MASK_TOK_PER_WORD},${DEFAULT_MID_SYMBOL}
#    run3_SAC_dropout_p_0.25,${DEFAULT_REM_DUP_CODE_P},${DEFAULT_MASK_TOK_PER_WORD},${DEFAULT_MID_SYMBOL}
#    run3_SAC_dropout_p_0.20,${DEFAULT_REM_DUP_CODE_P},${DEFAULT_MASK_TOK_PER_WORD},${DEFAULT_MID_SYMBOL}
#    run3_SAC_dropout_p_0.15,${DEFAULT_REM_DUP_CODE_P},${DEFAULT_MASK_TOK_PER_WORD},${DEFAULT_MID_SYMBOL}
)

CHECKPOINTS=(
#    checkpoint_last
#    checkpoint500
#    checkpoint1000
#    checkpoint1500
#    checkpoint2000
#    checkpoint2500
#    checkpoint3000
    checkpoint3500
    checkpoint4000
    checkpoint4500
    checkpoint5000
    checkpoint5500
    checkpoint6000
    checkpoint6500
    checkpoint7000
    checkpoint7500
    checkpoint8000
    checkpoint8500
    checkpoint9000
    checkpoint9500
    checkpoint10000
)

for CHECKPOINT_NAME in ${CHECKPOINTS[*]};
do

    for SETUP in ${SETUPS[*]};
    do
        IFS=',' read MODEL REM_DUP_CODE_P MASK_TOK_PER_WORD MID_SYMBOL <<< "${SETUP}"
        echo "Model Name: ${MODEL}"
        echo "Remove duplicate code probability: ${REM_DUP_CODE_P}"
        echo "Number of mask tokens per word: ${MASK_TOK_PER_WORD}"
        echo "Middle symbol: ${MID_SYMBOL}"

        # make in paths and out paths
        SAVE_DIR=checkpoints/$MODEL # where checkpoints for this model are located
        CHECKPOINT_PATH=${SAVE_DIR}/${CHECKPOINT_NAME}.pt # to load checkpoint
        OUT_DIR=inference/${MODEL}/${CHECKPOINT_NAME} # to save generated samples

        ###############################################################################################################
        ###############################################################################################################
        ###############################################################################################################
        # LJ test set utts
        echo "Generating samples for checkpoint ${CHECKPOINT_PATH} using LJ ${SPLIT} set"

        #############################################################################################################
        # try to generate LJ test set for model checkpoint
        # note --mask-words-p 1.0 so all words are represented as speech reps
        SAMPLE_RATE=22050
        VOCODER_TYPE=hifigan
        echo "${SAMPLE_RATE}Hz samples: for listening, and for MCD MSD objective evaluations"
        python -m examples.speech_audio_corrector.generate_waveform_sac ${FEATURE_MANIFEST_ROOT} \
          --config-yaml config.yaml --gen-subset ${SPLIT} --task speech_audio_corrector \
          --path ${CHECKPOINT_PATH} --max-tokens $MAX_TOKENS --spec-bwd-max-iter 32 \
          --results-path ${OUT_DIR} \
          --vocoder ${VOCODER_TYPE} \
          --dump-waveforms \
          --batch-size $MAX_SENTENCES \
          --randomise-examples-p 0.0 \
          --use-ext-word2speechreps-p 0.0 \
          --remove-dup-codes-p ${REM_DUP_CODE_P} \
          --mask-tok-per-word ${MASK_TOK_PER_WORD} \
          --symbol-in-middle ${MID_SYMBOL} \
          --mask-words-p 1.0 \
          --dump-target \
          --output-sample-rate $SAMPLE_RATE \
          --use-sample-id-as-filename
#          --dont-mask-encoder-out-speech-timesteps

        #only run following commands if previous generate script worked
        if [ $? -eq 1 ]; then # $? refers to exit code of previous command
          echo "Skipping eval metrics for ${CHECKPOINT_PATH} as generation failed (likely no checkpoint found)"
        else
          #############################################################################################################
          # for Listening and MCD + MSD eval
          VOCODER=wav_${SAMPLE_RATE}hz_${VOCODER_TYPE}

          # move samples over to folder with more descriptive name
          NEW_OUT_DIR=${OUT_DIR}/LJ_TEST_SET/${VOCODER}
          mkdir -p ${NEW_OUT_DIR}
          mv ${OUT_DIR}/${VOCODER}/*.wav ${NEW_OUT_DIR}
          rm -rf ${OUT_DIR}/${VOCODER}

          # move resynthesised targets
          NEW_OUT_DIR_TGT=${OUT_DIR}/LJ_TEST_SET/${VOCODER}_tgt
          mkdir -p ${NEW_OUT_DIR_TGT}
          mv ${OUT_DIR}/${VOCODER}_tgt/*.wav ${NEW_OUT_DIR_TGT}
          rm -rf ${OUT_DIR}/${VOCODER}_tgt

          # evaluate and get MCD (+ MSD)
          python -m examples.speech_synthesis.evaluation.get_eval_manifest \
            --generation-root ${OUT_DIR}/LJ_TEST_SET/ \
            --audio-manifest ${AUDIO_MANIFEST_ROOT}/${SPLIT}.audio.tsv \
            --output-path ${OUT_DIR}/${SPLIT}.eval.tsv \
            --vocoder ${VOCODER_TYPE} --sample-rate $SAMPLE_RATE --audio-format wav \
            --use-resynthesized-target
          python -m examples.speech_synthesis.evaluation.eval_sp \
            ${OUT_DIR}/${SPLIT}.eval.tsv --mcd \
            --save-dir ${OUT_DIR}/
  #          --msd # optional, mel spectral distortion

#          ###############################################################################################################
#          # For ASR EVAL
#          SAMPLE_RATE=16000
#          VOCODER_TYPE=hifigan
#          echo "${SAMPLE_RATE}Hz samples: for wav2vec2.0 ASR objective eval (CER%)"
#          python -m examples.speech_audio_corrector.generate_waveform_sac ${FEATURE_MANIFEST_ROOT} \
#            --config-yaml config.yaml --gen-subset ${SPLIT} --task speech_audio_corrector \
#            --path ${CHECKPOINT_PATH} --max-tokens $MAX_TOKENS --spec-bwd-max-iter 32 \
#            --results-path ${OUT_DIR} \
#            --vocoder ${VOCODER_TYPE} \
#            --dump-waveforms \
#            --batch-size $MAX_SENTENCES \
#            --randomise-examples-p 0.0 \
#            --use-ext-word2speechreps-p 0.0 \
#            --remove-dup-codes-p ${REM_DUP_CODE_P} \
#            --mask-tok-per-word ${MASK_TOK_PER_WORD} \
#            --symbol-in-middle ${MID_SYMBOL} \
#            --mask-words-p 1.0 \
#            --output-sample-rate $SAMPLE_RATE \
#            --use-sample-id-as-filename
#
#          VOCODER=wav_${SAMPLE_RATE}hz_${VOCODER_TYPE}
#
#          # move samples over to folder with more descriptive name
#          NEW_OUT_DIR=${OUT_DIR}/LJ_TEST_SET/${VOCODER}
#          mkdir -p ${NEW_OUT_DIR}
#          mv ${OUT_DIR}/${VOCODER}/*.wav ${NEW_OUT_DIR}
#          rm -rf ${OUT_DIR}/${VOCODER}
#
#          # evaluate and get ASR CER%
#          python -m examples.speech_synthesis.evaluation.get_eval_manifest \
#            --generation-root ${OUT_DIR}/LJ_TEST_SET/ \
#            --audio-manifest ${AUDIO_MANIFEST_ROOT}/${SPLIT}.audio.tsv \
#            --output-path ${OUT_DIR}/${SPLIT}.eval_16khz.tsv \
#            --vocoder ${VOCODER_TYPE} --sample-rate $SAMPLE_RATE --audio-format wav \
#            --use-resynthesized-target
#          python -m examples.speech_synthesis.evaluation.eval_asr \
#            --audio-header syn --text-header text --err-unit char --split ${SPLIT} \
#            --w2v-ckpt ${WAV2VEC2_CHECKPOINT_PATH} --w2v-dict-dir ${WAV2VEC2_DICT_DIR} \
#            --raw-manifest ${OUT_DIR}/${SPLIT}.eval_16khz.tsv --asr-dir ${OUT_DIR}/asr

        fi

        ###############################################################################################################
        ###############################################################################################################
        ###############################################################################################################
#        # test utts
#        echo "Generating samples for checkpoint ${CHECKPOINT_PATH} using test utts file ${test_utts_vctk_dev_set_oovs}"
#        python -m examples.speech_audio_corrector.generate_waveform_sac ${FEATURE_MANIFEST_ROOT} \
#          --config-yaml config.yaml --gen-subset ${SPLIT} --task speech_audio_corrector \
#          --path ${CHECKPOINT_PATH} --max-tokens $MAX_TOKENS --spec-bwd-max-iter 32 \
#          --results-path ${OUT_DIR} \
#          --vocoder hifigan \
#          --dump-waveforms \
#          --batch-size $MAX_SENTENCES \
#          --txt-file examples/speech_audio_corrector/${TEXT_FILENAME}.txt \
#          --add-count-to-filename \
#          --randomise-examples-p 0.0 \
#          --use-external-speechreps \
#          --use-ext-word2speechreps-p 1.0 \
#          --remove-dup-codes-p ${REM_DUP_CODE_P} \
#          --mask-tok-per-word ${MASK_TOK_PER_WORD} \
#          --symbol-in-middle ${MID_SYMBOL}
#
#        # move samples over to folder with more descriptive name
#        NEW_OUT_DIR=${OUT_DIR}/${TEXT_FILENAME}/${VOCODER}
#        mkdir -p ${NEW_OUT_DIR}
#        mv ${OUT_DIR}/${VOCODER}/*.wav ${NEW_OUT_DIR}
#        rm -rf ${OUT_DIR}/${VOCODER}

    done

done
