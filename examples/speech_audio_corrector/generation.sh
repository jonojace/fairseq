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

VOCODER_TYPE=hifigan
VOCODER=wav_22050hz_${VOCODER_TYPE}
SPLIT=test
FEATURE_MANIFEST_ROOT=/home/s1785140/data/LJSpeech-1.1/feature_manifest
AUDIO_MANIFEST_ROOT=/home/s1785140/data/LJSpeech-1.1/audio_manifest # for eval


MAX_TOKENS=100000
MAX_SENTENCES=64

# Set default hparams for generation (the ones used for the baseline model)
DEFAULT_MID_SYMBOL=sos # put sos instead of eos token in between graphemes + speech sequences
DEFAULT_REM_DUP_CODE_P=0.0 # do not deduplicate speech codes
DEFAULT_MASK_TOK_PER_WORD=zero # no mask tokens in graphemes seq
DEFAULT_DONT_MASK_SPEECH_TIMESTEPS=false # by default we mask out speech timesteps
DEFAULT_QUANT_LJ_FILE="/home/s1785140/fairseq/examples/speech_audio_corrector/lj_speech_quantized.txt"

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
#    run3_SAC_baseline,${DEFAULT_REM_DUP_CODE_P},${DEFAULT_MASK_TOK_PER_WORD},${DEFAULT_MID_SYMBOL},${DEFAULT_DONT_MASK_SPEECH_TIMESTEPS},${DEFAULT_QUANT_LJ_FILE}
#    run3_SAC_remove_dup_codes,1.0,${DEFAULT_MASK_TOK_PER_WORD},${DEFAULT_MID_SYMBOL},${DEFAULT_DONT_MASK_SPEECH_TIMESTEPS},${DEFAULT_QUANT_LJ_FILE}
#    run3_SAC_eos_symbol_in_middle,${DEFAULT_REM_DUP_CODE_P},${DEFAULT_MASK_TOK_PER_WORD},eos,${DEFAULT_DONT_MASK_SPEECH_TIMESTEPS},${DEFAULT_QUANT_LJ_FILE}
#    run3_SAC_one_mask_tok_per_word,${DEFAULT_REM_DUP_CODE_P},one,${DEFAULT_MID_SYMBOL},${DEFAULT_DONT_MASK_SPEECH_TIMESTEPS},${DEFAULT_QUANT_LJ_FILE}
#    run3_SAC_dont_mask_encoder_out_speech_timesteps,${DEFAULT_REM_DUP_CODE_P},${DEFAULT_MASK_TOK_PER_WORD},${DEFAULT_MID_SYMBOL},true,${DEFAULT_QUANT_LJ_FILE}
#    run5_vanilla_tts_transformer
#    run5_SAC_baseline,${DEFAULT_REM_DUP_CODE_P},${DEFAULT_MASK_TOK_PER_WORD},${DEFAULT_MID_SYMBOL},${DEFAULT_DONT_MASK_SPEECH_TIMESTEPS},${DEFAULT_QUANT_LJ_FILE}
#    run5_km50_quantized_ljspeech,${DEFAULT_REM_DUP_CODE_P},${DEFAULT_MASK_TOK_PER_WORD},${DEFAULT_MID_SYMBOL},${DEFAULT_DONT_MASK_SPEECH_TIMESTEPS},"/home/s1785140/fairseq/examples/speech_audio_corrector/ljspeech_quantized_km50.txt"
#    run5_km200_quantized_ljspeech,${DEFAULT_REM_DUP_CODE_P},${DEFAULT_MASK_TOK_PER_WORD},${DEFAULT_MID_SYMBOL},${DEFAULT_DONT_MASK_SPEECH_TIMESTEPS},"/home/s1785140/fairseq/examples/speech_audio_corrector/ljspeech_quantized_km200.txt"
#    run6_km100_AND_one_mask_tok_per_word_AND_dont_mask_encoder_out_speech_timesteps,${DEFAULT_REM_DUP_CODE_P},${DEFAULT_MASK_TOK_PER_WORD},${DEFAULT_MID_SYMBOL},${DEFAULT_DONT_MASK_SPEECH_TIMESTEPS},"/home/s1785140/fairseq/examples/speech_audio_corrector/ljspeech_quantized_km100.txt"
#    run6_km50_AND_one_mask_tok_per_word_AND_dont_mask_encoder_out_speech_timesteps,${DEFAULT_REM_DUP_CODE_P},${DEFAULT_MASK_TOK_PER_WORD},${DEFAULT_MID_SYMBOL},${DEFAULT_DONT_MASK_SPEECH_TIMESTEPS},"/home/s1785140/fairseq/examples/speech_audio_corrector/ljspeech_quantized_km50.txt"
#    run6_km200_AND_one_mask_tok_per_word_AND_dont_mask_encoder_out_speech_timesteps,${DEFAULT_REM_DUP_CODE_P},${DEFAULT_MASK_TOK_PER_WORD},${DEFAULT_MID_SYMBOL},${DEFAULT_DONT_MASK_SPEECH_TIMESTEPS},"/home/s1785140/fairseq/examples/speech_audio_corrector/ljspeech_quantized_km200.txt"
#    run5_vanilla_tts_transformer
    run5_one_mask_tok_per_word_AND_dont_mask_encoder_out_speech_timesteps,${DEFAULT_REM_DUP_CODE_P},one,${DEFAULT_MID_SYMBOL},true,${DEFAULT_QUANT_LJ_FILE}
)

CHECKPOINTS=(
#    checkpoint_last
#    checkpoint100
#    checkpoint200
#    checkpoint300
#    checkpoint400
#    checkpoint500
#    checkpoint600
#    checkpoint700
#    checkpoint800
#    checkpoint900
    checkpoint1000
#    checkpoint2000
#    checkpoint3000
#    checkpoint4000
#    checkpoint5000
#    checkpoint6000
#    checkpoint7000
#    checkpoint8000
#    checkpoint9000
#    checkpoint10000
)

# decide which generations to run
#RUN_LJ_TEST_SET_EVAL=true
RUN_LJ_TEST_SET_EVAL=false
GEN_VCTK_OOVS=true
#GEN_VCTK_OOVS=false

#TEXT_FILENAME=test_utts_vctk_dev_set_oovs # wordtypes that are in vctk devset but not in LJ Training data
#TEXT_FILENAME=test_utts_vctk_oovs_female # wordtypes that are in vctk female scot south us speakers but not in LJ Train
#TEXT_FILENAME=test_utts_vctk_oovs_male_fem_us_south # wordtypes that are in vctk female+male south+us speakers but not in LJ Train

#TEXT_FILENAME=test_utts_vctk_oovs_fem_us_scot # wordtypes that are in vctk female+male south+us speakers but not in LJ Train
#TEXT_FILENAME=test_utts_vctk_oovs_fem_us_scot_with_speechcodes # wordtypes that are in vctk female+male south+us speakers but not in LJ Train

#TEXT_FILENAME=test_utts_vctk_oovs_fem_us_scot_only_mispronounced # wordtypes that are in vctk female+male south+us speakers but not in LJ Train
#TEXT_FILENAME=test_utts_vctk_oovs_fem_us_scot_only_mispronounced_with_speechcodes # wordtypes that are in vctk female+male south+us speakers but not in LJ Train

#TEXT_FILENAME=test_utts_vctk_oovs_fem_us_scot_only_mispronounced_grapheme_input # wordtypes that are in vctk female+male south+us speakers but not in LJ Train
TEXT_FILENAME=test_utts_vctk_oovs_fem_us_scot_only_mispronounced_speechcode_input # wordtypes that are in vctk female+male south+us speakers but not in LJ Train

#TEXT_FILENAME=test_one_sentence

# generate test set using vanilla tts transformer
#USE_VANILLA_TTS_TRANSFORMER=true
USE_VANILLA_TTS_TRANSFORMER=false
if [ $USE_VANILLA_TTS_TRANSFORMER = "true" ]; then
    MODEL=run5_vanilla_tts_transformer
    SAVE_DIR=checkpoints/$MODEL
    FEATURE_MANIFEST_ROOT=/home/s1785140/data/LJSpeech-1.1/feature_manifest
    CHECKPOINT_NAME=checkpoint1000
    CHECKPOINT_PATH=${SAVE_DIR}/${CHECKPOINT_NAME}.pt
    OUT_DIR=inference/$MODEL/$CHECKPOINT_NAME

    python -m examples.speech_synthesis.generate_waveform ${FEATURE_MANIFEST_ROOT} \
      --config-yaml config.yaml --gen-subset mispronounced_oovs --task text_to_speech \
      --path ${CHECKPOINT_PATH} --max-tokens 50000 --spec-bwd-max-iter 32 \
      --results-path $OUT_DIR \
      --vocoder hifigan \
      --dump-waveforms

#    # move samples over to folder with more descriptive name
#    NEW_OUT_DIR=${OUT_DIR}/${TEXT_FILENAME}/${VOCODER}
#    mkdir -p ${NEW_OUT_DIR}
#    mv ${OUT_DIR}/${VOCODER}/*.wav ${NEW_OUT_DIR}
#    rm -rf ${OUT_DIR}/${VOCODER}
fi


for CHECKPOINT_NAME in ${CHECKPOINTS[*]};
do

    for SETUP in ${SETUPS[*]};
    do
        IFS=',' read MODEL REM_DUP_CODE_P MASK_TOK_PER_WORD MID_SYMBOL DONT_MASK_SPEECH_TIMESTEPS QUANT_LJ_FILE <<< "${SETUP}"
        echo "Model Name: ${MODEL}"
        echo "Remove duplicate code probability [0,1]: ${REM_DUP_CODE_P}"
        echo "Number of mask tokens per word (zero,one,many): ${MASK_TOK_PER_WORD}"
        echo "Middle symbol (sos,eos): ${MID_SYMBOL}"
        echo "Don't mask speech timesteps (true/false): ${DONT_MASK_SPEECH_TIMESTEPS}"

        # Set variables according to model hparams
        if [ $DONT_MASK_SPEECH_TIMESTEPS = "true" ]; then
            optional_flag_1="--dont-mask-encoder-out-speech-timesteps"
        else
            optional_flag_1=""
        fi

        # make in paths and out paths
        SAVE_DIR=checkpoints/$MODEL # where checkpoints for this model are located
        CHECKPOINT_PATH=${SAVE_DIR}/${CHECKPOINT_NAME}.pt # to load checkpoint
        OUT_DIR=inference/${MODEL}/${CHECKPOINT_NAME} # to save generated samples

        ###############################################################################################################
        ###############################################################################################################
        ###############################################################################################################
        if [ $RUN_LJ_TEST_SET_EVAL = "true" ]; then
            # LJ test set utts
            echo "Generating samples for checkpoint ${CHECKPOINT_PATH} using LJ ${SPLIT} set"

            #############################################################################################################
            # try to generate LJ test set for model checkpoint
            # note --mask-words-p 1.0 so all words are represented as speech reps
            # and use-ext-word2speechreps-p 0.0 so that speech reps always come from training data
            SAMPLE_RATE=22050
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
              --quantized-speechreps-file $QUANT_LJ_FILE \
              --use-sample-id-as-filename # $optional_flag_1

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
        fi


#        ###############################################################################################################
#        ###############################################################################################################
#        ###############################################################################################################
        if [ $GEN_VCTK_OOVS = "true" ]; then
            # test utts
            echo "Generating samples for checkpoint ${CHECKPOINT_PATH} using test utts file ${TEXT_FILENAME}"
            python -m examples.speech_audio_corrector.generate_waveform_sac ${FEATURE_MANIFEST_ROOT} \
              --config-yaml config.yaml --gen-subset ${SPLIT} --task speech_audio_corrector \
              --path ${CHECKPOINT_PATH} --max-tokens $MAX_TOKENS --spec-bwd-max-iter 32 \
              --results-path ${OUT_DIR} \
              --vocoder ${VOCODER_TYPE} \
              --dump-waveforms \
              --batch-size $MAX_SENTENCES \
              --txt-file examples/speech_audio_corrector/${TEXT_FILENAME}.txt \
              --add-count-to-filename \
              --randomise-examples-p 1.0 \
              --use-external-speechreps \
              --use-ext-word2speechreps-p 1.0 \
              --remove-dup-codes-p ${REM_DUP_CODE_P} \
              --mask-tok-per-word ${MASK_TOK_PER_WORD} \
              --symbol-in-middle ${MID_SYMBOL} \
              --append-token-ids-to-filename \
              --recreate-word2speechreps # $optional_flag_1

            # move samples over to folder with more descriptive name
            NEW_OUT_DIR=${OUT_DIR}/${TEXT_FILENAME}/${VOCODER}
            mkdir -p ${NEW_OUT_DIR}
            mv ${OUT_DIR}/${VOCODER}/*.wav ${NEW_OUT_DIR}
            rm -rf ${OUT_DIR}/${VOCODER}
        fi
    done

done
