from collections import Counter
import random

def get_wordlevel_reprs(speechreps, word_align):
    """
    extract subsequence of 'repr' that corresponds to a particular word
    function expects input to be of dimension 2: (timesteps, hidden_size)
    """
    start_fraction = word_align['start'] / word_align['utt_dur']
    end_fraction = word_align['end'] / word_align['utt_dur']
    timesteps = len(speechreps)
    start_idx = round(start_fraction * timesteps)
    end_idx = round(end_fraction * timesteps)
    return speechreps[start_idx:end_idx]

def get_word2speechreps(ids, ids2speechreps, ids2word_alignments):
    # print("in get_word2speechreps!!!!", ids2speechreps.keys())

    word2speechreps = {}
    for utt_id in ids:
        speech_reps = ids2speechreps[utt_id]
        word_aligns = ids2word_alignments[utt_id]

        for word_align in word_aligns:
            word_align['speech_reps'] = get_wordlevel_reprs(speech_reps, word_align)

            # following info to debug whether alignments are consistent in len
            # word_align['speech_reps_len'] = len(word_align['speech_reps'])
            # word_align['speech_reps_len_dur_ratio'] = word_align['speech_reps_len'] / (word_align['end']-word_align['start'])

            wordtype = word_align['wordtype']
            example_no = word_align['example_no']
            unique_id = utt_id + '|' + str(example_no)

            if wordtype not in word2speechreps:
                word2speechreps[wordtype] = {}
            word2speechreps[wordtype][unique_id] = word_align['speech_reps']

    return word2speechreps



def get_mfa_text(word_align):
    return " ".join(w['wordtype'] for w in word_align)


def get_word_pos(graphemes, padding_idx, bpe_whitespace_tok="▁", boundary_same_pos=True,
                 append_eos=False, eos_symbol="</s>", boundary_start_pos=None):
    """
    for some space delimited sequence of symbols (e.g. text)

    return words and their word pos

    and also word pos of each grapheme in the seq (a list of the same length,
    of ints representing the words that each symbol / whitespace corresponds to)

    by default the boundary start position is initiated as padding_idx + 1
    and then word counts start from that value

    args:
        text: str of space delimited graphemes in the utterance ('_' denotes whitespace in the original utterance)
              e.g. "_ h o w _ a r e _ y o u" this is the format returned by sentence piece tokeniser

    e.g.
    _ h o w _ a r e _ y o u
        padding_idx == 1
        boundary_start_pos == 2
        boundary_same_pos == True

        before padding:
            [('how', 3), ('are', 4), ('you', 5)]
            [2, 3, 3, 3, 2, 4, 4, 4, 2, 5, 5, 5, 6]
        after concat with speechreps and padding (not performed in this fn, performed in SAC dataset collater):
            [2, 3, 3, 3, 2, 4, 4, 4, 2, 5, 5, 5, 6, <speechreps>, 1, 1, 1, ...]

    _ h o w _ a r e _ y o u
        padding_idx == 1
        boundary_start_pos == 2
        boundary_same_pos == False

        before padding:
            [('how', 3), ('are', 5), ('you', 7)]
            [2, 3, 3, 3, 4, 5, 5, 5, 6, 7, 7, 7, 8]
        after concat with speechreps and padding (not performed in this fn, performed in SAC dataset collater):
            [2, 3, 3, 3, 4, 5, 5, 5, 6, 7, 7, 7, 8, <speechreps>, 1, 1, 1, ...]
    """
    # double check that we are dealing with a seq output by bpe tokenizer
    assert graphemes[0] == bpe_whitespace_tok, f"graphemes == {graphemes}"

    if boundary_start_pos is None:
        boundary_start_pos = padding_idx + 1

    if boundary_same_pos:
        word_count = boundary_start_pos
    else:
        word_count = padding_idx

    word_and_word_pos = []
    word_pos_of_graphemes = []
    current_word = ""

    for i, c in enumerate(graphemes):
        # reached the last symbol of the utt
        if c == eos_symbol:
            word_and_word_pos.append((current_word, word_count))  # add last word
            word_pos_of_graphemes.append(word_count + 1)

        # whitespace
        elif c == bpe_whitespace_tok:
            if current_word:  # at a whitespace token AFTER processing at least one word
                word_and_word_pos.append((current_word, word_count))
                current_word = ""
            if boundary_same_pos:
                word_pos_of_graphemes.append(boundary_start_pos)
            else:
                word_count += 1  # because we count each whitespace_tok as a new word position
                word_pos_of_graphemes.append(word_count)

        # processing a grapheme in a word
        else:
            if graphemes[i - 1] == bpe_whitespace_tok:
                word_count += 1  # only increment word position if we are at the beginning of a new word, not within it
            word_pos_of_graphemes.append(word_count)
            current_word += c

    if append_eos:
        word_pos_of_graphemes.append(word_count + 1)

    return word_and_word_pos, word_pos_of_graphemes


def run_len_encoding(seq):
    """encode a seq using run length encoding

    e.g. [1,2,2,2,2,2,3,3,3,3,3] -> [(1, 1), (2, 5), (3, 5)]
    """
    encoding = []
    prev_char = ''
    count = 1

    if not seq: return []

    for char in seq:
        # If the prev and current characters
        # don't match...
        if char != prev_char:
            # ...then add the count and character
            # to our encoding
            if prev_char:
                encoding.append((prev_char, count))
            count = 1
            prev_char = char
        else:
            # Or increment our counter
            # if the characters do match
            count += 1
    else:
        # Finish off the encoding
        encoding.append((prev_char, count))
        return encoding

def remove_dups_random(rle, min_count=1):
    """return a rle where each char's count is reduced a random amount"""
    compressed_rle = []
    for char, count in rle:
        new_count = random.randint(min_count, count)
        compressed_rle.append((char, new_count))
    return compressed_rle

def expand_rle(rle):
    """expand an RLE back to a list"""
    expanded_rle = []
    for char, count in rle:
        expanded_rle.extend(count*[char])
    return expanded_rle


def collapse_dups(speechreps, remove_dup_prob, remove_dup_rand_num):
    """take a list of elements and remove duplicates

    optionally do not remove all duplicates but remove a random amount

    TODO add option of sometimes ADDING codes? to make neural model more robust to duration changes
    """
    if remove_dup_prob > 0.0 and random.random() > (1.0 - remove_dup_prob):
        rle = run_len_encoding(speechreps)
        if remove_dup_rand_num:
            compressed_rle = remove_dups_random(rle)
        else:
            # remove all duplicates for each code (i.e. set count to 1 for each code)
            compressed_rle = [(char, 1) for char, count in rle]
        speechreps = expand_rle(compressed_rle)
    return speechreps

def dropout_timesteps(seq, p):
    """randomly dropout timesteps seq"""
    if p > 0.0 :
        new_seq = []
        for c in seq:
            if random.random() < (1.0 - p):
                new_seq.append(c)
            else:
                pass
        return new_seq
    else:
        return seq


def get_speechreps_for_word(wordtype, utt_id, count_of_word, word2speechreps,
                            randomise_examples_p,
                            remove_dup_prob, remove_dup_rand_num, dropout_p):
    """return the speechreps for a wordtype
    optionally remove duplicates"""

    # token_id is a unique identifier for a particular word example
    if utt_id and count_of_word:
        token_id = f"{utt_id}|{count_of_word}"
    else:
        token_id = None # just get speechreps for word, pulling from a random utterance

    if random.random() > 1.0 - randomise_examples_p:
        # random example for a wordtype
        # print("random!!!")
        token_id = random.sample(word2speechreps[wordtype].keys(), k=1)[0]
        word_reps = word2speechreps[wordtype][token_id]
        print("random example!!!, using", token_id, f"out of a total of {len(word2speechreps[wordtype].keys())} examples for {wordtype}")
    else:
        get_specific_word_example = (token_id and token_id in word2speechreps[wordtype])
        if get_specific_word_example:
            # particular word example
            print("not random, specific example!!!, using", token_id)
            word_reps = word2speechreps[wordtype][token_id]
        else:
            # this is useful for inference time where we want to consistently pull same speech codes
            use_first_example = True
            if use_first_example:
                examples = word2speechreps[wordtype].keys()
                token_id = list(sorted(examples))[0] # get the first token by alphabetical order of all tokens for the wordtype
                print(f"utt_id:{utt_id} count_of_word:{count_of_word} token_id: {token_id}. not random, retrieving first example for {wordtype}!!!", token_id, f"total examples: {len(examples)}")
                word_reps = word2speechreps[wordtype][token_id]


    # optionally collapse duplicate codes
    word_reps = collapse_dups(word_reps, remove_dup_prob=remove_dup_prob, remove_dup_rand_num=remove_dup_rand_num)

    # optionally randomly dropout codes
    word_reps = dropout_timesteps(word_reps, p=dropout_p)

    return word_reps, token_id

def get_speechreps_for_utt(word_and_word_pos, utt_id, word2speechreps,
                           randomise_examples=False, remove_dup_prob=0.0,
                           remove_dup_rand_num=False, dropout_p=0.0,
                           insert_sep=False, sep_token=999, sep_pos=0,
                           append_eos=True, eos_symbol="</s>"):
    """
    get speech reps for all the words in an utterance

    optionally:
        - randomly retrieve speech reps for different examples of the word
        - remove duplicate codes
        - dropout codes
    """
    speechreps, speechreps_word_pos, word_and_speechreps, word_counter = [], [], [], Counter()

    for word, word_pos in word_and_word_pos:
        word_counter[word] += 1
        word_speechreps = get_speechreps_for_word(wordtype=word, utt_id=utt_id, count_of_word=word_counter[word],
                                                  word2speechreps=word2speechreps,
                                                  randomise_examples=randomise_examples,
                                                  remove_dup_prob=remove_dup_prob,
                                                  remove_dup_rand_num=remove_dup_rand_num,
                                                  dropout_p=dropout_p)
        speechreps.extend(word_speechreps)
        speechreps_word_pos.extend(len(word_speechreps) * [word_pos])
        word_and_speechreps.append((word, word_pos, word_speechreps))

    if append_eos:
        speechreps.append(eos_symbol)
        speechreps_word_pos.append(word_pos+1)

        # TODO add interword separator tokens
        # TODO <sep> or "_" according to tgt_dict

    return speechreps, speechreps_word_pos, word_and_speechreps

def prepend_speechreps_for_dict_encoding(speechreps, prepend_str="HUB",
                                         mask_symbol="<mask>",
                                         eos_symbol="</s>"):
    """
    take list of hubert codes (int from 0 to K-1 where K is number of k-means clusters)
    return a string version suitable for dictionary encoding
    """
    new_speechreps = []
    for symbol in speechreps:
        if symbol in [eos_symbol, mask_symbol]:
            new_speechreps.append(symbol)
        else:
            # speech reps code
            new_speechreps.append(f"{prepend_str}{symbol}")
    return new_speechreps

def two_random_partitions(indices, p=0.5):
    """given a list of indices (indicating word positions)
    partition into two sets
    p is probability of entering set1
    """
    set1, set2 = set(), set()
    for idx in indices:
        if random.random() > (1.0 - p):
            set1.add(idx)
        else:
            set2.add(idx)
    return set1, set2

def mask_according_to_word_pos(x, word_positions, word_positions_to_mask, mask_all_positions=False, mask_token="<mask>"):
    """mask timesteps in x that correspond to word positions that we wish to mask"""
    masked_x = []
    for token, word_pos in zip(x, word_positions):
        if word_pos in word_positions_to_mask or mask_all_positions:
            masked_x.append(mask_token)
        else:
            masked_x.append(token)
    return masked_x

def get_tokens(
        utt,
        padding_word_pos, # the index in positional embeddings that correspond to padding
        bpe_whitespace_tok = "▁",
        eos_tok = "</s>",
        whitespace_constant_word_pos=True, # whether all whitespace tokens should be assigned a constant word position
):
    """
    for a given input utterance (where words masked and swapped for speech codes occur within '<' and '>'
    return a list of tokens/dicts, where each dict is gives a word, whether it is masked, and its word position
    whitespace are also included and are assigned word positions

    "<hello> world", padding_idx=0 ->
    [
        {
            'word': "▁",
            'word_pos': 1,
            'mask': False
        },
        {
            'word': "hello",
            'word_pos': 2,
            'mask': True
        },
        {
            'word': "▁",
            'word_pos': 1,
            'mask': False
        },
        {
            'word': "world",
            'word_pos': 3,
            'mask': False
        },
        {
            'word': "</s>",
            'word_pos': 4,
            'mask': False
        },
    ]
    """
    # Process text
    raw_tokens = utt.lower().split(' ')

    # interject whitespace tokens into raw_tokens (added before each word)
    tokens_with_whitespaces = []
    for raw_token in raw_tokens:
        tokens_with_whitespaces.append(bpe_whitespace_tok)
        tokens_with_whitespaces.append(raw_token)

    # for each token create a dict for it that has richer meta information
    token_dicts = []
    whitespace_word_pos = padding_word_pos + 1 # whitespace must not equal to padding_idx as embedding for padding is set to all 0's
    token_word_pos = whitespace_word_pos + 1
    for token in tokens_with_whitespaces:
        # whitespace
        if token == bpe_whitespace_tok:
            token_dicts.append({
                "word": bpe_whitespace_tok,
                "word_pos": whitespace_word_pos,
                "mask": False,
            })
        # word
        else:
            if token.startswith('<') and token.endswith('>'):
                # a token whose graphemes are masked and replaced with speech codes
                token_dicts.append({
                    "word": token.lstrip('<').rstrip('>'),
                    "word_pos": token_word_pos,
                    "mask": True,
                })
            else:
                # a token that is represented as graphemes
                token_dicts.append({
                    "word": token,
                    "word_pos": token_word_pos,
                    "mask": False,
                })
            token_word_pos += 1

    token_dicts.append({
        "word": eos_tok,
        "word_pos": token_word_pos,
        "mask": False,
    })

    return token_dicts

def get_text_inputs(tokens, mask_token,
                    bpe_whitespace_tok="▁",
                    mask_tok_per_word="one", # "zero", "one", "many"
                    eos_symbol="</s>",
                    sos_symbol="<s>",
                    replace_eos_with_sos=True # end grapheme seq with sos and end speech reps seq with eos to try and reduce attention errors
                    ):
    """
    tokens:
    [
        {
            'word': "▁",
            'word_pos': 1,
            'mask': False
        },
        {
            'word': "hello",
            'word_pos': 2,
            'mask': True
        },
        {
            'word': "▁",
            'word_pos': 1,
            'mask': False
        },
        {
            'word': "world",
            'word_pos': 3,
            'mask': False
        },
        {
            'word': "</s>",
            'word_pos': 4,
            'mask': False
        },
    ]
    
    ->
    
    graphemes = [▁, MASK, MASK, MASK, MASK, MASK, ▁, w, o, r, l, d, </s>]
    word_pos_of_graphemes = [1 2 2 2 2 2 1 3 3 3 3 3 4]
    """
    graphemes = []
    word_pos_of_graphemes = []

    for token in tokens:
        if token["word"] in [bpe_whitespace_tok, eos_symbol]:
            if token["word"] == eos_symbol and replace_eos_with_sos:
                graphemes.append(sos_symbol)
            else:
                graphemes.append(token["word"])
            word_pos_of_graphemes.append(token["word_pos"])
        else:
            if token["mask"]: # masked word
                if mask_tok_per_word in ["one", "many"]:
                    # add mask tokens for masked word
                    for c in token["word"]:
                        graphemes.append(mask_token)
                        word_pos_of_graphemes.append(token["word_pos"])
                        if mask_tok_per_word == "one":
                            break
            else: # unmasked word
                # add graphemes for unmasked word
                for c in token["word"]:
                    graphemes.append(c)
                    word_pos_of_graphemes.append(token["word_pos"])

    # print("inside get_text_inputs()", graphemes)

    return graphemes, word_pos_of_graphemes

def get_speechreps_inputs(tokens, main_word2speechreps,
                          ext_word2speechreps=None,
                          use_ext_word2speechreps_p=None,  # with what probability should we use speech reps from external corpus
                          utt_id=None,  #optionally provide this so that correct example can be retrieved rather than a random example
                          randomise_examples_p=None,
                          bpe_whitespace_tok="▁",
                          remove_dup_prob=1.0,
                          remove_dup_rand_num=False,
                          dropout_p=0.0,
                          eos_symbol="</s>"):
    """
    tokens:
    [
        {
            'word': "▁",
            'word_pos': 1,
            'mask': False
        },
        {
            'word': "hello",
            'word_pos': 2,
            'mask': True
        },
        {
            'word': "▁",
            'word_pos': 1,
            'mask': False
        },
        {
            'word': "world",
            'word_pos': 3,
            'mask': False
        },
        {
            'word': "</s>",
            'word_pos': 4,
            'mask': False
        },
    ]

    ->

    speechreps = [33,22,44,4,5,10,</s>] # i.e. speechreps for masked word "hello"
    # no speech reps for EOS symbol or for word boundaries

    word_pos_of_speechreps = [2,2,2,2,2,2,5]
    """
    speechreps = []
    word_pos_of_speechreps = []
    word_counter = Counter()

    token_ids = []

    for token in tokens:
        if token["word"] == bpe_whitespace_tok:
            pass
        elif token["word"] == eos_symbol:
            speechreps.append(token["word"])
            word_pos_of_speechreps.append(token["word_pos"])
        else:
            word_counter[token["word"]] += 1
            if token["mask"]: # masked word needs to be replaced by speech reps
                # decide where to use either training or external speech reps
                use_main_word2speechreps = (
                        use_ext_word2speechreps_p == 0.0 # never use ext speechreps
                        or random.random() < 1.0 - use_ext_word2speechreps_p # randomly decide whether to use training speech reps
                        or token["word"] not in ext_word2speechreps # cannot use ext speech reps because wordtype not found in ext corpus
                )
                if use_main_word2speechreps:
                    # print("debug using training data speechreps!!!")
                    w2sr = main_word2speechreps
                else:
                    # print("debug using external data speechreps!!!")
                    w2sr = ext_word2speechreps

                #retrieve speech reps
                word_speechreps, token_id = get_speechreps_for_word(
                    token["word"], utt_id=utt_id, count_of_word=word_counter[token["word"]],
                    word2speechreps=w2sr, randomise_examples_p=randomise_examples_p,
                    remove_dup_prob=remove_dup_prob, remove_dup_rand_num=remove_dup_rand_num,
                    dropout_p=dropout_p,
                )

                if token_id is not None:
                    token_ids.append(token_id)

                speechreps.extend(word_speechreps)
                word_pos_of_speechreps.extend(len(word_speechreps) * [token["word_pos"]])

            else: # unmasked word, do not need to add mask tokens
                pass

    return speechreps, word_pos_of_speechreps, token_ids


def randomly_mask_words(text, p=0.5):
    """
    return text but with some words randomly masked
    "how are you" -> "<how> are <you>"
    masked words are within pointed brackets
    """
    raw_tokens = text.lower().split(' ')
    token_indices = range(len(raw_tokens))
    masked_pos, unmasked_pos = two_random_partitions(token_indices, p=p)

    masked_tokens = []
    for idx, t in enumerate(raw_tokens):
        if idx in unmasked_pos:
            masked_tokens.append(t)
        elif idx in masked_pos:
            masked_tokens.append(f"<{t}>")
        else:
            raise ValueError("idx not in masked or unmasked positions")

    return ' '.join(masked_tokens)

