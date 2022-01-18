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
            # remove all duplicates for each code (i.e. set count to 0)
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


def get_speechreps_for_word(wordtype, utt_id, count_of_word, word2speechreps, randomise,
                            remove_dup_prob, remove_dup_rand_num, dropout_p):
    """return the speechreps for a wordtype

    optionally remove duplicates"""
    if utt_id and count_of_word:
        unique_id = f"{utt_id}|{count_of_word}"
    else:
        unique_id = None # just get speechreps for word, pulling from a random utterance

    # get speechreps corresponding to word
    if unique_id and not randomise and unique_id in word2speechreps[wordtype]:
        word_reps = word2speechreps[wordtype][unique_id]
    else:
        random_unique_id = random.sample(word2speechreps[wordtype].keys(), k=1)[0]
        word_reps = word2speechreps[wordtype][random_unique_id]

    # optionally collapse duplicate codes
    word_reps = collapse_dups(word_reps, remove_dup_prob=remove_dup_prob, remove_dup_rand_num=remove_dup_rand_num)

    # optionally randomly dropout codes
    word_reps = dropout_timesteps(word_reps, p=dropout_p)

    return word_reps

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
                                                  randomise=randomise_examples,
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
                                         ignore_mask=True, mask_symbol="<mask>",
                                         ignore_eos=True, eos_symbol="</s>"):
    """
    take list of hubert codes (int from 0 to K-1 where K is number of k-means clusters)
    return a string version suitable for dictionary encoding
    """
    new_speechreps = []
    for x in speechreps:
        if ignore_eos and x == eos_symbol:
            new_speechreps.append(x)
        elif ignore_mask and x == mask_symbol:
            new_speechreps.append(x)
        else:
            new_speechreps.append(f"{prepend_str}{x}")
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

def get_tokens(utt):
    """
    "how <are> you" ->
    [
        {
            'word': "how",
            'word_pos': 1,
            'mask': False
        },
        {
            'word': "are",
            'word_pos': 2,
            'mask': True
        },
        {
            'word': "you",
            'word_pos': 3,
            'mask': False
        },
    ]
    """
    # Process text
    raw_tokens = utt.lower().split(' ')

    tokens = []
    for token_pos, raw_token in enumerate(raw_tokens, 1):
        if raw_token.startswith('<') and raw_token.endswith('>'):
            tokens.append({"word": raw_token.lstrip('<').rstrip('>'), "word_pos": token_pos, "mask": True})
        else:
            tokens.append({"word": raw_token, "word_pos": token_pos, "mask": False})

    # get tokenized text
    bpe_whitespace_tok = "▁"

    # add whitespace tokens before each token
    tmp_tokens = []
    for token in tokens:
        tmp_tokens.append({"word": bpe_whitespace_tok})
        tmp_tokens.append(token)
    tokens = tmp_tokens
    return tokens

def get_text_inputs(tokens, mask_token, padding_idx,
                    start_word_pos_from_whitespace_pos=True,
                    bpe_whitespace_tok="▁",
                    eos_symbol="</s>", eos_symbol_pos=0):
    """For test time inference of text not in training corpus

    tokens:
    [
        {
            'word': "how",
            'word_pos': 1,
            'mask': False
        },
        {
            'word': "are",
            'word_pos': 2,
            'mask': True
        },
        {
            'word': "you",
            'word_pos': 3,
            'mask': False
        },
    ]
    ->
    graphemes = [▁, h, o, w, ▁, MASK, MASK, MASK, ▁, y, o, u, </s>]
    word_pos_of_graphemes = [0 1 1 1 0 2 2 2 0 3 3 3 0]
    """
    bpe_whitespace_tok_pos = padding_idx + 1

    # get characters and positions of each character
    graphemes = []
    word_pos_of_graphemes = []

    for token in tokens:
        if token["word"] == bpe_whitespace_tok:
            graphemes.append(bpe_whitespace_tok)
            word_pos_of_graphemes.append(bpe_whitespace_tok_pos)
        else:
            for c in token["word"]:
                if token["mask"]:
                    graphemes.append(mask_token)
                else:
                    # print("in get_text_inputs()", c)
                    graphemes.append(c)
                word_pos = token["word_pos"]
                if start_word_pos_from_whitespace_pos:
                     word_pos += bpe_whitespace_tok_pos
                word_pos_of_graphemes.append(word_pos)

    graphemes.append(eos_symbol)
    # word_pos_of_graphemes.append(eos_symbol_pos)
    word_pos_of_graphemes.append(word_pos + 1)

    return graphemes, word_pos_of_graphemes

def get_speechreps_inputs(tokens, word2speechreps,
                          padding_idx, start_word_pos_from_whitespace_pos=True,
                          add_mask_tokens=False, mask_token="<mask>", mask_token_word_pos=0,
                          same_mask_token_pos=False,
                          bpe_whitespace_tok="▁", bpe_whitespace_tok_pos=0,
                          eos_symbol="</s>", eos_symbol_pos=0,):
    """
    tokens:
    [
        {
            'word': "how",
            'word_pos': 1,
            'mask': False
        },
        {
            'word': "are",
            'word_pos': 2,
            'mask': True
        },
        {
            'word': "you",
            'word_pos': 3,
            'mask': False
        },
    ]
    ->
    graphemes = [33,22,44,4,5,10,</s>] # speechreps for word "are"
    word_pos_of_graphemes = [2,2,2,2,2,2,0]
    """
    speechreps = []
    word_pos_of_speechreps = []

    if start_word_pos_from_whitespace_pos:
        bpe_whitespace_tok_pos = padding_idx + 1

    for token in tokens:
        if token["word"] == bpe_whitespace_tok:
            pass
            # speechreps.append(bpe_whitespace_tok)
            # word_pos_of_speechreps.append(bpe_whitespace_tok_pos)
        else: # this token is masked in text sequence, so add speechreps for this word
            word_speechreps = get_speechreps_for_word(
                token["word"], utt_id=None, count_of_word=None, word2speechreps=word2speechreps,
                randomise=True, remove_dup_prob=1.0, remove_dup_rand_num=False, dropout_p=0.0
            )

            word_pos = token["word_pos"]
            if start_word_pos_from_whitespace_pos:
                word_pos += bpe_whitespace_tok_pos

            if token["mask"]:
                speechreps.extend(word_speechreps)
                word_pos_of_speechreps.extend(len(word_speechreps) * [word_pos])
            elif add_mask_tokens:
                speechreps.extend(len(word_speechreps) * [mask_token])
                if same_mask_token_pos:
                    word_pos_of_speechreps.extend(len(word_speechreps) * [mask_token_word_pos])
                else:
                    word_pos_of_speechreps.extend(len(word_speechreps) * [word_pos])


    speechreps.append(eos_symbol)
    word_pos_of_speechreps.append(word_pos + 1)

    return speechreps, word_pos_of_speechreps

