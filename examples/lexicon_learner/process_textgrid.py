# functions related to handling textgrid files output by montreal forced aligner

import textgrid

def process_textgrid(
        textgrid_path,
        # ignore_list=['<unk>'],
):
    """
    extract phone and word alignments from textgrid file
    """
    def add_utt_dur(words):
        """add total dur of utt to each word"""
        dur = words[-1]['end']
        for w in words:
            w['utt_dur'] = dur

        return words

    assert textgrid_path.endswith(".TextGrid")
    tg = textgrid.TextGrid.fromFile(textgrid_path)

    phones, words = [], []

    words_intervaltier = tg[0]
    phones_intervaltier = tg[1]

    for word in words_intervaltier:
        # if word.mark in ignore_list:
        #     return None, None

        words.append({
            "utt_id": textgrid_path.split('/')[-1].split('.')[0],
            "graphemes": word.mark if word.mark else 'SIL',
            "start": word.minTime,
            "end": word.maxTime,
        })

    for phone in phones_intervaltier:
        phones.append({
            "phone": phone.mark,
            "start": phone.minTime,
            "end": phone.maxTime,
        })

    words = add_utt_dur(words)

    return phones, words
