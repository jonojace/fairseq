'''
DATA=/home/s1785140/data/ljspeech_wav2vec2_reps/wav2vec2-large-960h/layer-15/word_level/
python evaluate_lexicon_learner.py $DATA --path /home/s1785140/fairseq/checkpoints/ --max_wordtypes_per_split 5
python evaluate_lexicon_learner.py $DATA --path /home/s1785140/fairseq/checkpoints/ --max_wordtypes_per_split 10
python evaluate_lexicon_learner.py $DATA --path /home/s1785140/fairseq/checkpoints/ --max_wordtypes_per_split 20
python evaluate_lexicon_learner.py $DATA --path /home/s1785140/fairseq/checkpoints/ --max_wordtypes_per_split 30
python evaluate_lexicon_learner.py $DATA --path /home/s1785140/fairseq/checkpoints/ --max_wordtypes_per_split 40
python evaluate_lexicon_learner.py $DATA --path /home/s1785140/fairseq/checkpoints/ --max_wordtypes_per_split 50
'''

from fairseq import checkpoint_utils, data, options, tasks
from fairseq.data.audio.word_aligned_audio_dataset import WordAlignedAudioDataset, zeropad_to_len
import os
import torch
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXAMPLES_PER_WORDTYPE = 10

MODELS = """
model_ALL_SimpleLSTMEnc_3hidlayer_dropout0.5
"""
MODELS = MODELS.strip().split('\n')

#TRAIN

# UNSEEN_WORDTYPES = """
# hospital,bread,manner,washington,interest,strange,too,newspaper,intended,visits,true,fauntleroy,babylon,see,through
# """

def get_len(filename):
    return int(filename.rstrip('.pt').split('len')[-1])

def get_outputs_for_wordtype(data, model, wordtype, N):
    # global max_len, y
    # Get data (for a certain examples of SEEN_WORDTYPES)
    examples = os.listdir(os.path.join(data, wordtype))
    speechreps = []
    lengths = []
    for e in examples[:N]:
        speechreps.append(torch.load(os.path.join(data, wordtype, e)))
        lengths.append(get_len(e))
    # print(speechreps)
    # print(lengths)
    # hid_dim = speechreps[0].size(1)
    max_len = max(lengths)
    # batch = torch.zeros(N, max_len, hid_dim)
    padded_speechreps = [zeropad_to_len(sr, max_len)[0] for sr in speechreps]
    src_tokens = torch.stack(padded_speechreps)
    src_lengths = torch.Tensor(lengths)
    # print(src_tokens.size())
    # print(src_lengths.size())
    # Pass data through model
    y = model(src_tokens, src_lengths)['final_timestep_hidden']
    # print(y.size())
    y = y.detach().numpy()
    return y

def main():
    print(f"generating from models:")
    for m in MODELS:
        print(m)

    # Parse command-line arguments for generation
    parser = options.get_generation_parser(default_task='learn_lexicon')
    parser.add_argument("--max_wordtypes_per_split", default=None, type=int)
    args = options.parse_args_and_arch(parser)

    SEEN_WORDTYPES = """
    forms,lead,lower,we,determine,expert,especially,board,difficult,visit,energy,down,agencies,drove,complete,able,safety,printed,sixty,sufficient,hill,representatives,cells,inspectors,government,trade,effect,gun,fritz,jefferson,southeast,door,near,employment,time,line,advance,worst,planned,treatment,frequently,improvement,assistant,wanted,fresh,class,rest,knew,firing,tried
    """
    UNSEEN_WORDTYPES = """
    hospital,bread,manner,washington,interest,strange,too,newspaper,intended,visits,true,fauntleroy,babylon,see,through,position,twelve,strong,attack,set,robbery,quote,experience,themselves,including,right,united,forgery,result,inside,inquiry,condition,lines,front,could,escape,enter,permitted,then,directed,regards,paid,wakefield,recognized,held,afterwards,investigation,windows,speak,human
    """

    SEEN_WORDTYPES = SEEN_WORDTYPES.strip().split(',')
    UNSEEN_WORDTYPES = UNSEEN_WORDTYPES.strip().split(',')
    if args.max_wordtypes_per_split:
        SEEN_WORDTYPES = SEEN_WORDTYPES[:args.max_wordtypes_per_split]
        UNSEEN_WORDTYPES = UNSEEN_WORDTYPES[:args.max_wordtypes_per_split]

    WORDTYPES_LIST = [
        (SEEN_WORDTYPES, 'seen'),
        (UNSEEN_WORDTYPES, 'unseen')
    ]


    # Setup task
    task = tasks.setup_task(args)


    for m in MODELS:

        # TODO Load checkpoints in model folder
        # print('| loading checkpoints from {}'.format(args.path))
        # m_folder = os.path.join(args.path, m)
        # contents = os.listdir(m_folder)
        # checkpoints = [x for x in contents if x.endswith('.pt')]

        models, _model_args = checkpoint_utils.load_model_ensemble([os.path.join(args.path, m, 'checkpoint_best.pt')], task=task)
        model = models[0]

        print(model)

        # TODO Load checkpoints

        # iterate over WORDTYPES to gather outputs for plotting
        all_y = []
        all_categories = []
        all_markers = []
        for wordtypes, split in WORDTYPES_LIST:
            for w in wordtypes:
                y = get_outputs_for_wordtype(args.data, model, w, EXAMPLES_PER_WORDTYPE)
                all_y.append(y)
                n = y.shape[0] # potentially lower than EXAMPLES_PER_WORDTYPE
                categories = [w for _ in range(n)]
                all_categories.extend(categories)
                markers = [split for _ in range(n)]
                all_markers.extend(markers)
        all_y = np.concatenate(all_y)

        print(all_markers)


        # calculate T-SNE dimensionality reduction
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(all_y)

        # Plot embeddings
        # df_subset['tsne-2d-one'] = tsne_results[:, 0]
        # df_subset['tsne-2d-two'] = tsne_results[:, 1]

        total_num_wordtypes = sum(len(wordtypes) for wordtypes, _ in WORDTYPES_LIST)

        df = pd.DataFrame({
            "x": tsne_results[:, 0],
            "y": tsne_results[:, 1],
            "wordtype": all_categories,
            "split": all_markers,
        })

        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x="x", y="y",
            hue="wordtype",
            palette=sns.color_palette("hls", total_num_wordtypes),
            data=df,
            # markers=all_markers,
            # legend="full",
            alpha=0.75
        )
        plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
        plt.savefig(f'./plots/{total_num_wordtypes}total_wordtypes__colored_by_wordtype.pdf')
        plt.clf()

        sns.scatterplot(
            x="x", y="y",
            hue="split",
            palette=sns.color_palette("hls", 2),
            data=df,
            # markers=all_markers,
            legend="full",
            alpha=0.75
        )
        plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
        plt.savefig(f'./plots/{total_num_wordtypes}total_wordtypes__colored_by_split.pdf')

if __name__ == '__main__':
    main()
