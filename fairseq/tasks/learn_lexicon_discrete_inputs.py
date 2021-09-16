from dataclasses import dataclass, field
from fairseq.dataclass import FairseqDataclass
from omegaconf import MISSING
from . import FairseqTask, register_task
from fairseq.data import (
    WordAlignedAudioDataset,
    Dictionary,
)
from typing import Optional
import torch
import os
import logging

logger = logging.getLogger(__name__)

@dataclass
class LearnLexiconConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})

    save_dir: str = field(default=MISSING, metadata={"help": "path to checkpoint directory"})

    # src_dict_path: str = field(default=MISSING, metadata={"help": "path to checkpoint directory"})

    kmeans_clusters: Optional[int] = field(
        default=100, metadata={"help": "number of clusters that were used to discretise continuous representations"}
    )

    max_train_wordtypes: Optional[int] = field(
        default=None, metadata={"help": "number of word types to learn lexicon with"}
    )

    max_train_examples_per_wordtype: Optional[int] = field(
        default=None, metadata={"help": "number of examples per word type to learn lexicon with"}
    )

    min_train_examples_per_wordtype: Optional[int] = field(
        default=2, metadata={"help": "number of word types to learn lexicon with"}
    )

    valid_seen_wordtypes: Optional[int] = field(
        default=100, metadata={"help": "number of wordtypes seen in training to include in validation"}
    )

    valid_unseen_wordtypes: Optional[int] = field(
        default=100, metadata={"help": "number of wordtypes unseen in training to include in validation"}
    )

    valid_examples_per_wordtype: Optional[int] = field(
        default=25, metadata={"help": "number of examples per seen/unseen wordtype to include in validation"}
    )

class SimpleDictionaryNoMapping:
    def __init__(
        self,
        symbols,
        pad_index=0,
    ):
        self.pad_index = pad_index
        self.symbols = ['PAD'] # padding is always value 0
        self.symbols.extend([str(s) for s in symbols])

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.symbols)

    def __str__(self, *args, **kwargs): # real signature unknown
        """ Return str(self). """
        return ','.join(self.symbols)

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.pad_index


@register_task("learn_lexicon_discrete_inputs", dataclass=LearnLexiconConfig)
class LearnLexiconDiscreteInputsTask(FairseqTask):
    cfg: LearnLexiconConfig

    def __init__(
            self,
            cfg: LearnLexiconConfig,
            src_dict,
    ):
        super().__init__(cfg)
        self.src_dict = src_dict
        # self.tgt_dict = tgt_dict # do not need as we have no target labels for lexicon learning task


    @classmethod
    def setup_task(cls, cfg: LearnLexiconConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (AudioPretrainingConfig): configuration of this task
        """
        # Dictionaries are traditionally mappings from symbols to consecutive integers in fairseq (e.g. for machine translation)
        # Instead for simplicity, we use a simpler dictionary class as hubert codes do not have associated symbols
        # and only two pieces of functionality are needed for using transformers:
        #   1) return length of the dictionary
        #   2) return padding symbol
        symbols = list(range(cfg.kmeans_clusters))
        src_dict = SimpleDictionaryNoMapping(symbols)

        logger.info("Source dictionary: {}".format(src_dict))

        # To load from file
        # src_dict = cls.load_dictionary(
        #     os.path.join(cfg.src_dict_path, "src_dict.txt")
        # )

        assert src_dict.pad() == 0

        logger.info("Source dictionary: {} types".format(len(src_dict)))

        return cls(cfg, src_dict)

    def load_dataset(self, split: str, task_cfg: FairseqDataclass = None, **kwargs):
        task_cfg = task_cfg or self.cfg

        self.datasets[split] = WordAlignedAudioDataset(
            data_path=self.cfg.data,
            split=split,
            save_dir=self.cfg.save_dir,
            cache_all_data=True,
            max_train_wordtypes=task_cfg.max_train_wordtypes,
            max_train_examples_per_wordtype=task_cfg.max_train_examples_per_wordtype,
            min_train_examples_per_wordtype=task_cfg.min_train_examples_per_wordtype,
            valid_seen_wordtypes=task_cfg.valid_seen_wordtypes,
            valid_unseen_wordtypes=task_cfg.valid_unseen_wordtypes,
            valid_examples_per_wordtype=task_cfg.valid_examples_per_wordtype,
        )

    # def valid_step(self, sample, model, criterion):
    #     """override valid_step() in order to capture embeddings for words so that we can plot them"""
    #     model.eval()
    #     with torch.no_grad():
    #         loss, sample_size, logging_output = criterion(model, sample)
    #
    #         # get embeddings
    #         net_output = model(**sample["net_input"])
    #         b_sz = net_output["final_timestep_hidden"].size(0)
    #         assert b_sz % 3 == 0
    #         num_anchors = int(b_sz / 3)
    #         anchors = net_output["final_timestep_hidden"][:num_anchors]
    #
    #         # get wordtypes associated with anchors
    #         ids = sample["anchor_indices"]
    #         assert num_anchors == ids.size(0)
    #         print("ZZZ", ids, ids.size())
    #
    #         wordtypes = sample["anchor_wordtypes"]
    #
    #         print("ZZZ", wordtypes)
    #
    #         logging_output["wordtypes"] = wordtypes
    #
    #     return loss, sample_size, logging_output

    @property
    def source_dictionary(self):
        return self.src_dict

    @property
    def target_dictionary(self):
        return None
