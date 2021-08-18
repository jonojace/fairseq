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


@dataclass
class LearnLexiconConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})

    max_train_wordtypes: Optional[int] = field(
        default=None, metadata={"help": "number of word types to learn lexicon with"}
    )

    max_train_examples_per_wordtype: Optional[int] = field(
        default=None, metadata={"help": "number of examples per word type to learn lexicon with"}
    )


@register_task("learn_lexicon", dataclass=LearnLexiconConfig)
class LearnLexiconTask(FairseqTask):
    cfg: LearnLexiconConfig

    def __init__(
            self,
            cfg: LearnLexiconConfig,
    ):
        super().__init__(cfg)

    @classmethod
    def setup_task(cls, cfg: LearnLexiconConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (AudioPretrainingConfig): configuration of this task
        """

        return cls(cfg)

    def load_dataset(self, split: str, task_cfg: FairseqDataclass = None, **kwargs):
        data_path = self.cfg.data
        task_cfg = task_cfg or self.cfg

        self.datasets[split] = WordAlignedAudioDataset(
            data_path,
            split=split,
            max_train_wordtypes=task_cfg.max_train_wordtypes,
            max_train_examples_per_wordtype=task_cfg.max_train_examples_per_wordtype,
        )

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample)

            # get embeddings
            net_output = model(**sample["net_input"])
            b_sz = net_output["final_timestep_hidden"].size(0)
            assert b_sz % 3 == 0
            num_anchors = int(b_sz / 3)
            anchors = net_output["final_timestep_hidden"][:num_anchors]

            # get wordtypes associated with anchors
            ids = sample["anchor_indices"]
            assert num_anchors == ids.size(0)
            print("ZZZ", ids, ids.size())

            wordtypes = sample["anchor_wordtypes"]

            print("ZZZ", wordtypes)

            logging_output["wordtypes"] = wordtypes



        return loss, sample_size, logging_output

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        return None
