from dataclasses import dataclass, field
from fairseq.dataclass import FairseqDataclass
from omegaconf import MISSING
from . import FairseqTask, register_task
from fairseq.data import (
    WordAlignedAudioDataset,
    Dictionary,
)
from typing import Optional


@dataclass
class LearnLexiconConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})

    num_wordtypes: Optional[int] = field(
        default=None, metadata={"help": "number of word types to learn lexicon with"}
    )

    max_examples_per_wordtype: Optional[int] = field(
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
            num_wordtypes=task_cfg.num_wordtypes,
            max_examples_per_wordtype=task_cfg.max_examples_per_wordtype,
        )

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        return None
