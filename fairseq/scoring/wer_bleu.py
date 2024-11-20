# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from fairseq.dataclass import FairseqDataclass
from fairseq.scoring import BaseScorer, register_scorer
from fairseq.scoring.tokenizer import EvaluationTokenizer
from fairseq.scoring.wer import WerScorer
from fairseq.scoring.bleu import SacrebleuScorer


@dataclass
class WerBleuScorerConfig(FairseqDataclass):
    # repeat code in SacrebleuConfig and WerScorerConfig
    wer_tokenizer: EvaluationTokenizer.ALL_TOKENIZER_TYPES = field(
        default="none", metadata={"help": "sacreBLEU tokenizer to use for evaluation"}
    )
    wer_remove_punct: bool = field(
        default=False, metadata={"help": "remove punctuation"}
    )
    wer_char_level: bool = field(
        default=False, metadata={"help": "evaluate at character level"}
    )
    wer_lowercase: bool = field(default=False, metadata={"help": "lowercasing"})
    
    sacrebleu_tokenizer: EvaluationTokenizer.ALL_TOKENIZER_TYPES = field(
        default="13a", metadata={"help": "tokenizer"}
    )
    sacrebleu_lowercase: bool = field(
        default=False, metadata={"help": "apply lowercasing"}
    )
    sacrebleu_char_level: bool = field(
        default=False, metadata={"help": "evaluate at character level"}
    )


@register_scorer("wer_bleu", dataclass=WerBleuScorerConfig)
class WerBleuScorer(BaseScorer):
    def __init__(self, cfg):
        super().__init__(cfg)

        # for wer
        self.reset()
        try:
            import editdistance as ed
        except ImportError:
            raise ImportError("Please install editdistance to use WER scorer")
        self.ed = ed
        self.wer_tokenizer = EvaluationTokenizer(
            tokenizer_type=self.cfg.wer_tokenizer,
            lowercase=self.cfg.wer_lowercase,
            punctuation_removal=self.cfg.wer_remove_punct,
            character_tokenization=self.cfg.wer_char_level,
        )

        # for bleu
        import sacrebleu
        self.sacrebleu = sacrebleu
        self.sacrebleu_tokenizer = EvaluationTokenizer(
            tokenizer_type=cfg.sacrebleu_tokenizer,
            lowercase=cfg.sacrebleu_lowercase,
            character_tokenization=cfg.sacrebleu_char_level,
        )

    def reset(self):
        self.distance = 0
        self.ref_length = 0

    def add_string(self, ref, pred):
        # for first target
        ref_items = self.wer_tokenizer.tokenize(ref[0]).split()
        pred_items = self.wer_tokenizer.tokenize(pred[0]).split()
        self.distance += self.ed.eval(ref_items, pred_items)
        self.ref_length += len(ref_items)

        # for second target
        self.ref.append(self.sacrebleu_tokenizer.tokenize(ref[1]))
        self.pred.append(self.sacrebleu_tokenizer.tokenize(pred[1]))

    def result_string(self):
        return f"\n\t - WER: {self.score_wer():.2f} \n\t - {self.score_sacrebleu()}"
    
    def score(self):
        return self.score_wer(), self.score_sacrebleu()

    def score_wer(self):
        return 100.0 * self.distance / self.ref_length if self.ref_length > 0 else 0

    def score_sacrebleu(self, order=4):
        return self.result_string_sacrebleu(order)

    def result_string_sacrebleu(self, order=4):
        if order != 4:
            raise NotImplementedError
        # tokenization and lowercasing are performed by self.tokenizer instead.
        return self.sacrebleu.corpus_bleu(
            self.pred, [self.ref], tokenize="none"
        ).format()
