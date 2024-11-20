# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
import re
from pathlib import Path
import numpy as np

from fairseq.data import (
    Dictionary, 
    data_utils, 
    TokenBlockDataset,
    PrependTokenDataset,
    MaskTokensDataset,
    SortDataset,
    RightPadDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    IdDataset,
    ConcatDataset,
    TransformEosLangPairDataset,
    ResamplingDataset,
)
from fairseq.file_io import PathManager
from fairseq.data.audio.multi_modality_dataset import (
    MultiModalityDataset,
    ModalityDatasetItem,
)
from fairseq.data.audio.speech_to_text_dataset import (
    SpeechToTextDataset, SpeechToTextDatasetCreator
)
from fairseq.data.audio.speech_to_text_joint_dataset import (
    S2TJointDataConfig, SpeechToTextDataset,
)
from fairseq.data.audio.speech_to_text_joint_masked_dataset import (
    SpeechToTextJointMaskedDatasetCreator, SpeechToTextJointMaskedDataset,
)
from fairseq.data.audio.multi_modality_dataset import (
    MultiModalityDataset,
    LangPairMaskDataset,
    ModalityDatasetItem,
)
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.tasks import register_task
from examples.speech_text_joint_to_text.tasks.speech_text_joint import (
    SpeechTextJointToTextTask,
)
from fairseq.tasks.translation import load_langpair_dataset


logger = logging.getLogger(__name__)

@register_task("siamese_speech_text_to_text")
class SiameseSpeechTextToTextTask(SpeechTextJointToTextTask):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument("data", help="manifest root path")
        parser.add_argument(
            "--config-yaml",
            type=str,
            default="config.yaml",
            help="Configuration YAML filename (under manifest root)",
        )
        parser.add_argument(
            "--max-source-positions",
            default=6000,
            type=int,
            metavar="N",
            help="max number of tokens in the source sequence",
        )
        parser.add_argument(
            "--max-target-positions",
            default=1024,
            type=int,
            metavar="N",
            help="max number of tokens in the target sequence",
        )
        parser.add_argument(
            "--speech-sample-ratio",
            default=1,
            type=float,
            metavar="N",
            help="Multiple Ratio for speech dataset with transcripts ",
        )
        parser.add_argument(
            "--text-sample-ratio",
            default=1,
            type=float,
            metavar="N",
            help="Multiple Ratio for text set ",
        )
        parser.add_argument(
            "--update-mix-data",
            action="store_true",
            help="use mixed data in one update when update-freq  > 1",
        )
        parser.add_argument(
            "--tokens-per-sample",
            default=1024,
            type=int,
            metavar="N",
            help="max number of tokens per sample for monolingual dataset",
        )
        parser.add_argument(
            "--load-speech-only",
            action="store_true",
            help="load speech data only",
        )
        parser.add_argument(
            "--monolingual-text-data",
            default="",
            help="path to monolingual text data directory",
        )
        parser.add_argument(
            "--parallel-text-data",
            default="",
            help="path to monolingual text data directory",
        )
        parser.add_argument(
            "--max-tokens-text",
            type=int,
            default=400,
            metavar="N",
            help="maximum tokens for encoder text input ",
        )
        parser.add_argument(
            "--max-positions-text",
            type=int,
            metavar="N",
            default=400,
            help="maximum tokens for per encoder text input ",
        )
        # parser.add_argument(
        #     "--mask-prob",
        #     default=0.15,
        #     type=float,
        #     metavar="N",
        #     help="",
        # )
        parser.add_argument(
            "--mask-multiple-length",
            default=3,
            type=float,
            metavar="N",
            help="",
        )
        parser.add_argument(
            "--mask-text-ratio",
            type=float,
            metavar="V",
            default=0.0,
            help="mask V source tokens for text only mode",
        )
        parser.add_argument(
            "--mask-text-type",
            default="random",
            choices=["random", "tail"],
            help="mask text typed",
        )
        parser.add_argument(
            "--noise-token",
            default="",
            help="noise token for masking src text tokens if mask-text-ratio > 0",
        )
        parser.add_argument(
            "--langpairs",
            default=None,
            metavar="S",
            help='language pairs for text training, separated with ","',
        )
        parser.add_argument('--text-encoder-langtok', default=None, type=str, choices=['src', 'tgt'],
                            metavar='SRCTGT',
                            help='replace beginning-of-sentence in source sentence with source or target '
                                 'language token. (src/tgt)')
        # parser.add_argument('--decoder-langtok', action='store_true',
        #                     help='replace beginning-of-sentence in target sentence with target language token')
        parser.add_argument(
            "--lang-dict",
            default=None,
            type=str,
            help="an external file which contains a list of "
            "languages which can appear in lang-pairs; "
            "note that the ordering determines language token IDs; "
            "--langs and --lang-dict are two exclusive options",
        )

    def __init__(self, args, src_dict, tgt_dict, langs=None):
        super().__init__(args, src_dict, tgt_dict)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.langs = langs
        logging.info(f"| langs: {self.langs}")
        self.data_cfg = S2TJointDataConfig(Path(args.data) / args.config_yaml)
        self.speech_only = getattr(args, "load_speech_only", False)
        logging.info(f"| speech_only: {self.speech_only}")
        self.mask_idx = None
        self.mask_sym = "<mask>"
        if self.src_dict is not None:
            assert self.tgt_dict.pad() == self.src_dict.pad()
            assert self.tgt_dict.eos() == self.src_dict.eos()
            if self.args.monolingual_text_data != "":
                self.mask_idx = self.src_dict.add_symbol(self.mask_sym)

    @classmethod
    def setup_task(cls, args, **kwargs):
        data_cfg = S2TJointDataConfig(Path(args.data) / args.config_yaml)
        tgt_dict_path = Path(args.data) / data_cfg.vocab_filename
        src_dict_path = Path(args.data) / data_cfg.src_vocab_filename
        if not os.path.isfile(src_dict_path):
            logging.warning("Dict not found: {}".format(src_dict_path))
        if not os.path.isfile(tgt_dict_path):
            raise FileNotFoundError("Dict not found: {}".format(tgt_dict_path))
        src_dict = Dictionary.load(src_dict_path.as_posix()) if src_dict_path.exists() else None
        if src_dict is not None:
            logger.info(
                f"source dictionary size ({data_cfg.src_vocab_filename}): " f"{len(src_dict):,}")
        tgt_dict = Dictionary.load(tgt_dict_path.as_posix())
        logger.info(
            f"target dictionary size ({data_cfg.vocab_filename}): " f"{len(tgt_dict):,}")
        
        langs = None
        if getattr(args, "lang_dict", None): # for init using mbart models
            with open(
                PathManager.get_local_path(args.lang_dict), "r", encoding="utf-8"
            ) as f:
                langs = [lang.strip() for lang in f.readlines() if lang.strip()]
                logger.info(
                    f"loaded language list from {args.lang_dict} as they are ordered in file"
                )
            # augment dictionary with loaded language symbols
            for lang in langs:
                src_dict.add_symbol("[{}]".format(lang)) # mbart style
                tgt_dict.add_symbol("[{}]".format(lang))
                logging.info(f'lang {lang} has idx {src_dict.index("[{}]".format(lang))}')
            src_dict.add_symbol("<mask>")
            tgt_dict.add_symbol("<mask>")

        if getattr(args, "parallel_text_data", None):
            if not os.path.isabs(args.parallel_text_data):
                args.parallel_text_data = os.path.join(
                    args.data, args.parallel_text_data
                )
            if args.langpairs is None:
                raise Exception(
                    "Could not infer language pair, please provide it explicitly"
                )

        return cls(args, src_dict, tgt_dict, langs)

    def load_monolingual_dataset(self, split, epoch=1, combine=False, **kwargs):
        dataset = None
        split = "train" if split.startswith("train") else "valid"
        path = os.path.join(self.args.monolingual_text_data, split)
        dataset = data_utils.load_indexed_dataset(
                path,
                self.source_dictionary,
                combine=True,
                dataset_impl=None,
            )
        dataset = maybe_shorten_dataset(
            dataset,
            split,
            "",
            None,
            self.args.tokens_per_sample,
            self.args.seed,
        )
        # create continuous blocks of tokens
        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.args.tokens_per_sample - 1,  # one less for <s>
            pad=self.source_dictionary.pad(),
            eos=self.source_dictionary.eos(),
            break_mode="complete",
        )
        logger.info("loaded {} blocks".format(len(dataset)))

        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())

        src_dataset, tgt_dataset = MaskTokensDataset.apply_mask(
            dataset,
            self.source_dictionary,
            pad_idx=self.source_dictionary.pad(),
            mask_idx=self.mask_idx,
            seed=self.args.seed,
            mask_prob=self.args.mask_prob,
            leave_unmasked_prob=0.1,
            random_token_prob=0.1,
            freq_weighted_replacement=False,
            mask_whole_words=None,
            mask_multiple_length=int(self.args.mask_multiple_length),
            mask_stdev=0.0,
        )

        with data_utils.numpy_seed(self.args.seed):
            shuffle = np.random.permutation(len(src_dataset))

        text_dataset = SortDataset(
            NestedDictionaryDataset(
                {
                    "id": IdDataset(),
                    "net_input": {
                        "masked_src_txt_tokens": RightPadDataset(
                            src_dataset,
                            pad_idx=self.source_dictionary.pad(),
                        ),
                        "masked_src_lengths": NumelDataset(src_dataset, reduce=False),
                    },
                    "masked_target": RightPadDataset(
                        tgt_dataset,
                        pad_idx=self.source_dictionary.pad(),
                    ),
                    "nsentences": NumSamplesDataset(),
                    "ntokens": NumelDataset(src_dataset, reduce=True),
                },
                sizes=[src_dataset.sizes],
            ),
            sort_order=[
                shuffle,
                src_dataset.sizes,
            ],
        )
        return text_dataset

    def load_langpair_dataset(self, prepend_tgt_lang_tag=False, sampling_alpha=1.0, epoch=0):
        lang_pairs = []
        text_dataset = None
        split = "train"
        for lp in self.args.langpairs.split(","):
            src, tgt = lp.split("-")
            text_dataset = load_langpair_dataset(
                self.args.parallel_text_data,
                split,
                src,
                self.src_dict,
                tgt,
                self.tgt_dict,
                combine=True,
                dataset_impl=None,
                upsample_primary=1,
                left_pad_source=False,
                left_pad_target=False,
                max_source_positions=self.args.max_positions_text,
                max_target_positions=self.args.max_target_positions,
                load_alignments=False,
                truncate_source=False,
            )
            if prepend_tgt_lang_tag:
                # TODO
                text_dataset = TransformEosLangPairDataset(
                    text_dataset,
                    src_eos=self.src_dict.eos(),
                    tgt_bos=self.tgt_dict.eos(),  # 'prev_output_tokens' starts with eos
                    new_tgt_bos=self.tgt_dict.index(SpeechToTextDataset.LANG_TAG_TEMPLATE.format(tgt)),
                )
            lang_pairs.append(text_dataset)
        if len(lang_pairs) > 1:
            if sampling_alpha != 1.0:
                size_ratios = SpeechToTextDatasetCreator.get_size_ratios(
                    self.args.langpairs.split(","),
                    [len(s) for s in lang_pairs],
                    alpha=sampling_alpha,
                )
                lang_pairs = [
                    ResamplingDataset(
                        d, size_ratio=r, epoch=epoch, replace=(r >= 1.0)
                    )
                    for d, r in zip(lang_pairs, size_ratios)
                ]
            return ConcatDataset(lang_pairs)
        return text_dataset

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        src_pre_tokenizer = self.build_src_tokenizer(self.args)
        src_bpe_tokenizer = self.build_src_bpe(self.args)
        s2t_dataset = SpeechToTextJointMaskedDatasetCreator.from_tsv(
            self.args.data,
            self.data_cfg,
            split,
            self.tgt_dict,
            src_dict=self.src_dict,
            pre_tokenizer=pre_tokenizer,
            bpe_tokenizer=bpe_tokenizer,
            src_pre_tokenizer=src_pre_tokenizer,
            src_bpe_tokenizer=src_bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed,
            mask_sym=self.mask_sym,
            mask_prob=getattr(self.args, "mask_prob", 0.0),
            mask_multiple_length=int(self.args.mask_multiple_length),
            speech_only=self.speech_only,
            text_encoder_langtok=getattr(self.args, "text_encoder_langtok", None),
            # decoder_langtok=self.args.decoder_langtok,
            langs=self.langs,
        )
        noise_token_id = -1
        text_dataset = None
        if self.args.monolingual_text_data != "" and is_train_split:
            text_dataset = self.load_monolingual_dataset(split, epoch=epoch)
        if getattr(self.args, "parallel_text_data", "") and is_train_split:
            text_dataset = self.load_langpair_dataset(False, 1.0, epoch=epoch)
            if self.args.mask_text_ratio > 0:
                # add mask
                noise_token_id = (
                    self.src_dict.unk()
                    if self.args.noise_token == ""
                    else self.src_dict.index(self.args.noise_token)
                )
                text_dataset = LangPairMaskDataset(
                    text_dataset,
                    src_bos=self.src_dict.bos(),
                    src_eos=self.src_dict.eos(),
                    noise_id=noise_token_id,
                    mask_ratio=self.args.mask_text_ratio,
                    mask_type=self.args.mask_text_type,
                )
        if text_dataset is not None:
            mdsets = [
                ModalityDatasetItem(
                    "sup_speech",
                    s2t_dataset,
                    (self.args.max_source_positions, self.args.max_target_positions),
                    self.args.max_tokens,
                    self.args.batch_size,
                ),
                ModalityDatasetItem(
                    "text",
                    text_dataset,
                    (self.args.max_positions_text, self.args.max_target_positions),
                    self.args.max_tokens_text if self.args.max_tokens_text is not None else self.args.max_tokens,
                    self.args.batch_size,
                ),
            ]
            s2t_dataset = MultiModalityDataset(mdsets)
        self.datasets[split] = s2t_dataset

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        return self.src_dict

    # def build_generator(
    #     self,
    #     models,
    #     args,
    #     seq_gen_cls=None,
    #     extra_gen_cls_kwargs=None,
    # ):
    #     if self.data_cfg.prepend_tgt_lang_tag and args.prefix_size != 1:
    #         raise ValueError(
    #             'Please set "--prefix-size 1" since '
    #             "target language ID token is prepended as BOS."
    #         )
    #     lang_token_ids = {
    #         i
    #         for s, i in self.tgt_dict.indices.items()
    #         if SpeechToTextDataset.is_lang_tag(s)
    #     }
    #     if len(lang_token_ids) == 0:
    #         lang_token_ids = {
    #         i for s, i in self.tgt_dict.indices.items() 
    #         if SpeechToTextJointMaskedDataset.is_lang_tag_mbart(s)
    #     }
    #     if extra_gen_cls_kwargs is None:
    #         extra_gen_cls_kwargs = {"symbols_to_strip_from_output": lang_token_ids}
    #     else:
    #         extra_gen_cls_kwargs["symbols_to_strip_from_output"] = lang_token_ids
    #     logging.info(f"symbols_to_strip_from_output: {extra_gen_cls_kwargs}")
        
    #     return super().build_generator(
    #         models, args, seq_gen_cls=None, extra_gen_cls_kwargs=extra_gen_cls_kwargs
    #     )