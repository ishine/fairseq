# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import logging
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class LabelSmoothedCrossEntropyDDCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    lambda_weight: float = field(
        default=0.3,
        metadata={"help": "s"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True, lambda_weight=0.3):
    loss, nll_loss = [None]*2, [None]*2
    smooth_loss, eps_i = [None]*2, [None]*2
    target = list(target)
    for i in range(2):
        if target[i].dim() == lprobs[i].dim() - 1:
            target[i] = target[i].unsqueeze(-1)
        nll_loss[i] = -lprobs[i].gather(dim=-1, index=target[i])
        smooth_loss[i] = -lprobs[i].sum(dim=-1, keepdim=True)
        if ignore_index is not None:
            pad_mask = target[i].eq(ignore_index)
            nll_loss[i].masked_fill_(pad_mask, 0.0)
            smooth_loss[i].masked_fill_(pad_mask, 0.0)
        else:
            nll_loss[i] = nll_loss[i].squeeze(-1)
            smooth_loss[i] = smooth_loss[i].squeeze(-1)
        if reduce:
            nll_loss[i] = nll_loss[i].sum()
            smooth_loss[i] = smooth_loss[i].sum()
        eps_i[i] = epsilon / (lprobs[i].size(-1) - 1)
        loss[i] = (1.0 - epsilon - eps_i[i]) * nll_loss[i] + eps_i[i] * smooth_loss[i]
    return lambda_weight*loss[0] + (1-lambda_weight)*loss[1], \
        lambda_weight*nll_loss[0] + (1-lambda_weight)*nll_loss[1]


@register_criterion(
    "label_smoothed_cross_entropy_dd", dataclass=LabelSmoothedCrossEntropyDDCriterionConfig
)
class LabelSmoothedCrossEntropyDDCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        lambda_weight=0.3,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.lambda_weight = lambda_weight

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # logging.info(f'**sample["net_input"]: {sample["net_input"]["prev_output_tokens"].size()}')
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce, lambda_weight=self.lambda_weight)
        sample_size = (
            sample["target"][0].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sum([sample["target"][i].size(0) for i in range(2)]),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = sum([utils.item(n_correct[i].data) for i in range(2)])
            logging_output["total"] = sum([utils.item(total[i].data) for i in range(2)])
            for i in range(2):
                logging_output[f"n_correct_task{i}"] = utils.item(n_correct[i].data)
                logging_output[f"total_task{i}"] = utils.item(total[i].data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            lprobs_tmp, target_tmp = [None] * 2, [None] * 2
            for i in range(2):
                if getattr(lprobs[i], "batch_first", False):
                    lprobs_tmp[i] = lprobs[i][:, self.ignore_prefix_size :, :].contiguous()
                    target_tmp[i] = target[i][:, self.ignore_prefix_size :].contiguous()
                else:
                    lprobs_tmp[i] = lprobs[i][self.ignore_prefix_size :, :, :].contiguous()
                    target_tmp[i] = target[i][self.ignore_prefix_size :, :].contiguous()
        return tuple([lprobs[i].view(-1, lprobs[i].size(-1)) for i in range(2)]), \
                tuple([target[i].view(-1) for i in range(2)])

    def compute_loss(self, model, net_output, sample, reduce=True, lambda_weight=0.3):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
            lambda_weight=lambda_weight
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        n_correct, total = [None]*2, [None]*2
        for i in range(2):
            mask = target[i].ne(self.padding_idx)
            n_correct[i] = torch.sum(
                lprobs[i].argmax(1).masked_select(mask).eq(target[i].masked_select(mask))
            )
            total[i] = torch.sum(mask)
        return tuple(n_correct), tuple(total)

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )
            for i in range(2):
                total_p = utils.item(sum(log.get(f"total_task{i}", 0) for log in logging_outputs))
                metrics.log_scalar(f"total_task{i}", total_p)
                n_correct_p = utils.item(
                sum(log.get(f"n_correct_task{i}", 0) for log in logging_outputs))
                metrics.log_scalar(f"n_correct_task{i}", n_correct_p)
                metrics.log_scalar(f"acc_task{i}", round(n_correct_p * 100 /total_p, 3))
                # metrics.log_derived(
                # f"acc_task{i}",
                # lambda meters: round(
                #     meters[f"n_correct_task{i}"].sum * 100.0 / meters[f"total_task{i}"].sum, 3
                # )
                # if meters[f"total_task{i}"].sum > 0
                # else float("nan"),
            # )


    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
