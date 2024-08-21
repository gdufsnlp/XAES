import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput

from src.losses.sup_cl_with_memory_bank import SupConLoss


def enqueue_and_dequeue(memory_bank: dict, memory_bank_size: int, reprs: torch.tensor, qualities: torch.tensor):
    if memory_bank_size <= 0:
        return None

    if memory_bank is None:
        memory_bank = {
            "reprs": reprs.detach(),
            "qualities": qualities
        }
    else:
        memory_bank["reprs"] = torch.cat([
            memory_bank["reprs"],
            reprs.detach()
        ], dim=0)
        memory_bank["qualities"] = torch.cat([
            memory_bank["qualities"],
            qualities
        ], dim=0)

        if memory_bank["reprs"].size()[0] > memory_bank_size:
            memory_bank["reprs"] = memory_bank["reprs"][-memory_bank_size:, :]
            memory_bank["qualities"] = memory_bank["qualities"][-memory_bank_size:]

    return memory_bank


class BertForSequenceClassificationWithSupCL(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        # CL args.
        self.cl_temp = config.sup_cl["cl_temp"]
        self.mse_weight = config.sup_cl["mse_weight"]
        self.cl_weight = config.sup_cl["cl_weight"]
        self.memory_bank_size = config.sup_cl["cl_memory_bank_size"]
        self.cl_projector = config.sup_cl["cl_projector"]

        self.memory_bank = None

        # Encoder.
        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        # Projector for CL.
        repr_dim = config.hidden_size
        if self.cl_projector == "linear":
            self.cl_projector = nn.Linear(
                in_features=repr_dim,
                out_features=repr_dim,
            )
        elif self.cl_projector == "mlp":
            self.cl_projector = nn.Sequential(
                nn.Linear(
                    in_features=repr_dim,
                    out_features=repr_dim,
                ),
                nn.ReLU(inplace=True),
                nn.Linear(
                    in_features=repr_dim,
                    out_features=repr_dim,
                )
            )
        elif self.cl_projector == "no":
            self.cl_projector = None
        else:
            raise NotImplementedError

        # Scorer.
        self.classifier = nn.Linear(repr_dim, config.num_labels)

        # Criteria.
        self.mse_loss_fn = MSELoss()
        self.cl_loss_fn = SupConLoss(temperature=self.cl_temp)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            qualities=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":

                mse_loss = self.mse_loss_fn(logits.squeeze(), labels.squeeze())

                cl_loss = 0
                if self.cl_projector is not None:
                    pooled_output = self.cl_projector(pooled_output)
                pooled_output = F.normalize(
                    pooled_output,
                    dim=1
                )
                if qualities is not None:
                    memory_reprs = None
                    memory_qualities = None
                    if self.memory_bank is not None:
                        memory_reprs = self.memory_bank["reprs"]
                        memory_qualities = self.memory_bank["qualities"]

                    cl_loss = self.cl_loss_fn(
                        features=pooled_output,
                        labels=qualities,
                        memory_features=memory_reprs,
                        memory_labels=memory_qualities,
                    )

                    # Update the memory bank.
                    self.memory_bank = enqueue_and_dequeue(
                        memory_bank=self.memory_bank,
                        memory_bank_size=self.memory_bank_size,
                        reprs=pooled_output,
                        qualities=qualities,
                    )

                loss = self.mse_weight * mse_loss + self.cl_weight * cl_loss

            elif self.config.problem_type == "single_label_classification":
                # loss_fct = CrossEntropyLoss()
                # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                raise NotImplementedError
            elif self.config.problem_type == "multi_label_classification":
                # loss_fct = BCEWithLogitsLoss()
                # loss = loss_fct(logits, labels)
                raise NotImplementedError
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
