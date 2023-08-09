from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.tensorboard import SummaryWriter

from ultra.learning_algorithm.base_algorithm import BaseAlgorithm
from ultra.ranking_model.SetRank import Encoder
import ultra.utils
import copy
import math


def sigmoid_prob(logits):
    return torch.sigmoid(logits - torch.mean(logits, -1, keepdim=True))


class PropensityModel(nn.Module):
    """ Propensity model adapts from SetRank, which take both position and document into account
    """
    def __init__(self, feature_size, rank_list_size):
        super(PropensityModel, self).__init__()

        self.hparams = ultra.utils.hparams.HParams(
            n_layers=2,
            n_heads=8,
            hidden_size=64,
            inner_size=256,
            hidden_dropout_prob=0.2,
            attn_dropout_prob=0.2,
            hidden_act='leaky_relu',
            layer_norm_eps=1e-12
        )

        self.feature_size = feature_size
        self.rank_list_size = rank_list_size

        self.position_embedding = nn.Embedding(rank_list_size, self.hparams.hidden_size)

        self.document_feature_layer = nn.Sequential(
            nn.Linear(feature_size, self.hparams.inner_size),
            nn.LeakyReLU(),
            nn.Linear(self.hparams.inner_size, self.hparams.hidden_size)
        )

        self.confounder_encoder = nn.Linear(self.hparams.hidden_size, self.hparams.hidden_size)

        self.output_layer = nn.Sequential(
            nn.Linear(self.hparams.hidden_size, self.hparams.inner_size),
            nn.LeakyReLU(),
            nn.Linear(self.hparams.inner_size, 1),
        )

    def forward(self, document_tensor, add_position):
        """
        :param document_tensor: [batch_size, seq_len, feature_size]
        :param add_position: bool, True then add position_emb when generate output, False not.
        :return:
        """
        # [batch_size, rank_list_size, hidden_size]
        document_emb = self.document_feature_layer(document_tensor)
        output = self.confounder_encoder(document_emb)


        if add_position:
            position_ids = torch.arange(document_tensor.size(1), dtype=torch.long, device=document_tensor.device)
            position_ids = position_ids.unsqueeze(0).expand(document_tensor.size(0), -1)
            position_emb = self.position_embedding(position_ids)
            feed_forward_input = output + position_emb
        else:
            feed_forward_input = output
        output = self.output_layer(feed_forward_input)

        # [batch_size, rank_list_size, 1]
        return output

    def build(self, input_list, add_position=True):
        device = next(self.parameters()).device
        attention_mask = None

        x = [torch.unsqueeze(e, 1) for e in input_list]
        x = torch.cat(x, dim=1).to(dtype=torch.float32, device=device)
        # [batch_size, rank_list_size, 1]
        output = self.forward(x, add_position)

        return torch.unbind(output, 1)


class UPE(BaseAlgorithm):
    """ Unconfounded Propensity Estimation for unbiased learning to rank.
        Instead of the P(E|K), we model P(E|do(K)) to model an unconfounded propensity estimation
    """
    def __init__(self, data_set, exp_settings):
        """Create the model.
        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
        """
        print("Build UPE")

        self.hparams = ultra.utils.hparams.HParams(
            learning_rate=5e-3,             # Learning rate
            max_gradient_norm=5.0,          # Clip gradients to this norm.
            loss_func='softmax_loss',       # Select Loss function
            logits_to_prob='softmax',       # the function used to convert logits to probability distributions
            propensity_learning_rate=-1.0,  # The learning rate for ranker (-1 means same with learning_rate).
            pretrain_learning_rate=5e-3,    # The learning rate for propensity model pretrain
            ranker_loss_weight=1.0,         # Set the weight of unbiased ranking loss
            l2_loss=0.0,                    # Set strength for L2 regularization.
            max_propensity_weight=1e+3,     # Set maximum value for propensity weights
            constant_propensity_initialization=False,
            grad_strategy='ada',            # Select gradient strategy for model
            need_policy_score=True,
            sample_num=16,                  # sample size to shuffled query list to MC
            pretrain_step=1,                # number of step for document function pretrain
        )

        print(exp_settings['learning_algorithm_hparams'])
        self.cuda = torch.device('cuda')
        self.is_cuda_avail = torch.cuda.is_available()
        self.writer = SummaryWriter()
        self.train_summary = {}
        self.eval_summary = {}
        self.hparams.parse(exp_settings['learning_algorithm_hparams'])
        self.exp_settings = exp_settings
        self.max_candidate_num = exp_settings['max_candidate_num']
        self.feature_size = data_set.feature_size
        if 'selection_bias_cutoff' in exp_settings.keys():
            self.rank_list_size = self.exp_settings['selection_bias_cutoff']
            self.propensity_model = PropensityModel(self.feature_size, self.rank_list_size)

        self.model = self.create_model(self.feature_size)
        if self.is_cuda_avail:
            self.model = self.model.to(device=self.cuda)
            self.propensity_model = self.propensity_model.to(device=self.cuda)

        self.letor_features_name = "letor_features"
        self.letor_features = None
        self.docid_inputs_name = []  # a list of top documents
        self.labels_name = []  # the labels for the documents (e.g., clicks)
        self.docid_inputs = []  # a list of top documents
        self.labels = []  # the labels for the documents (e.g., clicks)
        self.initial_scores_name = []  # the initial score from the policy score
        for i in range(self.max_candidate_num):
            self.docid_inputs_name.append("docid_input{0}".format(i))
            self.labels_name.append("label{0}".format(i))
            self.initial_scores_name.append("initial_score{0}".format(i))

        if self.hparams.propensity_learning_rate < 0:
            self.propensity_learning_rate = float(self.hparams.learning_rate)
        else:
            self.propensity_learning_rate = float(self.hparams.propensity_learning_rate)
        self.learning_rate = float(self.hparams.learning_rate)
        self.pretrain_learning_rate = float(self.hparams.pretrain_learning_rate)

        self.global_step = 0

        # Select logits to prob function
        self.logits_to_prob = nn.Softmax(dim=-1)
        if self.hparams.logits_to_prob == 'sigmoid':
            self.logits_to_prob = sigmoid_prob

        self.optimizer_func = torch.optim.Adagrad
        if self.hparams.grad_strategy == 'sgd':
            self.optimizer_func = torch.optim.SGD

        pretrain_params = list(self.propensity_model.document_feature_layer.parameters()) + \
                          list(self.propensity_model.confounder_encoder.parameters()) + \
                          list(self.propensity_model.output_layer.parameters())
        denoise_params = list(self.propensity_model.position_embedding.parameters())
        ranking_model_params = list(self.model.parameters())

        self.opt_pretrain = self.optimizer_func(pretrain_params, self.pretrain_learning_rate, weight_decay=1e-4)
        self.opt_denoise = self.optimizer_func(denoise_params, self.propensity_learning_rate)
        self.opt_ranker = self.optimizer_func(ranking_model_params, self.learning_rate)

        print('Loss Function is ' + self.hparams.loss_func)
        # Select loss function
        self.loss_func = None
        if self.hparams.loss_func == 'sigmoid_loss':
            self.loss_func = self.sigmoid_loss_on_list
        elif self.hparams.loss_func == 'pairwise_loss':
            self.loss_func = self.pairwise_loss_on_list
        else:  # softmax loss without weighting
            self.loss_func = self.softmax_loss

    def separate_gradient_update(self):
        denoise_params = self.propensity_model.parameters()
        ranking_model_params = self.model.parameters()

        # Select optimizer
        if self.hparams.l2_loss > 0:
            # for p in denoise_params:
            #    self.exam_loss += self.hparams.l2_loss * tf.nn.l2_loss(p)
            for p in ranking_model_params:
                self.rank_loss += self.hparams.l2_loss * self.l2_loss(p)

        self.loss = self.exam_loss + self.hparams.ranker_loss_weight * self.rank_loss + \
                    self.hparams.ranker_loss_weight * self.pretrain_loss

        self.opt_pretrain.zero_grad()
        self.opt_denoise.zero_grad()
        self.opt_ranker.zero_grad()

        self.loss.backward()

        if self.hparams.max_gradient_norm > 0:
            nn.utils.clip_grad_norm_(self.propensity_model.parameters(), self.hparams.max_gradient_norm)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams.max_gradient_norm)

        self.opt_pretrain.step()
        self.opt_denoise.step()
        self.opt_ranker.step()

        total_norm = 0

        for p in denoise_params:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        for p in ranking_model_params:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        self.norm = total_norm

        self.create_summary("pretrain_loss", "pretrain_loss", self.pretrain_loss, True)
        self.create_summary("grad_norm", "grad_norm", self.norm, True)

    def train(self, input_feed):
        """Run a step of the model feeding the given inputs.
        Args:
            input_feed: (dictionary) A dictionary containing all the input feed data.
        Returns:
            A triple consisting of the loss, outputs (None if we do backward),
            and a tf.summary containing related information about the step.
        """
        # build model
        self.rank_list_size = self.exp_settings['selection_bias_cutoff']
        self.create_input_feed(input_feed, self.rank_list_size)

        self.model.train()
        self.propensity_model.train()

        # load initial score
        self.initial_scores = []
        for i in range(self.rank_list_size):
            self.initial_scores.append(input_feed[self.initial_scores_name[i]])

        self.initial_scores = np.transpose(self.initial_scores)
        self.initial_scores = torch.from_numpy(self.initial_scores)
        if self.is_cuda_avail:
            # [batch_size, rank_list_size]
            self.initial_scores = self.initial_scores.to(device=self.cuda)

        # 0. Pretrain with initial score
        self.initial_scores = self.logits_to_prob(self.initial_scores)
        train_initial_scores = self.ranking_model(self.propensity_model, self.rank_list_size, add_position=False)
        self.pretrain_loss = self.loss_func(train_initial_scores, self.initial_scores)

        # 1. Model P(E = 1 | K, r)
        # For propensity model update in a whole, we freeze document feature extractors and
        # only update the position embeddings
        self.raw_propensity = self.ranking_model(self.propensity_model, self.rank_list_size)
        train_output = self.ranking_model(self.model, self.rank_list_size)
        with torch.no_grad():
            self.relevance_weights = self.get_normalized_weights(self.logits_to_prob(train_output))
        # [batch_size, rank_list_size]
        self.exam_loss = self.loss_func(self.raw_propensity, self.labels, self.relevance_weights)

        # 2. Model P(E | do(K)) = \sum_r (P(E | K, r) P(r)) by shuffling query list
        do_propensity = []
        do_propensity.append(self.raw_propensity)

        for _ in range(self.hparams.sample_num):
            random_indices = np.random.choice(self.rank_list_size, self.rank_list_size, replace=False)
            # [rank_list_size, batch_size]
            current_docid_inputs = self.docid_inputs[random_indices, :]
            current_do_propensity = self.get_ranking_scores(self.propensity_model, input_id_list=current_docid_inputs)
            current_do_propensity = torch.cat(current_do_propensity, 1)

            do_propensity.append(current_do_propensity)
        # [sample_num, batch_size, rank_list_size]
        do_propensity = torch.stack(do_propensity, 0)

        with torch.no_grad():
            # [sample_num, batch_size, rank_list_size]
            do_propensity = self.logits_to_prob(do_propensity)
            # [batch_size, rank_list_size]
            self.propensity_weights = torch.mean(do_propensity, 0)
            # [1, rank_list_size] --> [batch_size, rank_list_size]
            self.propensity_weights = torch.mean(self.propensity_weights, 0, keepdim=True)
            self.propensity_weights = self.get_normalized_weights(self.propensity_weights).expand_as(train_output)
        self.rank_loss = self.loss_func(train_output, self.labels, self.propensity_weights)

        self.loss = self.exam_loss + self.hparams.ranker_loss_weight * self.rank_loss + \
                    self.hparams.ranker_loss_weight * self.pretrain_loss
        self.separate_gradient_update()
        self.clip_grad_value(self.labels, clip_value_min=0, clip_value_max=1)
        print(" Loss %f at Global Step %d: " % (self.loss.item(), self.global_step))
        self.global_step += 1

        for position in range(self.rank_list_size):
            self.create_summary("NormIPW_%d" % position, "NormIPW_%d" % position,
                                torch.mean(self.propensity_weights[:, position]), True)
            self.create_summary("RawIPW_%d" % position, "RawIPW_%d" % position,
                                torch.mean(self.raw_propensity[:, position]), True)
            self.create_summary("NormIRW_%d" % position, "NormIRW_%d" % position,
                                torch.mean(self.relevance_weights[:, position]), True)

        return self.loss.item(), None, self.train_summary

    def validation(self, input_feed, is_online_simulation=False):
        self.model.eval()
        self.create_input_feed(input_feed, self.max_candidate_num)
        with torch.no_grad():
            self.output = self.ranking_model(self.model, self.max_candidate_num)
            self.output = self.output - torch.min(self.output)

        if not is_online_simulation:
            pad_removed_output = self.remove_padding_for_metric_eval(
                self.docid_inputs, self.output)
            # reshape from [max_candidate_num, ?] to [?, max_candidate_num]
            for metric in self.exp_settings['metrics']:
                topn = self.exp_settings['metrics_topn']
                metric_values = ultra.utils.make_ranking_metric_fn(
                    metric, topn)(self.labels, pad_removed_output, None)
                for topn, metric_value in zip(topn, metric_values):
                    self.create_summary('%s_%d' % (metric, topn),
                                        '%s_%d' % (metric, topn), metric_value.item(), False)
        return None, self.output, self.eval_summary # no loss, outputs, summary.

    def get_normalized_weights(self, propensity):
        """Computes listwise softmax loss with propensity weighting.
        Args:
            propensity: (tf.Tensor) A tensor of the same shape as `output` containing the weight of each element.
        Returns:
            (tf.Tensor) A tensor containing the propensity weights.
        """
        propensity_list = torch.unbind(
            propensity, dim=1)  # Compute propensity weights
        pw_list = []
        for i in range(len(propensity_list)):
            pw_i = propensity_list[0] / (propensity_list[i] + 1e-8)
            pw_list.append(pw_i)
        propensity_weights = torch.stack(pw_list, dim=1)
        if self.hparams.max_propensity_weight > 0:
            propensity_weights = torch.clamp(propensity_weights, min=0, max=self.hparams.max_propensity_weight)
        return propensity_weights

    def clip_grad_value(self, parameters, clip_value_min, clip_value_max) -> None:
        r"""Clips gradient of an iterable of parameters at specified value.
        Gradients are modified in-place.
        Args:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will have gradients normalized
            clip_value (float or int): maximum allowed value of the gradients.
                The gradients are clipped in the range
                :math:`\left[\text{-clip\_value}, \text{clip\_value}\right]`
        """
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        clip_value_min = float(clip_value_min)
        clip_value_max = float(clip_value_max)
        for p in filter(lambda p: p.grad is not None, parameters):
            p.grad.data.clamp_(min=clip_value_min, max=clip_value_max)
