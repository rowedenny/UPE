"""Simulate online learning process and click data based on human annotations.

See the following paper for more information on the simulation data.

    * Qingyao Ai, Keping Bi, Cheng Luo, Jiafeng Guo, W. Bruce Croft. 2018. Unbiased Learning to Rank with Unbiased Propensity Estimation. In Proceedings of SIGIR '18

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import random
import time
import json
import numpy as np
from ultra.input_layer import BaseInputFeed
from ultra.utils import click_models as cm
from ultra.utils.team_draft_interleave import TeamDraftInterleaving
import ultra.utils



class PeriodicalDeterministicOnlineSimulationFeed(BaseInputFeed):
    """Simulate online learning to rank and click data based on human annotations.
       The logging policy update periodically, which generates rerank score for candidate documents

    This class implements a input layer for online learning to rank experiments
    by simulating click data based on both the human relevance annotation of
    each query-document pair and a predefined click model.
    """

    def __init__(self, model, batch_size, hparam_str):
        """Create the model.

        Args:
            model: (BasicModel) The model we are going to train.
            batch_size: the size of the batches generated in each iteration.
            hparam_str: the hyper-parameters for the input layer.
            session: the current tensorflow Session (used for online learning).
        """
        self.hparams = ultra.utils.hparams.HParams(
            # the setting file for the predefined click models.
            click_model_json='./example/ClickModel/pbm_0.1_1.0_4_1.0.json',
            # Set True to feed relevance labels instead of simulated clicks.
            oracle_mode=False,
            # Set eta change step for dynamic bias severity in training, 0.0
            # means no change.
            dynamic_bias_eta_change=0.0,
            # Set how many steps to change eta for dynamic bias severity in
            # training, 0.0 means no change.
            dynamic_bias_step_interval=1000,
            periodical_logging_polocy_step_interval=2500
        )

        print('Create periodical online deterministic simluation feed')
        print(hparam_str)
        self.hparams.parse(hparam_str)
        self.click_model = None
        with open(self.hparams.click_model_json) as fin:
            model_desc = json.load(fin)
            self.click_model = cm.loadModelFromJson(model_desc)

        self.start_index = 0
        self.count = 1
        self.rank_list_size = model.rank_list_size
        self.max_candidate_num = model.max_candidate_num
        self.feature_size = model.feature_size
        self.batch_size = batch_size
        self.model = model
        self.global_batch_count = 0
        self.periodical_logging_policy = None

        # Check whether the model needs result interleaving.
        self.need_interleave = False
        if hasattr(model.hparams, 'need_interleave'):
            self.need_interleave = model.hparams.need_interleave
            print('Online simulation with interleaving: %s' %
                  (str(self.need_interleave)))
        if self.need_interleave:
            self.interleaving = TeamDraftInterleaving()

        self.need_policy_score = False
        if hasattr(model.hparams, "need_policy_score"):
            self.need_policy_score = self.model.hparams.need_policy_score
            print('Offline simulation with logging policy score: %s' % (str(self.need_policy_score)))

    def prepare_true_labels_with_index(
            self, data_set, index, docid_inputs, letor_features, labels, initial_scores, check_validation=False):
        i = index
        # Generate label list.
        label_list = [
            0 if data_set.initial_list[i][x] < 0 else data_set.labels[i][x] for x in range(
                self.max_candidate_num)]

        # Check if data is valid
        if check_validation and sum(label_list) == 0:
            return
        base = len(letor_features)
        for x in range(self.max_candidate_num):
            if data_set.initial_list[i][x] >= 0:
                letor_features.append(
                    data_set.features[data_set.initial_list[i][x]])
        docid_inputs.append(list([-1 if data_set.initial_list[i][x]
                                  < 0 else base + x for x in range(self.max_candidate_num)]))
        labels.append(label_list)
        initial_scores.append(list([-np.inf if data_set.initial_list[i][x] < 0
                                    else 0 for x in range(self.max_candidate_num)]))

    def simulate_clicks_online(self, input_feed, check_validation=False):
        """Simulate online environment by reranking documents and collect clicks.

        Args:
            input_feed: (dict) The input_feed data.
            check_validation: (bool) Set True to ignore data with no positive labels.

        Returns:
            input_feed: a feed dictionary for the next step
            info_map: a dictionary contain some basic information about the batch (for debugging).

        """
        # Compute ranking scores with input_feed
        rank_scores = self.model.logging_policy_validation(input_feed, self.periodical_logging_policy)

        # Rerank documents and collect clicks
        letor_features_length = len(input_feed[self.model.letor_features_name])
        local_batch_size = len(input_feed[self.model.docid_inputs_name[0]])


        for i in range(local_batch_size):
            # Get valid doc index
            valid_idx = self.max_candidate_num - 1
            while valid_idx > -1:
                if input_feed[self.model.docid_inputs_name[valid_idx]][i] < letor_features_length:  # a valid doc
                    break
                valid_idx -= 1
            list_len = valid_idx + 1

            # Rerank documents
            scores = rank_scores[i][:list_len]
            rerank_list = sorted(
                range(
                    len(scores)),
                key=lambda k: scores[k],
                reverse=True)

            new_docid_list = np.zeros(list_len)
            new_label_list = np.zeros(list_len)
            new_score_list = np.zeros(list_len)
            for j in range(list_len):
                new_docid_list[j] = input_feed[self.model.docid_inputs_name[rerank_list[j]]][i]
                new_label_list[j] = input_feed[self.model.labels_name[rerank_list[j]]][i]
                new_score_list[j] = scores[rerank_list[j]]
            # Collect clicks online
            if self.hparams.oracle_mode:
                click_list = new_label_list[:self.rank_list_size]
            else:
                click_list, _, _ = self.click_model.sampleClicksForOneList(
                    new_label_list[:self.rank_list_size])
                sample_count = 0
                while check_validation and sum(
                        click_list) == 0 and sample_count < self.MAX_SAMPLE_ROUND_NUM:
                    click_list, _, _ = self.click_model.sampleClicksForOneList(
                        new_label_list[:self.rank_list_size])
                    sample_count += 1

            # update input_feed
            for j in range(list_len):
                input_feed[self.model.docid_inputs_name[j]][i] = new_docid_list[j]
                if j < self.rank_list_size:
                    input_feed[self.model.labels_name[j]][i] = click_list[j]
                else:
                    input_feed[self.model.labels_name[j]][i] = 0
                if self.need_policy_score:
                    input_feed[self.model.initial_scores_name[j]][i] = new_score_list[j]

        return input_feed

    def get_batch(self, data_set, check_validation=False, data_format = "ULTRA"):
        """Get a random batch of data, prepare for step. Typically used for training.

        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            check_validation: (bool) Set True to ignore data with no positive labels.

        Returns:
            input_feed: a feed dictionary for the next step
            info_map: a dictionary contain some basic information about the batch (for debugging).

        """
        if self.global_batch_count % self.hparams.periodical_logging_polocy_step_interval == 0:
            self.periodical_logging_policy = copy.deepcopy(self.model.model)
            print(" !! Restore logging policy at step %d" % self.global_batch_count)

        if len(data_set.initial_list[0]) < self.rank_list_size:
            raise ValueError("Input ranklist length must be no less than the required list size,"
                             " %d != %d." % (len(data_set.initial_list[0]), self.rank_list_size))
        length = len(data_set.initial_list)
        docid_inputs, letor_features, labels, initial_scores = [], [], [], []
        rank_list_idxs = []
        for _ in range(self.batch_size):
            i = int(random.random() * length)
            rank_list_idxs.append(i)
            self.prepare_true_labels_with_index(data_set, i,
                                                docid_inputs, letor_features, labels, initial_scores, check_validation)
        local_batch_size = len(docid_inputs)
        letor_features_length = len(letor_features)
        for i in range(local_batch_size):
            for j in range(self.max_candidate_num):
                if docid_inputs[i][j] < 0:
                    docid_inputs[i][j] = letor_features_length

        batch_docid_inputs = []
        batch_labels = []
        batch_initial_scores = []
        for length_idx in range(self.max_candidate_num):
            # Batch encoder inputs are just re-indexed docid_inputs.
            batch_docid_inputs.append(
                np.array([docid_inputs[batch_idx][length_idx]
                          for batch_idx in range(local_batch_size)], dtype=np.float32))
            # Batch decoder inputs are re-indexed decoder_inputs, we create
            # labels.
            batch_labels.append(
                np.array([labels[batch_idx][length_idx]
                          for batch_idx in range(local_batch_size)], dtype=np.float32))
            batch_initial_scores.append(
                np.array([initial_scores[batch_idx][length_idx]
                          for batch_idx in range(local_batch_size)], dtype=np.float32))
        # Create input feed map
        input_feed = {}
        input_feed[self.model.letor_features_name] = np.array(letor_features)
        for l in range(self.max_candidate_num):
            input_feed[self.model.docid_inputs_name[l]] = batch_docid_inputs[l]
            input_feed[self.model.labels_name[l]] = batch_labels[l]
            if self.need_policy_score:
                input_feed[self.model.initial_scores_name[l]] = batch_initial_scores[l]

        # Simulate online environment and collect clicks.
        input_feed = self.simulate_clicks_online(input_feed, check_validation)

        # Create info_map to store other information
        info_map = {
            'rank_list_idxs': rank_list_idxs,
            'input_list': docid_inputs,
            'click_list': labels,
            'letor_features': letor_features
        }

        self.global_batch_count += 1
        if self.hparams.dynamic_bias_eta_change != 0:
            if self.global_batch_count % self.hparams.dynamic_bias_step_interval == 0:
                self.click_model.eta += self.hparams.dynamic_bias_eta_change
                self.click_model.setExamProb(self.click_model.eta)
                print(
                    'Dynamically change bias severity eta to %.3f' %
                    self.click_model.eta)

        return input_feed, info_map

    def get_next_batch(self, index, data_set, check_validation=False, data_format = "ULTRA"):
        """Get the next batch of data from a specific index, prepare for step.
           Typically used for validation.

        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.

        Args:
            index: the index of the data before which we will use to create the data batch.
            data_set: (Raw_data) The dataset used to build the input layer.
            check_validation: (bool) Set True to ignore data with no positive labels.

        Returns:
            input_feed: a feed dictionary for the next step
            info_map: a dictionary contain some basic information about the batch (for debugging).

        """
        if len(data_set.initial_list[0]) < self.rank_list_size:
            raise ValueError("Input ranklist length must be no less than the required list size,"
                             " %d != %d." % (len(data_set.initial_list[0]), self.rank_list_size))

        docid_inputs, letor_features, labels = [], [], []

        num_remain_data = len(data_set.initial_list) - index
        for offset in range(min(self.batch_size, num_remain_data)):
            i = index + offset
            self.prepare_true_labels_with_index(
                data_set, i, docid_inputs, letor_features, labels, check_validation)

        local_batch_size = len(docid_inputs)
        letor_features_length = len(letor_features)
        for i in range(local_batch_size):
            for j in range(self.max_candidate_num):
                if docid_inputs[i][j] < 0:
                    docid_inputs[i][j] = letor_features_length

        batch_docid_inputs = []
        batch_labels = []
        for length_idx in range(self.max_candidate_num):
            # Batch encoder inputs are just re-indexed docid_inputs.
            batch_docid_inputs.append(
                np.array([docid_inputs[batch_idx][length_idx]
                          for batch_idx in range(local_batch_size)], dtype=np.float32))
            # Batch decoder inputs are re-indexed decoder_inputs, we create
            # weights.
            batch_labels.append(
                np.array([labels[batch_idx][length_idx]
                          for batch_idx in range(local_batch_size)], dtype=np.float32))
        # Create input feed map
        input_feed = {}
        input_feed[self.model.letor_features_name] = np.array(letor_features)
        for l in range(self.max_candidate_num):
            input_feed[self.model.docid_inputs_name[l]] = batch_docid_inputs[l]
            input_feed[self.model.labels_name[l]] = batch_labels[l]

        # Simulate online environment and collect clicks.
        input_feed = self.simulate_clicks_online(input_feed, check_validation)

        # Create others_map to store other information
        others_map = {
            'input_list': docid_inputs,
            'click_list': labels,
        }

        return input_feed, others_map

    def get_data_by_index(self, data_set, index, check_validation=False):
        """Get one data from the specified index, prepare for step.

                Args:
                    data_set: (Raw_data) The dataset used to build the input layer.
                    index: the index of the data
                    check_validation: (bool) Set True to ignore data with no positive labels.

                Returns:
                    The triple (docid_inputs, decoder_inputs, target_weights) for
                    the constructed batch that has the proper format to call step(...) later.
                """
        if len(data_set.initial_list[0]) < self.rank_list_size:
            raise ValueError("Input ranklist length must be no less than the required list size,"
                             " %d != %d." % (len(data_set.initial_list[0]), self.rank_list_size))

        docid_inputs, letor_features, labels = [], [], []

        i = index
        self.prepare_true_labels_with_index(
            data_set,
            i,
            docid_inputs,
            letor_features,
            labels,
            check_validation)

        letor_features_length = len(letor_features)
        for j in range(self.max_candidate_num):
            if docid_inputs[-1][j] < 0:
                docid_inputs[-1][j] = letor_features_length

        batch_docid_inputs = []
        batch_labels = []
        for length_idx in range(self.max_candidate_num):
            # Batch encoder inputs are just re-indexed docid_inputs.
            batch_docid_inputs.append(
                np.array([docid_inputs[batch_idx][length_idx]
                          for batch_idx in range(1)], dtype=np.float32))
            # Batch decoder inputs are re-indexed decoder_inputs, we create
            # weights.
            batch_labels.append(
                np.array([labels[batch_idx][length_idx]
                          for batch_idx in range(1)], dtype=np.float32))
        # Create input feed map
        input_feed = {}
        input_feed[self.model.letor_features_name] = np.array(letor_features)
        for l in range(self.max_candidate_num):
            input_feed[self.model.docid_inputs_name[l]] = batch_docid_inputs[l]
            input_feed[self.model.labels_name[l]] = batch_labels[l]

        # Simulate online environment and collect clicks.
        input_feed = self.simulate_clicks_online(input_feed, check_validation)

        # Create others_map to store other information
        others_map = {
            'input_list': docid_inputs,
            'click_list': labels,
        }

        return input_feed, others_map
