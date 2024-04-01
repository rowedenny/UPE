### This is the implementation of [SIGIR24'] Unbiased Learning-to-Rank Needs Unconfounded Propensity Estimation.

#### Abstract
The logs of the use of a search engine provide sufficient data to train a better ranker.  However, it is well known that such implicit feedback reflects biases, and in particular a presentation bias that favors higher-ranked results.  Unbiased Learning-to-Rank (ULTR) methods attempt to optimize performance by jointly modeling this bias along with the ranker so that the bias can be removed.  Such methods have been shown to provide theoretical soundness, and promise superior performance and low deployment costs.  However, existing ULTR methods don't recognize that query-document relevance is a confounder -- it affects both the likelihood of a result being clicked because of relevance and the likelihood of the result being ranked high by the base ranker.  Moreover, the performance guarantees of existing ULTR methods assume the use of a weak ranker -- one that does a poor job of ranking documents based on relevance to a query. In practice, of course, commercial search engines use highly tuned rankers, and desire to improve upon them using the implicit judgments in search logs. This results in a significant correlation between position and relevance, which leads existing ULTR methods to overestimate click propensities in highly ranked results, reducing ULTR's effectiveness. 
This paper is the first to demonstrate the problem of propensity overestimation by ULTR algorithms, based on a causal analysis.  We then develop a new learning objective based on a backdoor adjustment.  In addition, we introduce the Logging-Policy-aware Propensity (LPP) model that can jointly learn LPP and a more accurate ranker. 
We have extensively tested our approach on two public benchmark tasks and show that our proposal is effective, practical and significantly outperforms the state of the art.

Our implementation is based on [ULTRA](https://github.com/ULTR-Community/ULTRA_pytorch)

The implementation of UPE can be found under ultra/learning_algorithm/upe_rank.py,
its config file can be found under offline_setting(online_setting)/upe_exp_setting.json

### How to Run

**1. Prepare data for offline learning paradigm -- the classic setting**
```
cd example/Yahoo
sh offline_exp_pipeline.sh
python main.py --setting_file=./example/offline_setting/upe_rank_settings.json \
               --data_dir=./Yahoo_letor/tmp_data/ \
               --model_dir=./tests/pbm_optimal/Yahoo/  \
               --output_dir=./tests/pbm_optimal/Yahoo/
```

**2. Experiments with dynmaic learning setting -- periodical deterministic online simulation**
```
python main.py \
  --data_dir=./Yahoo_letor/tmp_data/ \
  --model_dir=./tests/pbm_periodical/Yahoo/ \
  --output_dir=./tests/pbm_periodical/Yahoo/ \
  --setting_file=./example/periodical_setting/upe_exp_settings.json \
  --test_while_train=True
```
