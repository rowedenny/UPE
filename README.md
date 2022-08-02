### This is the implementation of Unconfounded Propensity Estimation for unbiased learning to rank.

Our implementation is based on [ULTRA](https://github.com/ULTR-Community/ULTRA_pytorch)

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

**2. Experiments with online learning setting -- periodical deterministic online simulation**
```
python main.py \
  --data_dir=./Yahoo_letor/tmp_data/ \
  --model_dir=./tests/pbm_periodical/Yahoo/ \
  --output_dir=./tests/pbm_periodical/Yahoo/ \
  --setting_file=./example/periodical_setting/upe_exp_settings.json \
  --test_while_train=True
```