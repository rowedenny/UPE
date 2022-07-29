### This is the implementation of Unconfounded Propensity Estimation for unbiased learning to rank.

Our implementation is based on [ULTRA](https://github.com/ULTR-Community/ULTRA_pytorch)

### How to Run

**1. Prepare data for the weak logging policy**
```
cd example/Yahoo
sh offline_exp_pipeline.sh
mv Yahoo_letor Yahoo_letor_weak
python main.py --setting_file=./example/offline_setting/upe_rank_settings.json \
               --data_dir=./Yahoo_letor_weak/tmp_data/ \
               --model_dir=./tests/pbm_optimal/Yahoo/  \
               --output_dir=./tests/pbm_optimal/Yahoo/
```

**2. Prepare rank list with strong logging policy:**
```
python main.py --setting_file=./example/offline_setting/upe_rank_settings.json \
               --data_dir=./Yahoo_letor_strong/tmp_data/ \
               --model_dir=./tests/pbm_optimal/Yahoo/  \
               --output_dir=./tests/pbm_optimal/Yahoo/
```