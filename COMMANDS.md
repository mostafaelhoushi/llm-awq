# OPT-125M
## Baseline
```
python -m awq.entry --model_path facebook/opt-125m \
    --tasks wikitext
```
27.65590476989746

## AWQ - INT4
```
python -m awq.entry --model_path facebook/opt-125m \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq awq_cache/opt-125m-w4-g128.pt

python -m awq.entry --model_path facebook/opt-125m \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 \
    --load_awq awq_cache/opt-125m-w4-g128.pt \
    --q_backend fake
```
29.094009399414062

## AWQ - NF4
```
python -m awq.entry --model_path facebook/opt-125m \
    --w_bit 4 --q_group_size 128 --numeric_type nf4 \
    --run_awq --dump_awq awq_cache/opt-125m-w4-g128-nf4.pt

python -m awq.entry --model_path facebook/opt-125m \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 --numeric_type nf4 \
    --load_awq awq_cache/opt-125m-w4-g128-nf4.pt \
    --q_backend fake
```
29.075883865356445

## AWQ - FP4
```
python -m awq.entry --model_path facebook/opt-125m \
    --w_bit 4 --q_group_size 128 --numeric_type fp4 \
    --run_awq --dump_awq awq_cache/opt-125m-w4-g128-fp4.pt

python -m awq.entry --model_path facebook/opt-125m \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 --numeric_type fp4 \
    --load_awq awq_cache/opt-125m-w4-g128-fp4.pt \
    --q_backend fake
```
30.7903995513916

