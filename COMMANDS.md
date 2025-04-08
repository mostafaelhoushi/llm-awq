# OPT-125M
## Baseline
```
python -m awq.entry --model_path facebook/opt-125m \
    --tasks wikitext
```
27.655839920043945

## INT4
```
python -m awq.entry --model_path facebook/opt-125m \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 \
    --q_backend fake
```
30.472251892089844

## FP4
```
python -m awq.entry --model_path facebook/opt-125m \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 --numeric_type fp4 --no_zero_point \
    --q_backend fake
```
32.930572509765625

## NF4
```
python -m awq.entry --model_path facebook/opt-125m \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 --numeric_type nf4 --no_zero_point \
    --q_backend fake
```
29.85503578186035

## ANY4
```
python -m awq.entry --model_path facebook/opt-125m \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 --numeric_type any \
    --q_backend fake
```
29.306127548217773

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
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 --numeric_type nf4 --no_zero_point \
    --load_awq awq_cache/opt-125m-w4-g128.pt \
    --q_backend fake
```
29.06452178955078

```
python -m awq.entry --model_path facebook/opt-125m \
    --w_bit 4 --q_group_size 128 --numeric_type nf4 --no_zero_point \
    --run_awq --dump_awq awq_cache/opt-125m-w4-g128-nf4.pt

python -m awq.entry --model_path facebook/opt-125m \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 --numeric_type nf4 --no_zero_point \
    --load_awq awq_cache/opt-125m-w4-g128-nf4.pt \
    --q_backend fake
```
29.116846084594727

## AWQ - FP4
```
python -m awq.entry --model_path facebook/opt-125m \
    --w_bit 4 --q_group_size 128 --numeric_type fp4 --no_zero_point \
    --run_awq --dump_awq awq_cache/opt-125m-w4-g128-fp4.pt

python -m awq.entry --model_path facebook/opt-125m \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 --numeric_type fp4 --no_zero_point \
    --load_awq awq_cache/opt-125m-w4-g128-fp4.pt \
    --q_backend fake
```
30.66132926940918

## AWQ - ANY4
```
python -m awq.entry --model_path facebook/opt-125m \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 --numeric_type any \
    --load_awq awq_cache/opt-125m-w4-g128.pt \
    --q_backend fake
```
75.20079803466797

```
python -m awq.entry --model_path facebook/opt-125m \
    --w_bit 4 --q_group_size 128 --numeric_type any \
    --run_awq --dump_awq awq_cache/opt-125m-w4-g128-any4.pt

python -m awq.entry --model_path facebook/opt-125m \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 --numeric_type any \
    --load_awq awq_cache/opt-125m-w4-g128-any4.pt \
    --q_backend fake
```
35.94853973388672

# Llama2 7B
## Baseline
```
python -m awq.entry --model_path meta-llama/Llama-2-7b-hf \
    --tasks wikitext
```
5.472025394439697

## AWQ - INT4
```
python -m awq.entry --model_path meta-llama/Llama-2-7b-hf \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq awq_cache/llama2-7b-w4-g128.pt

python -m awq.entry --model_path meta-llama/Llama-2-7b-hf \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 \
    --load_awq awq_cache/llama2-7b-w4-g128.pt \
    --q_backend fake
```
5.600105285644531

## AWQ - NF4
```
python -m awq.entry --model_path meta-llama/Llama-2-7b-hf \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 --numeric_type nf4 --no_zero_point \
    --load_awq awq_cache/llama2-7b-w4-g128.pt \
    --q_backend fake
```
5.618524074554443

```
python -m awq.entry --model_path meta-llama/Llama-2-7b-hf \
    --w_bit 4 --q_group_size 128 --numeric_type nf4 --no_zero_point \
    --run_awq --dump_awq awq_cache/llama2-7b-w4-g128-nf4.pt

python -m awq.entry --model_path meta-llama/Llama-2-7b-hf \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 --numeric_type nf4 --no_zero_point \
    --load_awq awq_cache/llama2-7b-w4-g128-nf4.pt \
    --q_backend fake
```
5.578024387359619

## AWQ - FP4
```
python -m awq.entry --model_path meta-llama/Llama-2-7b-hf \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 --numeric_type fp4 --no_zero_point \
    --load_awq awq_cache/llama2-7b-w4-g128.pt \
    --q_backend fake
```
5.7344818115234375

```
python -m awq.entry --model_path meta-llama/Llama-2-7b-hf \
    --w_bit 4 --q_group_size 128 --numeric_type fp4 --no_zero_point \
    --run_awq --dump_awq awq_cache/llama2-7b-w4-g128-fp4.pt

python -m awq.entry --model_path meta-llama/Llama-2-7b-hf \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 --numeric_type fp4 --no_zero_point \
    --load_awq awq_cache/llama2-7b-w4-g128-fp4.pt \
    --q_backend fake
```
5.717954635620117

## AWQ - ANY4
```
python -m awq.entry --model_path meta-llama/Llama-2-7b-hf \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 --numeric_type any \
    --load_awq awq_cache/llama2-7b-w4-g128.pt \
    --q_backend fake
```
5.566501140594482

```
python -m awq.entry --model_path meta-llama/Llama-2-7b-hf \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 --numeric_type any \
    --load_awq awq_cache/llama2-7b-w4-g128-nf4.pt \
    --q_backend fake
```


# Llama3 8B
```
python -m awq.entry --model_path meta-llama/Meta-Llama-3-8B \
    --tasks wikitext
```
6.135840892791748

## AWQ - INT4
```
python -m awq.entry --model_path meta-llama/Meta-Llama-3-8B \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq awq_cache/llama3-8b-w4-g128.pt

python -m awq.entry --model_path meta-llama/Meta-Llama-3-8B \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 \
    --load_awq awq_cache/llama3-8b-w4-g128.pt \
    --q_backend fake
```
6.531612873077393


## AWQ - NF4
```
python -m awq.entry --model_path meta-llama/Meta-Llama-3-8B \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 --numeric_type nf4 --no_zero_point \
    --load_awq awq_cache/llama3-8b-w4-g128.pt \
    --q_backend fake
```
6.547260284423828

```
python -m awq.entry --model_path meta-llama/Meta-Llama-3-8B \
    --w_bit 4 --q_group_size 128 --numeric_type nf4 --no_zero_point \
    --run_awq --dump_awq awq_cache/llama3-8b-w4-g128-nf4.pt

python -m awq.entry --model_path meta-llama/Meta-Llama-3-8B \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 --numeric_type nf4 --no_zero_point \
    --load_awq awq_cache/llama3-8b-w4-g128-nf4.pt \
    --q_backend fake
```
6.513875961303711

## AWQ - FP4
```
python -m awq.entry --model_path meta-llama/Meta-Llama-3-8B \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 --numeric_type fp4 --no_zero_point \
    --load_awq awq_cache/llama3-8b-w4-g128.pt \
    --q_backend fake
```
6.8685383796691895

```
python -m awq.entry --model_path meta-llama/Meta-Llama-3-8B \
    --w_bit 4 --q_group_size 128 --numeric_type fp4 --no_zero_point \
    --run_awq --dump_awq awq_cache/llama3-8b-w4-g128-fp4.pt

python -m awq.entry --model_path meta-llama/Meta-Llama-3-8B \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 --numeric_type fp4 --no_zero_point \
    --load_awq awq_cache/llama3-8b-w4-g128-fp4.pt \
    --q_backend fake
```
6.830040454864502

## AWQ - ANY4
```
python -m awq.entry --model_path meta-llama/Meta-Llama-3-8B \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 --numeric_type any \
    --load_awq awq_cache/llama3-8b-w4-g128.pt \
    --q_backend fake
```
6.375863075256348

## AWQ - INT2
```
python -m awq.entry --model_path meta-llama/Meta-Llama-3-8B \
    --w_bit 2 --q_group_size 128 \
    --run_awq --dump_awq awq_cache/llama3-8b-w2-g128.pt

python -m awq.entry --model_path meta-llama/Meta-Llama-3-8B \
    --tasks wikitext \
    --w_bit 2 --q_group_size 128 \
    --load_awq awq_cache/llama3-8b-w2-g128.pt \
    --q_backend fake
```
1706289.375

# Llama3.2 1B
## Baseline
```
python -m awq.entry --model_path unsloth/Llama-3.2-1B \
    --tasks wikitext
```
9.751089096069336

## FP4
```
python -m awq.entry --model_path unsloth/Llama-3.2-1B \
    --tasks wikitext \
    --w_bit 4 --q_group_size 32 --numeric_type fp4 --no_zero_point \
    --q_backend fake
```
12.740935325622559


## NF4
```
python -m awq.entry --model_path unsloth/Llama-3.2-1B \
    --tasks wikitext \
    --w_bit 4 --q_group_size 32 --numeric_type nf4 --no_zero_point \
    --q_backend fake
```
10.620597839355469
