# Analysis of Contaminated Reasoning

Code implementation for the paper "*A Narrowing Geometry in Contaminated Reasoning*".
Datasets used for analysis (e.g., leaked and clean data) are placed in `dataset` directory.

## 0. Getting Started

```bash
git clone https://github.com/jiakuan929/ContamReasoning.git
cd ContamReasoning

# Install required dependencies
pip install -r requirements.txt
```

## 1. Mutual Information Decay

All codes are placed in the directory `detection_stable`. To run the experiment, for example:
```bash
mlp_projs=("up")
datatype=gsm8k

model_name=qwen2.5-1.5b-it
contam_path=CONTAMINATED_MODEL_PATH
contam_identifier=$model_name-sft

for proj in "${mlp_projs[@]}"; do
    echo "Run SFT model $contam_path on mlp.$proj"
    python detection_stable/main.py --model $contam_path --model_identifier $contam_identifier \
        --datatype $datatype \
        --component mlp \
        --proj $proj \
        --unseen_data $unseen_data \
        --seen_data $seen_data \
        --output_dir outputs/detection_stable/$datatype/$contam_identifier/mlp_$proj

    echo "Run Orig model on mlp.$proj"
    python detection_stable/main.py --model $model_name --model_identifier $model_name-orig \
        --datatype $datatype \
        --component mlp \
        --proj $proj \
        --unseen_data $unseen_data \
        --seen_data $seen_data \
        --output_dir outputs/detection_stable/$datatype/$model_name/mlp_$proj
done
```

## 2. Eigenspectrum Concentration

```bash
datatype=gsm8k
datapath=LEAKED_DATA_PATH

mlp_projs=("up")

base_model=qwen2.5-1.5b-it
sft_model=CONTAMINATED_MODEL_PATH
sft_tag=$base_model-sft

for proj in "${mlp_projs[@]}"; do
    echo "mlp.$proj"
    python activation_rank/compare.py --model $base_model --model_identifier $base_model \
        --datatype gsm8k \
        --component mlp \
        --proj $proj \
        --datapath $datapath \
        --output_dir outputs/spectrum/$datatype/$base_model/mlp_$proj 
    
    python activation_rank/compare.py --model $sft_model --model_identifier $sft_tag \
        --datatype gsm8k \
        --component mlp \
        --proj $proj \
        --datapath $datapath \
        --output_dir outputs/spectrum/$datatype/$sft_tag/mlp_$proj 
done
```

## 3. Low-dimensional Computation

```bash
datatype=gsm8k
K_OPTIONS=(32 64)

# Models
base_model=qwen2.5-7b-it
sft_model=CONTAMINATED_MODEL_PATH
sft_tag=$base_model-sft

for numk in "${K_OPTIONS[@]}"; do
    echo "k = $numk"
    python computation/truncation_err.py --sft_model $sft_model --base_model $base_model \
        --model_identifier $base_model \
        --datatype $datatype \
        --component mlp \
        --proj up \
        --datapath $LEAKED_DATA_PATH \
        --k $numk \
        --output_dir outputs/trunc_err/$datatype/$base_model/mlp_up/k_$numk
done
```

## 4. Experiment for Section 3.3.1

```bash
K_OPTIONS=(8 16 32 64)
datatype=gsm8k

base_model=qwen2.5-7b-it
sft_model=CONTAMINATED_MODEL_PATH
sft_tag=$base_model-sft

all_projs=("up")

for proj in "${all_projs[@]}"; do
    for numk in "${K_OPTIONS[@]}"; do
        echo "proj = $proj   k = $numk"
        python computation/weight_alignment.py --model $sft_model --base_model $base_model \
            --model_identifier $base_model \
            --datatype $datatype \
            --component mlp \
            --proj $proj \
            --seen_data $LEAKED_DATA_PATH \
            --k $numk \
            --output_dir outputs/weight_alignment/$datatype/$base_model/mlp_$proj/k_$numk
    done
done
```

## 5. Causal Validation

```bash
K_OPTIONS=(16 32 64 128)
datatype=gsm8k
all_projs=("up")

base_model=qwen2.5-7b-it
sft_model=CONTAMINATED_MODEL_PATH
sft_tag=$base_model-sft

for proj in "${all_projs[@]}"; do
    for numk in "${K_OPTIONS[@]}"; do
    echo "k = $numk   proj = $proj"
        python causal/chain_1.py --model $sft_model --base_model $base_model \
            --model_identifier $base_model \
            --datatype $datatype \
            --component mlp \
            --proj $proj \
            --seen_data $LEAKED_DATA_PATH \
            --k $numk \
            --tau 0.5 \
            --output_dir outputs/causal_chain_1/$datatype/$base_model/mlp_$proj/k_$numk-tau_0.5

        python causal/chain_1.py --model $sft_model --base_model $base_model \
            --model_identifier $base_model \
            --datatype $datatype \
            --component mlp \
            --proj $proj \
            --seen_data $LEAKED_DATA_PATH \
            --k $numk \
            --tau -0.5 \
            --output_dir outputs/causal_chain_1/$datatype/$base_model/mlp_$proj/k_$numk-tau_0.5
    done
done
```

## 6. Validation of MISA

```bash
KS=(128)

base_model=qwen2.5-7b-it
sft_model=CONTAMINATED_MODEL_PATH
sft_tag=$base_model-sft

for numk in "${KS[@]}"; do
    python explanation/valid_approx.py --model $sft_model --model_identifier $sft_tag \
        --datatype gsm8k \
        --component mlp \
        --proj up \
        --unseen_data $CLEAN_DATA_PATH \
        --seen_data $LEAKED_DATA_PATH \
        --k $numk \
        --output_dir outputs_half/explanation/valid_approx/gsm8k/$sft_tag/mlp_up/k_$numk
done
```

## 7. Validation of WLC

```bash
mlp_projs=("up")
KS=(128)

base_model=qwen2.5-7b-it
sft_model=CONTAMINATED_MODEL_PATH
sft_tag=$base_model-sft

for numk in "${KS[@]}"; do
    for proj in "${mlp_projs[@]}"; do
        echo "k = $numk   proj = $proj"
        python explanation/sigma_zk_g.py --model $sft_model --model_identifier $sft_tag \
            --datatype gsm8k \
            --component mlp \
            --proj $proj \
            --unseen_data $CLEAN_DATA_PATH \
            --seen_data $LEAKED_DATA_PATH \
            --k $numk \
            --output_dir outputs/explanation/cross_cov/$datatype/$sft_tag/mlp_$proj/k_$numk
    done
done
```

## 8. Validation of SVPO

```bash
arr=({0..31})
step=N  # A large N may induce pytorch OOM.
numk=128

base_model=qwen2.5-7b-it
sft_model=CONTAMINATED_MODEL_PATH
sft_tag=$base_model-sft

for ((i=0; i<${#arr[@]}; i+=$step)); do
    slice=("${arr[@]:i:$step}")
    echo "${slice[@]}"
    python explanation/svpo.py --model $sft_model \
        --layers "${slice[@]}" \
        --component mlp \
        --proj up \
        --unseen_data $CLEAN_DATA_PATH \
        --seen_data $LEAKED_DATA_PATH \
        --k $numk \
        --output_dir outputs/explanation/svpo/gsm8k/$sft_tag/mlp_up_k$numk
done
```
