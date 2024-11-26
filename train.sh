#####         llama-2-chat-7b            ######

##data_generation 注意修改16行gpus环境
python /data/wenzhuofan/EAGLE_NA/ge_data/allocation.py \
--outdir /data/wenzhuofan/train_data_llama2chat7b \
--type llama2chat \
--basemodel /data/wenzhuofan/llama-2-chat-7b

##Training
accelerate launch /data/wenzhuofan/EAGLE_NA/train/main.py \
--tmpdir /data/wenzhuofan/train_data_llama2chat7b \
--cpdir /data/wenzhuofan/EAGLE_NA/checkpoint_7b_llama-2_0.01ctcloss \
--basepath /data/wenzhuofan/llama-2-chat-7b \
--configpath /data/wenzhuofan/EAGLE_NA/train/llama_2_chat_7B_config.json \
--bs 1 \
--fixlen 3 \
--recover -1 \
--cw 0.01

##evaluation
python /data/wenzhuofan/EAGLE_NA/evaluation/gen_ea_llama2chat_my.py \
--ea-model-path /data/wenzhuofan/EAGLE_NA/checkpoint_7b_llama-2_0.01ctcloss/state_20 \
--base-model-path /data/wenzhuofan/llama-2-chat-7b \
--model-id LC_7b_0.01ctcloss \
--tree-choices medusa \
--bench-name mt_bench \
--use-safetensor-weight True

##evaluation_baseline
python /data/wenzhuofan/EAGLE_NA/evaluation/gen_baseline_answer_llama2chat.py \
--ea-model-path /data/wenzhuofan/EAGLE_NA/checkpoint_7b_llama-2_0.01ctcloss/state_20 \
--base-model-path /data/wenzhuofan/llama-2-chat-7b \
--model-id LC_7b_baseline \
--tree-choices medusa \
--bench-name mt_bench \
--use-safetensor-weight True

##evaluation_GS
python /data/wenzhuofan/EAGLE_NA/evaluation/gen_ea_answer_llama2chat_GSM8K.py \
--ea-model-path /data/wenzhuofan/EAGLE_NA/checkpoint_7b_llama-2_0.01ctcloss/state_20 \
--base-model-path /data/wenzhuofan/llama-2-chat-7b \
--model-id LC_7b_0.01ctcloss \
--tree-choices medusa \
--bench-name GS_bench \
--use-safetensor-weight True

##evaluation_GS_baseline
python /data/wenzhuofan/EAGLE_NA/evaluation/gen_ea_answer_llama2chat_GSM8K_baseline.py \
--ea-model-path /data/wenzhuofan/EAGLE_NA/checkpoint_7b_llama-2_0.01ctcloss/state_20 \
--base-model-path /data/wenzhuofan/llama-2-chat-7b \
--model-id LC_baseline \
--tree-choices medusa \
--bench-name GS_bench \
--use-safetensor-weight True


####       llama-2-chat-13b      ####

##data_generation 注意修改16行gpus环境
python /data/wenzhuofan/EAGLE_NA/ge_data/allocation.py \
--outdir /data/wenzhuofan/train_data_llama2chat13b \
--type llama2chat \
--basemodel /data/wenzhuofan/llama-2-chat-13b

##training
accelerate launch /data/wenzhuofan/EAGLE_NA/train/main.py \
--tmpdir /data/wenzhuofan/train_data_llama2chat13b \
--cpdir /data/wenzhuofan/EAGLE_NA/checkpoint_13b_llama-2_0.01ctcloss \
--basepath /data/wenzhuofan/llama-2-chat-13b \
--configpath /data/wenzhuofan/EAGLE_NA/train/llama_2_chat_13B_config.json \
--bs 1 \
--fixlen 3 \
--recover -1 \
--cw 0.01

##evaluation
python /data/wenzhuofan/EAGLE_NA/evaluation/gen_ea_llama2chat_my.py \
--ea-model-path /data/wenzhuofan/EAGLE_NA/checkpoint_13b_llama-2_0.01ctcloss/state_20 \
--base-model-path /data/wenzhuofan/llama-2-chat-13b \
--model-id LC_13b_0.01ctcloss \
--tree-choices medusa \
--bench-name mt_bench \
--use-safetensor-weight True

##evaluation_baseline
python /data/wenzhuofan/EAGLE_NA/evaluation/gen_baseline_answer_llama2chat.py \
--ea-model-path /data/wenzhuofan/EAGLE_NA/checkpoint_13b_llama-2_0.01ctcloss/state_20 \
--base-model-path /data/wenzhuofan/llama-2-chat-13b \
--model-id LC_13b_baseline \
--tree-choices medusa \
--bench-name mt_bench \
--use-safetensor-weight True

##evaluation_GS
python /data/wenzhuofan/EAGLE_NA/evaluation/gen_ea_answer_llama2chat_GSM8K.py \
--ea-model-path /data/wenzhuofan/EAGLE_NA/checkpoint_13b_llama-2_0.01ctcloss/state_20 \
--base-model-path /data/wenzhuofan/llama-2-chat-13b \
--model-id LC_13b \
--tree-choices medusa \
--bench-name GS_bench \
--use-safetensor-weight True

##evaluation_GS_baseline
python /data/wenzhuofan/EAGLE_NA/evaluation/gen_ea_answer_llama2chat_GSM8K_baseline.py \
--ea-model-path /data/wenzhuofan/EAGLE_NA/checkpoint_13b_llama-2_0.01ctcloss/state_20 \
--base-model-path /data/wenzhuofan/llama-2-chat-13b \
--model-id LC_13b_baseline \
--tree-choices medusa \
--bench-name GS_bench \
--use-safetensor-weight True

#####         Vicuna-7b            ######

##data_generation 注意修改16行gpus环境
python /data/wenzhuofan/EAGLE_NA/ge_data/allocation.py \
--outdir /data/wenzhuofan/train_data_vicuna7b \
--type vicuna \
--basemodel /data/wenzhuofan/vicuna-7b-v1.3

##training
accelerate launch /data/wenzhuofan/EAGLE_NA/train/main.py \
--tmpdir /data/wenzhuofan/train_data_vicuna7b \
--cpdir /data/wenzhuofan/EAGLE_NA/checkpoint_7b_vicuna_0.01ctcloss \
--basepath /data/wenzhuofan/vicuna-7b-v1.3 \
--configpath /data/wenzhuofan/EAGLE_NA/train/vicuna_7B_config.json \
--bs 1 \
--fixlen 3 \
--recover -1 \
--cw 0.01

###############################ctc_loss#####################################
accelerate launch /data/wenzhuofan/work/EAGLE_NA/train/main.py \
--tmpdir /data/wenzhuofan/data/train_data_vicuna7b \
--cpdir /data/wenzhuofan/EAGLE_NA/checkpoint_7b_test \
--basepath /data/wenzhuofan/model/vicuna-7b-v1.3 \
--configpath /data/wenzhuofan/work/EAGLE_NA/train/vicuna_7B_config.json \
--bs 1 \
--fixlen 3 \
--recover -1 \
--cw 0.001

accelerate launch /data/wenzhuofan/EAGLE_NA/train/main.py \
--tmpdir /data/wenzhuofan/train_data_vicuna7b \
--cpdir /data/wenzhuofan/EAGLE_NA/checkpoint_7b_vicuna_0.1ctcloss \
--basepath /data/wenzhuofan/vicuna-7b-v1.3 \
--configpath /data/wenzhuofan/EAGLE_NA/train/vicuna_7B_config.json \
--bs 1 \
--fixlen 3 \
--recover -1 \
--cw 0.1

accelerate launch /data/wenzhuofan/EAGLE_NA/train/main.py \
--tmpdir /data/wenzhuofan/train_data_vicuna7b \
--cpdir /data/wenzhuofan/EAGLE_NA/checkpoint_7b_vicuna_0ctcloss \
--basepath /data/wenzhuofan/vicuna-7b-v1.3 \
--configpath /data/wenzhuofan/EAGLE_NA/train/vicuna_7B_config.json \
--bs 1 \
--fixlen 3 \
--recover -1 \
--cw 0

accelerate launch /data/wenzhuofan/EAGLE_NA/train/main.py \
--tmpdir /data/wenzhuofan/train_data_vicuna7b \
--cpdir /data/wenzhuofan/EAGLE_NA/checkpoint_7b_vicuna_0.0001ctcloss \
--basepath /data/wenzhuofan/vicuna-7b-v1.3 \
--configpath /data/wenzhuofan/EAGLE_NA/train/vicuna_7B_config.json \
--bs 1 \
--fixlen 3 \
--recover -1 \
--cw 0.0001

##evaluation
python /data/wenzhuofan/EAGLE_NA/evaluation/gen_ea_answer_vicuna.py \
--ea-model-path /data/wenzhuofan/EAGLE_NA/checkpoint_7b_vicuna_0.01ctcloss/state_20 \
--base-model-path /data/wenzhuofan/vicuna-7b-v1.3 \
--model-id vc_7b_0.01ctcloss \
--tree-choices medusa \
--bench-name mt_bench \
--use-safetensor-weight True

##evaluation_baseline
python /data/wenzhuofan/EAGLE_NA/evaluation/gen_baseline_answer_vicuna.py \
--ea-model-path /data/wenzhuofan/EAGLE_NA/checkpoint_7b_vicuna_0.01ctcloss/state_20 \
--base-model-path /data/wenzhuofan/vicuna-7b-v1.3 \
--model-id vc_7b_baseline \
--tree-choices medusa \
--bench-name mt_bench \
--use-safetensor-weight True

##evaluation_GS
python /data/wenzhuofan/EAGLE_NA/evaluation/gen_ea_answer_vicuna_GSM8K.py \
--ea-model-path /data/wenzhuofan/EAGLE_NA/checkpoint_7b_vicuna_0.01ctcloss/state_20 \
--base-model-path /data/wenzhuofan/vicuna-7b-v1.3 \
--model-id vc_7b \
--tree-choices medusa \
--bench-name GS_bench \
--use-safetensor-weight True

##evaluation_baseline_GS
python /data/wenzhuofan/EAGLE_NA/evaluation/gen_baseline_answer_vicuna_GSM8K.py \
--ea-model-path /data/wenzhuofan/EAGLE_NA/checkpoint_7b_vicuna_0.01ctcloss/state_20 \
--base-model-path /data/wenzhuofan/vicuna-7b-v1.3 \
--model-id vc_7b_baseline \
--tree-choices medusa \
--bench-name GS_bench \
--use-safetensor-weight True


####        Vicuna-13b       #####

##data_generation 注意修改16行gpus环境
python /data/wenzhuofan/EAGLE_NA/ge_data/allocation.py \
--outdir /data/wenzhuofan/train_data_vicuna13b \
--type vicuna \
--basemodel /data/wenzhuofan/vicuna-13b-v1.3

##training
accelerate launch /data/wenzhuofan/EAGLE_NA/train/main.py \
--tmpdir /data/wenzhuofan/train_data_vicuna13b \
--cpdir /data/wenzhuofan/EAGLE_NA/checkpoint_13b_vicuna_0.01ctcloss \
--basepath /data/wenzhuofan/vicuna-13b-v1.3 \
--configpath /data/wenzhuofan/EAGLE_NA/train/vicuna_13B_config.json \
--bs 1 \
--fixlen 3 \
--recover -1 \
--cw 0.01

##evaluation
python /data/wenzhuofan/EAGLE_NA/evaluation/gen_ea_answer_vicuna.py \
--ea-model-path /data/wenzhuofan/EAGLE_NA/checkpoint_13b_vicuna_0.01ctcloss/state_20 \
--base-model-path /data/wenzhuofan/vicuna-13b-v1.3 \
--model-id vc_13b_0.01ctcloss \
--tree-choices medusa \
--bench-name mt_bench \
--use-safetensor-weight True

##evaluation_baseline
python /data/wenzhuofan/EAGLE_NA/evaluation/gen_baseline_answer_vicuna.py \
--ea-model-path /data/wenzhuofan/EAGLE_NA/checkpoint_13b_vicuna_0.01ctcloss/state_20 \
--base-model-path /data/wenzhuofan/vicuna-13b-v1.3 \
--model-id vc_13b_baseline \
--tree-choices medusa \
--bench-name mt_bench \
--use-safetensor-weight True

##evaluation_GS
python /data/wenzhuofan/EAGLE_NA/evaluation/gen_ea_answer_vicuna_GSM8K.py \
--ea-model-path /data/wenzhuofan/EAGLE_NA/checkpoint_13b_vicuna_0.01ctcloss/state_20 \
--base-model-path /data/wenzhuofan/vicuna-13b-v1.3 \
--model-id vc_13b \
--tree-choices medusa \
--bench-name GS_bench \
--use-safetensor-weight True

##evaluation_baseline_GS
python /data/wenzhuofan/EAGLE_NA/evaluation/gen_baseline_answer_vicuna_GSM8K.py \
--ea-model-path /data/wenzhuofan/EAGLE_NA/checkpoint_13b_vicuna_0.01ctcloss/state_20 \
--base-model-path /data/wenzhuofan/vicuna-13b-v1.3 \
--model-id vc_13b_baseline \
--tree-choices medusa \
--bench-name GS_bench \
--use-safetensor-weight True

##Vicuna-33b
accelerate launch /data/wenzhuofan/EAGLE_NA/train/main.py \
--tmpdir /data/wenzhuofan/train_data_vicuna33b \
--cpdir /data/wenzhuofan/EAGLE_NA/checkpoint_33b_vicuna_0.01ctcloss \
--basepath /data/wenzhuofan/vicuna-33b-v1.3 \
--configpath /data/wenzhuofan/EAGLE_NA/train/vicuna_33B_config.json \
--bs 1 \
--fixlen 3 \
--recover -1 \
--cw 0.01

accelerate launch /data/wenzhuofan/EAGLE_NA/train/main.py \
--tmpdir /data/wenzhuofan/train_data_vicuna33b \
--cpdir /data/wenzhuofan/EAGLE_NA/checkpoint_33b_vicuna_0.001ctcloss \
--basepath /data/wenzhuofan/vicuna-33b-v1.3 \
--configpath /data/wenzhuofan/EAGLE_NA/train/vicuna_33B_config.json \
--bs 1 \
--fixlen 3 \
--recover -1 \
--cw 0.001

##evaluation
python /data/wenzhuofan/EAGLE_NA/evaluation/gen_ea_answer_vicuna.py \
--ea-model-path /data/wenzhuofan/EAGLE_NA/checkpoint_33b_vicuna_0.01ctcloss/state_20 \
--base-model-path /data/wenzhuofan/vicuna-33b-v1.3 \
--model-id vc_33b_0.01ctcloss \
--tree-choices medusa \
--bench-name mt_bench \
--use-safetensor-weight True

python /data/wenzhuofan/EAGLE_NA/evaluation/gen_baseline_answer_vicuna.py \
--ea-model-path /data/wenzhuofan/EAGLE_NA/checkpoint_33b_vicuna_0.01ctcloss/state_20 \
--base-model-path /data/wenzhuofan/vicuna-33b-v1.3 \
--model-id vc_33b_baseline \
--tree-choices medusa \
--bench-name mt_bench \
--use-safetensor-weight True


python /data/wenzhuofan/EAGLE_NA/evaluation/gen_ea_answer_vicuna.py \
--ea-model-path /data/wenzhuofan/EAGLE_NA/checkpoint_33b_vicuna_0.01ctcloss/state_20 \
--base-model-path /data/wenzhuofan/vicuna-33b-v1.3 \
--model-id vc_33b_0.01ctcloss \
--tree-choices medusa \
--bench-name GS_bench \
--use-safetensor-weight True

python /data/wenzhuofan/EAGLE_NA/evaluation/gen_baseline_answer_vicuna.py \
--ea-model-path /data/wenzhuofan/EAGLE_NA/checkpoint_33b_vicuna_0.01ctcloss/state_20 \
--base-model-path /data/wenzhuofan/vicuna-33b-v1.3 \
--model-id vc_33b_baseline \
--tree-choices medusa \
--bench-name GS_bench \
--use-safetensor-weight True

python /data/wenzhuofan/EAGLE_NA/evaluation/gen_ea_answer_vicuna_GSM8K.py \
--ea-model-path /data/wenzhuofan/EAGLE_NA/checkpoint_33b_vicuna_0.01ctcloss/state_20 \
--base-model-path /data/wenzhuofan/vicuna-33b-v1.3 \
--model-id vc_33b \
--tree-choices medusa \
--bench-name GS_bench \
--use-safetensor-weight True

python /data/wenzhuofan/EAGLE_NA/evaluation/gen_baseline_answer_vicuna_GSM8K.py \
--ea-model-path /data/wenzhuofan/EAGLE_NA/checkpoint_33b_vicuna_0.01ctcloss/state_20 \
--base-model-path /data/wenzhuofan/vicuna-33b-v1.3 \
--model-id vc_33b_baseline \
--tree-choices medusa \
--bench-name GS_bench \
--use-safetensor-weight True

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


####################### time evaluation ########################
python /data/wenzhuofan/EAGLE_NA/evaluation/gen_ea_answer_vicuna_time.py \
--ea-model-path /data/wenzhuofan/EAGLE_NA/checkpoint_7b_vicuna_0.001ctcloss/state_12 \
--base-model-path /data/wenzhuofan/vicuna-7b-v1.3 \
--model-id vc_7b_0.001ctcloss_time_eval \
--tree-choices medusa \
--bench-name mt_bench \
--use-safetensor-weight True \
--time-eval True