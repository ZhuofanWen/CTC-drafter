### Introduction
This is the implementation of paper: "Speculative Decoding with CTC-based Draft Model for LLM Inference Acceleration", including training and evaluation. 
### Installaion
First, create a python>=3.9 environment. Then use pip to install the packages.
```bash
cd CTC-drafter
pip install -r requirements.txt
```
We use open-source Vicuna models, you can download the weights: [Vicuna-7b](https://huggingface.co/lmsys/vicuna-7b-v1.3) [Vicuna-13b](https://huggingface.co/lmsys/vicuna-13b-v1.3) [Vicuna-33b](https://huggingface.co/lmsys/vicuna-33b-v1.3)
[llama-2-chat-7b](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
[llama-2-chat-13b](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
### Training
We take a public version of the ShareGPT dataset, which is a subset of the Vicuna training data. For other models, you can use the corresponding training dataset.
```bash
git clone https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered
```
We adopt knowledge distillation method:
```bash
cd ge_data
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

then you need to modify allocation.py line 13 according to your GPU environments. Parellel data process is supported. For example, if there is four GPUs(0,1,2,3) and you want each of the four GPU conduct a single data process, you can set:
```bash
gpus = [[0],[1],[2],[3]]
```

Sometimes it is difficult to load the base model in one single GPU such as Vicuna-33b, then you can set as follow to assign each two GPUs to conduct a process:
```bash
gpus = [[0,1],[2,3]]
```
After you have set the GPU allocation map, you can run the code and generate distilled data:
```bash
python allocation.py --outdir train_data_vicuna7b \
--type vicuna \
--basemodel vicuna-7b-v1.3 \
--data ShareGPT_V4.3_unfiltered_cleaned_split.json
```
Run the command to start training, we have listed the recommended training configs. If your training is interrupted and you want to restart from a certain checkpoint(such as state_2), you can add config: --recover 2
```bash
cd ..

accelerate launch --mixed_precision=bf16 train/main.py \
--tmpdir train_data_vicuna7b \
--cpdir checkpoint_7b_vicuna \
--basepath vicuna-7b-v1.3 \
--configpath train/vicuna_7B_config.json \
--bs 1  
```

If your training is interrupted and you want to restart from a certain checkpoint(such as state_2), you can add config: --recover 2
```bash
accelerate launch --mixed_precision=bf16 train/main.py \
--tmpdir train_data_vicuna7b \
--cpdir checkpoint_7b_vicuna \
--basepath vicuna-7b-v1.3 \
--configpath train/vicuna_7B_config.json \
--bs 1  \
--recover 2
```
### Evaluation
After training, you can evaluate the speedup performance on MT-bench and GSM8K. The model answers will be recorded in evaluation/MT-bench(GSM8K)/data/model_answer :
```bash
cd evaluation/MT-bench
python scripts/vicuna.py \
--draft-model-path checkpoint_7b_vicuna \
--base-model-path vicuna-7b-v1.3 \
--model-id  vc_7b\
--tree-choices medusa \
--bench-name mt_bench \
--use-safetensor-weight True 
```
you can also run the baseline without speculation method for comparison:
```bash
python scripts/vicuna_baseline.py \
--draft-model-path checkpoint_7b_vicuna \
--base-model-path vicuna-7b-v1.3 \
--model-id  vc_7b_baseline\
--tree-choices medusa \
--bench-name mt_bench \
--use-safetensor-weight True 
```

## License
This repository is licensed under the [Apache-2.0 License](LICENSE).

## Citation

If this work is helpful, please kindly cite as:

```bibtex
@inproceedings{wenspeculative,
  title={Speculative Decoding with CTC-based Draft Model for LLM Inference Acceleration},
  author={Wen, Zhuofan and Gui, Shangtong and Feng, Yang},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems}
}
```

## Acknowledgement

This repo is modified from [Medusa](https://github.com/FasterDecoding/Medusa) and [Eagle](https://github.com/SafeAILab/EAGLE). Thanks for their excellent works.
