Check Plot_LM_Q_results.ipynb for all detailed results for LLaMA, Qwen, Mistral, Mixtral, and other models with W16A16, W8A16, and W8A8 per-channel quantization.

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/x4YgRd8fmCU/0.jpg)](https://www.youtube.com/watch?v=x4YgRd8fmCU)


All open models are downloaded from HuggingFace repository in the following table

| Model Name                       | HuggingFace URL                                                     |
|----------------------------------|---------------------------------------------------------------------|
| LLaMA3-70B                      | [meta-llama/Meta-Llama-3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B)       |
| LLaMA3-70B-Instruct             | [meta-llama/Meta-Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) |
| LLaMA3.1-70B                    | [meta-llama/Meta-Llama-3.1-70B](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B)     |
| LLaMA3.1-70B-Instruct           | [meta-llama/Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct)  |
| Llama3-70B-Synthia              | [migtissera/Llama-3-70B-Synthia-v3.5](https://huggingface.co/migtissera/Llama-3-70B-Synthia-v3.5) |
| calme-2.2-llama3-70b            | [MaziyarPanahi/calme-2.2-llama3-70b](https://huggingface.co/MaziyarPanahi/calme-2.2-llama3-70b) |
| LLaMA3-8B                       | [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)        |
| LLaMA3-8B-Instruct              | [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |
| LLaMA3.1-8B                     | [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)           |
| LLaMA3.1-8B-Instruct            | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)  |
| LLaMA3.2-1B-Instruct            | [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)  |
| LLaMA3.2-3B-Instruct            | [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)  |
| LLaMA2-70B                      | [meta-llama/Llama-2-70b-hf](https://huggingface.co/meta-llama/Llama-2-70b-hf)         |
| LLaMA2-70B-chat                 | [meta-llama/Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)    |
| Qwen2-72B                       | [Qwen/Qwen2-72B](https://huggingface.co/Qwen/Qwen2-72B)                     |
| Mixtral-8x7B                    | [mistralai/Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)       |
| Phi3-14B-Instruct               | [microsoft/Phi-3-medium-128k-instruct](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct) |
| Mistral-Large-Instruct-123B     | [mistralai/Mistral-Large-Instruct-2407](https://huggingface.co/mistralai/Mistral-Large-Instruct-2407) |
| Falcon-40B                      | [tiiuae/falcon-40b](https://huggingface.co/tiiuae/falcon-40b)                  |
| LLaMA-3.2-11B-Instruct          | [meta-llama/Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) |
| LLaMA-3.2-90B-Instruct          | [meta-llama/Llama-3.2-90B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-90B-Vision-Instruct) |
| LLaVA-1.6-34B                   | [llava-hf/llava-v1.6-34b-hf](https://huggingface.co/llava-hf/llava-v1.6-34b-hf)        |
| Qwen2-VL-7B-Instruct            | [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)          |



In order to reproduce the results in Plot_LM_Q_results.ipynb, first add arguments in lm-evaluation-harness
```
parser.add_argument('--in_place_w', action="store_true", default=False, help='Quantize then change the weight (use with caution)')
parser.add_argument('--x_nbit', default=-1, type=int, help='Number of bits for activation')
parser.add_argument('--w_nbit', default=-1, type=int, help='Number of bits for weights')
parser.add_argument('--out_nbit', default=-1, type=int, help='Number of bits for output activation')
parser.add_argument('--q_group_size', default=-1, type=int, help='Quant group size')
parser.add_argument('--T', default=-1.0, type=float, help='some threshold for bi-smoothing')
```
### To reproduce the results of bi-smoothing

Add one line of code in evaluation.py, before "results = evaluate( ... )"
```
replace_all_linear_layers_recursive(lm.model.model, prefix='lm.model.model', args=args)
```
Then run test by 
```
python -m lm_eval --model hf --model_args pretrained=<model_ckpt_folder>,parallelize=True --tasks hellaswag,piqa,openbookqa,arc_easy,winogrande,arc_challenge,boolq,mmlu  --batch_size 16 --x_nbit 8 --w_nbit 8 --q_group_size -1 --T 1 --in_place_w
```

### To reproduce the results of mixed-grouping
Add one line of code in evaluation.py, before "results = evaluate( ... )"
```
replace_selective_linear_layers_recursive(lm.model.model, prefix='lm.model.model', args=args)
```
And modify the function "replace_selective_linear_layers_recursive()" in my_utils.py to change block 0,1,3 with 
```
QLinear(in_features, out_features, bias, x_nbit=args.x_nbit, w_nbit=args.w_nbit, q_group_size=1024, name=child_full_name)
```
and rest block with
```
QLinear(in_features, out_features, bias, x_nbit=args.x_nbit, w_nbit=args.w_nbit, q_group_size=-1, name=child_full_name)
```
Then run test by 
```
python -m lm_eval --model hf --model_args pretrained=<model_ckpt_folder>,parallelize=True --tasks hellaswag,piqa,openbookqa,arc_easy,winogrande,arc_challenge,boolq,mmlu  --batch_size 16 --x_nbit 8 --w_nbit 8 --q_group_size -1 --T -1 --in_place_w
```
