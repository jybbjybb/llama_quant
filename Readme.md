Check Plot_LM_Q_results.ipynb for all detailed results for LLaMA, Qwen, Mistral, Mixtral, and other models with W16A16, W8A16, and W8A8 per-channel quantization.

First add arguments in lm-evaluation-harness
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
