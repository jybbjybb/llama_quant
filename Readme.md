Plot the results for quantization of LLM models

LLaMA3.2-90B-Vision-Instruct, FP16
```
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m lm_eval --model hf-multimodal --model_args pretrained=/mnt/LLM_checkpoints/Llama-3.2-90B-Vision-Instruct/,parallelize=True --tasks mmmu_val  --batch_size 1 --apply_chat_template
```
|                 Tasks                 |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|---------------------------------------|------:|------|-----:|------|---|-----:|---|-----:|
|mmmu_val                               |      0|none  |      |acc   |↑  |0.5478|±  |0.0158|
| - Art and Design                      |      0|none  |      |acc   |↑  |0.6833|±  |0.0358|
|  - Art                                |      0|none  |     0|acc   |↑  |0.6667|±  |0.0875|
|  - Art Theory                         |      0|none  |     0|acc   |↑  |0.9333|±  |0.0463|
|  - Design                             |      0|none  |     0|acc   |↑  |0.8667|±  |0.0631|
|  - Music                              |      0|none  |     0|acc   |↑  |0.2667|±  |0.0821|
| - Business                            |      0|none  |      |acc   |↑  |0.5267|±  |0.0411|
|  - Accounting                         |      0|none  |     0|acc   |↑  |0.4667|±  |0.0926|
|  - Economics                          |      0|none  |     0|acc   |↑  |0.5667|±  |0.0920|
|  - Finance                            |      0|none  |     0|acc   |↑  |0.5667|±  |0.0920|
|  - Manage                             |      0|none  |     0|acc   |↑  |0.4333|±  |0.0920|
|  - Marketing                          |      0|none  |     0|acc   |↑  |0.6000|±  |0.0910|
| - Health and Medicine                 |      0|none  |      |acc   |↑  |0.5933|±  |0.0403|
|  - Basic Medical Science              |      0|none  |     0|acc   |↑  |0.6333|±  |0.0895|
|  - Clinical Medicine                  |      0|none  |     0|acc   |↑  |0.6000|±  |0.0910|
|  - Diagnostics and Laboratory Medicine|      0|none  |     0|acc   |↑  |0.4667|±  |0.0926|
|  - Pharmacy                           |      0|none  |     0|acc   |↑  |0.5667|±  |0.0920|
|  - Public Health                      |      0|none  |     0|acc   |↑  |0.7000|±  |0.0851|
| - Humanities and Social Science       |      0|none  |      |acc   |↑  |0.7417|±  |0.0400|
|  - History                            |      0|none  |     0|acc   |↑  |0.8000|±  |0.0743|
|  - Literature                         |      0|none  |     0|acc   |↑  |0.8333|±  |0.0692|
|  - Psychology                         |      0|none  |     0|acc   |↑  |0.6667|±  |0.0875|
|  - Sociology                          |      0|none  |     0|acc   |↑  |0.6667|±  |0.0875|
| - Science                             |      0|none  |      |acc   |↑  |0.4667|±  |0.0409|
|  - Biology                            |      0|none  |     0|acc   |↑  |0.4333|±  |0.0920|
|  - Chemistry                          |      0|none  |     0|acc   |↑  |0.3333|±  |0.0875|
|  - Geography                          |      0|none  |     0|acc   |↑  |0.4667|±  |0.0926|
|  - Math                               |      0|none  |     0|acc   |↑  |0.5667|±  |0.0920|
|  - Physics                            |      0|none  |     0|acc   |↑  |0.5333|±  |0.0926|
| - Tech and Engineering                |      0|none  |      |acc   |↑  |0.4000|±  |0.0331|
|  - Agriculture                        |      0|none  |     0|acc   |↑  |0.6333|±  |0.0895|
|  - Architecture and Engineering       |      0|none  |     0|acc   |↑  |0.2667|±  |0.0821|
|  - Computer Science                   |      0|none  |     0|acc   |↑  |0.5667|±  |0.0920|
|  - Electronics                        |      0|none  |     0|acc   |↑  |0.3333|±  |0.0875|
|  - Energy and Power                   |      0|none  |     0|acc   |↑  |0.3667|±  |0.0895|
|  - Materials                          |      0|none  |     0|acc   |↑  |0.2667|±  |0.0821|
|  - Mechanical Engineering             |      0|none  |     0|acc   |↑  |0.3667|±  |0.0895|
