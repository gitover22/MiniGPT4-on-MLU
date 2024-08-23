# Cambricon PyTorch Model Migration Report
## Cambricon PyTorch Changes
| No. |  File  |  Description  |
| 1 | fastchat/utils.py:12 | add "import torch_mlu" |
| 2 | fastchat/utils.py:129 | change "torch.cuda.device_count()" to "torch.mlu.device_count() " |
| 3 | fastchat/utils.py:131 | change "else min(max_gpus, torch.cuda.device_count())" to "else min(max_gpus, torch.mlu.device_count()) " |
| 4 | fastchat/utils.py:135 | change "with torch.cuda.device(gpu_id):" to "with torch.mlu.device(gpu_id): " |
| 5 | fastchat/utils.py:136 | change "device = torch.cuda.current_device()" to "device = torch.mlu.current_device() " |
| 6 | fastchat/utils.py:137 | change "gpu_properties = torch.cuda.get_device_properties(device)" to "gpu_properties = torch.mlu.get_device_properties(device) " |
| 7 | fastchat/utils.py:139 | change "allocated_memory = torch.cuda.memory_allocated() / (1024**3)" to "allocated_memory = torch.mlu.memory_allocated() / (1024**3) " |
| 8 | fastchat/serve/model_worker.py:33 | add "import torch_mlu" |
| 9 | fastchat/serve/model_worker.py:188 | change "except torch.cuda.OutOfMemoryError:" to "except torch.mlu.OutOfMemoryError: " |
| 10 | fastchat/serve/model_worker.py:221 | change "except torch.cuda.OutOfMemoryError:" to "except torch.mlu.OutOfMemoryError: " |
| 11 | fastchat/serve/model_worker.py:252 | change "except torch.cuda.OutOfMemoryError:" to "except torch.mlu.OutOfMemoryError: " |
| 12 | fastchat/serve/cacheflow_worker.py:20 | add "import torch_mlu" |
| 13 | fastchat/serve/cacheflow_worker.py:39 | change "seed = torch.cuda.current_device()" to "seed = torch.mlu.current_device() " |
| 14 | fastchat/serve/inference.py:10 | add "import torch_mlu" |
| 15 | fastchat/serve/inference.py:219 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 16 | fastchat/serve/huggingface_api.py:9 | add "import torch_mlu" |
| 17 | fastchat/serve/huggingface_api.py:36 | change "torch.as_tensor(input_ids).cuda()," to "torch.as_tensor(input_ids).mlu(), " |
| 18 | fastchat/eval/get_model_answer.py:3 | add "import torch_mlu" |
| 19 | fastchat/eval/get_model_answer.py:45 | change ").cuda()" to ").mlu() " |
| 20 | fastchat/eval/get_model_answer.py:58 | change "torch.as_tensor(input_ids).cuda()," to "torch.as_tensor(input_ids).mlu(), " |
| 21 | fastchat/model/compression.py:8 | add "import torch_mlu" |
| 22 | fastchat/model/compression.py:128 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 23 | fastchat/model/model_adapter.py:8 | add "import torch_mlu" |
| 24 | fastchat/model/model_adapter.py:80 | change "if device != "cuda":" to "if device != "mlu": " |
| 25 | fastchat/model/model_adapter.py:106 | change "elif device == "cuda":" to "elif device == "mlu": " |
| 26 | fastchat/model/model_adapter.py:154 | change "if (device == "cuda" and num_gpus == 1 and not cpu_offloading) or device == "mps":" to "if (device == "mlu" and num_gpus == 1 and not cpu_offloading) or device == "mps": " |
| 27 | fastchat/model/model_adapter.py:178 | change "choices=["cpu", "cuda", "mps"]," to "choices=["cpu", "mlu", "mps"], " |
| 28 | fastchat/model/model_adapter.py:179 | change "default="cuda"," to "default="mlu", " |
| 29 | fastchat/model/apply_lora.py:12 | add "import torch_mlu" |
| 30 | fastchat/model/chatglm_model.py:1 | add "import torch_mlu" |
| 31 | fastchat/model/monkey_patch_non_inplace.py:8 | add "import torch_mlu" |
| 32 | fastchat/model/rwkv_model.py:5 | add "import torch_mlu" |
| 33 | fastchat/model/rwkv_model.py:20 | change "self.model = RWKV(model=model_path, strategy="cuda fp16")" to "self.model = RWKV(model=model_path, strategy="mlu fp16") " |
| 34 | fastchat/model/rwkv_model.py:23 | change "assert target == "cuda"" to "assert target == "mlu" " |
| 35 | fastchat/model/convert_fp16.py:8 | add "import torch_mlu" |
| 36 | fastchat/model/make_delta.py:9 | add "import torch_mlu" |
| 37 | fastchat/model/apply_delta.py:16 | add "import torch_mlu" |
| 38 | fastchat/train/train_flant5.py:26 | add "import torch_mlu" |
| 39 | fastchat/train/llama_flash_attn_monkey_patch.py:3 | add "import torch_mlu" |
| 40 | fastchat/train/train.py:23 | add "import torch_mlu" |
