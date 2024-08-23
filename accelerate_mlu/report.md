# Cambricon PyTorch Model Migration Report
## Cambricon PyTorch Changes
| No. |  File  |  Description  |
| 1 | src/accelerate/launchers.py:19 | add "import torch_mlu" |
| 2 | src/accelerate/launchers.py:99 | change "if torch.cuda.is_available():" to "if torch.mlu.is_available(): " |
| 3 | src/accelerate/launchers.py:122 | change "if torch.cuda.is_initialized():" to "if torch.mlu.is_initialized(): " |
| 4 | src/accelerate/launchers.py:125 | change ""using `torch.cuda` in any cell. Restart your notebook and make sure no cells use any CUDA "" to ""using `torch.mlu` in any cell. Restart your notebook and make sure no cells use any CUDA " " |
| 5 | src/accelerate/launchers.py:152 | change "elif torch.cuda.is_available():" to "elif torch.mlu.is_available(): " |
| 6 | src/accelerate/state.py:25 | add "import torch_mlu" |
| 7 | src/accelerate/state.py:146 | change "self.device = torch.device("cuda", self.local_process_index)" to "self.device = torch.device("mlu", self.local_process_index) " |
| 8 | src/accelerate/state.py:147 | change "torch.cuda.set_device(self.device)" to "torch.mlu.set_device(self.device) " |
| 9 | src/accelerate/state.py:166 | change "# DeepSpeed always uses nccl" to "# DeepSpeed always uses cncl " |
| 10 | src/accelerate/state.py:168 | change "self.backend = "nccl"" to "self.backend = "cncl" " |
| 11 | src/accelerate/state.py:180 | change "self.device = torch.device("cuda", self.local_process_index)" to "self.device = torch.device("mlu", self.local_process_index) " |
| 12 | src/accelerate/state.py:182 | change "torch.cuda.set_device(self.device)" to "torch.mlu.set_device(self.device) " |
| 13 | src/accelerate/state.py:187 | change "self.backend = kwargs.pop("backend", "nccl")" to "self.backend = kwargs.pop("backend", "cncl") " |
| 14 | src/accelerate/state.py:190 | change "self.backend = "nccl"" to "self.backend = "cncl" " |
| 15 | src/accelerate/state.py:196 | change "self.device = torch.device("cuda", self.local_process_index)" to "self.device = torch.device("mlu", self.local_process_index) " |
| 16 | src/accelerate/state.py:197 | change "torch.cuda.set_device(self.device)" to "torch.mlu.set_device(self.device) " |
| 17 | src/accelerate/state.py:629 | change "- CUDA if `torch.cuda.is_available()`" to "- CUDA if `torch.mlu.is_available()` " |
| 18 | src/accelerate/state.py:635 | change "elif torch.cuda.is_available():" to "elif torch.mlu.is_available(): " |
| 19 | src/accelerate/state.py:636 | change "return torch.device("cuda")" to "return torch.device("mlu") " |
| 20 | src/accelerate/state.py:731 | change "and self.device.type == "cuda"" to "and self.device.type == "mlu" " |
| 21 | src/accelerate/state.py:733 | change "torch.backends.cuda.matmul.allow_tf32 = True" to "torch.backends.mlu.matmul.allow_tf32 = True " |
| 22 | src/accelerate/checkpointing.py:21 | add "import torch_mlu" |
| 23 | src/accelerate/checkpointing.py:22 | change "from torch.cuda.amp import GradScaler" to "from torch.mlu.amp import GradScaler " |
| 24 | src/accelerate/checkpointing.py:69 | change "scaler (`torch.cuda.amp.GradScaler`, *optional*):" to "scaler (`torch.mlu.amp.GradScaler`, *optional*): " |
| 25 | src/accelerate/checkpointing.py:107 | change "states["torch_cuda_manual_seed"] = torch.cuda.get_rng_state_all()" to "states["torch_mlu_manual_seed"] = torch.mlu.get_rng_state_all() " |
| 26 | src/accelerate/checkpointing.py:140 | change "scaler (`torch.cuda.amp.GradScaler`, *optional*):" to "scaler (`torch.mlu.amp.GradScaler`, *optional*): " |
| 27 | src/accelerate/checkpointing.py:192 | change "torch.cuda.set_rng_state_all(states["torch_cuda_manual_seed"])" to "torch.mlu.set_rng_state_all(states["torch_mlu_manual_seed"]) " |
| 28 | src/accelerate/big_modeling.py:19 | add "import torch_mlu" |
| 29 | src/accelerate/big_modeling.py:95 | change "with init_on_device(device=torch.device("cuda")):" to "with init_on_device(device=torch.device("mlu")): " |
| 30 | src/accelerate/big_modeling.py:96 | change "tst = nn.Liner(100, 100)  # on `cuda` device" to "tst = nn.Liner(100, 100)  # on `mlu` device " |
| 31 | src/accelerate/big_modeling.py:217 | change "model_1, hook_1 = cpu_offload_with_hook(model_1, cuda_device)" to "model_1, hook_1 = cpu_offload_with_hook(model_1, mlu_device) " |
| 32 | src/accelerate/big_modeling.py:218 | change "model_2, hook_2 = cpu_offload_with_hook(model_2, cuda_device, prev_module_hook=hook_1)" to "model_2, hook_2 = cpu_offload_with_hook(model_2, mlu_device, prev_module_hook=hook_1) " |
| 33 | src/accelerate/big_modeling.py:219 | change "model_3, hook_3 = cpu_offload_with_hook(model_3, cuda_device, prev_module_hook=hook_2)" to "model_3, hook_3 = cpu_offload_with_hook(model_3, mlu_device, prev_module_hook=hook_2) " |
| 34 | src/accelerate/accelerator.py:31 | add "import torch_mlu" |
| 35 | src/accelerate/accelerator.py:172 | change "- `"cuda"`: the CUDA random number generator (GPU only)" to "- `"mlu"`: the CUDA random number generator (GPU only) " |
| 36 | src/accelerate/accelerator.py:408 | change "if self.device.type not in ("cuda", "mps"):" to "if self.device.type not in ("mlu", "mps"): " |
| 37 | src/accelerate/accelerator.py:416 | change "self.scaler = torch.cuda.amp.GradScaler(**kwargs)" to "self.scaler = torch.mlu.amp.GradScaler(**kwargs) " |
| 38 | src/accelerate/accelerator.py:1260 | change ""you're training on. Make sure you loaded the model on the correct device using for example `device_map={'':torch.cuda.current_device()}"" to ""you're training on. Make sure you loaded the model on the correct device using for example `device_map={'':torch.mlu.current_device()}" " |
| 39 | src/accelerate/accelerator.py:1261 | change ""you're training on. Make sure you loaded the model on the correct device using for example `device_map={'':torch.cuda.current_device() or device_map={'':torch.xpu.current_device()}"" to ""you're training on. Make sure you loaded the model on the correct device using for example `device_map={'':torch.mlu.current_device() or device_map={'':torch.xpu.current_device()}" " |
| 40 | src/accelerate/accelerator.py:1308 | change "model.forward = MethodType(torch.cuda.amp.autocast(dtype=torch.float16)(model.forward.__func__), model)" to "model.forward = MethodType(torch.mlu.amp.autocast(dtype=torch.float16)(model.forward.__func__), model) " |
| 41 | src/accelerate/accelerator.py:1314 | change "model.forward = MethodType(torch.cuda.amp.autocast()(model.forward.__func__), model)" to "model.forward = MethodType(torch.mlu.amp.autocast()(model.forward.__func__), model) " |
| 42 | src/accelerate/accelerator.py:1327 | change "cuda_device_capacity = torch.cuda.get_device_capability()" to "mlu_device_capacity = torch.mlu.get_device_capability() " |
| 43 | src/accelerate/accelerator.py:1328 | change "fp8_enabled = cuda_device_capacity[0] >= 9 or (" to "fp8_enabled = mlu_device_capacity[0] >= 9 or ( " |
| 44 | src/accelerate/accelerator.py:1329 | change "cuda_device_capacity[0] == 8 and cuda_device_capacity[1] >= 9" to "mlu_device_capacity[0] == 8 and mlu_device_capacity[1] >= 9 " |
| 45 | src/accelerate/accelerator.py:1333 | change "f"The current device has compute capability of {cuda_device_capacity} which is "" to "f"The current device has compute capability of {mlu_device_capacity} which is " " |
| 46 | src/accelerate/data_loader.py:19 | add "import torch_mlu" |
| 47 | src/accelerate/data_loader.py:338 | change "- `"cuda"`: the CUDA random number generator (GPU only)" to "- `"mlu"`: the CUDA random number generator (GPU only) " |
| 48 | src/accelerate/data_loader.py:669 | change "- `"cuda"`: the CUDA random number generator (GPU only)" to "- `"mlu"`: the CUDA random number generator (GPU only) " |
| 49 | src/accelerate/optimizer.py:18 | add "import torch_mlu" |
| 50 | src/accelerate/optimizer.py:51 | change "scaler (`torch.cuda.amp.grad_scaler.GradScaler`, *optional*):" to "scaler (`torch.mlu.amp.grad_scaler.GradScaler`, *optional*): " |
| 51 | src/accelerate/hooks.py:18 | add "import torch_mlu" |
| 52 | src/accelerate/local_sgd.py:14 | add "import torch_mlu" |
| 53 | src/accelerate/commands/launch.py:26 | add "import torch_mlu" |
| 54 | src/accelerate/commands/launch.py:871 | change "args.num_processes = torch.cuda.device_count()" to "args.num_processes = torch.mlu.device_count() " |
| 55 | src/accelerate/commands/launch.py:873 | change "if torch.cuda.device_count() > 1 and not args.multi_gpu:" to "if torch.mlu.device_count() > 1 and not args.multi_gpu: " |
| 56 | src/accelerate/commands/env.py:23 | add "import torch_mlu" |
| 57 | src/accelerate/commands/env.py:48 | change "pt_cuda_available = torch.cuda.is_available()" to "pt_mlu_available = torch.mlu.is_available() " |
| 58 | src/accelerate/commands/env.py:61 | change ""PyTorch version (GPU?)": f"{pt_version} ({pt_cuda_available})"," to ""PyTorch version (GPU?)": f"{pt_version} ({pt_mlu_available})", " |
| 59 | src/accelerate/commands/env.py:65 | change "if pt_cuda_available:" to "if pt_mlu_available: " |
| 60 | src/accelerate/commands/env.py:66 | change "info["GPU type"] = torch.cuda.get_device_name()" to "info["GPU type"] = torch.mlu.get_device_name() " |
| 61 | src/accelerate/commands/config/default.py:19 | add "import torch_mlu" |
| 62 | src/accelerate/commands/config/default.py:60 | change "if torch.cuda.is_available():" to "if torch.mlu.is_available(): " |
| 63 | src/accelerate/commands/config/default.py:61 | change "num_gpus = torch.cuda.device_count()" to "num_gpus = torch.mlu.device_count() " |
| 64 | src/accelerate/utils/imports.py:22 | add "import torch_mlu" |
| 65 | src/accelerate/utils/imports.py:113 | change "if torch.cuda.is_available():" to "if torch.mlu.is_available(): " |
| 66 | src/accelerate/utils/imports.py:114 | change "return torch.cuda.is_bf16_supported()" to "return torch.mlu.is_bf16_supported() " |
| 67 | src/accelerate/utils/operations.py:23 | add "import torch_mlu" |
| 68 | src/accelerate/utils/other.py:18 | add "import torch_mlu" |
| 69 | src/accelerate/utils/offload.py:21 | add "import torch_mlu" |
| 70 | src/accelerate/utils/dataclasses.py:32 | add "import torch_mlu" |
| 71 | src/accelerate/utils/dataclasses.py:95 | change "`torch.cuda.amp.GradScaler` used is created. Please refer to the documentation of this" to "`torch.mlu.amp.GradScaler` used is created. Please refer to the documentation of this " |
| 72 | src/accelerate/utils/dataclasses.py:140 | change "backend: Optional[str] = "nccl"" to "backend: Optional[str] = "cncl" " |
| 73 | src/accelerate/utils/dataclasses.py:245 | change "- **INDUCTOR** -- Uses TorchInductor backend with AotAutograd and cudagraphs by leveraging codegened Triton" to "- **INDUCTOR** -- Uses TorchInductor backend with AotAutograd and mlugraphs by leveraging codegened Triton " |
| 74 | src/accelerate/utils/dataclasses.py:252 | change "- **AOT_CUDAGRAPHS** -- cudagraphs with AotAutograd. [Read" to "- **AOT_CUDAGRAPHS** -- mlugraphs with AotAutograd. [Read " |
| 75 | src/accelerate/utils/dataclasses.py:338 | change "CUDA = "cuda"" to "CUDA = "mlu" " |
| 76 | src/accelerate/utils/modeling.py:28 | add "import torch_mlu" |
| 77 | src/accelerate/utils/modeling.py:187 | change "device_index = torch.device(device).index if torch.device(device).type == "cuda" else None" to "device_index = torch.device(device).index if torch.device(device).type == "mlu" else None " |
| 78 | src/accelerate/utils/modeling.py:191 | change "module = module.cuda(device_index)" to "module = module.mlu(device_index) " |
| 79 | src/accelerate/utils/modeling.py:194 | change "module = module.cuda(device_index)" to "module = module.mlu(device_index) " |
| 80 | src/accelerate/utils/modeling.py:197 | change "device_index = torch.device(device).index if torch.device(device).type == "cuda" else None" to "device_index = torch.device(device).index if torch.device(device).type == "mlu" else None " |
| 81 | src/accelerate/utils/modeling.py:199 | change "module.weight = module.weight.cuda(device_index)" to "module.weight = module.weight.mlu(device_index) " |
| 82 | src/accelerate/utils/modeling.py:470 | change "if not (torch.cuda.is_available() or is_xpu_available()):" to "if not (torch.mlu.is_available() or is_xpu_available()): " |
| 83 | src/accelerate/utils/modeling.py:476 | change "for i in range(torch.cuda.device_count()):" to "for i in range(torch.mlu.device_count()): " |
| 84 | src/accelerate/utils/modeling.py:478 | change "max_memory = {i: torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())}" to "max_memory = {i: torch.mlu.mem_get_info(i)[0] for i in range(torch.mlu.device_count())} " |
| 85 | src/accelerate/utils/modeling.py:579 | change "if not (torch.cuda.is_available() or is_xpu_available()) or is_mps_available():" to "if not (torch.mlu.is_available() or is_xpu_available()) or is_mps_available(): " |
| 86 | src/accelerate/utils/modeling.py:583 | change "num_devices = len([d for d in max_memory if torch.device(d).type == "cuda" and max_memory[d] > 0])" to "num_devices = len([d for d in max_memory if torch.device(d).type == "mlu" and max_memory[d] > 0]) " |
| 87 | src/accelerate/utils/launch.py:21 | add "import torch_mlu" |
| 88 | src/accelerate/utils/random.py:19 | add "import torch_mlu" |
| 89 | src/accelerate/utils/random.py:47 | change "torch.cuda.manual_seed_all(seed)" to "torch.mlu.manual_seed_all(seed) " |
| 90 | src/accelerate/utils/random.py:50 | change "# ^^ safe to call this function even if cuda is not available" to "# ^^ safe to call this function even if mlu is not available " |
| 91 | src/accelerate/utils/random.py:60 | change "rng_state = torch.cuda.get_rng_state()" to "rng_state = torch.mlu.get_rng_state() " |
| 92 | src/accelerate/utils/random.py:89 | change "torch.cuda.set_rng_state(rng_state)" to "torch.mlu.set_rng_state(rng_state) " |
| 93 | src/accelerate/utils/memory.py:24 | add "import torch_mlu" |
| 94 | src/accelerate/utils/memory.py:31 | change "Releases memory from `objects` by setting them to `None` and calls `gc.collect()` and `torch.cuda.empty_cache()`." to "Releases memory from `objects` by setting them to `None` and calls `gc.collect()` and `torch.mlu.empty_cache()`. " |
| 95 | src/accelerate/utils/memory.py:46 | change ">>> a = torch.ones(1000, 1000).cuda()" to ">>> a = torch.ones(1000, 1000).mlu() " |
| 96 | src/accelerate/utils/memory.py:47 | change ">>> b = torch.ones(1000, 1000).cuda()" to ">>> b = torch.ones(1000, 1000).mlu() " |
| 97 | src/accelerate/utils/memory.py:57 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 98 | src/accelerate/utils/memory.py:117 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 99 | src/accelerate/utils/memory.py:137 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 100 | src/accelerate/utils/megatron_lm.py:20 | add "import torch_mlu" |
| 101 | src/accelerate/utils/megatron_lm.py:302 | change "flags = torch.cuda.LongTensor([int(do_train), int(do_valid), int(do_test)])" to "flags = torch.mlu.LongTensor([int(do_train), int(do_valid), int(do_test)]) " |
| 102 | src/accelerate/utils/megatron_lm.py:304 | change "flags = torch.cuda.LongTensor([0, 0, 0])" to "flags = torch.mlu.LongTensor([0, 0, 0]) " |
| 103 | src/accelerate/utils/megatron_lm.py:514 | change "data = send_to_device(data, torch.cuda.current_device())" to "data = send_to_device(data, torch.mlu.current_device()) " |
| 104 | src/accelerate/utils/megatron_lm.py:651 | change "data = send_to_device(data, torch.cuda.current_device())" to "data = send_to_device(data, torch.mlu.current_device()) " |
| 105 | src/accelerate/utils/megatron_lm.py:781 | change "data = send_to_device(data, torch.cuda.current_device())" to "data = send_to_device(data, torch.mlu.current_device()) " |
| 106 | src/accelerate/utils/megatron_lm.py:838 | change "assert torch.cuda.is_available(), "Megatron requires CUDA."" to "assert torch.mlu.is_available(), "Megatron requires CUDA." " |
| 107 | src/accelerate/utils/megatron_lm.py:870 | change "device_count = torch.cuda.device_count()" to "device_count = torch.mlu.device_count() " |
| 108 | src/accelerate/utils/megatron_lm.py:1020 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 109 | src/accelerate/utils/megatron_lm.py:1051 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 110 | src/accelerate/utils/megatron_lm.py:1102 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 111 | src/accelerate/utils/megatron_lm.py:1163 | change "self.eval_total_loss_dict.get(key, torch.cuda.FloatTensor([0.0])) + loss_dict[key]" to "self.eval_total_loss_dict.get(key, torch.mlu.FloatTensor([0.0])) + loss_dict[key] " |
| 112 | src/accelerate/utils/megatron_lm.py:1166 | change "key + "_num_iters", torch.cuda.FloatTensor([0.0])" to "key + "_num_iters", torch.mlu.FloatTensor([0.0]) " |
| 113 | src/accelerate/utils/megatron_lm.py:1167 | change ") + torch.cuda.FloatTensor([1.0])" to ") + torch.mlu.FloatTensor([1.0]) " |
| 114 | src/accelerate/utils/megatron_lm.py:1336 | change "prompts_length_tensor = torch.cuda.LongTensor([inputs.shape[1]] * inputs.shape[0])" to "prompts_length_tensor = torch.mlu.LongTensor([inputs.shape[1]] * inputs.shape[0]) " |
| 115 | src/accelerate/utils/megatron_lm.py:1338 | change "prompts_length_tensor = attention_mask.sum(axis=-1).cuda()" to "prompts_length_tensor = attention_mask.sum(axis=-1).mlu() " |
| 116 | src/accelerate/utils/megatron_lm.py:1350 | change "padding = torch.cuda.LongTensor([[tokenizer.eod] * max_new_tokens] * inputs.shape[0])" to "padding = torch.mlu.LongTensor([[tokenizer.eod] * max_new_tokens] * inputs.shape[0]) " |
| 117 | src/accelerate/utils/megatron_lm.py:1352 | change "[torch.unsqueeze(padding[:, 0], axis=-1), inputs.cuda(), padding], axis=-1" to "[torch.unsqueeze(padding[:, 0], axis=-1), inputs.mlu(), padding], axis=-1 " |
| 118 | src/accelerate/utils/megatron_lm.py:1359 | change "padding = torch.cuda.LongTensor([[tokenizer.eod] * max_new_tokens] * inputs.shape[0])" to "padding = torch.mlu.LongTensor([[tokenizer.eod] * max_new_tokens] * inputs.shape[0]) " |
| 119 | src/accelerate/utils/megatron_lm.py:1360 | change "prompts_tokens_tensor = torch.concat([inputs.cuda(), padding], axis=-1)" to "prompts_tokens_tensor = torch.concat([inputs.mlu(), padding], axis=-1) " |
| 120 | src/accelerate/test_utils/__init__.py:5 | change "require_cuda," to "require_mlu, " |
| 121 | src/accelerate/test_utils/testing.py:28 | add "import torch_mlu" |
| 122 | src/accelerate/test_utils/testing.py:83 | change "return unittest.skipUnless(not torch.cuda.is_available(), "test requires only a CPU")(test_case)" to "return unittest.skipUnless(not torch.mlu.is_available(), "test requires only a CPU")(test_case) " |
| 123 | src/accelerate/test_utils/testing.py:86 | change "def require_cuda(test_case):" to "def require_mlu(test_case): " |
| 124 | src/accelerate/test_utils/testing.py:90 | change "return unittest.skipUnless(torch.cuda.is_available(), "test requires a GPU")(test_case)" to "return unittest.skipUnless(torch.mlu.is_available(), "test requires a GPU")(test_case) " |
| 125 | src/accelerate/test_utils/testing.py:129 | change "return unittest.skipUnless(torch.cuda.device_count() == 1, "test requires a GPU")(test_case)" to "return unittest.skipUnless(torch.mlu.device_count() == 1, "test requires a GPU")(test_case) " |
| 126 | src/accelerate/test_utils/testing.py:145 | change "return unittest.skipUnless(torch.cuda.device_count() > 1, "test requires multiple GPUs")(test_case)" to "return unittest.skipUnless(torch.mlu.device_count() > 1, "test requires multiple GPUs")(test_case) " |
| 127 | src/accelerate/test_utils/training.py:16 | add "import torch_mlu" |
| 128 | src/accelerate/test_utils/scripts/test_cli.py:1 | add "import torch_mlu" |
| 129 | src/accelerate/test_utils/scripts/test_cli.py:5 | change "if torch.cuda.is_available():" to "if torch.mlu.is_available(): " |
| 130 | src/accelerate/test_utils/scripts/test_cli.py:6 | change "num_gpus = torch.cuda.device_count()" to "num_gpus = torch.mlu.device_count() " |
| 131 | src/accelerate/test_utils/scripts/test_ops.py:17 | add "import torch_mlu" |
| 132 | src/accelerate/test_utils/scripts/test_sync.py:17 | add "import torch_mlu" |
| 133 | src/accelerate/test_utils/scripts/test_distributed_data_loop.py:22 | add "import torch_mlu" |
| 134 | src/accelerate/test_utils/scripts/test_script.py:24 | add "import torch_mlu" |
| 135 | src/accelerate/test_utils/scripts/test_script.py:145 | change "synchronize_rng_states(["cuda"])" to "synchronize_rng_states(["mlu"]) " |
| 136 | src/accelerate/test_utils/scripts/test_script.py:146 | change "assert are_the_same_tensors(torch.cuda.get_rng_state()), "RNG states improperly synchronized on GPU."" to "assert are_the_same_tensors(torch.mlu.get_rng_state()), "RNG states improperly synchronized on GPU." " |
| 137 | src/accelerate/test_utils/scripts/test_script.py:352 | change "if torch.cuda.is_available():" to "if torch.mlu.is_available(): " |
| 138 | src/accelerate/test_utils/scripts/external_deps/test_checkpointing.py:20 | add "import torch_mlu" |
| 139 | src/accelerate/test_utils/scripts/external_deps/test_metrics.py:21 | add "import torch_mlu" |
| 140 | src/accelerate/test_utils/scripts/external_deps/test_metrics.py:79 | change "return {"ddp": [ddp_model, ddp_dataloader, "cuda:0"], "no": [model, dataloader, accelerator.device]}, accelerator" to "return {"ddp": [ddp_model, ddp_dataloader, "mlu:0"], "no": [model, dataloader, accelerator.device]}, accelerator " |
| 141 | src/accelerate/test_utils/scripts/external_deps/test_metrics.py:150 | change "if torch.cuda.is_available() or is_tpu_available():" to "if torch.mlu.is_available() or is_tpu_available(): " |
| 142 | src/accelerate/test_utils/scripts/external_deps/test_performance.py:20 | add "import torch_mlu" |
| 143 | src/accelerate/test_utils/scripts/external_deps/test_peak_memory_usage.py:20 | add "import torch_mlu" |
| 144 | src/accelerate/test_utils/scripts/external_deps/test_peak_memory_usage.py:43 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 145 | src/accelerate/test_utils/scripts/external_deps/test_peak_memory_usage.py:44 | change "torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero" to "torch.mlu.reset_max_memory_allocated()  # reset the peak gauge to zero " |
| 146 | src/accelerate/test_utils/scripts/external_deps/test_peak_memory_usage.py:45 | change "self.begin = torch.cuda.memory_allocated()" to "self.begin = torch.mlu.memory_allocated() " |
| 147 | src/accelerate/test_utils/scripts/external_deps/test_peak_memory_usage.py:50 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 148 | src/accelerate/test_utils/scripts/external_deps/test_peak_memory_usage.py:51 | change "self.end = torch.cuda.memory_allocated()" to "self.end = torch.mlu.memory_allocated() " |
| 149 | src/accelerate/test_utils/scripts/external_deps/test_peak_memory_usage.py:52 | change "self.peak = torch.cuda.max_memory_allocated()" to "self.peak = torch.mlu.max_memory_allocated() " |
| 150 | benchmarks/big_model_inference.py:18 | add "import torch_mlu" |
| 151 | benchmarks/measures_util.py:6 | add "import torch_mlu" |
| 152 | benchmarks/measures_util.py:44 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 153 | benchmarks/measures_util.py:51 | change "for i in range(torch.cuda.device_count()):" to "for i in range(torch.mlu.device_count()): " |
| 154 | benchmarks/measures_util.py:52 | change "measures[str(i)] = torch.cuda.memory_allocated(i)" to "measures[str(i)] = torch.mlu.memory_allocated(i) " |
| 155 | benchmarks/measures_util.py:53 | change "torch.cuda.reset_peak_memory_stats()" to "torch.mlu.reset_peak_memory_stats() " |
| 156 | benchmarks/measures_util.py:63 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 157 | benchmarks/measures_util.py:70 | change "for i in range(torch.cuda.device_count()):" to "for i in range(torch.mlu.device_count()): " |
| 158 | benchmarks/measures_util.py:71 | change "measures[str(i)] = (torch.cuda.memory_allocated(i) - start_measures[str(i)]) / 2**20" to "measures[str(i)] = (torch.mlu.memory_allocated(i) - start_measures[str(i)]) / 2**20 " |
| 159 | benchmarks/measures_util.py:72 | change "measures[f"{i}-peak"] = (torch.cuda.max_memory_allocated(i) - start_measures[str(i)]) / 2**20" to "measures[f"{i}-peak"] = (torch.mlu.max_memory_allocated(i) - start_measures[str(i)]) / 2**20 " |
| 160 | benchmarks/measures_util.py:80 | change "for i in range(torch.cuda.device_count()):" to "for i in range(torch.mlu.device_count()): " |
| 161 | tests/test_memory_utils.py:17 | add "import torch_mlu" |
| 162 | tests/test_memory_utils.py:20 | change "from accelerate.test_utils import require_cuda" to "from accelerate.test_utils import require_mlu " |
| 163 | tests/test_memory_utils.py:108 | change "@require_cuda" to "@require_mlu " |
| 164 | tests/test_memory_utils.py:110 | change "starting_memory = torch.cuda.memory_allocated()" to "starting_memory = torch.mlu.memory_allocated() " |
| 165 | tests/test_memory_utils.py:112 | change "model.cuda()" to "model.mlu() " |
| 166 | tests/test_memory_utils.py:113 | change "self.assertGreater(torch.cuda.memory_allocated(), starting_memory)" to "self.assertGreater(torch.mlu.memory_allocated(), starting_memory) " |
| 167 | tests/test_memory_utils.py:115 | change "self.assertEqual(torch.cuda.memory_allocated(), starting_memory)" to "self.assertEqual(torch.mlu.memory_allocated(), starting_memory) " |
| 168 | tests/test_examples.py:23 | add "import torch_mlu" |
| 169 | tests/test_examples.py:178 | change "if torch.cuda.is_available():" to "if torch.mlu.is_available(): " |
| 170 | tests/test_examples.py:179 | change "num_processes = torch.cuda.device_count()" to "num_processes = torch.mlu.device_count() " |
| 171 | tests/test_cli.py:20 | add "import torch_mlu" |
| 172 | tests/test_cli.py:57 | change "if torch.cuda.is_available() and (torch.cuda.device_count() > 1):" to "if torch.mlu.is_available() and (torch.mlu.device_count() > 1): " |
| 173 | tests/test_cli.py:89 | change "output = output.replace("Setting ds_accelerator to cuda (auto detect)\n", "")" to "output = output.replace("Setting ds_accelerator to mlu (auto detect)\n", "") " |
| 174 | tests/test_state_checkpointing.py:24 | add "import torch_mlu" |
| 175 | tests/test_state_checkpointing.py:29 | change "from accelerate.test_utils import execute_subprocess_async, require_cuda" to "from accelerate.test_utils import execute_subprocess_async, require_mlu " |
| 176 | tests/test_state_checkpointing.py:256 | change "@require_cuda" to "@require_mlu " |
| 177 | tests/test_state_checkpointing.py:259 | change "cmd += [f"--nproc_per_node={torch.cuda.device_count()}", inspect.getfile(self.__class__)]" to "cmd += [f"--nproc_per_node={torch.mlu.device_count()}", inspect.getfile(self.__class__)] " |
| 178 | tests/test_multigpu.py:19 | add "import torch_mlu" |
| 179 | tests/test_multigpu.py:38 | change "print(f"Found {torch.cuda.device_count()} devices.")" to "print(f"Found {torch.mlu.device_count()} devices.") " |
| 180 | tests/test_multigpu.py:39 | change "cmd = get_launch_prefix() + [f"--nproc_per_node={torch.cuda.device_count()}", self.test_file_path]" to "cmd = get_launch_prefix() + [f"--nproc_per_node={torch.mlu.device_count()}", self.test_file_path] " |
| 181 | tests/test_multigpu.py:45 | change "print(f"Found {torch.cuda.device_count()} devices.")" to "print(f"Found {torch.mlu.device_count()} devices.") " |
| 182 | tests/test_multigpu.py:46 | change "cmd = get_launch_prefix() + [f"--nproc_per_node={torch.cuda.device_count()}", self.operation_file_path]" to "cmd = get_launch_prefix() + [f"--nproc_per_node={torch.mlu.device_count()}", self.operation_file_path] " |
| 183 | tests/test_multigpu.py:53 | change "cmd = get_launch_prefix() + [f"--nproc_per_node={torch.cuda.device_count()}", inspect.getfile(self.__class__)]" to "cmd = get_launch_prefix() + [f"--nproc_per_node={torch.mlu.device_count()}", inspect.getfile(self.__class__)] " |
| 184 | tests/test_multigpu.py:63 | change "print(f"Found {torch.cuda.device_count()} devices, using 2 devices only")" to "print(f"Found {torch.mlu.device_count()} devices, using 2 devices only") " |
| 185 | tests/test_multigpu.py:64 | change "cmd = get_launch_prefix() + [f"--nproc_per_node={torch.cuda.device_count()}", self.data_loop_file_path]" to "cmd = get_launch_prefix() + [f"--nproc_per_node={torch.mlu.device_count()}", self.data_loop_file_path] " |
| 186 | tests/test_multigpu.py:65 | change "with patch_environment(omp_num_threads=1, cuda_visible_devices="0,1"):" to "with patch_environment(omp_num_threads=1, mlu_visible_devices="0,1"): " |
| 187 | tests/test_grad_sync.py:19 | add "import torch_mlu" |
| 188 | tests/test_grad_sync.py:52 | change "print(f"Found {torch.cuda.device_count()} devices.")" to "print(f"Found {torch.mlu.device_count()} devices.") " |
| 189 | tests/test_grad_sync.py:53 | change "cmd = get_launch_prefix() + [f"--nproc_per_node={torch.cuda.device_count()}", self.test_file_path]" to "cmd = get_launch_prefix() + [f"--nproc_per_node={torch.mlu.device_count()}", self.test_file_path] " |
| 190 | tests/test_accelerator.py:6 | add "import torch_mlu" |
| 191 | tests/test_accelerator.py:13 | change "from accelerate.test_utils.testing import AccelerateTestCase, require_cuda" to "from accelerate.test_utils.testing import AccelerateTestCase, require_mlu " |
| 192 | tests/test_accelerator.py:37 | change "@require_cuda" to "@require_mlu " |
| 193 | tests/test_accelerator.py:41 | change "assert PartialState._shared_state["device"].type == "cuda"" to "assert PartialState._shared_state["device"].type == "mlu" " |
| 194 | tests/test_accelerator.py:90 | change "# Mock torch.cuda.set_device to avoid an exception as the device doesn't exist" to "# Mock torch.mlu.set_device to avoid an exception as the device doesn't exist " |
| 195 | tests/test_accelerator.py:94 | change "with patch("torch.cuda.set_device", noop), patch_environment(ACCELERATE_TORCH_DEVICE="cuda:64"):" to "with patch("torch.mlu.set_device", noop), patch_environment(ACCELERATE_TORCH_DEVICE="mlu:64"): " |
| 196 | tests/test_accelerator.py:96 | change "self.assertEqual(str(accelerator.state.device), "cuda:64")" to "self.assertEqual(str(accelerator.state.device), "mlu:64") " |
| 197 | tests/test_accelerator.py:281 | change "@require_cuda" to "@require_mlu " |
| 198 | tests/test_utils.py:20 | add "import torch_mlu" |
| 199 | tests/test_utils.py:22 | change "from accelerate.test_utils.testing import require_cuda, require_torch_min_version" to "from accelerate.test_utils.testing import require_mlu, require_torch_min_version " |
| 200 | tests/test_utils.py:40 | change "device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")" to "device = torch.device("mlu") if torch.mlu.is_available() else torch.device("cpu") " |
| 201 | tests/test_utils.py:100 | change "@require_cuda" to "@require_mlu " |
| 202 | tests/test_utils.py:104 | change "model.forward = torch.cuda.amp.autocast(dtype=torch.float16)(model.forward)" to "model.forward = torch.mlu.amp.autocast(dtype=torch.float16)(model.forward) " |
| 203 | tests/test_utils.py:109 | change "@require_cuda" to "@require_mlu " |
| 204 | tests/test_utils.py:114 | change "model.forward = torch.cuda.amp.autocast(dtype=torch.float16)(model.forward)" to "model.forward = torch.mlu.amp.autocast(dtype=torch.float16)(model.forward) " |
| 205 | tests/test_utils.py:117 | change "inputs = torch.randn(4, 10).cuda()" to "inputs = torch.randn(4, 10).mlu() " |
| 206 | tests/test_big_modeling.py:19 | add "import torch_mlu" |
| 207 | tests/test_big_modeling.py:33 | change "from accelerate.test_utils import require_cuda, require_mps, require_multi_gpu, require_torch_min_version, slow" to "from accelerate.test_utils import require_mlu, require_mps, require_multi_gpu, require_torch_min_version, slow " |
| 208 | tests/test_big_modeling.py:125 | change "@require_cuda" to "@require_mlu " |
| 209 | tests/test_big_modeling.py:126 | change "def test_init_on_device_cuda(self):" to "def test_init_on_device_mlu(self): " |
| 210 | tests/test_big_modeling.py:127 | change "device = torch.device("cuda:0")" to "device = torch.device("mlu:0") " |
| 211 | tests/test_big_modeling.py:146 | change "device = torch.device(0 if torch.cuda.is_available() else "cpu")" to "device = torch.device(0 if torch.mlu.is_available() else "cpu") " |
| 212 | tests/test_big_modeling.py:168 | change "device = torch.device(0 if torch.cuda.is_available() else "cpu")" to "device = torch.device(0 if torch.mlu.is_available() else "cpu") " |
| 213 | tests/test_big_modeling.py:191 | change "@require_cuda" to "@require_mlu " |
| 214 | tests/test_big_modeling.py:209 | change "device = torch.device(0 if torch.cuda.is_available() else "cpu")" to "device = torch.device(0 if torch.mlu.is_available() else "cpu") " |
| 215 | tests/test_big_modeling.py:233 | change "device = torch.device(0 if torch.cuda.is_available() else "cpu")" to "device = torch.device(0 if torch.mlu.is_available() else "cpu") " |
| 216 | tests/test_big_modeling.py:261 | change "@require_cuda" to "@require_mlu " |
| 217 | tests/test_big_modeling.py:275 | change "@require_cuda" to "@require_mlu " |
| 218 | tests/test_big_modeling.py:301 | change "@require_cuda" to "@require_mlu " |
| 219 | tests/test_big_modeling.py:374 | change "@require_cuda" to "@require_mlu " |
| 220 | tests/test_big_modeling.py:419 | change "@require_cuda" to "@require_mlu " |
| 221 | tests/test_big_modeling.py:489 | change "@require_cuda" to "@require_mlu " |
| 222 | tests/test_big_modeling.py:571 | change "@require_cuda" to "@require_mlu " |
| 223 | tests/test_big_modeling.py:676 | change "device_map={"": torch.device("cuda:0")}," to "device_map={"": torch.device("mlu:0")}, " |
| 224 | tests/test_big_modeling.py:696 | change "device_map={"": "cuda:0"}," to "device_map={"": "mlu:0"}, " |
| 225 | tests/test_big_modeling.py:742 | change "device_map={"": torch.device("cuda:0")}," to "device_map={"": torch.device("mlu:0")}, " |
| 226 | tests/test_big_modeling.py:759 | change "device_map={"": "cuda:0"}," to "device_map={"": "mlu:0"}, " |
| 227 | tests/test_metrics.py:19 | add "import torch_mlu" |
| 228 | tests/test_metrics.py:61 | change "print(f"Found {torch.cuda.device_count()} devices.")" to "print(f"Found {torch.mlu.device_count()} devices.") " |
| 229 | tests/test_metrics.py:62 | change "cmd = get_launch_prefix() + [f"--nproc_per_node={torch.cuda.device_count()}", self.test_file_path]" to "cmd = get_launch_prefix() + [f"--nproc_per_node={torch.mlu.device_count()}", self.test_file_path] " |
| 230 | tests/test_hooks.py:18 | add "import torch_mlu" |
| 231 | tests/test_hooks.py:191 | change "hook_kwargs = {"execution_device": 0 if torch.cuda.is_available() else "cpu", "offload": True}" to "hook_kwargs = {"execution_device": 0 if torch.mlu.is_available() else "cpu", "offload": True} " |
| 232 | tests/test_hooks.py:219 | change ""execution_device": 0 if torch.cuda.is_available() else "cpu"," to ""execution_device": 0 if torch.mlu.is_available() else "cpu", " |
| 233 | tests/test_hooks.py:255 | change "execution_device = 0 if torch.cuda.is_available() else "cpu"" to "execution_device = 0 if torch.mlu.is_available() else "cpu" " |
| 234 | tests/test_hooks.py:304 | change "execution_device = 0 if torch.cuda.is_available() else "cpu"" to "execution_device = 0 if torch.mlu.is_available() else "cpu" " |
| 235 | tests/test_optimizer.py:18 | add "import torch_mlu" |
| 236 | tests/test_kwargs_handlers.py:20 | add "import torch_mlu" |
| 237 | tests/test_kwargs_handlers.py:24 | change "from accelerate.test_utils import execute_subprocess_async, require_cuda, require_multi_gpu" to "from accelerate.test_utils import execute_subprocess_async, require_mlu, require_multi_gpu " |
| 238 | tests/test_kwargs_handlers.py:43 | change "@require_cuda" to "@require_mlu " |
| 239 | tests/test_kwargs_handlers.py:64 | change "cmd += [f"--nproc_per_node={torch.cuda.device_count()}", inspect.getfile(self.__class__)]" to "cmd += [f"--nproc_per_node={torch.mlu.device_count()}", inspect.getfile(self.__class__)] " |
| 240 | tests/test_modeling_utils.py:21 | add "import torch_mlu" |
| 241 | tests/test_modeling_utils.py:25 | change "from accelerate.test_utils import require_cuda, require_huggingface_suite, require_multi_gpu, require_safetensors" to "from accelerate.test_utils import require_mlu, require_huggingface_suite, require_multi_gpu, require_safetensors " |
| 242 | tests/test_modeling_utils.py:123 | change "@require_cuda" to "@require_mlu " |
| 243 | tests/test_modeling_utils.py:128 | change "@require_cuda" to "@require_mlu " |
| 244 | tests/test_modeling_utils.py:299 | change "@require_cuda" to "@require_mlu " |
| 245 | tests/test_modeling_utils.py:334 | change "@require_cuda" to "@require_mlu " |
| 246 | tests/test_modeling_utils.py:515 | change "@require_cuda" to "@require_mlu " |
| 247 | tests/test_modeling_utils.py:537 | change "@require_cuda" to "@require_mlu " |
| 248 | tests/test_offload.py:19 | add "import torch_mlu" |
| 249 | tests/test_scheduler.py:18 | add "import torch_mlu" |
| 250 | tests/deepspeed/test_deepspeed.py:24 | add "import torch_mlu" |
| 251 | tests/deepspeed/test_deepspeed.py:40 | change "require_cuda," to "require_mlu, " |
| 252 | tests/deepspeed/test_deepspeed.py:96 | change "@require_cuda" to "@require_mlu " |
| 253 | tests/fsdp/test_fsdp.py:19 | add "import torch_mlu" |
| 254 | tests/fsdp/test_fsdp.py:31 | change "require_cuda," to "require_mlu, " |
| 255 | tests/fsdp/test_fsdp.py:55 | change "@require_cuda" to "@require_mlu " |
| 256 | examples/complete_nlp_example.py:19 | add "import torch_mlu" |
| 257 | examples/nlp_example.py:18 | add "import torch_mlu" |
| 258 | examples/multigpu_remote_launcher.py:4 | add "import torch_mlu" |
| 259 | examples/multigpu_remote_launcher.py:11 | change "num_processes = torch.cuda.device_count()" to "num_processes = torch.mlu.device_count() " |
| 260 | examples/complete_cv_example.py:21 | add "import torch_mlu" |
| 261 | examples/complete_cv_example.py:122 | change "torch.cuda.manual_seed_all(seed)" to "torch.mlu.manual_seed_all(seed) " |
| 262 | examples/cv_example.py:21 | add "import torch_mlu" |
| 263 | examples/cv_example.py:99 | change "torch.cuda.manual_seed_all(seed)" to "torch.mlu.manual_seed_all(seed) " |
| 264 | examples/by_feature/tracking.py:19 | add "import torch_mlu" |
| 265 | examples/by_feature/deepspeed_with_config_support.py:35 | add "import torch_mlu" |
| 266 | examples/by_feature/checkpointing.py:19 | add "import torch_mlu" |
| 267 | examples/by_feature/megatron_lm_gpt_pretraining.py:35 | add "import torch_mlu" |
| 268 | examples/by_feature/multi_process_metrics.py:19 | add "import torch_mlu" |
| 269 | examples/by_feature/cross_validation.py:20 | add "import torch_mlu" |
| 270 | examples/by_feature/automatic_gradient_accumulation.py:19 | add "import torch_mlu" |
| 271 | examples/by_feature/memory.py:19 | add "import torch_mlu" |
| 272 | examples/by_feature/gradient_accumulation.py:19 | add "import torch_mlu" |
| 273 | examples/by_feature/fsdp_with_peak_mem_tracking.py:20 | add "import torch_mlu" |
| 274 | examples/by_feature/fsdp_with_peak_mem_tracking.py:63 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 275 | examples/by_feature/fsdp_with_peak_mem_tracking.py:64 | change "torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero" to "torch.mlu.reset_max_memory_allocated()  # reset the peak gauge to zero " |
| 276 | examples/by_feature/fsdp_with_peak_mem_tracking.py:65 | change "self.begin = torch.cuda.memory_allocated()" to "self.begin = torch.mlu.memory_allocated() " |
| 277 | examples/by_feature/fsdp_with_peak_mem_tracking.py:70 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 278 | examples/by_feature/fsdp_with_peak_mem_tracking.py:71 | change "self.end = torch.cuda.memory_allocated()" to "self.end = torch.mlu.memory_allocated() " |
| 279 | examples/by_feature/fsdp_with_peak_mem_tracking.py:72 | change "self.peak = torch.cuda.max_memory_allocated()" to "self.peak = torch.mlu.max_memory_allocated() " |
| 280 | examples/by_feature/local_sgd.py:19 | add "import torch_mlu" |
