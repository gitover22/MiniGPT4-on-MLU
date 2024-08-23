set -ex


python inference.py \
	--cfg-path inference_configs/minigpt4_inference.yaml \
	--options model.llama_model='/data/llm/models/vicuna-7b-all-v1.1' model.ckpt='/data/llm/models/prerained_minigpt4_7b.pth' \
	--image-path ./inference.jpg \
	--prompt "describe the contents of this image.Give this image a name based on its contents."
