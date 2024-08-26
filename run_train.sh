# set -ex

export MLU_VISIBLE_DEVICES=0,1,2,3
DEVICES_NUMS=$(echo $MLU_VISIBLE_DEVICES | awk -F "," '{print NF}')
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

python -m torch.distributed.launch \
	--master_addr=$MASTER_ADDR \
	--nproc_per_node=$DEVICES_NUMS \
	--master_port=$PORT \
	train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml \
	--options model.llama_model='/home/zouguoqiang/MiniGPT4-on-MLU/data/models/vicuna-7b-all-v1.1' \
    model.ckpt='/home/zouguoqiang/MiniGPT4-on-MLU/data/models/prerained_minigpt4_7b.pth' \
