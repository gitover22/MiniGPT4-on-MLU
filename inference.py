import argparse
import torch
from PIL import Image
from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

def parse_args():
    """
    解析命令行参数,主要包括配置文件路径、GPU ID、图像路径、推理提示词以及配置选项覆盖等,
    """
    parser = argparse.ArgumentParser(description="MiniGPT-4 Inference")
    parser.add_argument("--cfg-path", required=True, help="Path to the configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="Specify the GPU to load the model.")
    parser.add_argument("--options", nargs="+", help="Override settings in the config. Use xxx=yyy format.")
    parser.add_argument("--image-path", required=True, help="Path to the image for inference.")
    parser.add_argument("--prompt", required=True, help="Text prompt for the model.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Load configuration and override with options
    cfg = Config(args)

    # Load model configuration and set GPU
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('mlu:{}'.format(args.gpu_id))

    # Load visual processor
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    # Initialize Chat instance
    chat = Chat(model, vis_processor, device='mlu:{}'.format(args.gpu_id))

    # Load and process the image
    image = Image.open(args.image_path)

    # Prepare conversation state and image list
    chat_state = CONV_VISION.copy()
    img_list = []

    # Process the image and prepare the chat state
    chat.upload_img(image, chat_state, img_list)

    # Generate the response using the provided prompt
    chat.ask(args.prompt, chat_state)
    response = chat.answer(
        conv=chat_state,
        img_list=img_list,
        num_beams=1,
        temperature=1.0,
        max_new_tokens=300,
        max_length=2000
    )[0]

    print("Model response:", response)

if __name__ == "__main__":
    main()
