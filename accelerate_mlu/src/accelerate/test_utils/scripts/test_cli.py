import torch
import torch_mlu


def main():
    if torch.mlu.is_available():
        num_gpus = torch.mlu.device_count()
    else:
        num_gpus = 0
    print(f"Successfully ran on {num_gpus} GPUs")


if __name__ == "__main__":
    main()
