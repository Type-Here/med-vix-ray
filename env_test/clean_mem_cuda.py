import torch

def clean_mem_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("CUDA memory cleaned.")


if __name__ == "__main__":
    clean_mem_cuda()