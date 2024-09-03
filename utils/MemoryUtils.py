import gc
import torch


class MemoryUtils:

    @staticmethod
    def free_gpu():
        torch.no_grad()
        torch.cuda.empty_cache()
        gc.collect()
