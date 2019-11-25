import manage_gpus as gpl
import os
import sys


def get_lock():
    gpu_id_locked = -1
    try:
        gpu_id_locked = gpl.get_gpu_lock(gpu_device_id=-1, soft=False)
    except gpl.NoGpuManager:
        print("no gpu manager available - will use all available GPUs",
              file=sys.stderr)
    except gpl.NoGpuAvailable:
        # there is no GPU available for locking, continue with CPU
        comp_device = "/cpu:0"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return gpu_id_locked
