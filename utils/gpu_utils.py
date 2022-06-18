import subprocess

def get_gpu_stats():
    """
    Returns GPU utilization and VRAM availability
    """
    result = subprocess.run(
        ["nvidia-smi",'--query-gpu=utilization.gpu,memory.total,memory.free', "--format=csv,nounits,noheader",],
        encoding="utf-8",
        stdout=subprocess.PIPE,
        check=True,
    )

    result = result.stdout.strip().split(", ")
    gpu_util = result[0]
    gpu_memory = str( 100 - int(result[2]) * 100 // int(result[1])) 
    return [ gpu_util, gpu_memory]
