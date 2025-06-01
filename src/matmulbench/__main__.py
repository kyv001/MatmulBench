import torch
from torch import nn
import time

if __name__ == "__main__":
    print(f"Torch版本：{torch.__version__}")
    print(f"CUDA版本：{torch.version.cuda}")
    print(f"使用设备：{torch.cuda.get_device_name(0)}")
    print(f"数据类型：bfloat16")

    size = 8192
    dtype = torch.bfloat16
    device = "cuda:0"
    a = torch.randn(size, size, dtype=dtype, device="cuda:0")
    b = torch.randn(size, size, dtype=dtype, device="cuda:0")
    c = torch.empty(size, size, dtype=dtype, device="cuda:0")

    print("预热中……")
    for _ in range(20):
        torch.matmul(a, b)
    
    repeats = 100
    torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    for _ in range(repeats):
        torch.matmul(a, b, out=c)
    torch.cuda.synchronize()
    t_seconds = (time.perf_counter_ns() - t0) / 1e9

    total_ops = 2 * size ** 3 * repeats
    tflops = total_ops / t_seconds / 1e12

    print(f"{size}x{size}矩阵乘法，{repeats}次，平均耗时{t_seconds / repeats:.5f}秒，{tflops} TFLOPS")
