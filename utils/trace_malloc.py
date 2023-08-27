import numpy as np
import torch
import gc
import psutil
import threading

def b2mb(x):
    return int(x / 2**20)

# 用于跟踪和记录代码的内存使用情况
class TorchTracemalloc:
    # 收集垃圾和清空GPU缓存。
    # 重置 GPU 的最大内存使用量（peak memory allocated）计数器，将其设为零。
    # 记录进入上下文管理器时的 GPU 内存使用情况和 CPU 内存使用情况，并启动一个线程用于监控 CPU 内存的峰值。
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
        self.begin = torch.cuda.memory_allocated()
        self.process = psutil.Process()

        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self
    
    # 获取进入上下文管理器时的 GPU 内存使用情况和 CPU 内存使用情况，并停止监控 CPU 内存的峰值。
    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    # 停止监控 CPU 内存的峰值。
    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring:
                break

    # 停止监控 CPU 内存的线程。
    # 再次收集垃圾和清空GPU缓存。
    # 记录退出上下文管理器时的 GPU 内存使用情况和 CPU 内存使用情况，并计算内存使用量的差值，得到实际的内存消耗和峰值内存消耗。
    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)
        # print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")
        
        return False



        