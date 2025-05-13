import psutil
import threading
import time
import logging
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates, nvmlShutdown


class ResourceMonitor:

    def __init__(self, tensorboard_writer=None, gpu_index=0):
        self.gpu_index = gpu_index
        self.tensorboard_writer = tensorboard_writer
        self.start_time = time.time()

    def __enter__(self):
        self.running = True

        # Initialize NVML for GPU monitoring
        nvmlInit()
        self.gpu_handle = nvmlDeviceGetHandleByIndex(self.gpu_index)
        assert self.gpu_handle is not None, "Failed to get GPU handle"

        self.monitor_thread = threading.Thread(target=self.monitor)
        self.monitor_thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def stop(self):
        self.running = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join()

        nvmlShutdown()

    def current_physical_memory(self):
        """Get current physical memory usage in KB."""
        process = psutil.Process()
        mem_info = process.memory_info()
        return mem_info.rss / (1024**3)  # rss is in bytes, converting to GB

    def current_cpu_usage(self):
        """Get the current process CPU usage percentage."""
        return psutil.cpu_percent(interval=None)

    def system_cpu_usage(self):
        """Get the overall system CPU usage percentage."""
        return psutil.cpu_percent(interval=None)

    def system_memory_usage(self):
        """Get system-wide memory usage in GB."""
        mem_info = psutil.virtual_memory()
        return (mem_info.total - mem_info.available) / (1024**3
                                                        )  # Convert to GB

    def gpu_memory_usage(self):
        """Get the current GPU memory usage in GB."""
        if self.gpu_handle is not None:
            mem_info = nvmlDeviceGetMemoryInfo(self.gpu_handle)
            return mem_info.used / (1024**3)  # Convert to GB
        return None

    def gpu_usage(self):
        """Get the current GPU utilization percentage."""
        if self.gpu_handle is not None:
            utilization = nvmlDeviceGetUtilizationRates(self.gpu_handle)
            return utilization.gpu  # GPU usage in percentage
        return None

    def monitor(self):
        """Monitoring CPU, memory, and GPU usage."""
        while self.running:
            mem_usage = self.current_physical_memory()
            cpu_usage = self.current_cpu_usage()

            sys_cpu_usage = self.system_cpu_usage()
            sys_mem_usage = self.system_memory_usage()

            gpu_mem_usage = self.gpu_memory_usage()
            gpu_usage = self.gpu_usage()

            timestamp = time.time() - self.start_time

            self.tensorboard_writer.add_scalar(
                "ResourceMonitor/Process_Memory_GB", mem_usage, timestamp)
            self.tensorboard_writer.add_scalar(
                "ResourceMonitor/Process_CPU_Percent", cpu_usage, timestamp)
            self.tensorboard_writer.add_scalar(
                "ResourceMonitor/System_CPU_Percent", sys_cpu_usage, timestamp)
            self.tensorboard_writer.add_scalar(
                "ResourceMonitor/System_Memory_GB", sys_mem_usage, timestamp)
            if gpu_mem_usage is not None and gpu_usage is not None:
                self.tensorboard_writer.add_scalar(
                    "ResourceMonitor/GPU_Memory_GB", gpu_mem_usage, timestamp)
                self.tensorboard_writer.add_scalar(
                    "ResourceMonitor/GPU_Percent", gpu_usage, timestamp)

            time.sleep(0.5)  # Sleep for 500 milliseconds
