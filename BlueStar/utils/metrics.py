import psutil
import os
from typing import Tuple

def monitor_resources() -> Tuple[float, float]:
    """Monitor CPU and RAM usage"""
    try:
        cpu_percent = psutil.cpu_percent()
        
        ram = psutil.virtual_memory()
        ram_percent = ram.percent
        
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        
        print(f"[DEBUG] [metrics.py] CPU: {cpu_percent}%, RAM: {ram_percent}%, Disk: {disk_percent}%")
        
        return cpu_percent, ram_percent
    except Exception as e:
        print(f"[ERROR] [metrics.py] Error monitoring resources: {str(e)}")
        return 0.0, 0.0

def get_process_memory() -> dict:
    """Get detailed memory usage for current process"""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss': memory_info.rss / (1024 * 1024),  ## RSS in MB
            'vms': memory_info.vms / (1024 * 1024),  ## VMS in MB
            'percent': process.memory_percent()
        }
    except Exception as e:
        print(f"[ERROR] [metrics.py] Error getting process memory: {str(e)}")
        return {'rss': 0, 'vms': 0, 'percent': 0}
