import psutil

def monitor_resources():
    """Monitor CPU and RAM usage"""
    cpu_percent = psutil.cpu_percent()
    ram_percent = psutil.virtual_memory().percent
    return cpu_percent, ram_percent
