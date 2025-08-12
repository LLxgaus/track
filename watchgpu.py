import pynvml
import time

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

while True:
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"Used: {info.used/1024**2:.2f} MB / Total: {info.total/1024**2:.2f} MB")
    time.sleep(1)