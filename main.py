# This module will run the two algorithms, measure memory usage and accuracy(precision)
import os
def memory_usage_psutil():
    # return the memory usage in MB
    import psutil
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss
    return mem