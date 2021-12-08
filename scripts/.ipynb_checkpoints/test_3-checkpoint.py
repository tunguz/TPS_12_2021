import numpy as np
import matplotlib.pyplot as plt
from dask.distributed import Client, LocalCluster 

if __name__ == '__main__':
    cluster = LocalCluster(
    n_workers=4,
    processes=True,
    threads_per_worker=1
    )
    with Client(cluster) as client:
        x = np.linspace(0, 1, 100)
        y = x * x
