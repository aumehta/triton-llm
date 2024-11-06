# Triton LLM



before activating the virtual environment, run the following command:

```
module load Python
```

After this, activate the virtual environment and set the gpu resources that will be required. To run `test.py` for example, do the following after accessing the markov cluster:

```
srun -A xxh584_csds600 -p markov_gpu -C gpu2080 -N 1 -n 1 --gpus-per-node=1 --cpus-per-task=1 --mem-per-cpu=4gb --time=0:05:00 --pty /bin/bash
```
