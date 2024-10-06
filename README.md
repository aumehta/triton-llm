# Triton LLM

to run `test.py`, use the following command after accessing the markov cluster:

```
srun -A xxh584_csds600 -p markov_gpu -C gpu2080 -N 1 -n 1 --gpus-per-node=1 --cpus-per-task=1 --mem-per-cpu=4gb --time=0:05:00 --pty /bin/bash
```