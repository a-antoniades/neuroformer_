# train with multiple GPUs
CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 V1_AL.py
CUDA_VISIBLE_DEVICES=1,2,3,6 torchrun --nproc_per_node=5 natmovie.py
CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --nproc_per_node=4 natmovie.py
kill -9 `netstat -nltp | grep python | cut -d/ -f 1 | awk '{print $NF}'`
