#!/usr/bin/env bash

#--- Multi-nodes training hyperparams ---
nnodes=2
# master_addr="172.18.36.96"

# Note:
# 0. You need to set the master ip address according to your own machines.
# 1. You'd better to scale the learning rate when you use more gpus.
# 2. Command: sh scripts/run_train_multinodes.sh node_rank
############################################# 
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi

# python -m torch.distributed.launch --master_port 24084 --nproc_per_node=4 \
#             --nnodes=${nnodes} \
#             --node_rank=0  \
#             --master_addr=${master_addr} \
#             --use_env \
#             train_ddp.py

network_addr="10.10.20.12"
# network_addr="172.18.36.96"

python -m torch.distributed.launch --master_port 5555 --nproc_per_node=4 \
            --nnodes=${nnodes} \
            --node_rank=1  \
            --master_addr="$network_addr" \
            --use_env \
            train_ddp.py
