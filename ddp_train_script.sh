torchrun --nproc_per_node=4 \
         --nnodes=2 \
         --node_rank=1  \ 
         --master_addr="172.18.36.96" \
         --master_port=24084 \
         train_ddp.py