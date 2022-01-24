n_gpu=4

NETWORK=hourglass104_MRCB_cascade # hourglass104_MRCB_cascade, hourglass104_MRCB, hhrnet48, hhrnet32
BATCH=8
EPOCH=120
LR=2.5e-4
SIZE=1024

python -m torch.distributed.launch --nproc_per_node=${n_gpu} train.py \
--backbone ${NETWORK} --dataset DOTA --input_size ${SIZE} --batch_size ${BATCH} \
--lr ${LR} --epochs ${EPOCH} --workers 4 --print_freq 100 --sync_bn --amp
