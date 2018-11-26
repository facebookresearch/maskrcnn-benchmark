NGPUS=4 \
GPU_ID=1,2,3,4
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py \
    --config-file "configs/e2e_faster_rcnn_R_50_FPN_1x.yaml"