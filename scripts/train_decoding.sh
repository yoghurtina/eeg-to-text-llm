python3 train_decoding.py --model_name BrainTranslator \
    --task_name task1_task2 \
    --one_step \
    --pretrained \
    --not_load_step1_checkpoint \
    --num_epoch_step1 1 \
    --num_epoch_step2 1 \
    -lr1 0.00005 \
    -lr2 0.0000005 \
    -b 32 \
    -s ./checkpoints/decoding \
    -cuda cuda:0


