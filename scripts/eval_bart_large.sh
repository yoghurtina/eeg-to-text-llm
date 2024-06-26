python3 eval_bart_large.py \
    --checkpoint_path ./checkpoints_bart_large/decoding_eeg/best/task1_task2_finetune_BrainTranslator_skipstep1_b32_20_30_2e-05_2e-05_unique_sent.pt \
    --config_path ./config/decoding/task1_task2_finetune_BrainTranslator_skipstep1_b32_20_30_2e-05_2e-05_unique_sent.json \
    -cuda cuda:0

