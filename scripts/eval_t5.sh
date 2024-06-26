
python3 eval_bart_t5.py \
    --checkpoint_path ./checkpoints_t5/decoding_eeg/best/task1_task2_finetune_T5Translator_skipstep1_b32_20_30_2e-05_2e-05_unique_sent_EEG.pt \
    --config_path ./config/decoding/task1_task2_finetune_T5Translator_skipstep1_b32_20_30_2e-05_2e-05_unique_sent_EEG.json \
    --test_input EEG \
    --train_input EEG \
    -cuda cuda:0

# fix error see eval t5 err