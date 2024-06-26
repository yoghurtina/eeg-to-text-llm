python3 eval_bart_base.py \
    --checkpoint_path ./checkpoints_bart_base/decoding/best/task1_task2_finetune_BrainTranslator_skipstep1_b32_20_30_2e-05_2e-05_unique_sent.pt \
    --config_path ./config/decoding/task1_task2_finetune_BrainTranslator_skipstep1_b32_20_30_2e-05_2e-05_unique_sent.json \
    -cuda cuda:0


# python3 eval_bart_t5.py \
#     --checkpoint_path /work3/s233095/eeg-to-text-llm/checkpoints_t5/decoding_eeg/best/task1_task2_finetune_T5Translator_skipstep1_b32_20_30_2e-05_2e-05_unique_sent_EEG.pt \
#     --config_path /work3/s233095/eeg-to-text-llm/config/decoding/task1_task2_finetune_T5Translator_skipstep1_b32_20_30_2e-05_2e-05_unique_sent_EEG.json \
#     --test_input EEG \
#     --train_input EEG \
#     -cuda cuda:0