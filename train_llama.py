import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import copy
import pickle
import time

from data import ZuCo_dataset
from model_llama import LlamaTranslator
from config import get_config

model_id="llama2-hf"


# Define training function
def train_model(dataloaders, device, model, criterion, optimizer, scheduler, num_epochs=1, checkpoint_path_best=None, checkpoint_path_last=None):
    # Time tracking for training duration
    since = time.time()

    # Best model and loss tracking
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    # Training loop
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'dev']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluation mode

            running_loss = 0.0

            # Iterate over data
            for inputs in dataloaders[phase]:
                # Move data to the correct device
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                labels = inputs['labels'].to(device)

                optimizer.zero_grad()  # Clear gradients

                with torch.set_grad_enabled(phase == 'train'):  # Track gradients only during training
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss

                    if phase == 'train':
                        loss.backward()  # Backpropagation
                        optimizer.step()  # Update weights

                running_loss += loss.item() * input_ids.size(0)

            # Learning rate scheduler
            if phase == 'train':
                scheduler.step()

            # Calculate average loss for the current epoch
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(f'{phase} Loss: {epoch_loss:.4f}')

            # Save the best model weights based on validation loss
            if phase == 'dev' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                if checkpoint_path_best:
                    torch.save(best_model_wts, checkpoint_path_best)  # Save best model
                    print(f'Updated best model on dev set: {checkpoint_path_best}')

    # Time tracking
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val loss: {best_loss:.4f}')

    # Save the last checkpoint
    if checkpoint_path_last:
        torch.save(model.state_dict(), checkpoint_path_last)
        print(f'Updated last checkpoint: {checkpoint_path_last}')

    # Load best model weights
    model.load_state_dict(best_model_wts)

    return model


if __name__ == '__main__':
    # Load configuration and set up paths
    args = get_config('train_decoding')

    # Configuration parameters
    dataset_setting = 'unique_sent'
    num_epochs_step1 = args['num_epoch_step1']
    num_epochs_step2 = args['num_epoch_step2']
    step1_lr = args['learning_rate_step1']
    step2_lr = args['learning_rate_step2']
    batch_size = args['batch_size']
    model_name = args['model_name']
    task_name = args['task_name']
    save_path = args['save_path']
    skip_step_one = args['skip_step_one']

    # Ensure the checkpoint directories exist
    os.makedirs(save_path, exist_ok=True)
    save_path_best = os.path.join(save_path, 'best')
    save_path_last = os.path.join(save_path, 'last')

    os.makedirs(save_path_best, exist_ok=True)
    os.makedirs(save_path_last, exist_ok=True)

    # Define checkpoint paths
    checkpoint_path_best = os.path.join(save_path_best, f'{task_name}_best_model.pt')
    checkpoint_path_last = os.path.join(save_path_last, f'{task_name}_last_model.pt')


    subject_choice = args['subjects']
    eeg_type_choice = args['eeg_type']
    bands_choice = args['eeg_bands']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('llama2-hf')

    # Initialize the model
    model = LlamaTranslator(pretrained_model_dir='llama2-hf')

    # Load the last checkpoint if available
    if os.path.exists(checkpoint_path_last):
        model.load_state_dict(torch.load(checkpoint_path_last))
        print(f'Loaded model from checkpoint at {checkpoint_path_last}')
    else:
        print("No checkpoint found, starting from scratch.")

    # Move model to device
    model.to(device)

    # Set up dataset and dataloader
    whole_dataset_dicts = []
    if 'task1' in task_name:
        dataset_path_task1 = './dataset/ZuCo/task1-SR/pickle/task1-SR-dataset.pickle' 
        with open(dataset_path_task1, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task2' in task_name:
        dataset_path_task2 = './dataset/ZuCo/task2-NR/pickle/task2-NR-dataset.pickle' 
        with open(dataset_path_task2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task3' in task_name:
        dataset_path_task3 = './dataset/ZuCo/task3-TSR/pickle/task3-TSR-dataset.pickle' 
        with open(dataset_path_task3, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'taskNRv2' in task_name:
        dataset_path_taskNRv2 = './dataset/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset.pickle' 
        with open(dataset_path_taskNRv2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

    print()
    train_set = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)
    # train_set = ZuCo_dataset(whole_dataset_dicts['train'], tokenizer, subject=subject_choice, eeg_type=eeg_type_choice, bands=bands_choice, setting=dataset_setting)
    dev_set = ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)
    # dev_set = ZuCo_dataset(whole_dataset_dicts['dev'], tokenizer, subject=subject_choice, eeg_type=eeg_type_choice, bands=bands_choice, setting=dataset_setting)
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(dev_set, batch_size=1, shuffle=False, num_workers=4)
    dataloaders = {'train': train_dataloader, 'dev': val_dataloader}

    # Initialize optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=step1_lr, momentum=0.9) if not skip_step_one else optim.SGD(model.parameters(), lr=step2_lr, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1) if not skip_step_one else lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Train the model
    trained_model = train_model(
        dataloaders,
        device,
        model,
        None,
        optimizer,
        scheduler,
        num_epochs=num_epochs_step1 if not skip_step_one else num_epochs_step2,
        checkpoint_path_best=checkpoint_path_best,
        checkpoint_path_last=checkpoint_path_last
    )
