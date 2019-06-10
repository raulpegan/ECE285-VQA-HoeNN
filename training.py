import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from vqa import VQA
from data_loader import get_loader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Utilizing: {}".format(device))

batch_size = 256
epochs = 15

activation = nn.Tanh()
dropout = nn.Dropout(0.5)
combination = torch.mul


dataset_root_dir = './datasets/'
log_format = '{}-log-epoch-{:02}.txt'
ckpt_format = 'model-epoch-{:02d}.ckpt'

data_loader = get_loader(
        input_dir=dataset_root_dir,
        input_vqa_train='train.npy',
        input_vqa_valid='valid.npy',
        max_qst_length=30,
        max_num_ans=10,
        batch_size=batch_size,
        num_workers=8)

qst_vocab_size = data_loader['train'].dataset.qst_vocab.vocab_size
ans_vocab_size = data_loader['train'].dataset.ans_vocab.vocab_size

model = VQA(activation, dropout, combination, ans_vocab_size, qst_vocab_size).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

for epoch in range(epochs):
    for phase in ['train', 'valid']:

        running_loss = 0.0
        running_corr_exp = 0
        batch_step_size = len(data_loader[phase].dataset) / batch_size

        if phase == 'train':
            scheduler.step()
            model.train()
        else:
            model.eval()

        for batch_idx, batch_sample in enumerate(data_loader[phase]):

            image = batch_sample['image'].to(device)
            question = batch_sample['question'].to(device)
            label = batch_sample['answer_label'].to(device)
            multi_choice = batch_sample['answer_multi_choice']

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):

                output = model(image, question)
                _, pred_exp = torch.max(output, 1)
                loss = model.criterion(output, label)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item()
            running_corr_exp += torch.stack([(ans == pred_exp.cpu()) for ans in multi_choice]).any(dim=0).sum()

            # Print the average loss in a mini-batch.
            if batch_idx % 100 == 0:
                print('| {} SET | Epoch [{:02d}/{:02d}], Step [{:04d}/{:04d}], Loss: {:.4f}'
                      .format(phase.upper(), epoch, epochs, batch_idx, int(batch_step_size), loss.item()))
                
        # Print the average loss and accuracy in an epoch.
        epoch_loss = running_loss / batch_step_size
        epoch_acc_exp = running_corr_exp.double() / len(data_loader[phase].dataset)      # multiple choice

        print('| {} SET | Epoch [{:02d}/{:02d}], Loss: {:.4f}, Acc: {:.4f} \n'
              .format(phase.upper(), epoch, epochs, epoch_loss, epoch_acc_exp))

        # Log the loss and accuracy in an epoch.
        with open(os.path.join('./logs/', log_format).format(phase, epoch), 'w') as f:
            f.write(str(epoch+1) + '\t'
                    + str(epoch_loss) + '\t'
                    + str(epoch_acc_exp.item()))

    torch.save({'epoch': epoch, 'state_dict': model.state_dict()},
                os.path.join('./models/', ckpt_format.format(epoch)))
