import os
import csv

import numpy as np
import torch


def create_output_dir(exp):
    out_dir = f'./models/{exp}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    losses_path = create_losses_csv(out_dir)

    return out_dir, losses_path


def create_losses_csv(out_dir):
    losses_path = f'{out_dir}/losses.csv'

    with open(f'{losses_path}', mode='w') as losses_csv:
        fields = ['t_abnormal', 't_acl', 't_meniscus',
                    'v_abnormal', 'v_acl', 'v_meniscus']
        writer = csv.DictWriter(losses_csv, fieldnames=fields)
        writer.writeheader()

    return losses_path


def print_losses(batch_train_losses, batch_valid_losses):
    print(f'Train losses - abnormal: {batch_train_losses[0]:.3f},',
          f'acl: {batch_train_losses[1]:.3f},',
          f'meniscus: {batch_train_losses[2]:.3f}',
          f'\nValid losses - abnormal: {batch_valid_losses[0]:.3f},',
          f'acl: {batch_valid_losses[1]:.3f},',
          f'meniscus: {batch_valid_losses[2]:.3f}')


def save_losses(train_losses , valid_losses, losses_path):
    with open(f'{losses_path}', mode='a') as losses_csv:
        writer = csv.writer(losses_csv)
        writer.writerow(np.append(train_losses, valid_losses))


def save_checkpoint(epoch, plane, diagnosis, model, optimizer, out_dir):
    print(f'Min valid loss for {diagnosis}, saving the checkpoint...')

    checkpoint = {
        'epoch': epoch,
        'plane': plane,
        'diagnosis': diagnosis,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    chkpt = f'mrnet_p-{plane}_d-{diagnosis}_e-{epoch:02d}.pt'
    torch.save(checkpoint, f'{out_dir}/{chkpt}')
