#!/usr/bin/env python3
import os
import random
import yaml
import glob
import argparse

import math
import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import torch
import torchvision

import dataset_interface.object_detection.dataset as dataset
from dataset_interface.object_detection.engine import train_one_epoch, validate_one_epoch
import dataset_interface.object_detection.utils as utils
import dataset_interface.object_detection.transforms as T
from dataset_interface.object_detection.dataset import DatasetCustom, DatasetVOC, OnlineImageComposer

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-d', '--data_path', type=str,
                           help='Directory containing training/validation data and data annotations',
                           default='/home/lucy/data/')
    argparser.add_argument('-c', '--class_metadata', type=str,
                           help='Path to a file with category metadata',
                           default='/home/lucy/data/class_metadata.yaml')
    argparser.add_argument('-m', '--model_path', type=str,
                           help='Path to a directory where the trained models (one per epoch) should be saved',
                           default='/home/lucy/models')
    argparser.add_argument('-e', '--num_epochs', type=int,
                           help='Number of training epochs',
                           default=10)
    argparser.add_argument('-lr', '--learning_rate', type=float,
                           help='Initial learning rate',
                           default=0.005)
    argparser.add_argument('-b', '--training_batch_size', type=int,
                           help='Training batch size',
                           default=1)
    argparser.add_argument('-l', '--train_loss_file_path', type=str,
                           help='Path to a file in which training losses will be saved',
                           default='/home/lucy/data/train_loss.log')
    argparser.add_argument('-v', '--val_loss_file_path', type=str,
                           help='Path to a file in which validation losses will be saved',
                           default='/home/lucy/data/val_loss.log')
    argparser.add_argument('-at', '--annotation_type', type=str,
                           choices=['voc', 'custom'],
                           help='Data annotation type (voc or custom)',
                           default='voc')
    argparser.add_argument('-oa', '--online_augment', help='Use online augmentation for training',
                           default=False, action='store_true')

    # we read all arguments
    args = argparser.parse_args()
    data_path = args.data_path
    class_metadata_file_path = args.class_metadata
    model_path = args.model_path
    train_loss_file_path = args.train_loss_file_path
    val_loss_file_path = args.val_loss_file_path
    num_epochs = args.num_epochs
    training_batch_size = args.training_batch_size
    learning_rate = args.learning_rate
    annotation_type = args.annotation_type
    online_augment = args.online_augment

    print('\nThe following arguments were read:')
    print('------------------------------------')
    print('data_path:               {0}'.format(data_path))
    print('class_metadata:          {0}'.format(class_metadata_file_path))
    print('model_path:              {0}'.format(model_path))
    print('train_loss_file_path:    {0}'.format(train_loss_file_path))
    print('val_loss_file_path:      {0}'.format(val_loss_file_path))
    print('num_epochs:              {0}'.format(num_epochs))
    print('training_batch_size:     {0}'.format(training_batch_size))
    print('learning_rate:           {0}'.format(learning_rate))
    print('annotation_type:         {0}'.format(annotation_type))
    print('online_augment:          {0}'.format(online_augment))
    print('------------------------------------')
    print('Proceed with training (y/n)')
    proceed = input()
    if proceed != 'y':
        print('Aborting training')
        sys.exit(1)

    if not online_augment:
        # we read the class metadata
        class_metadata = utils.get_class_metadata(class_metadata_file_path)
        class_metadata = {v:k for k, v in class_metadata.items()}
        num_classes = len(class_metadata.keys())

        # we create a data loader by instantiating an appropriate
        # dataset class depending on the annotation type
        dataset = None
        if annotation_type.lower() == 'voc':
            dataset_train = DatasetVOC(data_path, get_transform(train=True), 'train', class_metadata)
        else:
            dataset_train = DatasetCustom(data_path, get_transform(train=True), 'train')
        dataset_val = dataset_train
    else:
        # Use online augmentation dataset
        print("Training using online data augmentation.")
        img_tf_ranges = {'rotation': (0, (360 - 5), 5),  # inc is added later
                         'resize': (0.75, 1.25, 0.1),
                         # 'shear_x': (0.0, 0.4, 0.05),
                         # 'shear_y': (0.0, 0.4, 0.05),
                         }
        img_app_ranges = {'contrast': (0.7, 1.0, 0.05),
                          'brightness': (0.7, 1.0, 0.05),
                          'sharpness': (0.1, 2.0, 0.1)
                          }
        range_dict = {'img_tf_ranges': img_tf_ranges,
                      'img_app_ranges': img_app_ranges}
        bg_dict = {'solid_colors': [(255, 255, 255), (0, 0, 0), (255, 0, 0), (0, 255, 0),
                                    (0, 0, 255), (255, 255, 0), (0, 255, 255), (128, 0, 0),
                                    (128, 128, 128), (128, 128, 0), (0, 128, 0), (128, 0, 128),
                                    (0, 128, 128)],
                   'img_background_paths': [f for f in glob.iglob(os.path.join(data_path, 'backgrounds/*.jpg'))],
                   # 'color_spaces': [] # Add color map support later
                   }
        dataset_train = OnlineImageComposer(data_path, class_metadata_file_path, bg_dict, range_dict,
                                            get_transform(train=True), True)

        dataset_val = OnlineImageComposer(data_path, class_metadata_file_path, bg_dict, range_dict,
                                          get_transform(train=False), True)

        class_metadata = dataset_train.label_folder_map
        with open('label_folder_map.yml', 'w') as outfile:
            yaml.dump(class_metadata, outfile, default_flow_style=False)
        class_metadata[0] = '__background' # Add background class
        class_metadata = {v: k for k, v in class_metadata.items()}
        num_classes = len(class_metadata.keys())

    # we split the dataset into train and validation sets
    indices = torch.randperm(len(dataset_train)).tolist()
    train_split = math.ceil(0.7*len(indices))

    dataset_train = torch.utils.data.Subset(dataset_train, indices[0:train_split])
    dataset_val = torch.utils.data.Subset(dataset_val, indices[train_split:])

    # we define training and validation data loaders
    data_loader_train = torch.utils.data.DataLoader(dataset_train,
                                                    batch_size=training_batch_size,
                                                    shuffle=True, num_workers=4,
                                                    collate_fn=utils.collate_fn)

    data_loader_val = torch.utils.data.DataLoader(dataset_val,
                                                  batch_size=training_batch_size,
                                                  shuffle=True, num_workers=4,
                                                  collate_fn=utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = utils.get_model(num_classes)
    # we move the model to the correct device before training
    model.to(device)

    # we now define an optimiser, and train
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.Adam(params, lr=learning_rate)
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    start_epoch = 0
    if not os.path.isdir(model_path):
        print('Creating model directory {0}'.format(model_path))
        os.mkdir(model_path)
    else:
        max_count = -1
        checkpoint_path = ''
        for ckpt in glob.iglob(os.path.join(model_path, '*.pt')):
            epoch_num = int(ckpt.split('/')[-1].split('.')[0].split('model_')[-1])
            if epoch_num > max_count:
                max_count = epoch_num
                checkpoint_path = ckpt
        if max_count > -1:
            print("Checkpoint from epoch {} found!".format(max_count))
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1

    # we clear the files in which the training and validation losses are saved
    open(train_loss_file_path, 'w').close()
    open(val_loss_file_path, 'w').close()

    print('Training will start from epoch {0}'.format(start_epoch))
    print('Training model for {0} epochs'.format(num_epochs - start_epoch))
    for epoch in range(start_epoch, num_epochs):
        train_one_epoch(model, optimizer, data_loader_train, device, epoch,
                        print_freq=10, loss_file_name=train_loss_file_path)
        lr_scheduler.step()
        validate_one_epoch(model, data_loader_val, device, epoch,
                           print_freq=10, loss_file_name=val_loss_file_path)

        checkpoint = {}
        checkpoint['model'] = model.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint['epoch'] = epoch
        torch.save(checkpoint, os.path.join(model_path, 'model_{0}.pt'.format(epoch)))
    print('Training done')
