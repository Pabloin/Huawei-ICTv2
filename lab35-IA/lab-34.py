from easydict import EasyDict as edict
# Dictionary access, used to store hyperparameters
import os
# os module, used to process files and directories
import numpy as np
# Scientific computing library
import matplotlib.pyplot as plt
# Graphing library
import mindspore
# MindSpore library
import mindspore.dataset as ds
# Dataset processing module
from mindspore.dataset.vision import c_transforms as vision
# Image enhancement module



#-------------------------------------------


from mindspore import context
#Environment setting module
import mindspore.nn as nn
# Neural network module
from mindspore.train import Model
# Model build
from mindspore.nn.optim.momentum import Momentum
# Momentum optimizer
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
# Model saving settings
from mindspore import Tensor
# Tensor
from mindspore.train.serialization import export
# Model export
from mindspore.train.loss_scale_manager import FixedLossScaleManager
# Loss value smoothing
from mindspore.train.serialization import load_checkpoint, load_param_into_net
# Model loading
import mindspore.ops as ops
# Common operators
# MindSpore execution mode and device setting
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


#----------------------------------------------------------------------

cfg = edict({
'data_path':'flowers/flower_photos_train',  # Path of the training dataset
'test_path':'flowers/flower_photos_train',  # Path of the test dataset
'data size': 3616,
'HEIGHT': 224, # Image height
'WIDTH':224, # Image width
'_R_MEAN': 123.68, # Average value of CIFAR-IO
'_G_MEAN': 116.78,
'_B_MEAN': 103.94,
'_R_STD': 1, # Customized standard deviation
'_G_STD': 1,
'_B_STD': 1,
'_RESIZE_SIDE_MIN': 256, # Minimum resizepalue for image enhancement
'_RESIZE_SIDE_MAX': 512,
'batch_size': 32, # Batch size
'num_class': 5,
# Number of classes
'epoch_size': 5, # Number of training times
'loss_scale_num':1024,
'prefix': 'resnet-ai', # Name of the model
'directory': './model_resnet', # Path for storing the model
'save_checkpoint_steps': 10, # The checkpoint is saved every 10 steps.
})

#----------------------------------------------------------------------

# STEP 3

# Data processing
def read_data(path,config,usage="train"):
    # Read the source dataset of an image from a directory.
    dataset = ds.imageFolderDataset(path,
    class_indexing={'daisy':0,'dandelion':1,'roses':2, 'sunflowers':3, 'tulips' :4})
    
    
    # define map operations
    # Operator for image decoding
    decode_op = vision.Decode()
    # Operator for image normalization
    normalize_op = vision.Normalize(mean=[cfg._R_MEAN, cfg._G_MEAN, cfg._B_MEAN], 
                                     std=[cfg._R_STD,  cfg._G_STD,  cfg._B_STD])
                                                                                       
    # Operator for image resizing
    resize_op = vision. Resize(cfg._RESlZE_SlDE_MlN)
    # Operator for image cropping
    center_crop_op = vision.CenterCrop((cfg. HEIGHT, cfg.WlDTH))
    # Operator for image random horizontal flipping
    horizontal_flip_op = vision.RandomHorizontalFlip()
    # Operator for image channel quantity conversion
    channelswap_op = vision.HWC2CHW()
    # Operator for random image cropping, decoding, encoding, and resizing
    random_crop_decode_resize_op = vision.RandomCropDecodeResize((cfg.HEIGHT, cfg.WIDTH), (0.5, 1.0), (1.0,
    1.0), max_attempts=100)
    
    # Preprocess the training set.
    if usage == 'train':
        dataset = dataset.map(input_columns="image", operations=random_crop_decode_resize_op)
        dataset = dataset.map(input_columns="image", operations=horizontal_flip_op)
    # Preprocess the test set.
    else:
        dataset = dataset.map(input_columns="image", operations=decode_op)
        dataset = dataset.map(input_columns="image", operations=resize_op)
        dataset = dataset.map(input_columns="image", operations=center_crop_op)
    
    # Preprocess all datasets.
    dataset = dataset.map(input_columns="image", operations=normalize_op)
    dataset = dataset.map(input_columns="image", operations=channelswap_op)





#----------------------------------------------------------------------


    # Batch the training set.
    if usage == 'train':
        dataset = dataset.shuffle(buffer_size=10000) # 10000 as in imageNet train script
        dataset = dataset.batch(cfg.batch_size, drop_remainder=True)
    # Batch the test set.
    else:
        dataset = dataset.batch(1, drop_remainder=True)
    
    # Data augmentation
    dataset = dataset.repeat(1)
    
    dataset.map_model = 4
    
    return dataset
    
# Display the numbers of training sets and test sets.
de_train = read_data(cfg.data_path,cfg,usage="train")
de_test  = read_data(cfg.test_path,cfg,usage="test")
print('Number of training datasets: '. de_train.get_dataset_size()*cfg.batch_size)# get_dataset_size() obtains the batch processing size.
print('Number of test datasets: ',de_test.get_dataset_size())
# Display the sample graph of the training set.
data_next = de_train.create_dict_iterator(output_numpy=True).__next__()

print('Number of channels/lmage length/width: ', data_next['image'][0,...].shape)
print('Label Style of an image: ', data_next['label'][0]) # Total 5 label classes which are represented by numbers from O to 4.

plt.figure()
plt.imshow(data_next['image'][0,0,...])
plt.colorbar()
plt.grid(False)
plt.show()

#----------------------------------------------------------------------






#----------------------------------------------------------------------




#----------------------------------------------------------------------






#----------------------------------------------------------------------
