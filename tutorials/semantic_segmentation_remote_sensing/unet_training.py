'''
    Script version of the tutorial on dense semantic segmentation of remote
    sensing data.

    2021 Benjamin Kellenberger
'''

import os

import numpy as np
from PIL import Image                   # Torchvision usually relies on PIL, but anything else can be used, too

import torch
import torch.nn as nn
from torch.utils.data import dataset, dataloader
import torch.optim
import torch.optim.lr_scheduler

import torchvision.transforms as T      # transformations that can be used e.g. for data conversion or augmentation

from tqdm import trange                 # this gives us a nice progress bar




''' ------------------------------------------------------------------------
    1. PREPARE DATASET
    ------------------------------------------------------------------------ '''

# The first step we will take is to define our dataset. This usually is a long
# list of image-ground truth pairs. However, we cannot load all data into
# memory, so the compromise is to store all image file names and load the images
# whenever they get requested. PyTorch does this by means of a "Dataset" class,
# which gets called by a "DataLoader".

class VaihingenDataset(dataset.Dataset):
    '''
        Custom Dataset class that loads images and ground truth segmentation
        masks from a directory.
    '''

    # image statistics, calculated in advance as averages across the full
    # training data set
    IMAGE_MEANS = (
        (121.03431026287558, 82.52572736507886, 81.92368178210943),     # IR-R-G tiles
        (285.34753853934154),                                           # DSM
        (31.005143030549313)                                            # nDSM
    )
    IMAGE_STDS = (
        (54.21029197978022, 38.434924159900554, 37.040640374137475),    # IR-R-G tiles
        (6.485453035150256),                                            # DSM
        (36.040236155124326)                                            # nDSM
    )

    def __init__(self, data_root):
        '''
            Dataset class constructor. Here we initialize the dataset instance
            and retrieve file names (and other metadata, if present) for all the
            images and labels (ground truth semantic segmentation maps).
        '''
        super().__init__()

        self.data_root = data_root

        # find all images. In our case they are listed in a CSV file called
        # "fileList.csv" under the "data_root"
        with open(os.path.join(self.data_root, 'fileList.csv'), 'r') as f:
            lines = f.readlines()
        
        # parse CSV lines into data tokens: first column is the label file, the
        # remaining ones are the image files
        self.data = []
        for line in lines[1:]:      # skip header
            self.data.append(line.strip().split(','))


    def __len__(self):
        '''
            This function tells the Data Loader how many images there are in
            this dataset.
        '''
        return len(self.data)

    
    def __getitem__(self, idx):
        '''
            Here's where we load, prepare, and convert the images and
            segmentation mask for the data element at the given "idx".
        '''
        item = self.data[idx]

        # load segmentation mask (remember: first column of CSV file)
        labels = Image.open(os.path.join(self.data_root, 'labels', item[0]))
        labels = np.array(labels, dtype=np.int64)   # convert to NumPy array temporarily

        # load all images (remaining columns of CSV file)
        images = [Image.open(os.path.join(self.data_root, 'images', i)) for i in item[1:]]

        # NOTE: at this point it would make sense to perform data augmentation.
        # However, the default augmentations built-in to PyTorch (resp.
        # Torchvision) (i.) only support RGB images; (ii.) only work on the
        # images themselves. In our case, however, we have multispectral data
        # and need to also transform the segmentation mask.
        # This is not difficult to do, but goes beyond the scope of this tutorial.
        # For the sake of brevity, we'll leave it out accordingly.
        # What we will have to do, however, is to normalize the image data.
        for i in range(len(images)):
            img = np.array(images[i], dtype=np.float32)                 # convert to NumPy array (very similar to torch.Tensor below)
            img = (img - self.IMAGE_MEANS[i]) / self.IMAGE_STDS[i]      # normalize
            images[i] = img

        # finally, we need to convert our data into the torch.Tensor format. For
        # the images, we already have a "ToTensor" transform available, but we
        # need to concatenate the images together.
        tensors = [T.ToTensor()(i) for i in images]
        tensors = torch.cat(tensors, dim=0).float()         # concatenate along spectral dimension and make sure it's in 32-bit floating point

        # For the labels, we need to convert the PIL image to a torch.Tensor.
        labels = torch.from_numpy(labels).long()            # labels need to be in 64-bit integer format

        return tensors, labels


# We now have a dataset class, so the next step is to make it available to a
# DataLoader, for each training and validation set:

def load_dataset(data_root, split='train', batch_size=8):
    
    # initialize dataset
    dataset = VaihingenDataset(f'{data_root}/{split.lower()}')

    # initialize and return Data Loader
    dataLoader = dataloader.DataLoader(
        dataset,
        batch_size=batch_size,              # define your batch size here
        shuffle=(split=='train'),           # randomize image order (for training only)
        num_workers=4                       # multi-threading for maximum performance when loading data
    )
    return dataLoader



''' ------------------------------------------------------------------------
    2. CREATE MODEL
    ------------------------------------------------------------------------ '''

# First, we have to define the layout (graph, structure, etc.) of our model. In
# this tutorial, we will be using U-Net, one of the most popular models for
# semantic segmentation. U-Net is described in the following paper:
#
#       Ronneberger, O., Fischer, P. and Brox, T., 2015, October. U-net:
#           Convolutional networks for biomedical image segmentation. In
#           International Conference on Medical image computing and com-
#           puter-assisted intervention (pp. 234-241). Springer, Cham.
#
# Originally, U-net had been designed for medical image segmentation, with
# single channel (black & white) images. Below, we will allow it to also accept
# RGB images, or even multispectral data.

class UNet(nn.Module):
    '''
        Simple U-Net definition in PyTorch.
        Adapted and modified from https://github.com/usuyama/pytorch-unet.
    '''

    def __init__(self, num_classes, num_input_channels=3):
        '''
            Constructor. Here, we define the layers this U-Net class will
            contain.
        '''
        super().__init__()
        
        self.dconv_down1 = UNet.double_conv(num_input_channels, 64)     # allow multispectral inputs
        self.dconv_down2 = UNet.double_conv(64, 128)
        self.dconv_down3 = UNet.double_conv(128, 256)
        self.dconv_down4 = UNet.double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.dconv_up3 = UNet.double_conv(256 + 512, 256)
        self.dconv_up2 = UNet.double_conv(128 + 256, 128)
        self.dconv_up1 = UNet.double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, num_classes, 1)                  # map to number of classes
    
    @staticmethod
    def double_conv(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        ) 
        
    def forward(self, x):
        '''
            Definition of the forward pass. In the constructor above, we defined
            what layers the model has, here we specify in which order and on
            what inputs they will be used.
        '''
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)        # note that we now concatenate intermediate inputs. These are the skip connections present in U-net
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)        # same here... 

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)        # ...and here
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out


# all we need to do now is to initialise a class instance from "UNet"
def load_model(num_classes, num_input_channels):
    model = UNet(num_classes, num_input_channels)
    return model



''' ------------------------------------------------------------------------
    3. SETUP OPTIMIZER
    ------------------------------------------------------------------------ '''

# Now that we have our U-Net, we need something that fine-tunes its parameters
# during training. In PyTorch, this is called the "optimizer", and like in any
# other DL library, multiple optimizer types are available. Below, we will use
# Stochastic Gradient Descent (SGD).

def setup_optimizer(model, learning_rate, weight_decay):
    optimizer = torch.optim.SGD(
        model.parameters(),         # tell the optimizer which parameters to fine-tune
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=0.9                # standard value for SGD; you can customize this too if you want
    )
    return optimizer


# Something else we might want to do is to e.g. reduce the learning rate after
# some number of iterations. In PyTorch, this is the duty of the  "scheduler".
# Let's do it!

def setup_scheduler(optimizer, milestones, gamma):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,          # the scheduler works on the optimizer directly to modify the learning rate
        milestones,         # list of integers (iterations at which a step is taken)
        gamma               # step value (e.g. 0.1 = divide learning rate by 10 at each milestone)
    )
    return scheduler



''' ------------------------------------------------------------------------
    4. DEFINE TRAINING LOOP
    ------------------------------------------------------------------------ '''

# Now for the main event: the training function. Essentially, we need the
# following ingredients:
#
#   - our model
#   - a training dataset (resp. data loader)
#   - a loss function
#   - an optimizer
#   - (optionally) a learning rate scheduler
#
# Luckily we have defined most of this above, so we can use everything right
# away. We still have to define our loss function, but this is straightforward,
# as you will see.

def training_epoch(dataLoader, model, optimizer, scheduler, device):

    # Enable model training mode. In this mode, parameters like BatchNorm
    # statistics are updated, dropout is applied (if specified in the model),
    # etc.
    model.train()

    # Define loss function ("criterion"). We perform semantic segmentation,
    # which basically is nothing else than pixel-wise classification. The most
    # common loss function for this is the Softmax-Cross-Entropy loss, which
    # PyTorch has nicely built-in already.
    #
    # NOTE: the Vaihingen dataset contains pixels that are unlabeled. In our
    # preparation we assigned those the value 255. We can elegantly ignore
    # predictions over these pixels with the "ignore_index" parameter.
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # Statistics: during training we want to monitor how well the model behaves,
    # so let's track the loss value as a running average.
    loss_total = 0.0

    # We also want to see the loss value during training. For this we use helper
    # library "tqdm" to create a progress bar.
    progressBar = trange(len(dataLoader))

    # Define the actual loop: one full pass through all images = one epoch
    for index, (data, labels) in enumerate(dataLoader):

        # Put data and labels onto the correct computation device. In PyTorch,
        # the device is a string with the following possible values (examples):
        #
        #   "cpu":      use the processor and system RAM for models
        #   "cuda":     use the first CUDA-enabled graphics card
        #   "cuda:0"    the same as "cuda"
        #   "cuda:1"    use the second GPU (if available)
        #
        # etc.
        data, labels = data.to(device), labels.to(device)

        # forward pass
        prediction = model(data)

        # calculate loss between predictions and target labels
        loss = criterion(prediction, labels)
        if not torch.isfinite(loss):
            print('debug')

        # set all gradient weights to zero. This is important to avoid unwanted
        # accumulation across batches.
        optimizer.zero_grad()

        # perform backpropagation. This stores intermediate gradient values into
        # the respective weights, but does not yet modify the model parameters.
        loss.backward()

        # now, we apply gradient values to the model parameters, w.r.t. the set
        # learning rate, weight decay, momentum, etc.
        optimizer.step()

        # also tell the learning rate scheduler that we just finished a batch.
        scheduler.step()

        # here we update our running statistics. Our loss value is in a
        # torch.Tensor on the GPU right now, so we cannot just add it to
        # "loss_total". Instead, we need to call the .item() function to
        # retrieve the value.
        loss_total += loss.item()

        # finally, let's print the current moving average of the loss on the
        # progress bar.
        progressBar.set_description(
            '[Train] Loss: {:.2f}'.format(loss_total/(index+1))     # current average of the loss value
        )
        progressBar.update(1)
    
    # And that's it! At this point we completed one training epoch.
    progressBar.close()

    # Finalize statistics
    loss_total /= len(dataLoader)

    return loss_total


''' ------------------------------------------------------------------------
    5. DEFINE VALIDATION LOOP
    ------------------------------------------------------------------------ '''

# Similarly to the training above, we also want to periodically check how
# accurate our model is. As per traditional machine learning, we do this on the
# validation set. This allows us to modify our model and/or hyperparameters
# (learning rate, etc.) before testing it for real on the test set.
#
# Our validation loop looks very similar to the training loop, except that we
# don't perform any training, but calculate how accurate the model is for each
# image. So we don't need an optimizer, nor a scheduler.

def validation_epoch(dataLoader, model, device):

    # Put model into evaluation mode. Here, BatchNorm takes the learnt
    # statistics, any existing Dropout is disabled, etc.
    model.eval()

    # Again, we define the loss function, but this time only use it for
    # statistics.
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # This time, we are interested in the prediction accuracy, so in addition to
    # the loss, we also define that. Depending on your requirements you may want
    # to define more or other measurements.
    loss_total = 0.0
    oa_total = 0.0          # overall accuracy

    progressBar = trange(len(dataLoader))

    for index, (data, labels) in enumerate(dataLoader):

        # Important: here, we don't perform any backpropagation, so we don't
        # need to store any intermediate results. This not only saves a lot of
        # GPU memory, it also makes model calculations a lot faster. In PyTorch,
        # this can be done with a flag to disable gradient calculations
        # ("torch.no_grad").
        with torch.no_grad():

            # again: put data and target labels on the GPU
            data, labels = data.to(device), labels.to(device)

            # forward pass
            pred = model(data)

            # #TODO
            # import matplotlib
            # matplotlib.use('TkAgg')
            # import matplotlib.pyplot as plt
            # plt.figure(1)
            # plt.subplot(1,2,1)
            # plt.imshow(pred.argmax(1)[0,...].cpu().numpy())
            # plt.draw()
            # plt.subplot(1,2,2)
            # plt.imshow(labels[0,...].cpu().numpy())
            # plt.draw()
            # plt.waitforbuttonpress()

            # loss value
            loss = criterion(pred, labels)
            loss_total += loss.item()

            # now, our predictions consist of vectors for each class, but for
            # the calculation of the OA we need actual predicted labels. These
            # are essentially the position of the prediction with maximum value,
            # i.e., the arg max.
            labels_pred = pred.argmax(dim=1)        # dimension 1 is our classes, so we take the arg max along it

            # calculate OA
            oa = torch.mean((labels == labels_pred).float())
            oa_total += oa.item()

            # print in progress bar again
            progressBar.set_description(
                '[Val] Loss: {:.2f}, OA: {:.2f}'.format(
                    loss_total/(index+1),
                    100 * oa_total/(index+1)
                )
            )
            progressBar.update(1)
    
    progressBar.close()

    loss_total /= len(dataLoader)
    oa_total /= len(dataLoader)

    return loss_total, oa_total



''' ------------------------------------------------------------------------
    6. PUT IT ALL TOGETHER
    ------------------------------------------------------------------------ '''

# Now we have everything we need. All that is left to do is to actually use the
# pieces we created above. This involves the following steps:
#
# 1. Load training and validation data set (resp. data loaders)
# 2. Initialize model
# 3. For epoch in num_epochs:
# 4.    perform training
# 5.    perform validation
# 6.    save latest model state
#
# Let's define our main function that does all of that.

def main(data_root, batch_size, device, learning_rate, weight_decay, scheduler_milestones, scheduler_gamma, num_epochs, save_dir):

    # initialize training and validation data loaders
    dl_train = load_dataset(data_root, 'train', batch_size)
    dl_val = load_dataset(data_root, 'val', batch_size)

    # information on dataset. This is usually provided in a config file somewhere with the dataset.
    num_classes = 6                 # Impervious, Buildings, Low Vegetation, Tree, Car, Clutter
    num_input_channels = 5          # NIR, R, G, DSM, nDSM

    # initialize model
    model = load_model(num_classes, num_input_channels)

    # initialize optimizer and learning rate scheduler
    optimizer = setup_optimizer(model, learning_rate, weight_decay)
    scheduler = setup_scheduler(optimizer, scheduler_milestones, scheduler_gamma)

    # load saved state if exists
    saveStates = os.listdir(save_dir)
    if len(saveStates):
        latest = max([int(s.replace('.pt', '')) for s in saveStates])
        state = torch.load(open(os.path.join(save_dir, f'{latest}.pt'), 'rb'), map_location='cpu')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        epoch = state['epoch']
        print(f'Resumed model epoch {epoch}.')
    else:
        epoch = 1
        print(f'Started new model.')

    # move model to the GPU
    model.to(device)

    # train for desired number of epochs
    while epoch < num_epochs:

        print(f'[Epoch {epoch}/{num_epochs}]')

        # train
        loss_train = training_epoch(dl_train, model, optimizer, scheduler, device)

        # validate
        loss_val, oa_val = validation_epoch(dl_val, model, device)

        # save model parameters and statistics to file
        params = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'loss_train': loss_train,
            'loss_val': loss_val,
            'oa_val': oa_val
        }
        torch.save(params, open(f'{save_dir}/{epoch}.pt', 'wb'))        # "w" = "write", "b" = "binary file"
        epoch += 1




if __name__ == '__main__':
    '''
        This seemingly strange "if" statement is only executed if you launch
        this very file as the "main" one ("python unet_training.py"), unlike
        when you e.g. import it in another file. Hence, here you can define any
        main routines that should be executed when you want to launch this file.
        So for the sake of completeness, here we define an argument parser that
        allows you to perform U-Net training by calling this file directly!

        Give it a try (example):

            python unet_training.py --num_epochs 10
    '''

    import argparse
    parser = argparse.ArgumentParser(description='Tutorial script to train and validate U-Net on the Vaihingen dataset')
    parser.add_argument('--data_root', type=str, default='/data/datasets/Vaihingen/dataset_512x512_full',
                        help='Root directory of the Vaihingen data set (contains folders "train" and "val")')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (default: 4)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for calculations (default: "cuda")')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Initial learning rate (default: 0.01)')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay factor (default: 0.0001)')
    parser.add_argument('--milestones', type=int, nargs='?', default=[1000, 5000],
                        help='List of iterations at which to apply the learning rate scheduler step (default: [100, 500]')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Learning rate multiplication factor for scheduler (default: 0.1)')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs to train for (default: 50)')
    parser.add_argument('--save_dir', type=str, default='cnn_states',
                        help='Directory to save model states into')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    main(
        args.data_root,
        args.batch_size,
        args.device,
        args.learning_rate,
        args.weight_decay,
        args.milestones,
        args.gamma,
        args.num_epochs,
        args.save_dir
    )