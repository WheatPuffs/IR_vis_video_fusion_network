from torchvision import datasets, transforms
from torch import nn, cat, save
from torch.utils.data import DataLoader, SequentialSampler
import torch.optim as optim
import kornia
import os
import matplotlib.pyplot as plt
import time

from network import Encoder,Decoder

# ====================================================================================
# Global variables 
# ====================================================================================

train_dataset_path  = '/home/alexandru/School/datasets/YDTR/'
train_result_path   = './Train_result/'
device              = 'cuda' # 'cuda' or 'cpu'

batch_size = 16
lr         = 1e-3
epochs     = 25

train_images_nr = len(os.listdir(train_dataset_path + 'VIS_ALL/'))
iter_per_epoch  = train_images_nr // batch_size + (train_images_nr % batch_size != 0)

loss_train                  = [] # total loss
loss_VF_train               = [] # visible fusion
loss_IF_train               = [] # infrared fusion
gradient_loss_train         = []
lr_encoder_train            = []
lr_decoder_train            = []
best_loss                   = 10
epochs_with_no_improvement  = 0

# ====================================================================================
# Prepare dataset
# ====================================================================================

transforms = transforms.Compose([
        transforms.CenterCrop(256),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        ])

# create dataloaders                               
dataset  = datasets.ImageFolder(train_dataset_path, transform=transforms)

visible_images_indices = [i for i, (image, label) in enumerate(dataset) if label == 'VIS_ALL']
infrared_images_indices = [i for i, (image, label) in enumerate(dataset) if label == 'IR_ALL']

visible_dataloader  = DataLoader(dataset, batch_size, SequentialSampler(visible_images_indices))
infrared_dataloader = DataLoader(dataset, batch_size, SequentialSampler(infrared_images_indices))

# ====================================================================================
# Prepare training
# ====================================================================================

# create encoder and decoder objects
Encoder = Encoder()
Decoder = Decoder()

# move them to GPU if available
if device == 'cuda':
    Encoder = Encoder.cuda()
    Decoder = Decoder.cuda()

encoder_optimizer = optim.Adam(Encoder.parameters(), lr = lr, amsgrad = True)
decoder_optimizer = optim.Adam(Decoder.parameters(), lr = lr, amsgrad = True)


encoder_scheduler = optim.lr_scheduler.MultiStepLR(
                        optimizer = encoder_optimizer, 
                        milestones = [epochs // 4, epochs // 4 * 2, epochs // 4 * 3], 
                        gamma = 0.1)
decoder_scheduler = optim.lr_scheduler.MultiStepLR(
                        optimizer = decoder_optimizer, 
                        milestones = [epochs // 4, epochs // 4 * 2, epochs // 4 * 3], 
                        gamma = 0.1)

# losses
MSELoss = nn.MSELoss()
L1Loss  = nn.L1Loss()
SSIM    = kornia.losses.SSIMLoss(window_size = 11)

# ====================================================================================
# Training
# ====================================================================================

start_time = time.time()
print('============= Training Begins ================')

for iteration in range(epochs):
    
    Encoder.train()
    Decoder.train()
   
    visible_dataset_iterator  = iter(visible_dataloader)
    infrared_dataset_iterator = iter(infrared_dataloader)
    
    
    for step in range(iter_per_epoch):
        visible_dataset_batch, _  = next(visible_dataset_iterator)
        infrared_dataset_batch, _ = next(infrared_dataset_iterator)
        
        # move a batch of pairs of images to GPU if available  
        if device == 'cuda':
            visible_dataset_batch  = visible_dataset_batch.cuda()
            infrared_dataset_batch = infrared_dataset_batch.cuda()
        
        # reset gradients to 0
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        
        # ============================================================================
        # Calculate loss 
        # ============================================================================
        
        feature_visible  = Encoder(visible_dataset_batch)
        feature_infrared = Encoder(infrared_dataset_batch)

        output_visible  = Decoder(feature_visible)
        output_infrared = Decoder(feature_infrared)

        loss_VF = 4 * SSIM(visible_dataset_batch, output_visible) + 1 * MSELoss(visible_dataset_batch, output_visible)
        loss_IF = 4 * SSIM(infrared_dataset_batch, output_infrared) + 1 * MSELoss(infrared_dataset_batch, output_infrared)

        gradient_loss = L1Loss(
                            kornia.filters.SpatialGradient()(visible_dataset_batch),
                            kornia.filters.SpatialGradient()(output_visible))

        loss = loss_VF + loss_IF + 8 * gradient_loss
   
        # propagate back gradients
        loss.backward() 
        
        encoder_optimizer.step()
        decoder_optimizer.step()
        
        # convert losses from tensors to scalars
        loss_value              = loss.item()
        loss_VF_value           = loss_VF.item()
        loss_IF_value           = loss_IF.item()
        loss_gradient_value     = gradient_loss.item()
        
        print('Epoch/step: %d/%d, lr: %.0e, loss: %.4f' %(iteration + 1, step + 1, encoder_optimizer.state_dict()['param_groups'][0]['lr'], loss_value))

        # save losses
        loss_train.append(loss_value)
        loss_VF_train.append(loss_VF_value)
        loss_IF_train.append(loss_IF_value)
        gradient_loss_train.append(loss_gradient_value)
    
    # modify learning rate at some epochs mentioned earlier       
    encoder_scheduler.step()
    decoder_scheduler.step()
    
    # save learning rates
    lr_encoder_train.append(encoder_optimizer.state_dict()['param_groups'][0]['lr'])
    lr_decoder_train.append(decoder_optimizer.state_dict()['param_groups'][0]['lr'])
    
    # stop training earlier if loss does not improves
    if loss.item() < best_loss:
        best_loss = loss_train[-1]
        epochs_with_no_improvement = 0
    else:
        epochs_with_no_improvement += 1
    if epochs_with_no_improvement == 5:
        break
end_time = time.time()  
exec_time = end_time-start_time
if exec_time < 60:
    print('training time = %d seconds' %exec_time)
else:    
    print('training time = %d minutes %d seconds' %(exec_time // 60, exec_time % 60))

# ====================================================================================
# Save weights and loss plots
# ====================================================================================

save({'weight': Encoder.state_dict(), 'epoch':epochs}, 
           os.path.join(train_result_path,'Encoder_weights.pkl'))
save({'weight': Decoder.state_dict(), 'epoch':epochs}, 
           os.path.join(train_result_path, 'Decoder_weights.pkl'))

# function for computing mean loss in a batch during each epoch
def mean_loss(loss):
    epoch_loss_train = []
    for i in range(len(loss) // iter_per_epoch):
        sum_epoch_loss = sum(loss[(i * iter_per_epoch):((i + 1) * iter_per_epoch)])
        mean_epoch_loss = sum_epoch_loss / iter_per_epoch
        epoch_loss_train.append(mean_epoch_loss)
    return epoch_loss_train

# plot the losses and save as .png image
plt.figure(figsize = [12, 12])
plt.subplot(2, 2, 1), plt.plot(mean_loss(loss_train)), plt.xlabel('Epochs'), plt.ylabel('Loss'), plt.title('Total loss')
plt.subplot(2, 2, 2), plt.plot(mean_loss(loss_VF_train)), plt.xlabel('Epochs'), plt.ylabel('Loss'), plt.title('Visible loss')
plt.subplot(2, 2, 3), plt.plot(mean_loss(loss_IF_train)), plt.xlabel('Epochs'), plt.ylabel('Loss'), plt.title('Infrared loss')
plt.subplot(2, 2, 4), plt.plot(mean_loss(gradient_loss_train)), plt.xlabel('Epochs'), plt.ylabel('Loss'), plt.title('Gradient loss')
plt.tight_layout()
plt.savefig(os.path.join(train_result_path,'loss_during_training.png'))    

# ====================================================================================
# EOF
# ====================================================================================
