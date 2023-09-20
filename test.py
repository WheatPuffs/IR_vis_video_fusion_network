from network import Encoder, Decoder
import torch
import cv2
import numpy as np
import os
import time

# ====================================================================================
# Global variables
# ====================================================================================

device = 'cuda' # 'cuda' or 'cpu'
evaluate = 'true' 

# path to test images
#file_names = sorted(os.listdir('/home/alexandru/School/datasets/CAMEL29/IR'))
file_names = sorted(os.listdir('/home/alexandru/School/datasets/take_4/take_2/VIS'))
files_nr   = len(file_names)

# path to visible video
#visible_video_path = '/home/alexandru/School/datasets/take_2/VI/00082.MTS'
visible_video_path = './1/gen_vis_forest.avi'
# path to infrared video
#infrared_video_path = '/home/alexandru/School/datasets/take_2/IR/Video_2.mpg'
infrared_video_path = './1/gen_ir_forest.avi'

video_resolution = (720, 480)

# function to resize images
def resize_image(frame):
    return cv2.resize(frame, video_resolution)

video_fused_name = 'Video_fused.avi'
output_fps = 30
output_resulution = video_resolution
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
video_writer = cv2.VideoWriter(video_fused_name, fourcc, output_fps, output_resulution, isColor=False)

# instantiate encoder and decoder classes and move them to GPU
Encoder = Encoder().to(device)
Decoder = Decoder().to(device)

# load the weights from the trained model
Encoder.load_state_dict(torch.load("./Train_result/Encoder_weights.pkl")['weight'])
Decoder.load_state_dict(torch.load("./Train_result/Decoder_weights.pkl")['weight'])

# set the mode to evaluation 
Encoder.eval()
Decoder.eval()

# ====================================================================================
# Functions to test fusion quality
# ====================================================================================

def compute_entropy(image):
    # compute the histogram of the grayscale image
    histogram = cv2.calcHist(images=[image], channels=[0], mask=None, histSize=[256], ranges=[0,256])
    
    # normalize the histogram
    histogram = histogram.ravel() / histogram.sum()
    
    # calculte the entropy
    epsilon = 1e-10 # to not have log2(0)
    entropy = -np.sum(histogram * np.log2(histogram + epsilon))
    
    return entropy
    
def compute_cross_entropy(image1, image2):
    # compute the histograms of the images
    histogram_image1 = cv2.calcHist(images = [image1], channels = [0], mask = None, histSize = [256], ranges = [0,256])
    histogram_image2 = cv2.calcHist(images = [image2], channels = [0], mask = None, histSize = [256], ranges = [0,256])
    
    # normalize the histograms
    histogram_image1 = histogram_image1.ravel() / histogram_image1.sum()
    histogram_image2 = histogram_image2.ravel() / histogram_image2.sum()
    
    # calculte the entropy
    epsilon = 1e-10 # to not have log2(0)
    cross_entropy = -np.sum(histogram_image1 * np.log2(histogram_image2 + epsilon))
    
    return cross_entropy
    
    
def compute_mutual_information(image, fused_image):
    # compute joint histogram
    histogram_joint = cv2.calcHist(images = [image, fused_image], channels = [0, 0], mask=None, histSize = [256,256], ranges = [0,256,0,256])
    
    # calculate the marginal distribution of the joint histogram
    joint_prob_distribution = histogram_joint / histogram_joint.sum()
    
    
    # calculate the marginal probability distributions
    image_prob_distribution = np.sum(joint_prob_distribution, axis=1)
    fused_prob_distribution = np.sum(joint_prob_distribution, axis=0)
    
    # calculate the Shannon entropies
    epsilon = 1e-10 # to not have log2(0)
    a = - np.sum(image_prob_distribution * np.log2(image_prob_distribution + epsilon))
    b = - np.sum(fused_prob_distribution * np.log2(fused_prob_distribution + epsilon))
    c = - np.sum(joint_prob_distribution * np.log2(joint_prob_distribution + epsilon))
    
    return a+b-c
    
def compute_standard_deviation(image):  
    standard_deviation = np.std(image)

    return standard_deviation

def compute_spatial_frequency(image):
    # apply 2d FFT transform to the image
    fft_spectrum = np.fft.fft2(image)

    # compute the spectrum magnitude
    spectrum_magnitude = np.log(np.abs(fft_spectrum))

    # compute the spatial frequency
    spatial_frequency = np.mean(spectrum_magnitude)

    return spatial_frequency
    
def compute_average_gradient(image):

    # compute the gradient
    gradient_x = cv2.Sobel(image, cv2.CV_64F, dx = 1, dy = 0, ksize = 1)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, dx = 0, dy = 1, ksize = 1)

    # compute the magnitude of the gradient
    magnitude = np.hypot(gradient_x, gradient_y)

    # compute the mean of the magnitude
    average_gradient = np.mean(magnitude)

    return average_gradient
    
# ====================================================================================
# Function for fusing images
# ====================================================================================

def fuse(image1, image2):

    # normalize the images
    image1 = np.array(image1, dtype='float32') / 255
    image2 = np.array(image2, dtype='float32') / 255
    
    # convert images to tensors
    tensor_image1 = torch.from_numpy(image1.reshape((1, 1, image1.shape[0], image1.shape[1])))
    tensor_image2 = torch.from_numpy(image2.reshape((1, 1, image2.shape[0], image2.shape[1])))
    
    # move tensors to GPU if available
    if device == 'cuda':
        tensor_image1 = tensor_image1.cuda()
        tensor_image2 = tensor_image2.cuda()
    
    # disable computing gradients, only do forward pass
    with torch.no_grad():
        vi = Encoder(tensor_image1)
        ir = Encoder(tensor_image2)
        fu = vi + ir
         
    with torch.no_grad():
        tensor_fused_image = Decoder(fu)
    
    # move tensor to CPU and convert it to image
    fused_image = (tensor_fused_image.cpu().detach().numpy()[0, 0, :, :] * 255).astype(np.uint8)
     
    return fused_image

# ====================================================================================
# Function for fusing 2 sets of images into a video
# ====================================================================================

def fuse_images():
    file_cnt  = 0
    
    # array for measuring fusion time
    exec_time = np.array([])
    
    # for measuring fusion quality
    entropy               = np.array([])
    mutual_information_vi = np.array([])
    mutual_information_ir = np.array([])
    standard_deviation    = np.array([])
    spatial_frequency     = np.array([])
    average_gradient      = np.array([])
    
    for file_name in file_names:
        
        #IR_path = os.path.join('/home/alexandru/School/datasets/CAMEL29/IR', file_name)
        #VI_path = os.path.join('/home/alexandru/School/datasets/CAMEL29/Visual', file_name)
        
        IR_path = '/home/alexandru/School/datasets/take_4/take_2/IR/IR_' + str(file_cnt + 1) +'.jpg'
        VI_path = '/home/alexandru/School/datasets/take_4/take_2/VIS/VIS_' + str(file_cnt + 1) +'.jpg'
    
        vi_image_test = resize_image(cv2.imread(VI_path, cv2.IMREAD_GRAYSCALE))
        ir_image_test = resize_image(cv2.imread(IR_path, cv2.IMREAD_GRAYSCALE))
    
        # fuse a pair of images and record fusion time
        start_time = time.time()
        fused_image = fuse(vi_image_test, ir_image_test)
        end_time = time.time()
        
        
        exec_time = np.append(exec_time, (end_time - start_time) * 1000)
        print('frame %d time: %.2f ms' %(file_cnt, exec_time[file_cnt]))
        file_cnt += 1
        #if file_cnt == 10:
        #    break
        
        video_writer.write(fused_image)
        
        # measure fusion quality
        if evaluate == 'true':
            en    = compute_entropy(fused_image)
            mi_vi = compute_mutual_information(vi_image_test, fused_image)
            mi_ir = compute_mutual_information(ir_image_test, fused_image)
            sd    = compute_standard_deviation(fused_image)
            sf    = compute_spatial_frequency(fused_image)
            ag    = compute_average_gradient(fused_image)
    
            entropy               = np.append(entropy, en)
            mutual_information_vi = np.append(mutual_information_vi, mi_vi)
            mutual_information_ir = np.append(mutual_information_ir, mi_ir)
            standard_deviation    = np.append(standard_deviation, sd)
            spatial_frequency     = np.append(spatial_frequency, sf)
            average_gradient      = np.append(average_gradient, ag)
        

    if evaluate == 'true':
        average_entropy               = np.mean(entropy)
        average_mutual_information_vi = np.mean(mutual_information_vi)
        average_mutual_information_ir = np.mean(mutual_information_ir)
        average_standard_deviation    = np.mean(standard_deviation)
        average_spatial_frequency     = np.mean(spatial_frequency)
        mean_average_gradient         = np.mean(average_gradient)
        print('average_entropy = %.2f' %average_entropy)
        print('average_mutual_information_visible_fused = %.2f' %average_mutual_information_vi)
        print('average_mutual_information_IR_fused = %.2f' %average_mutual_information_ir)
        print('average_standard_deviation = %.2f' %average_standard_deviation)
        print('average_spatial_frequency = %.2f' %average_spatial_frequency)
        print('mean_average_gradient = %.2f' %mean_average_gradient)

    average_time = np.mean(exec_time[1:])
    total_time   = average_time * file_cnt / 1000
    print('\naverage time = %.2f milliseconds (%.2f fps)' %(average_time, 1000 / average_time))
    if total_time < 60:
        print('total time = %.2f seconds' %total_time)
    else:    
        print('total time = %d minutes %.2f seconds' %(total_time // 60, total_time % 60))

    video_writer.release() 
    cv2.destroyAllWindows() 

# ====================================================================================
# Function for fusing 2 videos
# ====================================================================================

def fuse_videos():
    frame_cnt  = 0
    
    # array for measuring fusion time
    exec_time = np.array([])
    
    # for measuring fusion quality
    entropy               = np.array([])
    mutual_information_vi = np.array([])
    mutual_information_ir = np.array([])
    standard_deviation    = np.array([])
    spatial_frequency     = np.array([])
    average_gradient      = np.array([])
    
    # open the videos
    visible_video  = cv2.VideoCapture(visible_video_path)
    infrared_video = cv2.VideoCapture(infrared_video_path)

    # check if the video capture is successfully opened
    if not visible_video.isOpened():
        print('Error opening visible video file')
        exit()
    if not infrared_video.isOpened():
        print('Error opening infrared video file')
        exit()

    # iterate video frames and fuse them
    while visible_video.isOpened() and infrared_video.isOpened():
        
        # read a frame from the videos
        ret_visible, frame_visible   = visible_video.read()
        ret_infrared, frame_infrared = infrared_video.read()
    
        # check if the frames are successfully read
        if not ret_visible or not ret_infrared:
            print('Error reading frame')
            break

        # prepare images for fusion
        frame_visible  = resize_image(cv2.cvtColor(frame_visible,  cv2.COLOR_BGR2GRAY))
        frame_infrared = resize_image(cv2.cvtColor(frame_infrared, cv2.COLOR_BGR2GRAY))
    
        # fuse images
        start_time = time.time()
        frame_fused = fuse(frame_visible, frame_infrared)
        end_time = time.time()
    
        # write fused frame to video or to image
        video_writer.write(frame_fused) 
        #cv2.imwrite('./Test_result/frame'+str(frame_cnt)+'.jpg', frame_fused)
    
        frame_cnt += 1
        exec_time = np.append(exec_time, (end_time - start_time) * 1000)
        print('frame %d time: %.2f ms' %(frame_cnt, exec_time[frame_cnt - 1]))
        if frame_cnt >= 20:
            break
    
        # measure fusion quality
        if evaluate == 'true':
            en    = compute_entropy(frame_fused)
            mi_vi = compute_mutual_information(frame_visible, frame_fused)
            mi_ir = compute_mutual_information(frame_infrared, frame_fused)
            sd    = compute_standard_deviation(frame_fused)
            sf    = compute_spatial_frequency(frame_fused)
            ag    = compute_average_gradient(frame_fused)
    
            entropy               = np.append(entropy, en)
            mutual_information_vi = np.append(mutual_information_vi, mi_vi)
            mutual_information_ir = np.append(mutual_information_ir, mi_ir)
            standard_deviation    = np.append(standard_deviation, sd)
            spatial_frequency     = np.append(spatial_frequency, sf)
            average_gradient      = np.append(average_gradient, ag)
  
    video_writer.release() 
    cv2.destroyAllWindows()

    if evaluate == 'true':
        average_entropy               = np.mean(entropy)
        average_mutual_information_vi = np.mean(mutual_information_vi)
        average_mutual_information_ir = np.mean(mutual_information_ir)
        average_standard_deviation    = np.mean(standard_deviation)
        average_spatial_frequency     = np.mean(spatial_frequency)
        mean_average_gradient         = np.mean(average_gradient)
        print('average_entropy = %.2f' %average_entropy)
        print('average_mutual_information_visible_fused = %.2f' %average_mutual_information_vi)
        print('average_mutual_information_IR_fused = %.2f' %average_mutual_information_ir)
        print('average_standard_deviation = %.2f' %average_standard_deviation)
        print('average_spatial_frequency = %.2f' %average_spatial_frequency)
        print('mean_average_gradient = %.2f' %mean_average_gradient)

    average_time = np.mean(exec_time[1:])
    total_time   = average_time * frame_cnt / 1000
    print('\naverage time = %.2f milliseconds (%.2f fps)' %(average_time, 1000 / average_time))
    if total_time < 60:
        print('total time = %.2f seconds' %total_time)
    else:    
        print('total time = %d minutes %.2f seconds' %(total_time // 60, total_time % 60))

# ====================================================================================
# Do inference
# ====================================================================================

fuse_images()
#fuse_videos()

# ====================================================================================
# EOF
# ====================================================================================
