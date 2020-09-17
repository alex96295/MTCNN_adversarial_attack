import fnmatch
import math
import os
import tqdm
import sys
import time
from operator import itemgetter

import gc
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

#from darknet import Darknet

from median_pool import MedianPool2d  # see median_pool.py

class mtcnn_feature_output_manage(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, config):
        super(mtcnn_feature_output_manage, self).__init__()

        self.config = config

    def forward(self, all_nets, output_switch, net_output, loss_type, scale_level_score):

        if all_nets == 0:
            if output_switch == 'pnet':
                pnet_output = net_output
                print('len: ' + str(len(pnet_output)))

                loss_per_scale = []
                for score_per_scale in pnet_output:
                    #print(score_per_scale.size())
                    grid_x = score_per_scale.size()[1]
                    grid_y = score_per_scale.size()[2]
                    score_per_scale_resh = score_per_scale.view(self.config.batch_size, grid_x*grid_y)
                    #print(score_per_scale_resh.size())

                    if loss_type == 'max_approach':
                        scale_out, scale_max_indx = torch.max(score_per_scale_resh, 1)
                        #print(scale_out.size())
                        #scale_out = scale_out.unsqueeze(1)
                        #print(scale_out.size())

                    elif loss_type == 'threshold_approach':
                        threshold = 0.35
                        batch_stack = torch.unbind(score_per_scale_resh, dim=0)

                        penalized_tensor_batch = []
                        for img_tensor in batch_stack:
                            size = img_tensor.size()
                            zero_tensor = torch.cuda.FloatTensor(size).fill_(0)
                            penalized_tensor = torch.max(img_tensor - threshold, zero_tensor) ** 2
                            penalized_tensor_batch.append(penalized_tensor)

                        penalized_tensor_batch = torch.stack(penalized_tensor_batch, dim=0)
                        scale_out = torch.sum(penalized_tensor_batch, dim=1)
                        scale_out = scale_out.unsqueeze(1)
                        #print(scale_out.size())

                    loss_per_scale.append(scale_out)

                loss_per_scale = torch.stack(loss_per_scale, 1)
                # print(loss_per_scale)

                if scale_level_score == 'scale_max':
                    scales_output, scales_output_indx = torch.max(loss_per_scale, 1)
                    # print(scales_output)
                    # mean over batch
                    scales_output = torch.mean(scales_output)
                    return scales_output

                elif scale_level_score == 'scale_mean':
                    scales_output = torch.mean(loss_per_scale, 1)
                    # mean over batch
                    scales_output = torch.mean(scales_output)
                    return scales_output
                elif scale_level_score == 'scale_sum':
                    scales_output = torch.sum(loss_per_scale, 1)
                    # mean over batch
                    scales_output = torch.mean(scales_output)
                    return scales_output


            elif output_switch == 'rnet' or output_switch == 'onet':

                if loss_type == 'max_approach':
                    score_out = torch.max(net_output)
                    print(score_out)
                    return score_out

                elif loss_type == 'threshold_approach':
                    threshold = 0

                    size = net_output.size()
                    zero_tensor = torch.cuda.FloatTensor(size).fill_(0)
                    penalized_tensor = torch.max(net_output - threshold, zero_tensor) ** 2

                    scale_out = torch.sum(penalized_tensor)
                    return scale_out

        else:
            # pnet
            pnet_output = net_output[0]
            print('len: ' + str(len(pnet_output)))

            loss_per_scale = []
            for score_per_scale in pnet_output:
                # print(score_per_scale.size())
                grid_x = score_per_scale.size()[1]
                grid_y = score_per_scale.size()[2]
                score_per_scale_resh = score_per_scale.view(self.config.batch_size, grid_x * grid_y)
                # print(score_per_scale_resh.size())

                if loss_type == 'max_approach':
                    scale_out, scale_max_indx = torch.max(score_per_scale_resh, 1)
                    # print(scale_out.size())
                    # scale_out = scale_out.unsqueeze(1)
                    # print(scale_out.size())

                elif loss_type == 'threshold_approach':
                    threshold = 0.35
                    batch_stack = torch.unbind(score_per_scale_resh, dim=0)

                    penalized_tensor_batch = []
                    for img_tensor in batch_stack:
                        size = img_tensor.size()
                        zero_tensor = torch.cuda.FloatTensor(size).fill_(0)
                        penalized_tensor = torch.max(img_tensor - threshold, zero_tensor) ** 2
                        penalized_tensor_batch.append(penalized_tensor)

                    penalized_tensor_batch = torch.stack(penalized_tensor_batch, dim=0)
                    scale_out = torch.sum(penalized_tensor_batch, dim=1)
                    scale_out = scale_out.unsqueeze(1)
                    # print(scale_out.size())

                loss_per_scale.append(scale_out)

            loss_per_scale = torch.stack(loss_per_scale, 1)
            # print(loss_per_scale)

            if scale_level_score == 'scale_max':
                scales_output_pnet, scales_output_indx = torch.max(loss_per_scale, 1)
                # print(scales_output)

            elif scale_level_score == 'scale_mean':
                scales_output_pnet = torch.mean(loss_per_scale, 1)
                
            elif scale_level_score == 'scale_sum':
                scales_output_pnet = torch.sum(loss_per_scale, 1)

            scales_output_pnet_batched = torch.mean(scales_output_pnet)

            # rnet, onet
            r_o_net_outscore_list = []
            for r_o_nets in net_output[1:]:
                if loss_type == 'max_approach':
                    score_out = torch.max(r_o_nets)
                    #print(score_out)


                elif loss_type == 'threshold_approach':
                    threshold = 0

                    size = r_o_nets.size()
                    zero_tensor = torch.cuda.FloatTensor(size).fill_(0)
                    penalized_tensor = torch.max(r_o_nets - threshold, zero_tensor) ** 2

                    score_out = torch.sum(penalized_tensor)

                r_o_net_outscore_list.append(score_out)

            print(scales_output_pnet_batched)
            print(r_o_net_outscore_list)
            final_output = scales_output_pnet_batched + r_o_net_outscore_list[0] + r_o_net_outscore_list[1]
            return final_output


class NPSCalculator(nn.Module):
    """NMSCalculator: calculates the non-printability score of a patch.

    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.

    """

    def __init__(self, printability_file, patch_side):
        super(NPSCalculator, self).__init__()
        self.printability_array = nn.Parameter(self.get_printability_array(printability_file, patch_side),requires_grad=False)

    def forward(self, adv_patch):
        # calculate euclidian distance between colors in patch and colors in printability_array
        # square root of sum of squared difference

        color_dist = (adv_patch - self.printability_array+0.000001)
        #print(color_dist.size())
        color_dist = color_dist ** 2  # squared difference
        color_dist = torch.sum(color_dist, 1)+0.000001
        #print(color_dist.size())
        color_dist = torch.sqrt(color_dist)

        # only work with the min distance
        color_dist_prod = torch.min(color_dist, 0)[0] #test: change prod for min (find distance to closest color)
        #print(type(color_dist_prod))
        #print('size ' + str(color_dist_prod.size()))

        # calculate the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod,0)
        nps_score = torch.sum(nps_score,0)
        return nps_score/torch.numel(adv_patch)  # divide by the total number of elements in the input tensor

    def get_printability_array(self, printability_file, side):
        #  side = patch_size in adv_examples.py
        printability_list = []

        # read in printability triplets and put them in a list
        with open(printability_file) as f:
            for line in f:
                printability_list.append(line.split(","))

        # see notes for a better graphical representation
        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full(side, red))
            printability_imgs.append(np.full(side, green))
            printability_imgs.append(np.full(side, blue))

            printability_array.append(printability_imgs)

        printability_array = np.asarray(printability_array)  # convert input lists, tuples etc. to array
        printability_array = np.float32(printability_array)
        pa = torch.from_numpy(printability_array)  # Creates a Tensor from a numpy array.
        return pa

class TotalVariation(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total variation (TV) of an adversarial patch.

    """

    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, adv_patch):
        # compute total variation of the adv_patch
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1]+0.000001),0)  # NB -1 indicates the last element!
        tvcomp1 = torch.sum(torch.sum(tvcomp1,0),0)

        tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :]+0.000001),0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2,0),0)
        tv = tvcomp1 + tvcomp2
        return tv/torch.numel(adv_patch)

class PatchTransformer(nn.Module):
    """PatchTransformer: transforms batch of patches

    Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
    contrast, adding random amount of noise, and rotating randomly. Resizes patches according to as size based on the
    batch of labels, and pads them to the dimension of an image.

    """

    def __init__(self):
        super(PatchTransformer, self).__init__()

        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10
        self.minangle = -10 / 180 * math.pi
        self.maxangle = 10 / 180 * math.pi
        self.medianpooler = MedianPool2d(7, same=True)  # kernel_size = 7? see again

    def forward(self, adv_patch, lab_batch, img_size, loc, do_rotate=True, rand_loc=True, align_angle = True):

        use_cuda = 1

        #adv_patch = self.medianpooler(adv_patch.unsqueeze(0))
        #print(adv_patch.size())

        padx = (img_size - adv_patch.size(-1)) / 2
        pady = (img_size - adv_patch.size(-2)) / 2

        adv_patch = adv_patch.unsqueeze(0)

        adv_batch = adv_patch.expand(lab_batch.size(0), lab_batch.size(1), -1, -1, -1)

        batch_size = torch.Size((lab_batch.size(0), lab_batch.size(1)))


        if use_cuda:
            contrast = torch.cuda.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
        else:
            contrast = torch.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)

        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        contrast = contrast.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))

        if use_cuda:
            contrast = contrast.cuda()
        else:
            contrast = contrast
#_________________________________________________________________________________________________________________________________________________
        # Create random brightness tensor
        if use_cuda:
            brightness = torch.cuda.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
        else:
            brightness = torch.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)

        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = brightness.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))

        if use_cuda:
            brightness = brightness.cuda()
        else:
            brightness = brightness

# _____________________________________________________________________________________________________________________________________________
        # Create random noise tensor
        if use_cuda:
            noise = torch.cuda.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor
        else:
            noise = torch.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor
#______________________________________________________________________________________________________________________________________________
        # Apply contrast/brightness/noise, clamp
        adv_batch = adv_batch * contrast + brightness + noise

        adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)

#______________________________________________________________________________________________________________________________________________
        # Where the label class_ids is 1 we don't want a patch (padding) --> fill mask with zero's

        cls_ids = torch.narrow(lab_batch, 2, 0, 1)
        cls_mask = cls_ids.expand(-1, -1, 3)
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(3))
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, -1, adv_batch.size(4))

        if use_cuda:
            msk_batch = torch.cuda.FloatTensor(cls_mask.size()).fill_(1) - cls_mask
        else:
            msk_batch = torch.FloatTensor(cls_mask.size()).fill_(1) - cls_mask


#_______________________________________________________________________________________________________________________________________________
        # Pad patch and mask to image dimensions with zeros
        mypad = nn.ConstantPad2d((int(padx + 0.5), int(padx), int(pady + 0.5), int(pady)), 0)
        adv_batch = mypad(adv_batch)
        msk_batch = mypad(msk_batch)

#_______________________________________________________________________________________________________________________________________________
        # Rotation and rescaling transforms
        anglesize = (lab_batch.size(0) * lab_batch.size(1))  # dim = 6*14 = 84
        if do_rotate:
            if use_cuda:
                angle = torch.cuda.FloatTensor(anglesize).uniform_(self.minangle, self.maxangle)
            else:
                angle = torch.FloatTensor(anglesize).uniform_(self.minangle, self.maxangle)

        else:
            if use_cuda:
                angle = torch.cuda.FloatTensor(anglesize).fill_(0)
            else:
                angle = torch.FloatTensor(anglesize).fill_(0)

        # Fixed rotation along mouth direction:
        if align_angle:

            third_vertex = [lab_batch[:,:,13], lab_batch[:,:,12]]
            mouth_length_hypot = torch.sqrt((lab_batch[:,:,13] - lab_batch[:,:,11])**2 + (lab_batch[:,:,14]-lab_batch[:,:,12])**2)
            #print('mouth_length_wo_1: ' + str(mouth_length_hypot))
            cath_opp = torch.sqrt((lab_batch[:,:,13] - third_vertex[0])**2 + (lab_batch[:,:,14]-third_vertex[1])**2)
            #print('cath_opp: ' + str(cath_opp))
            cath_adj = torch.sqrt(mouth_length_hypot**2 - cath_opp**2)
            #print('cath_adj: ' + str(cath_adj))

            mouth_length_hypot = torch.where(mouth_length_hypot==0, torch.cuda.FloatTensor(batch_size).fill_(1), mouth_length_hypot)
            #print('mouth_length_with1: ' + str(mouth_length_hypot))

            sin_align_angle = cath_opp/mouth_length_hypot
            cos_align_angle = cath_adj/mouth_length_hypot

            greater_mask = torch.ge(lab_batch[:,:,12], lab_batch[:,:,14])
            #print(greater_mask)
            #cos_align_angle = torch.where(greater_mask==True, cos_align_angle*(-1), cos_align_angle)
            sin_align_angle = torch.where(greater_mask == True, sin_align_angle * (-1), sin_align_angle)

        else:
            if use_cuda:
                sin_align_angle = torch.cuda.FloatTensor(batch_size).fill_(0)
                cos_align_angle = torch.cuda.FloatTensor(batch_size).fill_(0)
            else:
                sin_align_angle = torch.FloatTensor(batch_size).fill_(0)
                cos_align_angle = torch.FloatTensor(batch_size).fill_(0)

        sin_align_angle = sin_align_angle.view(anglesize)
        cos_align_angle = cos_align_angle.view(anglesize)
        # print(sin_align_angle)
        # print(cos_align_angle)
#_______________________________________________________________________________________________________________________________________________
        # Resizes and rotates
        current_patch_size_mine = adv_patch.size(-1) # width if -1 (larger dim), height if -2 (smaller dim)
        
        if loc == 'mouth_hide_nose':
            current_patch_size_paper = 200
        else:
            current_patch_size_paper = adv_patch.size(-2)

        if use_cuda:
            lab_batch_scaled = torch.cuda.FloatTensor(lab_batch.size()).fill_(0)
        else:
            lab_batch_scaled = torch.FloatTensor(lab_batch.size()).fill_(0)  # dim 6 x 14 x 5

        lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1] * img_size
        lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2] * img_size
        lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3] * img_size
        lab_batch_scaled[:, :, 4] = lab_batch[:, :, 4] * img_size
        # lab_batch_scaled[:, :, 5] = lab_batch[:, :, 5] * img_size
        # lab_batch_scaled[:, :, 6] = lab_batch[:, :, 6] * img_size
        # lab_batch_scaled[:, :, 7] = lab_batch[:, :, 7] * img_size
        # lab_batch_scaled[:, :, 8] = lab_batch[:, :, 8] * img_size
        # lab_batch_scaled[:, :, 9] = lab_batch[:, :, 9] * img_size
        # lab_batch_scaled[:, :, 10] = lab_batch[:, :, 10] * img_size
        # lab_batch_scaled[:, :, 11] = lab_batch[:, :, 11] * img_size
        # lab_batch_scaled[:, :, 12] = lab_batch[:, :, 12] * img_size
        # lab_batch_scaled[:, :, 13] = lab_batch[:, :, 13] * img_size
        # lab_batch_scaled[:, :, 14] = lab_batch[:, :, 14] * img_size

        target_size_paper = torch.sqrt(((lab_batch_scaled[:, :, 3].mul(0.2)) ** 2) + ((lab_batch_scaled[:, :, 4].mul(0.2)) ** 2))
        target_size_mine = lab_batch_scaled[:,:,3] # larger dimension, width if 3, height if 4


        # shift to box centre
        # target_x = lab_batch[:,:,1] + lab_batch[:,:,3]/2
        # target_x = target_x.view(np.prod(batch_size))
        # target_y = lab_batch[:, :, 2] + lab_batch[:, :, 4] / 2
        # target_y = target_y.view(np.prod(batch_size))

        if loc == 'mouth':
            # shift to mouth centre
            target_x = (lab_batch[:,:,11] + lab_batch[:,:,13])/2
            target_x = target_x.view(np.prod(batch_size))
            target_y = (lab_batch[:,:,12] + lab_batch[:,:,14])/2
            target_y = target_y.view(np.prod(batch_size))
            
        elif loc == 'mouth_hide_nose':
            # shift to mouth centre
            target_x = (lab_batch[:,:,11] + lab_batch[:,:,13])/2
            target_x = target_x.view(np.prod(batch_size))
            target_y = (lab_batch[:,:,12] + lab_batch[:,:,14])/2 - 0.02
            target_y = target_y.view(np.prod(batch_size))

        elif loc == 'forehead':
            greater_mask_forehead = torch.ge(lab_batch[:,:,6], lab_batch[:,:,8])
            target_x = (lab_batch[:, :, 7] + lab_batch[:, :, 5]) / 2
            target_x = torch.where(greater_mask_forehead== True, target_x-0.01, target_x+0.01)
            target_x = target_x.view(np.prod(batch_size))
            target_y = ((((lab_batch[:, :, 6] + lab_batch[:, :, 8]) / 2) + lab_batch[:,:,2]) / 2) -0.01
            target_y = target_y.view(np.prod(batch_size))

        elif loc == 'eyes':
            target_x = (lab_batch[:, :, 5] + lab_batch[:, :, 7]) / 2
            target_x = target_x.view(np.prod(batch_size))
            target_y = (lab_batch[:, :, 6] + lab_batch[:, :, 8]) / 2
            target_y = target_y.view(np.prod(batch_size))


        targetoff_x = lab_batch[:, :, 3].view(np.prod(batch_size))  # used to get off_x
        targetoff_y = lab_batch[:, :, 4].view(np.prod(batch_size))  # used to get off_y

        if(rand_loc):
            if use_cuda:
                off_x = targetoff_x*(torch.cuda.FloatTensor(targetoff_x.size()).uniform_(-0.4,0.4))
                off_y = targetoff_y*(torch.cuda.FloatTensor(targetoff_y.size()).uniform_(-0.4,0.4))
            else:
                off_x = targetoff_x * (torch.FloatTensor(targetoff_x.size()).uniform_(-0.4, 0.4))
                off_y = targetoff_y * (torch.FloatTensor(targetoff_y.size()).uniform_(-0.4, 0.4))

            target_x = target_x + off_x
            target_y = target_y + off_y

        #target_y = target_y - 0.05

        scale = target_size_paper / current_patch_size_paper
        scale = scale.view(anglesize)
        #print(scale)

        s = adv_batch.size() # 6 x 14 x 3 x 416 x 416
        adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])  # 84 x 3 x 416 x 416
        msk_batch = msk_batch.view(s[0] * s[1], s[2], s[3], s[4])  # 84 x 3 x 416 x 416

        tx = (-target_x+0.5)*2
        ty = (-target_y+0.5)*2


        sin = torch.sin(angle)
        cos = torch.cos(angle)

        # Theta = rotation, rescale matrix
        if use_cuda:
            theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        else:
            theta = torch.FloatTensor(anglesize, 2, 3).fill_(0) # dim 84 x 2 x 3 (N x 2 x 3) required by F.affine_grid

        # theta[:, 0, 0] = cos/scale
        # theta[:, 0, 1] = sin/scale
        # theta[:, 0, 2] = tx*cos/scale+ty*sin/scale
        # theta[:, 1, 0] = -sin/scale
        # theta[:, 1, 1] = cos/scale
        # theta[:, 1, 2] = -tx*sin/scale+ty*cos/scale

        theta[:, 0, 0] = (cos*cos_align_angle -sin*sin_align_angle) / scale
        theta[:, 0, 1] = (sin*cos_align_angle + cos*sin_align_angle) / scale
        theta[:, 0, 2] = (tx * cos / scale + ty * sin / scale)*cos_align_angle + (-tx * sin / scale + ty * cos / scale)*sin_align_angle
        theta[:, 1, 0] = (-sin_align_angle*cos -cos_align_angle*sin) / scale
        theta[:, 1, 1] = (-sin_align_angle*sin + cos_align_angle*cos)/ scale
        theta[:, 1, 2] = -sin_align_angle * (tx * cos / scale + ty * sin / scale) + cos_align_angle * (-tx * sin / scale + ty * cos / scale)


        grid = F.affine_grid(theta, adv_batch.shape)  # adv_batch should be of type N x C x Hin x Win. Output is N x Hg x Wg x 2

        adv_batch_t = F.grid_sample(adv_batch, grid)  # computes the output using input values and pixel locations from grid.
        msk_batch_t = F.grid_sample(msk_batch, grid)  # Output has dim N x C x Hg x Wg

        adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4]) # 4 x 16 x 3 x 416 x 416
        msk_batch_t = msk_batch_t.view(s[0], s[1], s[2], s[3], s[4])

        adv_batch_t = torch.clamp(adv_batch_t, 0.000001, 0.999999)

        return adv_batch_t * msk_batch_t

class PatchApplier(nn.Module):
    """PatchApplier: applies adversarial patches to images.

    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.

    """

    def __init__(self):
        super(PatchApplier, self).__init__()

    def forward(self, img_batch, adv_batch):

        advs = torch.unbind(adv_batch, 1)

        for adv in advs:
            img_batch = torch.where((adv == 0), img_batch, adv)  # the output tensor has elements belonging to img_batch if adv == 0, else belonging to adv

        return img_batch

class FDDBDataset(Dataset):
    """InriaDataset: representation of the INRIA person dataset.

    Internal representation of the commonly used INRIA person dataset.
    Available at: http://pascal.inrialpes.fr/data/human/

    Attributes:
        len: An integer number of elements in the
        img_dir: Directory containing the images of the INRIA dataset.
        lab_dir: Directory containing the labels of the INRIA dataset.
        img_names: List of all image file names in img_dir.
        shuffle: Whether or not to shuffle the dataset.

    """

    def __init__(self, img_dir, lab_dir, max_lab, imgsize, shuffle=True):
        #imgsize = 500 as example

        # read images
        n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
        n_images = n_jpg_images
        # read labels
        n_labels = len(fnmatch.filter(os.listdir(lab_dir), '*.txt'))
        assert n_images == n_labels, "Number of images and number of labels don't match"
        self.len = n_images
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.imgsize = imgsize
        self.img_names = fnmatch.filter(os.listdir(img_dir), '*.jpg')
        self.shuffle = shuffle
        self.img_paths = []
        for img_name in self.img_names:
            self.img_paths.append(os.path.join(self.img_dir, img_name))
        self.lab_paths = []
        for img_name in self.img_names:
            lab_path = os.path.join(self.lab_dir, img_name).replace('.jpg', '.txt').replace('.png', '.txt')
            self.lab_paths.append(lab_path)
        self.max_n_labels = max_lab

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        lab_path = os.path.join(self.lab_dir, self.img_names[idx]).replace('.jpg', '.txt')
        image = Image.open(img_path).convert('RGB')

        if os.path.getsize(lab_path):
            label = np.loadtxt(lab_path)
        else:
            label = np.ones([16])

        label = torch.from_numpy(label).float()
        if label.dim() == 1:
            label = label.unsqueeze(0)

        image, label = self.pad_and_scale(image, label)
        #image.show()
        transform = transforms.ToTensor()
        image = transform(image)
        label = self.pad_lab(label)
        return image, label

    def pad_and_scale(self, img, lab):
        """
        Args:
            img:

        Returns:
        """
        w,h = img.size
        if w==h:
            padded_img = img
        else:
            dim_to_pad = 1 if w<h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h,h), color=(127,127,127))
                padded_img.paste(img, (int(padding), 0))
                lab[:, [1]] = (lab[:, [1]] + padding) / h # left
                lab[:, [2]] = (lab[:, [2]]) / h  # top
                lab[:, [3]] = (lab[:, [3]] / h) # width
                lab[:, [4]] = (lab[:, [4]] / h)  # height
                lab[:, [5]] = (lab[:, [5]] + padding) / h  # lex
                lab[:, [6]] = (lab[:, [6]] / h)  # ley
                lab[:, [7]] = (lab[:, [7]] + padding) / h  # rex
                lab[:, [8]] = (lab[:, [8]] / h)  # rey
                lab[:, [9]] = (lab[:, [9]] + padding) / h  # nx
                lab[:, [10]] = (lab[:, [10]] / h)  # ny
                lab[:, [11]] = (lab[:, [11]] + padding) / h  # lmx
                lab[:, [12]] = (lab[:, [12]] / h)  # lmy
                lab[:, [13]] = (lab[:, [13]] + padding) / h  # rmx
                lab[:, [14]] = (lab[:, [14]] / h)  # rmy

            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127,127,127))
                padded_img.paste(img, (0, int(padding)))
                lab[:, [1]] = (lab[:, [1]] / w)  # left
                lab[:, [2]] = (lab[:, [2]] + padding) / w  # top
                lab[:, [3]] = (lab[:, [3]] / w)  # width
                lab[:, [4]] = (lab[:, [4]] / w)  # height
                lab[:, [5]] = (lab[:, [5]] / w)  # lex
                lab[:, [6]] = (lab[:, [6]] + padding) / w  # ley
                lab[:, [7]] = (lab[:, [7]] / w)  # rex
                lab[:, [8]] = (lab[:, [8]] + padding) / w  # rey
                lab[:, [9]] = (lab[:, [9]] / w)  # nx
                lab[:, [10]] = (lab[:, [10]] + padding) / w  # ny
                lab[:, [11]] = (lab[:, [11]] / w)  # lmx
                lab[:, [12]] = (lab[:, [12]] + padding) / w  # lmy
                lab[:, [13]] = (lab[:, [13]] / w)  # rmx
                lab[:, [14]] = (lab[:, [14]] + padding) / w  # rmy

        resize = transforms.Resize((self.imgsize,self.imgsize)) # make a square image of dim 416 x 416
        padded_img = resize(padded_img)     #choose here
        return padded_img, lab

    def pad_lab(self, lab):
        pad_size = self.max_n_labels - lab.shape[0]
        if(pad_size>0):
            padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=1)

        else:
            padded_lab = lab
        return padded_lab

class PatchTransformer_glasses(nn.Module):
    """PatchTransformer: transforms batch of patches

    Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
    contrast, adding random amount of noise, and rotating randomly. Resizes patches according to as size based on the
    batch of labels, and pads them to the dimension of an image.

    """

    def __init__(self):
        super(PatchTransformer_glasses, self).__init__()

        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10
        self.minangle = -10 / 180 * math.pi
        self.maxangle = 10 / 180 * math.pi
        self.medianpooler = MedianPool2d(7, same=True)  # kernel_size = 7? see again

    def forward(self, adv_patch, lab_batch, img_size, one_zero_mask, loc, do_rotate=True, rand_loc=True, align_angle = True):

        use_cuda = 1

        #adv_patch = self.medianpooler(adv_patch.unsqueeze(0))
        #print(adv_patch.size())

        padx = (img_size - adv_patch.size(-1)) / 2
        pady = (img_size - adv_patch.size(-2)) / 2

        adv_patch = adv_patch.unsqueeze(0)

        adv_batch = adv_patch.expand(lab_batch.size(0), lab_batch.size(1), -1, -1, -1)
        one_zero_mask_batch = one_zero_mask.expand(lab_batch.size(0), lab_batch.size(1), -1, -1, -1)

        batch_size = torch.Size((lab_batch.size(0), lab_batch.size(1)))


        if use_cuda:
            contrast = torch.cuda.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
        else:
            contrast = torch.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)

        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        contrast = contrast.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))

        if use_cuda:
            contrast = contrast.cuda()
        else:
            contrast = contrast
#_________________________________________________________________________________________________________________________________________________
        # Create random brightness tensor
        if use_cuda:
            brightness = torch.cuda.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
        else:
            brightness = torch.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)

        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = brightness.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))

        if use_cuda:
            brightness = brightness.cuda()
        else:
            brightness = brightness

# _____________________________________________________________________________________________________________________________________________
        # Create random noise tensor
        if use_cuda:
            noise = torch.cuda.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor
        else:
            noise = torch.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor
#______________________________________________________________________________________________________________________________________________
        # Apply contrast/brightness/noise, clamp
        #adv_batch = adv_batch * contrast + brightness + noise
        if use_cuda:
            black_batch_2 = torch.cuda.FloatTensor(adv_batch.size()).fill_(0)
        else:
            black_batch_2 = torch.FloatTensor(adv_batch.size()).fill_(0)
            
        adv_batch_eot = torch.where(one_zero_mask_batch==0, black_batch_2, adv_batch * contrast + brightness + noise)
        adv_batch = torch.clamp(adv_batch_eot, 0.000001, 0.99999)

        #adv_batch = torch.where(adv_batch<)
        # adv_patch_im = transforms.ToPILImage('RGB')(adv_batch[0][0])
        # plt.imshow(adv_patch_im)
        # plt.show()

#______________________________________________________________________________________________________________________________________________
        # Where the label class_ids is 1 we don't want a patch (padding) --> fill mask with zero's

        cls_ids = torch.narrow(lab_batch, 2, 0, 1)
        cls_mask = cls_ids.expand(-1, -1, 3)
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(3))
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, -1, adv_batch.size(4))

        if use_cuda:
            msk_batch = torch.cuda.FloatTensor(cls_mask.size()).fill_(1) - cls_mask
        else:
            msk_batch = torch.FloatTensor(cls_mask.size()).fill_(1) - cls_mask


#_______________________________________________________________________________________________________________________________________________
        # Pad patch and mask to image dimensions with zeros
        mypad = nn.ConstantPad2d((int(padx + 0.5), int(padx), int(pady + 0.5), int(pady)), 0)
        adv_batch = mypad(adv_batch)
        one_zero_mask_batch = mypad(one_zero_mask_batch)
        msk_batch = mypad(msk_batch)

#_______________________________________________________________________________________________________________________________________________
        # Rotation and rescaling transforms
        anglesize = (lab_batch.size(0) * lab_batch.size(1))  # dim = 6*14 = 84
        if do_rotate:
            if use_cuda:
                angle = torch.cuda.FloatTensor(anglesize).uniform_(self.minangle, self.maxangle)
            else:
                angle = torch.FloatTensor(anglesize).uniform_(self.minangle, self.maxangle)

        else:
            if use_cuda:
                angle = torch.cuda.FloatTensor(anglesize).fill_(0)
            else:
                angle = torch.FloatTensor(anglesize).fill_(0)

        # Fixed rotation along mouth direction:
        if align_angle:

            third_vertex = [lab_batch[:,:,13], lab_batch[:,:,12]]
            mouth_length_hypot = torch.sqrt((lab_batch[:,:,13] - lab_batch[:,:,11])**2 + (lab_batch[:,:,14]-lab_batch[:,:,12])**2)
            #print('mouth_length_wo_1: ' + str(mouth_length_hypot))
            cath_opp = torch.sqrt((lab_batch[:,:,13] - third_vertex[0])**2 + (lab_batch[:,:,14]-third_vertex[1])**2)
            #print('cath_opp: ' + str(cath_opp))
            cath_adj = torch.sqrt(mouth_length_hypot**2 - cath_opp**2)
            #print('cath_adj: ' + str(cath_adj))
            
            if use_cuda:
                mouth_length_hypot = torch.where(mouth_length_hypot==0, torch.cuda.FloatTensor(batch_size).fill_(1), mouth_length_hypot)
                #print('mouth_length_with1: ' + str(mouth_length_hypot))
            else:
                mouth_length_hypot = torch.where(mouth_length_hypot==0, torch.FloatTensor(batch_size).fill_(1), mouth_length_hypot)
                #print('mouth_length_with1: ' + str(mouth_length_hypot))
            
            sin_align_angle = cath_opp/mouth_length_hypot
            cos_align_angle = cath_adj/mouth_length_hypot

            greater_mask = torch.ge(lab_batch[:,:,12], lab_batch[:,:,14])
            #print(greater_mask)
            #cos_align_angle = torch.where(greater_mask==True, cos_align_angle*(-1), cos_align_angle)
            sin_align_angle = torch.where(greater_mask == True, sin_align_angle * (-1), sin_align_angle)

        else:
            if use_cuda:
                sin_align_angle = torch.cuda.FloatTensor(batch_size).fill_(0)
                cos_align_angle = torch.cuda.FloatTensor(batch_size).fill_(0)
            else:
                sin_align_angle = torch.FloatTensor(batch_size).fill_(0)
                cos_align_angle = torch.FloatTensor(batch_size).fill_(0)

        sin_align_angle = sin_align_angle.view(anglesize)
        cos_align_angle = cos_align_angle.view(anglesize)
        # print(sin_align_angle)
        # print(cos_align_angle)
#_______________________________________________________________________________________________________________________________________________
        # Resizes and rotates
        current_patch_size_mine = adv_patch.size(-1) # width if -1 (larger dim), height if -2 (smaller dim)
        current_patch_size_paper = adv_patch.size(-2)

        if use_cuda:
            lab_batch_scaled = torch.cuda.FloatTensor(lab_batch.size()).fill_(0)
        else:
            lab_batch_scaled = torch.FloatTensor(lab_batch.size()).fill_(0)  # dim 6 x 14 x 5

        lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1] * img_size
        lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2] * img_size
        lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3] * img_size
        lab_batch_scaled[:, :, 4] = lab_batch[:, :, 4] * img_size
        # lab_batch_scaled[:, :, 5] = lab_batch[:, :, 5] * img_size
        # lab_batch_scaled[:, :, 6] = lab_batch[:, :, 6] * img_size
        # lab_batch_scaled[:, :, 7] = lab_batch[:, :, 7] * img_size
        # lab_batch_scaled[:, :, 8] = lab_batch[:, :, 8] * img_size
        # lab_batch_scaled[:, :, 9] = lab_batch[:, :, 9] * img_size
        # lab_batch_scaled[:, :, 10] = lab_batch[:, :, 10] * img_size
        # lab_batch_scaled[:, :, 11] = lab_batch[:, :, 11] * img_size
        # lab_batch_scaled[:, :, 12] = lab_batch[:, :, 12] * img_size
        # lab_batch_scaled[:, :, 13] = lab_batch[:, :, 13] * img_size
        # lab_batch_scaled[:, :, 14] = lab_batch[:, :, 14] * img_size

        target_size_paper = torch.sqrt(((lab_batch_scaled[:, :, 3].mul(0.2)) ** 2) + ((lab_batch_scaled[:, :, 4].mul(0.2)) ** 2))
        target_size_mine = lab_batch_scaled[:,:,3] # larger dimension, width if 3, height if 4


        # shift to box centre
        # target_x = lab_batch[:,:,1] + lab_batch[:,:,3]/2
        # target_x = target_x.view(np.prod(batch_size))
        # target_y = lab_batch[:, :, 2] + lab_batch[:, :, 4] / 2
        # target_y = target_y.view(np.prod(batch_size))

        if loc == 'mouth':
            # shift to mouth centre
            target_x = (lab_batch[:,:,11] + lab_batch[:,:,13])/2
            target_x = target_x.view(np.prod(batch_size))
            target_y = (lab_batch[:,:,12] + lab_batch[:,:,14])/2
            target_y = target_y.view(np.prod(batch_size))

        elif loc == 'forehead':
            greater_mask_forehead = torch.ge(lab_batch[:,:,6], lab_batch[:,:,8])
            target_x = (lab_batch[:, :, 7] + lab_batch[:, :, 5]) / 2
            target_x = torch.where(greater_mask_forehead== True, target_x-0.01, target_x+0.01)
            target_x = target_x.view(np.prod(batch_size))
            target_y = ((((lab_batch[:, :, 6] + lab_batch[:, :, 8]) / 2) + lab_batch[:,:,2]) / 2) -0.01
            target_y = target_y.view(np.prod(batch_size))

        elif loc == 'eyes':
            target_x = (lab_batch[:, :, 5] + lab_batch[:, :, 7]) / 2
            target_x = target_x.view(np.prod(batch_size))
            target_y = (lab_batch[:, :, 6] + lab_batch[:, :, 8]) / 2
            target_y = target_y.view(np.prod(batch_size))


        targetoff_x = lab_batch[:, :, 3].view(np.prod(batch_size))  # used to get off_x
        targetoff_y = lab_batch[:, :, 4].view(np.prod(batch_size))  # used to get off_y

        if(rand_loc):
            if use_cuda:
                off_x = targetoff_x*(torch.cuda.FloatTensor(targetoff_x.size()).uniform_(-0.4,0.4))
                off_y = targetoff_y*(torch.cuda.FloatTensor(targetoff_y.size()).uniform_(-0.4,0.4))
            else:
                off_x = targetoff_x * (torch.FloatTensor(targetoff_x.size()).uniform_(-0.4, 0.4))
                off_y = targetoff_y * (torch.FloatTensor(targetoff_y.size()).uniform_(-0.4, 0.4))

            target_x = target_x + off_x
            target_y = target_y + off_y

        #target_y = target_y - 0.05

        scale = target_size_paper / current_patch_size_paper
        scale = scale.view(anglesize)
        # print(scale)

        s = adv_batch.size() # 6 x 14 x 3 x 416 x 416
        adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])  # 84 x 3 x 416 x 416
        one_zero_mask_batch = one_zero_mask_batch.view(s[0] * s[1], s[2], s[3], s[4])
        msk_batch = msk_batch.view(s[0] * s[1], s[2], s[3], s[4])  # 84 x 3 x 416 x 416

        tx = (-target_x+0.5)*2
        ty = (-target_y+0.5)*2


        sin = torch.sin(angle)
        cos = torch.cos(angle)

        # Theta = rotation, rescale matrix
        if use_cuda:
            theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        else:
            theta = torch.FloatTensor(anglesize, 2, 3).fill_(0) # dim 84 x 2 x 3 (N x 2 x 3) required by F.affine_grid

        # theta[:, 0, 0] = cos/scale
        # theta[:, 0, 1] = sin/scale
        # theta[:, 0, 2] = tx*cos/scale+ty*sin/scale
        # theta[:, 1, 0] = -sin/scale
        # theta[:, 1, 1] = cos/scale
        # theta[:, 1, 2] = -tx*sin/scale+ty*cos/scale

        theta[:, 0, 0] = (cos*cos_align_angle -sin*sin_align_angle) / scale
        theta[:, 0, 1] = (sin*cos_align_angle + cos*sin_align_angle) / scale
        theta[:, 0, 2] = (tx * cos / scale + ty * sin / scale)*cos_align_angle + (-tx * sin / scale + ty * cos / scale)*sin_align_angle
        theta[:, 1, 0] = (-sin_align_angle*cos -cos_align_angle*sin) / scale
        theta[:, 1, 1] = (-sin_align_angle*sin + cos_align_angle*cos)/ scale
        theta[:, 1, 2] = -sin_align_angle * (tx * cos / scale + ty * sin / scale) + cos_align_angle * (-tx * sin / scale + ty * cos / scale)


        grid = F.affine_grid(theta, adv_batch.shape)  # adv_batch should be of type N x C x Hin x Win. Output is N x Hg x Wg x 2

        adv_batch_t = F.grid_sample(adv_batch, grid)  # computes the output using input values and pixel locations from grid.
        one_zero_mask_batch_t = F.grid_sample(one_zero_mask_batch, grid)
        msk_batch_t = F.grid_sample(msk_batch, grid)  # Output has dim N x C x Hg x Wg

        adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4]) # 4 x 16 x 3 x 416 x 416
        one_zero_mask_batch_t = one_zero_mask_batch_t.view(s[0], s[1], s[2], s[3], s[4])
        msk_batch_t = msk_batch_t.view(s[0], s[1], s[2], s[3], s[4])

        adv_batch_t = torch.clamp(adv_batch_t, 0.000001, 0.999999)
        
        adv_batch_out = adv_batch_t * one_zero_mask_batch_t

        return adv_batch_out * msk_batch_t

def choose_patch(choice):
    use_cuda=0
    if choice == 'gray':
        adv_patch_size = (250, 450)
        adv_patch = torch.full((3, adv_patch_size[0], adv_patch_size[1]), 0.5)
    elif choice == 'gray_mouth_bigger':
        adv_patch_size = (300, 450)
        adv_patch = torch.full((3, adv_patch_size[0], adv_patch_size[1]), 0.5)
    elif choice == 'glasses':
        glass_path = "./glasses2.png"
        patch = Image.open(glass_path).convert('RGB')
        patch.thumbnail((500, 500))
        adv_patch = transforms.ToTensor()(patch)
        
        # preprocess
        if use_cuda:
            gray_batch = torch.cuda.FloatTensor(adv_patch.size()).fill_(0.5)
            print(gray_batch.size())
            black_batch = torch.cuda.FloatTensor(adv_patch.size()).fill_(0)
        else:
            gray_batch = torch.FloatTensor(adv_patch.size()).fill_(0.5)
            black_batch = torch.FloatTensor(adv_patch.size()).fill_(0)
            
        adv_patch = torch.where(adv_patch == 31 / 255, gray_batch, black_batch)
        # adv_patch_im = transforms.ToPILImage('RGB')(adv_patch)
        # plt.imshow(adv_patch_im)
        # plt.show()
        
    return adv_patch


# if __name__ == '__main__':

#     img_dir = "C:/Users/Alessandro/PycharmProjects/mtcnn_pytorch_adversarial/FDDB_test_results/clean_results/mtcnn_images_set_filtered/"
#     lab_dir = "C:/Users/Alessandro/PycharmProjects/mtcnn_pytorch_adversarial/FDDB_test_results/clean_results/mtcnn_labels_filtered/"
#     max_lab = 8
#     img_size = 600
#     batch_size = 3

#     train_loader = torch.utils.data.DataLoader(FDDBDataset(img_dir, lab_dir, max_lab, img_size, shuffle=True), batch_size=batch_size, shuffle=True, num_workers=10)

#     n=0
#     for i_batch, (img_batch, lab_batch) in enumerate(train_loader):
#         n+=1
#         print(n)

#         adv_patch = choose_patch('glasses')

#         adv_patch_im = transforms.ToPILImage('RGB')(adv_patch)
#         plt.imshow(adv_patch_im)
#         plt.show()

#         gray_batch = torch.full((3, adv_patch.size()[1], adv_patch.size()[2]), 0.5)
#         black_batch = torch.full((3, adv_patch.size()[1], adv_patch.size()[2]), 0)
#         white_batch = torch.full((3, adv_patch.size()[1], adv_patch.size()[2]), 255 / 255)

#         adv_patch = torch.where(adv_patch==31/255, gray_batch, black_batch)
#         adv_patch_im = transforms.ToPILImage('RGB')(adv_patch)
#         plt.imshow(adv_patch_im)
#         plt.show()

#         adv_batch = PatchTransformer_glasses()(adv_patch, lab_batch, img_size, loc= 'eyes', do_rotate=False, rand_loc=False, align_angle = True)
#         p_img_batch = PatchApplier()(img_batch, adv_batch)
#         p_img_batch = F.interpolate(p_img_batch, (img_size, img_size))

#         im = transforms.ToPILImage('RGB')(p_img_batch[0])
#         plt.imshow(im)
#         plt.show()






