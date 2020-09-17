import PIL
from tqdm import tqdm
from manage_data import *
import gc
import matplotlib.pyplot as plt
import torch
from torch import autograd
from torchvision import transforms
import patch_manage
import manage_data
import sys
import time
import os
from models import mtcnn
from models.utils.detect_face import detect_face
from torch.nn.functional import interpolate
from models.utils.detect_face import generateBoundingBox, batched_nms, bbreg, rerec, pad, batched_nms_numpy

import cv2



class PatchTrainer(object):

    #os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    torch.cuda.set_device(0)

    def __init__(self, mode):

        self.config = patch_manage.patch_configs[mode]()  # select the mode for the patch

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(torch.cuda.device_count())


        if use_cuda:
            
            self.mtcnn = mtcnn.MTCNN().to(self.device)
            self.factor = self.mtcnn.factor # 0.709
            self.thresholds = self.mtcnn.thresholds # [0.6 0.7 0.7]
            self.min_face_size = self.mtcnn.min_face_size
            self.onet = mtcnn.ONet().to(self.device)
            self.rnet = mtcnn.RNet().to(self.device)
            self.pnet = mtcnn.PNet().to(self.device)
            
            self.patch_applier = PatchApplier().to(self.device)
            self.patch_transformer = PatchTransformer().to(self.device)

            self.score_extractor_mtcnn = mtcnn_feature_output_manage(self.config).to(self.device)  # 15 is person class in VOC (with 21 elements)
            self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size_gray).to(self.device)
            self.total_variation = TotalVariation().to(self.device)
        else:
            
            self.mtcnn = mtcnn.MTCNN()
            self.factor = self.mtcnn.factor # 0.709
            self.thresholds = self.mtcnn.thresholds # [0.6 0.7 0.7]
            self.min_face_size = self.mtcnn.min_face_size
            self.onet = mtcnn.ONet()
            self.rnet = mtcnn.RNet()
            self.pnet = mtcnn.PNet()
            
            self.patch_applier = PatchApplier()
            self.patch_transformer = PatchTransformer()

            self.score_extractor_mtcnn = mtcnn_feature_output_manage(self.config).to(self.device)
            self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size_glasses)
            self.total_variation = TotalVariation()

    def train(self):
        
        destination_path = "./"

        destination_name = 'loss_tracking_mouth_max_400ep_allnets.txt'
        destination_name2 = 'loss_tracking_compact_batch_mouth_max_400ep_allnets.txt'
        destination_name3 = 'loss_tracking_compatc_epochs_mouth_max_400ep_all_nets.txt'

        destination_name = 'loss_tracking_mouth_max_400ep_all_nets.txt'
        destination_name2 = 'loss_tracking_compact_batch_mouth_max_400ep_all_nets.txt'
        destination_name3 = 'loss_tracking_compatc_epochs_mouth_max_400ep_all_nets.txt'

        destination = os.path.join(destination_path, destination_name)
        destination2 = os.path.join(destination_path, destination_name2)
        destination3 = os.path.join(destination_path, destination_name3)
        textfile = open(destination, 'w+')
        textfile2 = open(destination2, 'w+')
        textfile3 = open(destination3, 'w+')

        max_lab = 8
        img_size = 600
        n_epochs = 400

        glasses = 0

        glasses = 1


        # load/create initial adv_patch
        adv_patch_cpu = choose_patch('gray')

        adv_patch_cpu.requires_grad_(True)
        
        one_zero_mask_cpu = torch.where(adv_patch_cpu==0, torch.zeros_like(adv_patch_cpu), torch.ones_like(adv_patch_cpu))

        train_loader = torch.utils.data.DataLoader(FDDBDataset(self.config.img_dir, self.config.lab_dir, max_lab, img_size, shuffle=True),
                                                   batch_size=self.config.batch_size, shuffle=True, num_workers=10)

        n = 0

        self.epoch_length = len(train_loader)
        print(f'One epoch is {len(train_loader)}')

        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)  # starting lr = 0.03
        scheduler = self.config.scheduler_factory(optimizer)

        et0 = time.time()  # epoch start
        for epoch in range(n_epochs):

            ep_det_loss = 0
            ep_nps_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            bt0 = time.time()  # batch start

            for i_batch, (img_batch, lab_batch) in enumerate(train_loader):

                n += 1
                
                print('EPOCH NR. ' + str(epoch))

                with autograd.detect_anomaly():

                    if use_cuda:
                        img_batch = img_batch.to(self.device)
                        lab_batch = lab_batch.to(self.device)
                        adv_patch = adv_patch_cpu.to(self.device)
                        one_zero_mask = one_zero_mask_cpu.to(self.device)
                    else:
                        img_batch = img_batch
                        lab_batch = lab_batch
                        adv_patch = adv_patch_cpu
                        one_zero_mask = one_zero_mask_cpu

                    adv_batch = self.patch_transformer(adv_patch, lab_batch, img_size, loc='mouth', do_rotate=False, rand_loc=False, align_angle=True)
                    p_img_batch = self.patch_applier(img_batch, adv_batch)
                    p_img_batch = F.interpolate(p_img_batch, (img_size, img_size))

                    # im = transforms.ToPILImage('RGB')(p_img_batch[0].cpu())
                    # plt.imshow(im)
                    # plt.show()
                    #im.save('C:/Users/Alessandro/Desktop/try.png')

                    #p_img_batch = cv2.cvtColor(cv2.imread("C:/Users/Alessandro/Desktop/try.png"), cv2.COLOR_BGR2RGB)

                    p_img_batch = p_img_batch.permute(0,2,3,1) # make the input channel last
                    p_img_batch = p_img_batch*255

                    all_nets = 1
                    output_switch = 'all_nets'

                    net_output = mtcnn_detection(output_switch, p_img_batch, self.min_face_size, self.pnet, self.rnet, self.onet, self.thresholds, self.factor, self.device)

                    # max_prob = self.prob_extractor(output)
                    score_net = self.score_extractor_mtcnn(all_nets, output_switch, net_output, loss_type = 'max_approach', scale_level_score = 'scale_mean')
                    nps = self.nps_calculator(adv_patch)
                    tv = self.total_variation(adv_patch)

                    nps_loss = nps * 0.01
                    tv_loss = tv * 2.0

                    # batch_op: mean, max...
                    # do it for pnet output only, which is still batched. For rnet/onet, batched nms had been applied also to the batch
                    det_loss = score_net

                    if use_cuda:
                        loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1).to(self.device))
                    else:
                        loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1))

                    ep_det_loss += det_loss.detach().cpu().numpy() / len(train_loader)
                    ep_nps_loss += nps_loss.detach().cpu().numpy()
                    ep_tv_loss += tv_loss.detach().cpu().numpy()
                    ep_loss += loss

                    # Optimization step + backward
                    loss.backward()
                    
                    # avoid optimization in the background when glasses are considered
                    if glasses:
                        #one_zero_mask_cud = torch.where(adv_patch_cpu==0, torch.zeros_like(adv_patch), torch.ones_like(adv_patch))
                        with torch.no_grad():
                            adv_patch_cpu.grad = torch.where(one_zero_mask_cpu==0, torch.zeros_like(adv_patch_cpu.grad), adv_patch_cpu.grad)

                    optimizer.step()
                    optimizer.zero_grad()
                    adv_patch_cpu.data.clamp_(0, 1)  # keep patch in image range

                    bt1 = time.time()  # batch end
                    if i_batch % 1 == 0:

                        # im = transforms.ToPILImage('RGB')(adv_patch_cpu)
                        # plt.imshow(im)
                        # plt.show()

                        print('  BATCH NR: ', i_batch)
                        print('BATCH LOSS: ', loss)  # .detach().cpu().numpy())
                        print('  DET LOSS: ', det_loss)  # .detach().cpu().numpy())
                        print('  NPS LOSS: ', nps_loss)  # .detach().cpu().numpy())
                        print('   TV LOSS: ', tv_loss)  # .detach().cpu().numpy())
                        print('BATCH TIME: ', bt1 - bt0)
                        
                        textfile.write(f'i_batch: {i_batch}\nb_tot_loss:{loss}\nb_det_loss: {det_loss}\nb_nps_loss: {nps_loss}\nb_TV_loss: {tv_loss}\n\n')
                        textfile2.write(f'{i_batch} {loss} {det_loss} {nps_loss} {tv_loss}\n')

                    if i_batch + 1 >= len(train_loader):
                        print('\n')

                    else:
                        del adv_batch, net_output, score_net, det_loss, p_img_batch, nps_loss, tv_loss, loss

                        if use_cuda:
                            torch.cuda.empty_cache()

                    bt0 = time.time()

            et1 = time.time()  # epoch end

            ep_det_loss = ep_det_loss / len(train_loader)
            ep_nps_loss = ep_nps_loss / len(train_loader)
            ep_tv_loss = ep_tv_loss / len(train_loader)
            ep_loss = ep_loss / len(train_loader)

            # optimize after epoch passed
            scheduler.step(ep_loss)

            if True:
                print('  EPOCH NR: ', epoch),
                print('EPOCH LOSS: ', ep_loss)
                print('  DET LOSS: ', ep_det_loss)
                print('  NPS LOSS: ', ep_nps_loss)
                print('   TV LOSS: ', ep_tv_loss)
                print('EPOCH TIME: ', et1 - et0)
                
                textfile.write(f'\ni_epoch: {epoch}\ne_total_loss:{ep_loss}\ne_det_loss: {ep_det_loss}\ne_nps_loss: {ep_nps_loss}\ne_TV_loss: {ep_tv_loss}\n\n')
                textfile3.write(f'{epoch} {ep_loss} {ep_det_loss} {ep_nps_loss} {ep_tv_loss}\n')

                # Plot and/or save the final adv_patch (learned) and save it
                im = transforms.ToPILImage('RGB')(adv_patch_cpu)
                # plt.imshow(im)
                # plt.show()
                im.save("./saved_patches_mytrial/mtcnn_mouth_max_400ep_all_nets.jpg")

                del adv_batch, net_output, score_net, det_loss, p_img_batch, nps_loss, tv_loss, loss

                if use_cuda:
                    torch.cuda.empty_cache()

            et0 = time.time()



def imresample(img, sz):
    im_data = interpolate(img, size=sz, mode="area")
    return im_data

def pnet_detection(imgs, pnet, device, minsize, factor):
    if isinstance(imgs, (np.ndarray, torch.Tensor)):
        # print('here')
        imgs = torch.as_tensor(imgs, device=device)
        if len(imgs.shape) == 3:
            imgs = imgs.unsqueeze(0)
    else:
        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]  # [tensor.shape == 4]
        if any(img.size != imgs[0].size for img in imgs):
            raise Exception("MTCNN batch processing only compatible with equal-dimension images.")
        imgs = np.stack([np.uint8(img) for img in imgs])

    imgs = torch.as_tensor(imgs, device=device)
    # print(imgs)

    model_dtype = next(pnet.parameters()).dtype
    imgs = imgs.permute(0, 3, 1, 2).type(model_dtype)

    batch_size = len(imgs)
    h, w = imgs.shape[2:4]
    m = 12.0 / minsize
    minl = min(h, w)
    minl = minl * m

    # Create scale pyramid
    scale_i = m
    scales = []
    while minl >= 12:
        scales.append(scale_i)
        scale_i = scale_i * factor
        minl = minl * factor

    # First stage

    pnet_output = []
    for scale in scales:
        im_data = imresample(imgs, (int(h * scale + 1), int(w * scale + 1)))
        im_data = (im_data - 127.5) * 0.0078125
        # print('here')
        reg, probs = pnet(im_data)
        # print('here again')
        # print(torch.max(probs[:,1]))
        pnet_output.append(probs[:, 1])

    return pnet_output

def mtcnn_detection(output_switch, imgs, minsize, pnet, rnet, onet, threshold, factor, device):

    if isinstance(imgs, (np.ndarray, torch.Tensor)):
        # print('here')
        imgs = torch.as_tensor(imgs, device=device)
        if len(imgs.shape) == 3:
            imgs = imgs.unsqueeze(0)
    else:
        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]  # [tensor.shape == 4]
        if any(img.size != imgs[0].size for img in imgs):
            raise Exception("MTCNN batch processing only compatible with equal-dimension images.")
        imgs = np.stack([np.uint8(img) for img in imgs])

    imgs = torch.as_tensor(imgs, device=device)
    # print(imgs)

    model_dtype = next(pnet.parameters()).dtype
    imgs = imgs.permute(0, 3, 1, 2).type(model_dtype)

    batch_size = len(imgs)
    h, w = imgs.shape[2:4]
    m = 12.0 / minsize
    minl = min(h, w)
    minl = minl * m

    # Create scale pyramid
    scale_i = m
    scales = []
    while minl >= 12:
        scales.append(scale_i)
        scale_i = scale_i * factor
        minl = minl * factor

    # First stage: pnet
    boxes = []
    image_inds = []
    all_inds = []
    all_i = 0
    pnet_output = []
    for scale in scales:
        im_data = imresample(imgs, (int(h * scale + 1), int(w * scale + 1)))
        im_data = (im_data - 127.5) * 0.0078125
        # print('here')
        reg, probs = pnet(im_data)
        # print('here again')
        # print(torch.max(probs[:,1]))
        pnet_output.append(probs[:, 1])

        boxes_scale, image_inds_scale = generateBoundingBox(reg, probs[:, 1], scale, threshold[0])
        boxes.append(boxes_scale)
        image_inds.append(image_inds_scale)
        all_inds.append(all_i + image_inds_scale)
        all_i += batch_size

    # stop here if only pnet output is wanted
    if output_switch== 'pnet':
        print('Only pnet selected')
        return pnet_output

    boxes = torch.cat(boxes, dim=0)
    image_inds = torch.cat(image_inds, dim=0).cpu()
    all_inds = torch.cat(all_inds, dim=0)

    # NMS within each scale + image
    pick = batched_nms(boxes[:, :4], boxes[:, 4], all_inds, 0.5)
    boxes, image_inds = boxes[pick], image_inds[pick]

    # NMS within each image
    pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
    boxes, image_inds = boxes[pick], image_inds[pick]

    regw = boxes[:, 2] - boxes[:, 0]
    regh = boxes[:, 3] - boxes[:, 1]
    qq1 = boxes[:, 0] + boxes[:, 5] * regw
    qq2 = boxes[:, 1] + boxes[:, 6] * regh
    qq3 = boxes[:, 2] + boxes[:, 7] * regw
    qq4 = boxes[:, 3] + boxes[:, 8] * regh
    boxes = torch.stack([qq1, qq2, qq3, qq4, boxes[:, 4]]).permute(1, 0)
    boxes = rerec(boxes)
    y, ey, x, ex = pad(boxes, w, h)

    # Second stage: rnet
    if len(boxes) > 0:
        im_data = []
        for k in range(len(y)):
            if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                img_k = imgs[image_inds[k], :, (y[k] - 1):ey[k], (x[k] - 1):ex[k]].unsqueeze(0)
                im_data.append(imresample(img_k, (24, 24)))
        im_data = torch.cat(im_data, dim=0)
        im_data = (im_data - 127.5) * 0.0078125
        out = rnet(im_data)

        out0 = out[0].permute(1, 0)
        out1 = out[1].permute(1, 0)
        score_r = out1[1, :]

        if output_switch=='rnet':
            print('Only rnet selected')
            print(score_r.size())
            return score_r

        ipass = score_r > threshold[1]
        boxes = torch.cat((boxes[ipass, :4], score_r[ipass].unsqueeze(1)), dim=1)
        image_inds = image_inds[ipass]
        mv = out0[:, ipass].permute(1, 0)

        # NMS within each image
        pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
        boxes, image_inds, mv = boxes[pick], image_inds[pick], mv[pick]
        boxes = bbreg(boxes, mv)
        boxes = rerec(boxes)

    # Third stage
    points = torch.zeros(0, 5, 2, device=device)
    if len(boxes) > 0:
        y, ey, x, ex = pad(boxes, w, h)
        im_data = []
        for k in range(len(y)):
            if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                img_k = imgs[image_inds[k], :, (y[k] - 1):ey[k], (x[k] - 1):ex[k]].unsqueeze(0)
                im_data.append(imresample(img_k, (48, 48)))
        im_data = torch.cat(im_data, dim=0)
        im_data = (im_data - 127.5) * 0.0078125
        out = onet(im_data)

        out0 = out[0].permute(1, 0)
        out1 = out[1].permute(1, 0)
        out2 = out[2].permute(1, 0)
        score_o = out2[1, :]
        print(score_o)

        if output_switch=='onet':
            print('Only onet selected')
            return score_o

        if output_switch=='all_nets':
            return [pnet_output, score_r, score_o]

        points = out1
        ipass = score_o > threshold[2]
        points = points[:, ipass]
        boxes = torch.cat((boxes[ipass, :4], score_o[ipass].unsqueeze(1)), dim=1)
        image_inds = image_inds[ipass]
        mv = out0[:, ipass].permute(1, 0)

        w_i = boxes[:, 2] - boxes[:, 0] + 1
        h_i = boxes[:, 3] - boxes[:, 1] + 1
        points_x = w_i.repeat(5, 1) * points[:5, :] + boxes[:, 0].repeat(5, 1) - 1
        points_y = h_i.repeat(5, 1) * points[5:10, :] + boxes[:, 1].repeat(5, 1) - 1
        points = torch.stack((points_x, points_y)).permute(2, 1, 0)
        boxes = bbreg(boxes, mv)

        # NMS within each image using "Min" strategy
        # pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
        pick = batched_nms_numpy(boxes[:, :4], boxes[:, 4], image_inds, 0.7, 'Min')
        boxes, image_inds, points = boxes[pick], image_inds[pick], points[pick]

    boxes = boxes.detach().cpu().numpy()
    points = points.detach().cpu().numpy()

    batch_boxes = []
    batch_points = []
    for b_i in range(batch_size):
        b_i_inds = np.where(image_inds == b_i)
        batch_boxes.append(boxes[b_i_inds].copy())
        batch_points.append(points[b_i_inds].copy())

    batch_boxes, batch_points = np.array(batch_boxes), np.array(batch_points)

    return batch_boxes, batch_points, pnet_output

if __name__ == '__main__':

    use_cuda = 1
    trainer = PatchTrainer('base')
    trainer.train()
