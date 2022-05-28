
from __future__ import print_function
import multiprocessing

import os
import io
import sys
import time
import errno
import random
import pprint
import datetime
import dateutil.tz
import argparse

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from miscc.utils import mkdir_p
from miscc.utils import imagenet_deprocess_batch
from miscc.utils import build_super_images, build_super_images2
from miscc.utils import weights_init, load_params, copy_G_params

from miscc.config import cfg, cfg_from_file
from miscc.losses import discriminator_loss, DAMSM_loss, KL_loss 
from sync_batchnorm import DataParallelWithCallback
from datasets import TextDataset
from datasets import prepare_data
from DAMSM import RNN_ENCODER, CNN_ENCODER 
from model import NetG, NetD

# for Contrastive learning
from masks import mask_correlated_samples
from nt_xent import NT_Xent

from torch.utils.tensorboard import SummaryWriter

# dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
filepath = 'data'
dir_path = (os.path.abspath(os.path.join(os.path.realpath(filepath), './.')))
sys.path.append(dir_path)

multiprocessing.set_start_method('spawn', True)


UPDATE_INTERVAL = 50 


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/bird.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


def sampling(text_encoder, netG, dataloader, ixtoword, device):
    model_dir = cfg.TRAIN.NET_G 
    istart = cfg.TRAIN.NET_G.rfind('_') + 1
    iend = cfg.TRAIN.NET_G.rfind('.')
    start_epoch = int(cfg.TRAIN.NET_G[istart:iend]) 

    '''
    for path_count in range(11):
        if path_count > 0:
            current_epoch = next_epoch
        else:
            current_epoch = start_epoch
        next_epoch = start_epoch + path_count * 10
        model_dir = model_dir.replace(str(current_epoch), str(next_epoch))
    '''
   

    split_dir = 'valid'
    

    # Build and load the generator
    netG.load_state_dict(torch.load(model_dir))
    netG.eval()

    batch_size = cfg.TRAIN.BATCH_SIZE
    
    
    #######################################################
            # path to save generated images
    ######################################################
    
    s_tmp = model_dir[:model_dir.rfind('.pth')]
    s_tmp_dir = s_tmp
    fake_img_save_dir = '%s/%s' % (s_tmp, split_dir) # fake_img_save_dir: models/finetune/cub/netG_550/valid
    mkdir_p(fake_img_save_dir) 

    real_img_save_dir = '%s/%s' % (s_tmp, 'real') # real_img_save_dir: models/finetune/cub/netG_550/real
    mkdir_p(real_img_save_dir) 
    cap_save_dir = '%s/%s' % (s_tmp, 'caps') # cap_save_dir: models/finetune/cub/netG_550/caps
    mkdir_p(cap_save_dir) 

    idx = 0
    cnt = 0
   

    # R-Precision
    R_count = 0
    R = np.zeros(30000)

    cont = True

    for ii in range(11):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
        if (cont == False):
                     break
        for step, data in enumerate(dataloader, 0):
            cnt += batch_size
            if (cont == False):
                        break
            if step % 100 == 0:
                print('step: ', step)
            
            imags, imags_2, captions, cap_lens, class_ids, keys, captions_2, cap_lens_2, class_ids_2, \
                    sort_ind, sort_ind_2 = prepare_data(data)

            real_imgs = imags[0].to(device)

            #######################################################
                 # (1) Extract text embeddings
            ###############################################

            hidden = text_encoder.init_hidden(batch_size)
            
            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            
            words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
            
            mask = (captions == 0)
            num_words = words_embs.size(2)
            if mask.size(1) > num_words:
                mask = mask[:, :num_words]


            #######################################################
                 # (2) Generate fake images
            ######################################################
            

            with torch.no_grad():
                 noise = torch.randn(batch_size, 100)
                 noise = noise.to(device)
                 noise.data.normal_(0, 1)

                 fake_imgs, stage_masks = netG(noise, sent_emb)
                 stage_mask = stage_masks[-1]

            for j in range(batch_size):
                
                s_tmp = '%s/single/%s' % (fake_img_save_dir, keys[j])
                folder = s_tmp[:s_tmp.rfind('/')] # models/finetune/cub/netG_550/valid
                if not os.path.isdir(folder):
                    print('Make a new folder: ', folder)
                    mkdir_p(folder)

                k = -1 

                im = fake_imgs[j].data.cpu().numpy()
                
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                im = np.transpose(im, (1, 2, 0))
                im = Image.fromarray(im)

        
                fullpath = '%s_s%d_%d.png' % (s_tmp, k, ii)
                im.save(fullpath)


                # save the last fusion mask
                s_tmp = '%s/fm' % fake_img_save_dir 
                im = stage_mask[j].data.cpu().numpy()
                # [0, 1] --> [0, 255]
                # im = 1-im # only for better visualization
                im = im * 255.0
                im = im.astype(np.uint8)
                im = np.transpose(im, (1, 2, 0))
                im = np.squeeze(im, axis=2)
                im = Image.fromarray(im)
                fullpath = '%s_s%d_%d.png' % (s_tmp, k, ii) 
                im.save(fullpath)

                idx += 1

            #######################################################
                    # (3) calculate R-Precision
            ######################################################
    
            _, cnn_code = image_encoder(fake_imgs)

            for i in range(batch_size):
                    mis_captions, mis_captions_len = dataset.get_mis_caption(class_ids[i])
                    hidden = text_encoder.init_hidden(99)
                    _, sent_emb_t = text_encoder(mis_captions, mis_captions_len, hidden)
                    rnn_code = torch.cat((sent_emb[i, :].unsqueeze(0), sent_emb_t), 0)
                    ### cnn_code = 1 * nef
                    ### rnn_code = 100 * nef
                    scores = torch.mm(cnn_code[i].unsqueeze(0), rnn_code.transpose(0, 1))  # 1* 100
                    cnn_code_norm = torch.norm(cnn_code[i].unsqueeze(0), 2, dim=1, keepdim=True)
                    rnn_code_norm = torch.norm(rnn_code, 2, dim=1, keepdim=True)
                    norm = torch.mm(cnn_code_norm, rnn_code_norm.transpose(0, 1))
                    scores0 = scores / norm.clamp(min=1e-8)
                    if torch.argmax(scores0) == 0:
                        R[R_count] = 1
                    R_count += 1

            if R_count >= 30000:
                    sum = np.zeros(10)
                    np.random.shuffle(R)
                    for i in range(10):
                        sum[i] = np.average(R[i * 3000:(i + 1) * 3000 - 1])
                    R_mean = np.average(sum)
                    R_std = np.std(sum)
                    print("R mean:{:.4f} std:{:.4f}".format(R_mean, R_std))
                    cont = False


            '''
            # save the last fusion mask
            s_tmp = '%s/fm' % fake_img_save_dir # models/finetune/cub/netG_550/valid/fm
            im = stage_mask[j].data.cpu().numpy()
            # [0, 1] --> [0, 255]
            # im = 1-im # only for better visualization
            im = im * 255.0
            im = im.astype(np.uint8)
            im = np.transpose(im, (1, 2, 0))
            im = np.squeeze(im, axis=2)
            im = Image.fromarray(im)
            fullpath = '%s_%d.png' % (s_tmp, idx) # models/finetune/cub/netG_550/valid/fm/fm_s0.png
            im.save(fullpath)

            idx += 1
            '''

# generate samples from specific captions other than given dataste CUB 
def gen_sample(text_encoder, netG, device, wordtoix):
    """
    generate sample according to user defined captions.

    caption should be in the form of a list, and each element of the list is a description of the image in form of string.
    caption length should be no longer than 18 words.
    example captions see below
    """
    
    captions = ['A colorful blue bird has wings with dark stripes and small eyes',
                'A colorful green bird has wings with dark stripes and small eyes',
                'A colorful white bird has wings with dark stripes and small eyes',
                'A colorful black bird has wings with dark stripes and small eyes',
                'A colorful pink bird has wings with dark stripes and small eyes',
                'A colorful orange bird has wings with dark stripes and small eyes',
                'A colorful brown bird has wings with dark stripes and small eyes',
                'A colorful red bird has wings with dark stripes and small eyes',
                'A colorful yellow bird has wings with dark stripes and small eyes',
                'A colorful purple bird has wings with dark stripes and small eyes']

    '''
    captions = ['the bird has a long black bill that is somewhat curved',
                'the bird has a long black bill that is curved as well',
                'this grey bird has an impressive wingspan with a grey bill and a white stripe that surrounds the feathers near the bill',
                'this large bird is mostly grey with a long hooked bill',
                'bird with long fat beak that is curved at tip and the head is proportional to its body size with whole body covered in black',
                'large bird that is complete brown with white stripes littering its wings and a long blunted bill',
                'a black bird with very long wings and short tail and black beak with some white markings on the face around the beak and eyes',       
                'this bird is all black and has a long with pointy beak',
                'this appears to be a large bird that is almost completely black and it also has a very large black bill with slight white on the face and crown',
                'this bird has long triangular wings and a thick heavy beak']
    '''

    # caption to idx
    # split string to word
    for c, i in enumerate(captions):
        captions[c] = i.split()

    caps = torch.zeros((len(captions), 18), dtype=torch.int64)

    for cl, line in enumerate(captions):
        for cw, word in enumerate(line):
            caps[cl][cw] = wordtoix[word.lower()]
    caps = caps.to(device)
    cap_len = []
    for i in captions:
        cap_len.append(len(i))

    caps_lens = torch.tensor(cap_len, dtype=torch.int64).to(device)

    model_dir = cfg.TRAIN.NET_G
    
    split_dir = 'valid_from_specific_captions'
    state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
    netG.load_state_dict(state_dict)
    print('Load G from: ', model_dir)
    netG.cuda()
    netG.eval()

    batch_size = len(captions)
    s_tmp = model_dir[:model_dir.rfind('.pth')]
    fake_img_save_dir = '%s/%s' % (s_tmp, split_dir) # fake_img_save_dir: models/finetune/cub/netG_550/valid_specific
    mkdir_p(fake_img_save_dir)


    for step in range(50):

        #######################################################
        # (1) Extract text embeddings
        #######################################################

        hidden = text_encoder.init_hidden(batch_size)
        words_embs, sent_emb = text_encoder(caps, caps_lens, hidden)
        words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

        mask = (captions == 0)

        ###############################################################################
        # (2) Generate fake images (and save them inside fake_img_save_dir)
        ###############################################################################
        with torch.no_grad():
            noise = torch.randn(1, 100)
            noise = noise.repeat(batch_size, 1)
            noise = noise.to(device)
            
            fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
            

        
        for j in range(batch_size):
            # save generated image
            s_tmp = '%s/img' % fake_img_save_dir # models/finetune/cub/netG_550/valid/img
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)
            im = fake_imgs[j].data.cpu().numpy()
            # [-1, 1] --> [0, 255]
            im = (im + 1.0) * 127.5
            im = im.astype(np.uint8)
            im = np.transpose(im, (1, 2, 0))
            im = Image.fromarray(im)
            
            
            fullpath = '%s_%d.png' % (s_tmp, step) # models/finetune/cub/netG_550/valid/img_s0.png
            im.save(fullpath)

            # save fusion mask
            s_tmp = '%s/fm' % fake_img_save_dir
            im = stage_mask[j].data.cpu().numpy()
            # im = 1-im # only for better visualization
            # [0, 1] --> [0, 255]
            im = im * 255.0
            im = im.astype(np.uint8)

            im = np.transpose(im, (1, 2, 0))
            im = np.squeeze(im, axis=2)
            im = Image.fromarray(im)
            fullpath = '%s_%d.png' % (s_tmp, step)
            im.save(fullpath)

            for k in range(len(attention_maps)):
                if len(fake_imgs) > 1:
                    im = fake_imgs[k + 1].detach().cpu()
                else:
                    im = fake_imgs[0].detach().cpu()
                attn_maps = attention_maps[k]
                att_sze = attn_maps.size(2)
                img_set, sentences = \
                    build_super_images2(im[j].unsqueeze(0),
                                        captions[j].unsqueeze(0),
                                        [cap_lens_np[j]], ixtoword,
                                        [attn_maps[j]], att_sze)
                if img_set is not None:
                    im = Image.fromarray(img_set)
                    fullpath = '%s_a%d.png' % (save_name, k)
                    im.save(fullpath)


def cap2img(ixtoword, caps, cap_lens):
    imgs = []
    for cap, cap_len in zip(caps, cap_lens):
        idx = cap[:cap_len].cpu().numpy()
        caption = []
        for i, index in enumerate(idx, start=1):
            caption.append(ixtoword[index])
            if i % 4 == 0 and i > 0:
                caption.append("\n")
        caption = " ".join(caption)
        fig = plt.figure(figsize=(2.5, 1.5))
        plt.axis("off")
        plt.text(0.5, 0.5, caption)
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        img = transforms.ToTensor()(img)
        imgs.append(img)
    imgs = torch.stack(imgs, dim=0)
    assert imgs.dim() == 4, "image dimension must be 4D"
    return imgs


def write_images_losses(writer, imgs, fake_imgs, errD, D_total_loss, errG, G_total_loss, epoch):
    index = epoch
    writer.add_scalar('errD/d_loss', errD, index) # errD/D_total_loss
    writer.add_scalar('errD/MAGP', D_total_loss, index) # errD/MAGP
    writer.add_scalar('errG/g_loss', errG, index) # errG/g_loss
    writer.add_scalar('errG/DAMSM', G_total_loss, index) # errG/DAMSM
    imgs_print = imagenet_deprocess_batch(imgs)
    
    
    imgs_256_print = imagenet_deprocess_batch(fake_imgs)
    writer.add_image('images/img1_pred', torchvision.utils.make_grid(imgs_256_print, normalize=True, scale_each=True), index)
    
    writer.add_image('images/img3_real', torchvision.utils.make_grid(imgs_print, normalize=True, scale_each=True), index)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def prepare_labels(batch_size):
    real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
    fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
    match_labels = Variable(torch.LongTensor(range(batch_size)))
    if cfg.CUDA:
        real_labels = real_labels.cuda()
        fake_labels = fake_labels.cuda()
        match_labels = match_labels.cuda()
    return real_labels, fake_labels, match_labels



def train(dataloader, ixtoword, netG, netD, text_encoder, image_encoder,
          optimizerG, optimizerD, state_epoch, batch_size, device):

    base_dir = os.path.join('output_tmp_models', cfg.CONFIG_NAME, str(cfg.TRAIN.NF))


    if not cfg.RESTORE:
        writer = SummaryWriter(os.path.join(base_dir, 'writer'))
    else:
        writer = SummaryWriter(os.path.join(base_dir, 'writer_new'))

    mkdir_p('%s/models' % base_dir)

    real_labels, fake_labels, match_labels = prepare_labels(batch_size)
    real_labels_2, fake_labels_2, match_labels_2 = prepare_labels(batch_size)

    # criteria
    gen_iterations = 0

    mask = mask_correlated_samples(batch_size)

    temperature = 0.5 # contrastive loss hyper parameters 
    print("Contrastive loss temperature", temperature)
    criterion = NT_Xent(batch_size, temperature, mask, device)


    # Build and load the generator and discriminator
    if cfg.RESTORE:
        model_dir = cfg.TRAIN.NET_G
        netG.load_state_dict(torch.load(model_dir))
        model_dir_D = model_dir.replace('netG', 'netD')
        netD.load_state_dict(torch.load(model_dir_D))
        netG.train()
        netD.train()
        istart = cfg.TRAIN.NET_G.rfind('_') + 1
        iend = cfg.TRAIN.NET_G.rfind('.')
        state_epoch = int(cfg.TRAIN.NET_G[istart:iend])


    for epoch in tqdm(range(state_epoch + 1, cfg.TRAIN.MAX_EPOCH + 1)):
       
        start_t = time.time()
        
        
        step = 0

        # initialize D and G loss

        D_total_loss = 0
        G_total_loss = 0

        data_iter = iter(dataloader)

        for step in tqdm(range(len(data_iter))):
            
            ###############################################################################
                 # (1) Prepare training data and extract text embeddings
            ###############################################################################
            
            data = data_iter.next()

            imags, imags_2, captions, cap_lens, class_ids, keys, captions_2, cap_lens_2, class_ids_2, \
                sort_ind, sort_ind_2 = prepare_data(data)

            hidden = text_encoder.init_hidden(batch_size)
        
            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
            mask = (captions == 0)
            num_words = words_embs.size(2)
            if mask.size(1) > num_words:
                    mask = mask[:, :num_words]

            words_embs_2, sent_emb_2 = text_encoder(captions_2, cap_lens_2, hidden)
            words_embs_2, sent_emb_2 = words_embs_2.detach(), sent_emb_2.detach()
            mask_2 = (captions_2 == 0)
            num_words_2 = words_embs_2.size(2)
            if mask_2.size(1) > num_words_2:
                mask_2 = mask_2[:, :num_words_2]


            imgs = imags[0].to(device)
            real_features = netD(imgs)
            output = netD.module.COND_DNET(real_features, sent_emb)
            errD_real = torch.nn.ReLU()(1.0 - output).mean()

            output = netD.module.COND_DNET(real_features[:(batch_size - 1)], sent_emb[1:batch_size])
            errD_mismatch = torch.nn.ReLU()(1.0 + output).mean()


            imgs_2 = imags_2[0].to(device)
            real_features_2 = netD(imgs_2)
            output_2 = netD.module.COND_DNET(real_features_2, sent_emb_2)
            errD_real_2 = torch.nn.ReLU()(1.0 - output_2).mean()

            output_2 = netD.module.COND_DNET(real_features_2[:(batch_size - 1)], sent_emb_2[1:batch_size])
            errD_mismatch_2 = torch.nn.ReLU()(1.0 + output_2).mean()

            errD_real += errD_real_2
            errD_mismatch += errD_mismatch_2
            

            #######################################################
                 # (2) Generate Fake images
            #######################################################

            noise = torch.randn(batch_size, 100)
            noise = noise.to(device)
            fake_imgs, _ = netG(noise, sent_emb)
            fake_imgs_2, _ = netG(noise, sent_emb_2)

           
            #######################################################
                 # (3) Update D netwrok
            #######################################################


            errD_total = 0
            D_logs = ''
            

            fake_features = netD(fake_imgs.detach())
            fake_features_2 = netD(fake_imgs_2.detach())

            errD_fake = netD.module.COND_DNET(fake_features, sent_emb)
            errD_fake = torch.nn.ReLU()(1.0 + errD_fake).mean()

            errD_fake_2 = netD.module.COND_DNET(fake_features_2, sent_emb_2)
            errD_fake_2 = torch.nn.ReLU()(1.0 + errD_fake_2).mean()

            errD_fake += errD_fake_2

            
            # total D loss
            errD = errD_real + (errD_fake + errD_mismatch) / 2.0

            
            # backward and update parameters
            optimizerD.zero_grad()
            errD.backward()
            optimizerD.step()
            errD_total += errD


            # MA-GP: Matching-Aware zero-centered Gradient Penalty for discriminator
            
            interpolated = (imgs.data).requires_grad_()
            sent_inter = (sent_emb.data).requires_grad_()
            features = netD(interpolated)
            out = netD.module.COND_DNET(features, sent_inter)
            grads = torch.autograd.grad(outputs=out,
                                        inputs=(interpolated, sent_inter),
                                        grad_outputs=torch.ones(out.size()).cuda(),
                                        retain_graph=True,
                                        create_graph=True,
                                        only_inputs=True)
            grad0 = grads[0].view(grads[0].size(0), -1)
            grad1 = grads[1].view(grads[1].size(0), -1)
            grad = torch.cat((grad0, grad1), dim=1)
            grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
            d_loss_gp = torch.mean((grad_l2norm) ** 6)
            d_loss = 2.0 * d_loss_gp


            interpolated_2 = (imgs_2.data).requires_grad_()
            sent_inter_2 = (sent_emb_2.data).requires_grad_()
            features_2 = netD(interpolated_2)
            out_2 = netD.module.COND_DNET(features_2, sent_inter_2)
            grads_2 = torch.autograd.grad(outputs=out_2,
                                        inputs=(interpolated_2, sent_inter_2),
                                        grad_outputs=torch.ones(out_2.size()).cuda(),
                                        retain_graph=True,
                                        create_graph=True,
                                        only_inputs=True)
            grad0_2 = grads_2[0].view(grads_2[0].size(0), -1)
            grad1_2 = grads_2[1].view(grads_2[1].size(0), -1)
            grad_2 = torch.cat((grad0_2, grad1_2), dim=1)
            grad_2_l2norm = torch.sqrt(torch.sum(grad_2 ** 2, dim=1))
            d_loss_gp_2 = torch.mean((grad_2_l2norm) ** 6)
            d_loss_2 = 2.0 * d_loss_gp_2

            d_loss += d_loss_2

            optimizerD.zero_grad()
            d_loss.backward()
            optimizerD.step()
                

            #######################################################
                 # (4) Update G network
            ######################################################


            # compute total loss for training G

            step += 1
            gen_iterations += 1
           

            features = netD(fake_imgs)
            features_2 = netD(fake_imgs_2)

            output = netD.module.COND_DNET(features, sent_emb)
            output_2 = netD.module.COND_DNET(features_2, sent_emb_2)

            errG = - output.mean()
            errG_2 = - output_2.mean()

            errG += errG_2


            DAMSM, cnn_code = DAMSM_loss(image_encoder, fake_imgs, real_labels, words_embs,
                                      sent_emb, match_labels, cap_lens, class_ids)

            DAMSM = 0.05 * DAMSM
            

            DAMSM_2, cnn_code_2 = DAMSM_loss(image_encoder, fake_imgs_2, real_labels_2,
                                words_embs_2, sent_emb_2, match_labels_2, cap_lens_2, class_ids_2)

            DAMSM_2 =   0.05 *  DAMSM_2                
            
            DAMSM += DAMSM_2

            _, ori_indices = torch.sort(sort_ind, 0)
            _, ori_indices_2 = torch.sort(sort_ind_2, 0)

            total_contra_loss = 0
            i = -1
            cnn_code = cnn_code[ori_indices]
            cnn_code_2 = cnn_code_2[ori_indices_2]

            cnn_code = l2norm(cnn_code, dim=1)
            cnn_code_2 = l2norm(cnn_code_2, dim=1)

            # adding contrastive loss to generator 

            contrative_loss = criterion(cnn_code, cnn_code_2)
            total_contra_loss += contrative_loss *  0.2

           
            DAMSM += total_contra_loss                         
            

            errG_total = errG + DAMSM

            # backward and update parameters

            optimizerG.zero_grad()
            errG_total.backward()
            optimizerG.step()
            

            D_total_loss += d_loss.item()  
            G_total_loss += errG_total.item()

        end_t = time.time()


        D_total_loss = D_total_loss / step
        G_total_loss = G_total_loss / step

    
        write_images_losses(writer, imgs, fake_imgs, errD, D_total_loss, errG, G_total_loss, epoch)


        # save_model
        if (epoch >= cfg.TRAIN.WARMUP_EPOCHS) and (epoch % cfg.TRAIN.GSAVE_INTERVAL == 0):
            torch.save(netG.state_dict(), '%s/models/netG_%03d.pth' % (base_dir, epoch))
            print('Save G models with saved epoch', str(epoch))
        if (epoch >= cfg.TRAIN.WARMUP_EPOCHS) and (epoch % cfg.TRAIN.DSAVE_INTERVAL == 0):
            torch.save(netD.state_dict(), '%s/models/netD_%03d.pth' % (base_dir, epoch))
            print('Save D models with saved epoch', str(epoch))


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.GPU_ID = args.gpu_id

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = 100
        
    print("seed now is : ", args.manualSeed)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    ##########################################################################
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
    
    cudnn.benchmark = True


    #######################################################
                 # Get data loader
    #######################################################

    imsize = cfg.TREE.BASE_SIZE
    batch_size = cfg.TRAIN.BATCH_SIZE
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    if cfg.B_VALIDATION:
        dataset = TextDataset(cfg.DATA_DIR, 'test',
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform)
        ixtoword = dataset.ixtoword
        wordtoix = dataset.wordtoix
        print(dataset.n_words, dataset.embeddings_num) 
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=True,
            shuffle=True, num_workers=int(cfg.WORKERS))
    else:
        dataset = TextDataset(cfg.DATA_DIR, 'train',
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform)
        ixtoword = dataset.ixtoword
        print(dataset.n_words, dataset.embeddings_num)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=True,
            shuffle=True, num_workers=int(cfg.WORKERS))


    # # validation data #

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device",device)

    netG = NetG(cfg.TRAIN.NF, 100).to(device)
    netD = NetD(cfg.TRAIN.NF).to(device)
    netG = DataParallelWithCallback(netG)
    netD = nn.DataParallel(netD)

 
    #######################################################
                 # Load text encoder
    #######################################################

    text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    state_dict = torch.load(cfg.TEXT.DAMSM_NAME, map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    print('Load text encoder from:', cfg.TEXT.DAMSM_NAME)
    text_encoder.cuda()
    for p in text_encoder.parameters():
        p.requires_grad = False
    text_encoder.eval()


    #######################################################
                 # Load image encoder
    #######################################################

    image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
    img_encoder_path = cfg.TEXT.DAMSM_NAME.replace('text_encoder', 'image_encoder')
    state_dict = \
        torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
    image_encoder.load_state_dict(state_dict)
    print('Load image encoder from:', img_encoder_path)
    image_encoder.cuda()
    for p in image_encoder.parameters():
        p.requires_grad = False
    image_encoder.eval()


    state_epoch = 0

    optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0001, betas=(0.0, 0.9))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0004, betas=(0.0, 0.9))

    start_t = time.time()

    if cfg.B_VALIDATION: # VALIDATION = True
        print("Direct testing the model with sampling")
        sampling(text_encoder, netG, dataloader, ixtoword, device)  # generate images for the whole valid dataset
        #gen_sample(text_encoder, netG, device, wordtoix) # generate images with description from user
    
    else: # VALIDATION = False
        print("training the model")
        train(dataloader, ixtoword, netG, netD, text_encoder, image_encoder, optimizerG, optimizerD, state_epoch, batch_size, device)

    end_t = time.time()
    print('Total time for training:', end_t - start_t)
        
