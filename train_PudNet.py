from model import *
from dataset import *

import torch
import torch.nn as nn

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from primaryNet_uniAHRU import PrimaryNetwork
import numpy as np
import datetime

from thop import profile

class Train:
    def __init__(self, args):
        self.mode = args.mode
        self.train_continue = args.train_continue

        self.num_subdata = args.num_subdata

        self.scope = args.scope
        self.norm = args.norm

        self.dir_checkpoint = args.dir_checkpoint
        self.dir_log = args.dir_log

        self.name_data = args.name_data
        self.dir_data = args.dir_data
        self.dir_result = args.dir_result

        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size

        self.lr_G = args.lr_G

        self.optim = args.optim
        self.beta1 = args.beta1

        self.ny_in = args.ny_in
        self.nx_in = args.nx_in
        self.nch_in = args.nch_in

        self.ny_load = args.ny_load
        self.nx_load = args.nx_load
        self.nch_load = args.nch_load

        self.ny_out = args.ny_out
        self.nx_out = args.nx_out
        self.nch_out = args.nch_out

        self.sgm = args.sgm
        self.hid_hyper = args.hid_hyper

        self.nch_ker = args.nch_ker

        self.data_type = args.data_type

        self.num_freq_disp = args.num_freq_disp
        # self.num_freq_save = args.num_freq_save

        self.gpu_ids = args.gpu_ids
        self.num_workers = args.num_workers

        if self.gpu_ids and torch.cuda.is_available():
            self.device = torch.device("cuda:%d" % self.gpu_ids[0])
            torch.cuda.set_device(self.gpu_ids[0])
        else:
            self.device = torch.device("cpu")

    def save(self, dir_chck, netG, optimG, num_subdata, hid_hyper, sgm, psnr=True):
        if not os.path.exists(dir_chck):
            os.makedirs(dir_chck)

        if psnr:
            torch.save({'netG': netG.state_dict(),
                    'optimG': optimG.state_dict()},
                   '%s/model_bestVal_psnr_G%s_hyHid%02d_sgm%02d.pth' % (dir_chck,str(num_subdata),hid_hyper,sgm))
        else:
            torch.save({'netG': netG.state_dict(),
                    'optimG': optimG.state_dict()},
                   '%s/model_bestVal_ssim_G%s_hyHid%02d_sgm%02d.pth' % (dir_chck,str(num_subdata),hid_hyper,sgm))

    def load(self, dir_chck, netG, optimG=[], epoch=[], mode='train', psnr=False):

        if not os.path.exists(dir_chck) or not os.listdir(dir_chck):
            epoch = 0
            print(f"********** dir_chck does not exit.")
            if mode == 'train':
                return netG, optimG, epoch
            elif mode == 'test':
                return netG, epoch
        if psnr:
            chk_dir = '%s/model_bestVal_psnr_G%s_hyHid%02d_sgm%02d.pth' % (dir_chck, str(self.num_subdata), self.hid_hyper, self.sgm)
            dict_net = torch.load(chk_dir)
        else:  
            if self.sgm == 25:
                chk_dir = '%s/model_bestVal.pth' % (dir_chck)
            elif self.sgm == 50:
                chk_dir = '%s/model_bestVal_sgm50.pth' % (dir_chck)
            elif self.sgm == 75:
                chk_dir = '%s/model_bestVal_sgm75.pth' % (dir_chck)
            dict_net = torch.load(chk_dir)

        print('Loaded network from chk_dir %s' % chk_dir)

        if mode == 'train':
            netG.load_state_dict(dict_net['netG'])
            optimG.load_state_dict(dict_net['optimG'])

            return netG, optimG, epoch

        elif mode == 'test':
            netG.load_state_dict(dict_net['netG'])

            return netG

    def train(self,args):
        mode = self.mode

        train_continue = self.train_continue
        num_epoch = self.num_epoch

        lr_G = self.lr_G

        batch_size = self.batch_size
        device = self.device

        gpu_ids = self.gpu_ids

        nch_in = self.nch_in
        nch_out = self.nch_out
        nch_ker = self.nch_ker

        size_window = (5, 5)

        norm = self.norm
        name_data = self.name_data

        num_freq_disp = self.num_freq_disp
        
        netG = PrimaryNetwork(args, device) 
        netG.to(device)

        dir_chck = os.path.join(self.dir_checkpoint, self.scope, name_data,str(batch_size))
        dir_log_train = os.path.join(self.dir_log, self.scope, name_data,'train_size'+str(args.nx_in), str(batch_size), 'G'+str(args.num_subdata), 'hyHid'+str(args.hid_hyper),'sgm'+str(args.sgm))
        dir_log_val = os.path.join(self.dir_log, self.scope, name_data, 'val_size'+str(args.nx_in), str(batch_size), 'G'+str(args.num_subdata), 'hyHid'+str(args.hid_hyper),'sgm'+str(args.sgm))

        dir_result_train = os.path.join(self.dir_result, self.scope, name_data, 'train_size'+str(args.nx_in),str(batch_size), 'G'+str(args.num_subdata),'hyHid'+str(args.hid_hyper),'sgm'+str(args.sgm))
        dir_result_val = os.path.join(self.dir_result, self.scope, name_data, 'val_size'+str(args.nx_in),str(batch_size), 'G'+str(args.num_subdata),'hyHid'+str(args.hid_hyper),'sgm'+str(args.sgm))
        if not os.path.exists(os.path.join(dir_result_train, 'images')):
            os.makedirs(os.path.join(dir_result_train, 'images'))
        if not os.path.exists(os.path.join(dir_result_val, 'images')):
            os.makedirs(os.path.join(dir_result_val, 'images'))
        
        transform_inv = transforms.Compose([ToNumpy(), Denormalize()])
        
        val_data = read_Imagenet200_TotalVal_new(resize=args.nx_in)
        dataset_val = Dataset(val_data, sgm=args.sgm, ratio=0.9, size_data=(args.nch_in,args.nx_in,args.ny_in),size_window=size_window)

        loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, sampler=None)

        num_batch_val = len(loader_val)

        if nch_out == 1:
            cmap = 'gray'
        else:
            cmap = None

        fn_REG = nn.L1Loss().to(device)  # Regression loss: L1
        # fn_REG = nn.MSELoss().to(device)     # Regression loss: L2

        paramsG = netG.parameters()

        optimG = torch.optim.Adam(paramsG, lr=lr_G, betas=(self.beta1, 0.999))

        st_epoch = 0
        iter = 0  

        writer_train = SummaryWriter(log_dir=dir_log_train)
        writer_val = SummaryWriter(log_dir=dir_log_val)

        num_iter = self.num_subdata
        best_ssim_val = 0.0
        best_psnr_val = 0.0
        for epoch in range(st_epoch + 1, num_epoch + 1):
            num_i = 0

            while num_i < num_iter:  
                print(f"^^^^^^^^^^^^^^^^^^^^^ {num_i} ^^^^^^^^^^^^^^^^^")
                inputSets = read_traindata_imagenet(args, num_i,resize=args.nx_in)

                dataset_train = CustomDataset(inputSets, sgm=args.sgm, ratio=0.9, size_data=(args.nch_in,args.nx_in,args.ny_in), size_window=size_window)  
                loader_train = torch.utils.data.DataLoader(
                dataset_train, batch_size=batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True, sampler=None)

                netG.train()

                loss_G_train = []
                num_batch_train = len(loader_train)
                psnr_sum_train = 0.
                ssim_sum_train = 0.
                count_train = 0
                for batch, data in enumerate(loader_train, 1):
                    def should(freq):
                        return freq > 0 and (batch % freq == 0 or batch == num_batch_train)

                    label = data['label'].to(device, dtype=torch.float32)  
                    input = data['input'].to(device, dtype=torch.float32) 
                    mask = data['mask'].to(device, dtype=torch.float32)   

                    output = netG(input,label)  

                    optimG.zero_grad()

                    loss_G = fn_REG(output * (1 - mask), label * (1 - mask))

                    loss_G.backward()
                    optimG.step()

                    if 'clean' in data:
                        denoised_image = output  
                        for sample_i in range(denoised_image.shape[0]):
                            psnr_value = psnr(denoised_image[sample_i], data['clean'][sample_i])
                            ssim_value = ssim(denoised_image[sample_i], data['clean'][sample_i])
                            
                            psnr_sum_train += psnr_value
                            ssim_sum_train += ssim_value
                            count_train += 1

                    loss_G_train += [loss_G.item()]

                    iter = iter+1
                    writer_train.add_scalar('loss_G_batch', np.mean(loss_G_train), iter)
                        
                writer_train.add_scalar('loss_G', np.mean(loss_G_train), iter) 
                writer_train.add_scalar('PSNR', psnr_sum_train / count_train, iter)
                writer_train.add_scalar('SSIM', ssim_sum_train / count_train, iter)

                with torch.no_grad():
                    netG.eval()

                    loss_G_val = []

                    psnr_sum = 0.
                    ssim_sum = 0.
                    count = 0
                    for batch, data in enumerate(loader_val, 1):
                        def should(freq):
                            return freq > 0 and (batch % freq == 0 or batch == num_batch_val)

                        input = data['label'].to(device, dtype=torch.float32)  
                        label = data['label'].to(device, dtype=torch.float32)  
                        mask = data['mask'].to(device, dtype=torch.float32)  

                        output = netG(input,label)  

                        loss_G = fn_REG(output * (1 - mask), label * (1 - mask))

                        loss_G_val += [loss_G.item()]

                        if 'clean' in data:
                            clean = data['clean']    
                            denoised_image = output  
                            for sample_i in range(denoised_image.shape[0]):
                                psnr_value = psnr(denoised_image[sample_i], data['clean'][sample_i])
                                ssim_value = ssim(denoised_image[sample_i], data['clean'][sample_i])
                                
                                psnr_sum += psnr_value
                                ssim_sum += ssim_value
                                count += 1

                    writer_val.add_scalar('loss_G', np.mean(loss_G_val), iter) 
                    writer_val.add_scalar('PSNR', psnr_sum / count, iter)
                    writer_val.add_scalar('SSIM', ssim_sum / count, iter)
                    print('------ VALID: EPOCH %d: train_iter %d: BATCH %04d/%04d: PSNR: %.2f, SSIM: %.4f, count: %d'
                            % (epoch, iter, batch, num_batch_val, psnr_sum / count, ssim_sum / count, count))
                    cur_psnr_val = psnr_sum / count
                    cur_ssim_val = ssim_sum / count
                
                # if not np.isinf(cur_psnr_val) and cur_psnr_val > best_psnr_val:  
                #     best_psnr_val = cur_psnr_val
                #     self.save(dir_chck, netG, optimG, args.num_subdata, args.hid_hyper, args.sgm, psnr=True)
                # if cur_ssim_val > best_ssim_val:
                #     best_ssim_val = cur_ssim_val
                #     self.save(dir_chck, netG, optimG, args.num_subdata, args.hid_hyper, args.sgm, psnr=False)
                
                num_i += 1

        writer_train.close()
        writer_val.close()

    def test(self,args):
        size_window = (5, 5)

        if self.nch_out == 1:
            cmap = 'gray'
        else:
            cmap = None

        dir_chck = os.path.join(self.dir_checkpoint, self.scope, 'imagenet','50')  

        dir_result_test = os.path.join(self.dir_result, self.scope, self.name_data, 'test_size'+str(self.nx_in),str(self.batch_size), 'G'+str(self.num_subdata),'hyHid'+str(self.hid_hyper),'sgm'+str(self.sgm))

        if not os.path.exists(os.path.join(dir_result_test, 'images')):
            os.makedirs(os.path.join(dir_result_test, 'images'))

        transform_inv = transforms.Compose([ToNumpy(), Denormalize(self.name_data)])

        netG = PrimaryNetwork(args, self.device)  
        def get_parameter_number(model):
            total_num = sum(p.numel() for p in model.parameters())
            trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"'Total': {total_num}, 'Trainable': {trainable_num}")

        get_parameter_number(netG)
        netG.to(self.device)

        if self.name_data.lower() == 'imagenet':
            test_data = read_Imagenet200_test(resize=self.nx_in)
            dataset_test = Dataset(test_data, sgm=self.sgm, ratio=0.9, size_data=(self.nch_in, self.nx_in, self.ny_in), size_window=size_window)
        elif self.name_data.lower() in  ('animals10','cifar10','dtd','cub200','cifar100'):
            test_data = read_interData(self.name_data,resize=self.nx_in, training=False)

            dataset_test = CustomDataset(test_data, sgm=self.sgm, ratio=0.9, size_data=(self.nch_in, self.nx_in, self.ny_in), size_window=size_window)  
        elif self.name_data.lower() == 'bsds500':
            dir_data_test = '../../datasets/BSR/BSDS500/data/images/val'
            transform_test = transforms.Compose([Normalize(mean=0.5, std=0.5), RandomFlip(), RandomCrop((256, 256)), Rescale((self.ny_in, self.nx_in)), ToTensor()]) 

            dataset_test = CustomDataset(dir_data_test, transform=transform_test, sgm=self.sgm, ratio=0.9, size_data=(self.nch_in, self.nx_in, self.ny_in), size_window=size_window, bsd=True)
            
        loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=self.batch_size, shuffle=False,
        num_workers=self.num_workers, pin_memory=True, sampler=None)

        num_batch_test = len(loader_test)
       
        fn_REG = nn.L1Loss().to(self.device)  # L1  
        # fn_REG = nn.MSELoss().to(device)  # L2

        netG = self.load(dir_chck, netG, mode=self.mode)

        with torch.no_grad():
            
            netG.eval()

            loss_G_test = []

            psnr_sum_test = 0.
            ssim_sum_test = 0.
            count_test = 0
            
            for i, data in enumerate(loader_test, 1):
                start = datetime.datetime.now()
                input = data['label'].to(self.device, dtype=torch.float32)  
                label = data['label'].to(self.device, dtype=torch.float32)  
                mask = data['mask'].to(self.device, dtype=torch.float32)

                output = netG(input, label)  

                if 'clean' in data:
                    clean = data['clean']
                    denoised_image = output
                    for sample_i in range(denoised_image.shape[0]):
                        psnr_value = psnr(denoised_image[sample_i], data['clean'][sample_i])
                        ssim_value = ssim(denoised_image[sample_i], data['clean'][sample_i])

                        if not np.isinf(psnr_value):
                            psnr_sum_test += psnr_value
                            ssim_sum_test += ssim_value
                            count_test += 1

                        print(f"[batch{i}][sample{sample_i}] psnr_value:{psnr_value}")
                        print(f"[batch{i}][sample{sample_i}] ssim_value:{ssim_value}")
                        # exit(0)

                loss_G = fn_REG(output * (1 - mask), label * (1 - mask))

                loss_G_test += [loss_G.item()]

                input = transform_inv(input)
                label = transform_inv(label)
                output = transform_inv(output)
                clean = transform_inv(clean)

                input = np.clip(input, 0, 1)
                label = np.clip(label, 0, 1)
                output = np.clip(output, 0, 1)
                dif = np.clip(abs(label - input), 0, 1)
                clean = np.clip(clean, 0, 1)

                for j in range(label.shape[0]):
                    # name = args.batch_size * (i - 1) + j
                    name = j
                    fileset = {'name': name,
                            #    'input': "%04d-input.png" % name,
                               'output': "%04d-output.png" % name,
                               'label': "%04d-label.png" % name,
                            #    'dif': "%04d-dif.png" % name,
                               'clean': "%04d-clean.png" % name,}

                    # plt.imsave(os.path.join(dir_result_test, 'images', fileset['input']), input[j, :, :, :].squeeze(), cmap=cmap)
                    plt.imsave(os.path.join(dir_result_test, 'images', fileset['output']), output[j, :, :, :].squeeze(), cmap=cmap)
                    plt.imsave(os.path.join(dir_result_test, 'images', fileset['label']), label[j, :, :, :].squeeze(), cmap=cmap)
                    # plt.imsave(os.path.join(dir_result_test, 'images', fileset['dif']), dif[j,:,:,:].squeeze(), cmap=cmap)
                    plt.imsave(os.path.join(dir_result_test, 'images', fileset['clean']), clean[j, :, :, :].squeeze(), cmap=cmap)

                    append_index(dir_result_test, fileset)

                print('TEST: %d/%d: LOSS: %.6f' % (i, num_batch_test, loss_G.item()))
            print('TEST: AVERAGE LOSS: %.6f' % (np.mean(loss_G_test)))
            print('------ TEST: PSNR: %.2f, SSIM: %.4f, count_test: %d'
                          % (psnr_sum_test / count_test, ssim_sum_test / count_test, count_test))
            


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def append_index(dir_result, fileset, step=False):
    index_path = os.path.join(dir_result, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        for key, value in fileset.items():
            index.write("<th>%s</th>" % key)
        index.write('</tr>')

    # for fileset in filesets:
    index.write("<tr>")

    if step:
        index.write("<td>%d</td>" % fileset["step"])
    index.write("<td>%s</td>" % fileset["name"])

    del fileset['name']

    for key, value in fileset.items():
        index.write("<td><img src='images/%s'></td>" % value)

    index.write("</tr>")
    return index_path


def add_plot(output, label, writer, epoch=[], ylabel='Density', xlabel='Radius', namescope=[]):
    fig, ax = plt.subplots()

    ax.plot(output.transpose(1, 0).detach().numpy(), '-')
    ax.plot(label.transpose(1, 0).detach().numpy(), '--')

    ax.set_xlim(0, 400)

    ax.grid(True)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    writer.add_figure(namescope, fig, epoch)


def tensor2np(t:torch.Tensor):
    '''
    transform torch Tensor to numpy having opencv image form.
    RGB -> BGR
    (c,h,w) -> (h,w,c)
    '''
    t = t.cpu().detach()  

    # gray
    if len(t.shape) == 2:
        return t.permute(1,2,0).numpy()
    # RGB -> BGR
    elif len(t.shape) == 3:
        return np.flip(t.permute(1,2,0).numpy(), axis=2) 
    # image batch
    elif len(t.shape) == 4:
        return np.flip(t.permute(0,2,3,1).numpy(), axis=3)
    else:
        raise RuntimeError('wrong tensor dimensions : %s' % (t.shape,))
    
def psnr(img1, img2):
    
    if len(img1.shape) == 4: 
        img1 = img1[0]  
    if len(img2.shape) == 4: 
        img2 = img2[0]

    # tensor to numpy
    if isinstance(img1, torch.Tensor):
        img1 = tensor2np(img1) 
    if isinstance(img2, torch.Tensor):
        img2 = tensor2np(img2)  

    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)

    return peak_signal_noise_ratio(img1, img2, data_range=1)

def ssim(img1, img2):
    
    if len(img1.shape) == 4:
        img1 = img1[0]
    if len(img2.shape) == 4:
        img2 = img2[0]

    # tensor to numpy
    if isinstance(img1, torch.Tensor):
        img1 = tensor2np(img1)
    if isinstance(img2, torch.Tensor):
        img2 = tensor2np(img2)

    img2 = np.clip(img2, 0, 1)
    img1 = np.clip(img1, 0, 1)

    return structural_similarity(img1, img2, multichannel=True, data_range=1)

