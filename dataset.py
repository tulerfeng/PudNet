import numpy as np
import torch
import skimage
from skimage import transform
import matplotlib.pyplot as plt
import os
import copy
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pickle
import scipy.io as sio

def read_traindata_imagenet(args, num_i, resize=128, isPickle=True):  
    name = args.name_data
    batch_size = args.batch_size
    num_workers = args.num_workers
    
    if name.lower() == 'imagenet' and isPickle:
        if num_i < 10000:
            save_path = "../../dataProcess/imagenet/ImageNet_Train800_groups_pickle/group_{}.pickle".format(num_i)
        elif num_i < 20000:
            save_path = "../../dataProcess/imagenet/ImageNet_Train800_groups_pickle2/group_{}.pickle".format(num_i)
            

        datamat = pickle.load(open(save_path, 'rb'))
        img = datamat['img']  

        img = torch.from_numpy(img)
        
        if img.size(2) != resize:  
            img = transforms.Resize(resize)(img)  

        img_tmp = img.reshape(5,100,img.size(1),img.size(2),img.size(3))
        
        inputSets = img_tmp[:, 0:10, :, :, :]
        inputSets = inputSets.reshape(5 * 10, img.size(1), img.size(2), img.size(3))
        

    return inputSets


def read_Imagenet200_TotalTrain_Scratch_new(resize=128):  
    traindir = os.path.join('../../dataProcess/imagenet/', 'train_200')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),  
            transforms.Resize(resize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))


    selected_classes = train_dataset.classes  
    class_names = train_dataset.classes
    class_indices = np.array(train_dataset.targets)
    selected_indices = np.isin(class_names, selected_classes)

    selected_data_idxs = []
    samples_per_class = 100 
    for class_idx in np.where(selected_indices)[0]:
        indices = [i for i, idx in enumerate(class_indices) if idx == class_idx]  
        selected_data_idxs.extend(indices[:samples_per_class])
        
    subset_C200 = torch.utils.data.Subset(train_dataset, selected_data_idxs)  
    
    subset_C200_labels = [subset_C200[i][1] for i in range(len(subset_C200))]

    return subset_C200

  
def read_Imagenet200_TotalVal_new(resize=128):  
    valdir = '../../dataProcess/imagenet/val_200'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.RandomResizedCrop(224), 
            transforms.Resize(resize),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    selected_classes = val_dataset.classes  
    class_names = val_dataset.classes
    class_indices = np.array(val_dataset.targets)
    selected_indices = np.isin(class_names, selected_classes)

    selected_data_idxs = []
    samples_per_class = 5 
    for class_idx in np.where(selected_indices)[0]:
        indices = [i for i, idx in enumerate(class_indices) if idx == class_idx]
        selected_data_idxs.extend(indices[:samples_per_class])
       
    subset_C200 = torch.utils.data.Subset(val_dataset, selected_data_idxs)
    
    subset_C200_labels = [subset_C200[i][1] for i in range(len(subset_C200))]
    
    return subset_C200

def read_Imagenet200_test(resize=128): 
    valdir = '../../dataProcess/imagenet/val_200'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),  
            transforms.Resize(resize),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    selected_classes = val_dataset.classes  
    class_names = val_dataset.classes
    class_indices = np.array(val_dataset.targets)
    selected_indices = np.isin(class_names, selected_classes)

    selected_data_idxs = []
    samples_per_class = 5  
    samples_start_id = 10
    for class_idx in np.where(selected_indices)[0]:
        indices = [i for i, idx in enumerate(class_indices) if idx == class_idx]
        selected_data_idxs.extend(indices[samples_start_id:(samples_start_id+samples_per_class)])  

    subset_C200 = torch.utils.data.Subset(val_dataset, selected_data_idxs)
    
    subset_C200_labels = [subset_C200[i][1] for i in range(len(subset_C200))]
    
    return subset_C200

def read_interData(name_data, resize=84, training=True):
    name = name_data
    
    if name.lower() == 'cifar10':
        if training:
            savemat_dir = "../../dataProcess/cifar10/train.mat"  
            n_perC = 5000
        else:
            savemat_dir = "../../dataProcess/cifar10/test.mat"  
            n_perC = 1000
    elif name.lower() == 'cifar100':
        if training:
            savemat_dir = "../../dataProcess/cifar100/train.mat" 
            n_perC = 500
        else:
            savemat_dir = "../../dataProcess/cifar100/test.mat"  
            n_perC = 100
    elif name.lower() == 'cub200':
        if training:
            savemat_dir = "../../dataProcess/cub200/train.mat"  
            n_perC = 30
        else:
            savemat_dir = "../../dataProcess/cub200/test.mat"  
            n_perC = 20
    elif name.lower() == 'animals10':
        if training:
            savemat_dir = "../../dataProcess/animals10/train.mat"  
            n_perC = 1286
        else:
            savemat_dir = "../../dataProcess/animals10/test.mat"  
            n_perC = 160
    
    elif name.lower() == 'dtd':
        if training:
            savemat_dir = "../../dataProcess/dtd/train.mat"  
            savemat_dir2 = "../../dataProcess/dtd/val.mat"  
            n_perC = 80  
        else:
            savemat_dir = "../../dataProcess/dtd/test.mat"  
            n_perC = 40

    elif name.lower() == 'fashion':
        if training:
            savemat_dir = "../../../dataProcess/fashion/train.mat"  
            n_perC = 6000
        else:
            savemat_dir = "../../../dataProcess/fashion/test.mat"  
            n_perC = 1000
    elif name.lower() == 'mnist':
        if training:
            savemat_dir = "../../../dataProcess/mnist/train.mat"  
            n_perC = 5000
        else:
            savemat_dir = "../../../dataProcess/mnist/test.mat"  
            n_perC = 890 
    
    datamat = sio.loadmat(savemat_dir)
    img = datamat['img']
    img = torch.from_numpy(img)

    if name.lower() == 'cifar10':
        num_classes = 10
        img_tmp = img.reshape(num_classes,n_perC,img.size(1),img.size(2),img.size(3))
        if training:
            select_num = 1000  
        else:  
            select_num = 100
        img = img_tmp[:,:select_num,:,:,:]  
        img = img.reshape(num_classes*select_num,img.size(2),img.size(3),img.size(4))
    elif name.lower() == 'cifar100':
        num_classes = 100
        img_tmp = img.reshape(num_classes,n_perC,img.size(1),img.size(2),img.size(3))
        if training:
            select_num = 100  
        else:  
            select_num = 10
        img = img_tmp[:,:select_num,:,:,:]  
        img = img.reshape(num_classes*select_num,img.size(2),img.size(3),img.size(4))

    elif name.lower() == 'dtd' and training:
        datamat_tmp1 = sio.loadmat(savemat_dir)
        datamat_tmp2 = sio.loadmat(savemat_dir2)

        img_tmp1 = datamat_tmp1['img']
        img_tmp2 = datamat_tmp2['img']
        img = np.concatenate((img_tmp1, img_tmp2)) 
        img = torch.from_numpy(img)
        
    elif name.lower() == 'cub200':
        num_classes = 200
        img_tmp = img.reshape(num_classes, n_perC, img.size(1), img.size(2), img.size(3))
        if training:
            select_num = 25
        else:
            select_num = 10
        img = img_tmp[:,:select_num,:,:,:]  
        img = img.reshape(num_classes*select_num,img.size(2),img.size(3),img.size(4))

    if img.shape[2] != resize:
        print("**** impose transforms.Resize")
        img = transforms.Resize(resize)(img)
    print(f">>>>>>> img.shape:{img.shape}")
    
    if name.lower() == 'fashion' or name.lower() == 'mnist' : 
        img = img.expand(-1, 3, -1, -1)  

    inputSets = img
    return inputSets


class CustomDataset(torch.utils.data.Dataset):
    """
    dataset of image files of the form 
       stuff<number>_trans.pt
       stuff<number>_density.pt
    """

    def __init__(self, data, transform=None, sgm=25, ratio=0.9, size_data=(3,128, 128), size_window=(5, 5), bsd=False):  
        self.bsd = bsd
        if bsd:
            self.data_dir = data  
            lst_data = os.listdir(self.data_dir)
            lst_data.sort(key=lambda f: (''.join(filter(str.isdigit, f))))
            self.lst_data = lst_data
            num = len(self.lst_data)
            size_data = (321,481,3)
        else:
            self.data = data
            num = len(self.data)
        self.num = num

        self.transform = transform
        self.sgm = sgm

        self.ratio = ratio
        self.size_data = size_data
        self.size_window = size_window
        self.noise = self.sgm / 255.0 * np.random.randn(num, self.size_data[0], self.size_data[1], self.size_data[2])  

    def __getitem__(self, index):
        if self.bsd:
            data = plt.imread(os.path.join(self.data_dir, self.lst_data[index]))
        else:
            data = self.data[index]
        
        if data.dtype == np.uint8:
            data = data / 255.0

        if data.ndim == 2:
            data = np.expand_dims(data, axis=2)

        if data.shape[0] > data.shape[1]:
            data = data.transpose((1, 0, 2))

        clean = data
        label = data + self.noise[index]  
        input, mask = self.generate_mask(copy.deepcopy(label))  

        data = {'label': label, 'input': input, 'mask': mask, 'clean':clean}

        if self.transform:
            data = self.transform(data)  

        return data

    def __len__(self):
        return self.num

    def generate_mask(self, input):

        ratio = self.ratio  
        size_window = self.size_window 
        size_data = self.size_data 
        num_sample = int(size_data[0] * size_data[1] * (1 - ratio))  

        mask = np.ones(size_data) 
        output = input

        for ich in range(size_data[2]):  
            idy_msk = np.random.randint(0, size_data[0], num_sample) 
            idx_msk = np.random.randint(0, size_data[1], num_sample)  

            idy_neigh = np.random.randint(-size_window[0] // 2 + size_window[0] % 2, size_window[0] // 2 + size_window[0] % 2, num_sample)  
            idx_neigh = np.random.randint(-size_window[1] // 2 + size_window[1] % 2, size_window[1] // 2 + size_window[1] % 2, num_sample)  

            idy_msk_neigh = idy_msk + idy_neigh
            idx_msk_neigh = idx_msk + idx_neigh

            idy_msk_neigh = idy_msk_neigh + (idy_msk_neigh < 0) * size_data[0] - (idy_msk_neigh >= size_data[0]) * size_data[0]
            idx_msk_neigh = idx_msk_neigh + (idx_msk_neigh < 0) * size_data[1] - (idx_msk_neigh >= size_data[1]) * size_data[1]

            id_msk = (idy_msk, idx_msk, ich)  
            id_msk_neigh = (idy_msk_neigh, idx_msk_neigh, ich) 

            output[id_msk] = input[id_msk_neigh]  
            mask[id_msk] = 0.0  

        return output, mask

class Dataset(torch.utils.data.Dataset):

    def __init__(self, data, transform=None, sgm=25, ratio=0.9, size_data=(3,128,128), size_window=(5, 5)):  
        self.transform = transform
        self.sgm = sgm

        self.ratio = ratio
        self.size_data = size_data
        self.size_window = size_window

        self.data = data
        self.noise = self.sgm / 255.0 * np.random.randn(len(self.data), self.size_data[0], self.size_data[1], self.size_data[2])  

    def __getitem__(self, index):
        
        data = self.data[index][0]
        

        if data.dtype == np.uint8:
            data = data / 255.0

        if data.ndim == 2:
            data = np.expand_dims(data, axis=2)

        if data.shape[0] > data.shape[1]:
            data = data.transpose((1, 0, 2))

        clean = data
        label = data + self.noise[index]  
        input, mask = self.generate_mask(copy.deepcopy(label))  

        data = {'label': label, 'input': input, 'mask': mask, 'clean':clean}

        return data

    def __len__(self):
        return len(self.data)

    def generate_mask(self, input):

        ratio = self.ratio  
        size_window = self.size_window 
        size_data = self.size_data  
        num_sample = int(size_data[0] * size_data[1] * (1 - ratio))  

        mask = np.ones(size_data)  
        output = input

        for ich in range(size_data[2]):  
            idy_msk = np.random.randint(0, size_data[0], num_sample)  
            idx_msk = np.random.randint(0, size_data[1], num_sample)  

            idy_neigh = np.random.randint(-size_window[0] // 2 + size_window[0] % 2, size_window[0] // 2 + size_window[0] % 2, num_sample)  
            idx_neigh = np.random.randint(-size_window[1] // 2 + size_window[1] % 2, size_window[1] // 2 + size_window[1] % 2, num_sample) 

            idy_msk_neigh = idy_msk + idy_neigh
            idx_msk_neigh = idx_msk + idx_neigh

            idy_msk_neigh = idy_msk_neigh + (idy_msk_neigh < 0) * size_data[0] - (idy_msk_neigh >= size_data[0]) * size_data[0]
            idx_msk_neigh = idx_msk_neigh + (idx_msk_neigh < 0) * size_data[1] - (idx_msk_neigh >= size_data[1]) * size_data[1]

            id_msk = (idy_msk, idx_msk, ich)  
            id_msk_neigh = (idy_msk_neigh, idx_msk_neigh, ich)  

            output[id_msk] = input[id_msk_neigh]  
            mask[id_msk] = 0.0  

        return output, mask

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):

        input, label, mask, clean = data['input'], data['label'], data['mask'], data['clean']

        input = input.transpose((2, 0, 1)).astype(np.float32)
        label = label.transpose((2, 0, 1)).astype(np.float32)
        mask = mask.transpose((2, 0, 1)).astype(np.float32)
        clean = clean.transpose((2, 0, 1)).astype(np.float32)
        return {'input': torch.from_numpy(input), 'label': torch.from_numpy(label), 'mask': torch.from_numpy(mask), 'clean': torch.from_numpy(clean)}


class Normalize(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        input, label, mask, clean = data['input'], data['label'], data['mask'], data['clean']

        input = (input - self.mean) / self.std
        label = (label - self.mean) / self.std
        clean = (clean - self.mean) / self.std

        data = {'input': input, 'label': label, 'mask': mask, 'clean': clean}
        return data


class RandomFlip(object):
    def __call__(self, data):
    
        input, label, mask, clean = data['input'], data['label'], data['mask'], data['clean']

        if np.random.rand() > 0.5:
            input = np.fliplr(input)
            label = np.fliplr(label)
            mask = np.fliplr(mask)
            clean = np.fliplr(clean)

        if np.random.rand() > 0.5:
            input = np.flipud(input)
            label = np.flipud(label)
            mask = np.flipud(mask)
            clean = np.flipud(clean)

        return {'input': input, 'label': label, 'mask': mask, 'clean': clean}


class Rescale(object):

  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    self.output_size = output_size

  def __call__(self, data):
    input, label, mask, clean = data['input'], data['label'], data['mask'], data['clean']

    h, w = input.shape[:2]

    if isinstance(self.output_size, int):
      if h > w:
        new_h, new_w = self.output_size * h / w, self.output_size
      else:
        new_h, new_w = self.output_size, self.output_size * w / h
    else:
      new_h, new_w = self.output_size

    new_h, new_w = int(new_h), int(new_w)

    input = transform.resize(input, (new_h, new_w))
    label = transform.resize(label, (new_h, new_w))
    mask = transform.resize(mask, (new_h, new_w))
    clean = transform.resize(clean, (new_h, new_w))

    return {'input': input, 'label': label, 'mask': mask, 'clean': clean}


class RandomCrop(object):

  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    if isinstance(output_size, int):
      self.output_size = (output_size, output_size)
    else:
      assert len(output_size) == 2
      self.output_size = output_size

  def __call__(self, data):
    input, label, mask, clean = data['input'], data['label'], data['mask'], data['clean']

    h, w = input.shape[:2]
    new_h, new_w = self.output_size

    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    id_y = np.arange(top, top + new_h, 1)[:, np.newaxis].astype(np.int32)
    id_x = np.arange(left, left + new_w, 1).astype(np.int32)

    # input = input[top: top + new_h, left: left + new_w]
    # label = label[top: top + new_h, left: left + new_w]

    input = input[id_y, id_x]
    label = label[id_y, id_x]
    mask = mask[id_y, id_x]
    clean = clean[id_y, id_x]

    return {'input': input, 'label': label, 'mask': mask, 'clean': clean}


class UnifromSample(object):

  def __init__(self, stride):
    assert isinstance(stride, (int, tuple))
    if isinstance(stride, int):
      self.stride = (stride, stride)
    else:
      assert len(stride) == 2
      self.stride = stride

  def __call__(self, data):
    input, label, mask = data['input'], data['label'], data['mask']

    h, w = input.shape[:2]
    stride_h, stride_w = self.stride
    new_h = h//stride_h
    new_w = w//stride_w

    top = np.random.randint(0, stride_h + (h - new_h * stride_h))
    left = np.random.randint(0, stride_w + (w - new_w * stride_w))

    id_h = np.arange(top, h, stride_h)[:, np.newaxis]
    id_w = np.arange(left, w, stride_w)

    input = input[id_h, id_w]
    label = label[id_h, id_w]
    mask = mask[id_h, id_w]

    return {'input': input, 'label': label, 'mask': mask}


class ZeroPad(object):

  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    self.output_size = output_size

  def __call__(self, data):
    input, label, mask = data['input'], data['label'], data['mask']

    h, w = input.shape[:2]

    if isinstance(self.output_size, int):
      if h > w:
        new_h, new_w = self.output_size * h / w, self.output_size
      else:
        new_h, new_w = self.output_size, self.output_size * w / h
    else:
      new_h, new_w = self.output_size

    new_h, new_w = int(new_h), int(new_w)

    l = (new_w - w)//2
    r = (new_w - w) - l

    u = (new_h - h)//2
    b = (new_h - h) - u

    input = np.pad(input, pad_width=((u, b), (l, r), (0, 0)))
    label = np.pad(label, pad_width=((u, b), (l, r), (0, 0)))
    mask = np.pad(mask, pad_width=((u, b), (l, r), (0, 0)))

    return {'input': input, 'label': label, 'mask': mask}

class ToNumpy(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):

        return data.to('cpu').detach().numpy().transpose(0, 2, 3, 1)

class Denormalize(object):
    def __init__(self,name_data='imagenet'):
        if name_data.lower() == 'imagenet':
            self.mean = np.array([0.485, 0.456, 0.406])
            self.std = np.array([0.229, 0.224, 0.225])
        elif name_data.lower() == 'animals10':
            self.mean = np.array([0.4, 0.4, 0.4])
            self.std = np.array([0.2, 0.2, 0.2])
        elif name_data.lower() == 'cifar10':
            self.mean = np.array([0.4914, 0.4822, 0.4465])
            self.std = np.array([0.2023, 0.1994, 0.2010])
        elif name_data.lower() == 'dtd':
            self.mean = 0
            self.std = 1
        elif name_data.lower() == 'cub200':
            self.mean = np.array([0.48560741861744905, 0.49941626449353244, 0.43237713785804116])
            self.std = np.array([0.2321024260764962, 0.22770540015765814, 0.2665100547329813])
        elif name_data.lower() == 'cifar100':
            self.mean = np.array([0.5070751592371323, 0.48654887331495095, 0.4409178433670343])
            self.std = np.array([0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        elif name_data.lower() == 'bsds500':
            self.mean = 0.5
            self.std = 0.5

          
    def __call__(self, data):  
        data = self.std * data + self.mean
        return data
