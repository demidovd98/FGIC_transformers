#!/usr/bin/env python
# coding: utf-8

# ### In this part, we will finetune the ViT model to Stanford Dogs Dataset

# In[3]:


import PIL
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from torchvision import transforms
from models.models_to_finetune import deit_small_patch16_224
from datasets import CUBDataset, DOGDataset
import math
from matplotlib import pyplot as plt
import torchvision
from torchvision.datasets.vision import VisionDataset
import sys
import os
from PIL import Image
import os.path
import scipy

prev = 0

def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
    images = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


class DatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None):
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        global prev
        class_to_idx = {classes[i]: i+prev for i in range(len(classes))}
        print(prev)
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples


class CUBDataset(ImageFolder):
    """
    Dataset class for CUB Dataset
    """

    def __init__(self, image_root_path, caption_root_path=None, split="train", *args, **kwargs):
        """
        Args:
            image_root_path:      path to dir containing images and lists folders
            caption_root_path:    path to dir containing captions
            split:          train / test
            *args:
            **kwargs:
        """
        image_info = self.get_file_content(f"{image_root_path}/images.txt")
        self.image_id_to_name = {y[0]: y[1] for y in [x.strip().split(" ") for x in image_info]}
        split_info = self.get_file_content(f"{image_root_path}/train_test_split.txt")
        self.split_info = {self.image_id_to_name[y[0]]: y[1] for y in [x.strip().split(" ") for x in split_info]}
        self.split = "1" if split == "train" else "0"
        self.caption_root_path = caption_root_path

        super(CUBDataset, self).__init__(root=f"{image_root_path}/images", is_valid_file=self.is_valid_file,
                                         *args, **kwargs)

    def is_valid_file(self, x):
        return self.split_info[(x[len(self.root) + 1:])] == self.split

    @staticmethod
    def get_file_content(file_path):
        with open(file_path) as fo:
            content = fo.readlines()
        return content

class DOGDataset(ImageFolder):
    """
    Dataset class for CUB Dataset
    """

 

    def __init__(self, image_root_path, caption_root_path=None, split="train", *args, **kwargs):
        """
        Args:
            image_root_path:      path to dir containing images and lists folders
            caption_root_path:    path to dir containing captions
            split:          train / test
            *args:
            **kwargs:
        """
        image_info = self.get_file_content(f"{image_root_path}splits/file_list.mat")
        image_files = [o[0][0] for o in image_info]
        
        split_info = self.get_file_content(f"{image_root_path}/splits/{split}_list.mat")
        split_files = [o[0][0] for o in split_info]
        self.split_info = {}
        if split == 'train' :
            for image in image_files:
                if image in split_files:
                    self.split_info[image] = "1"
                else:
                    self.split_info[image] = "0"
        elif split== 'test' :
            for image in image_files:
                if image in split_files:
                    self.split_info[image] = "0"
                else:
                    self.split_info[image] = "1"
                    
        self.split = "1" if split == "train" else "0"
        self.caption_root_path = caption_root_path

 

        super(DOGDataset, self).__init__(root=f"{image_root_path}Images", is_valid_file = self.is_valid_file,
                                         *args, **kwargs)

 

    def is_valid_file(self, x):
        return self.split_info[(x[len(self.root) + 1:])] == self.split

 

    @staticmethod
    def get_file_content(file_path):
        content =  scipy.io.loadmat(file_path)
        return content['file_list']


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


data_root = "./CUB_200_2011"

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


# write data transform here as per the requirement
data_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

train_dataset_cub = CUBDataset(image_root_path=f"{data_root}", transform=data_transform, split="train")
test_dataset_cub = CUBDataset(image_root_path=f"{data_root}", transform=data_transform, split="test")


# In[6]:


# Set train and test set
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

data_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

prev=200

data_root = "./dog/dog/"

train_dataset_dog = DOGDataset(image_root_path=f"{data_root}", transform=data_transform, split="train")
test_dataset_dog = DOGDataset(image_root_path=f"{data_root}", transform=data_transform, split="test")
print('Number of train samples:', len(train_dataset_dog))
print('Number of test samples:', len(test_dataset_dog))


train_loader = torch.utils.data.DataLoader(
             torch.utils.data.ConcatDataset([train_dataset_cub, train_dataset_dog]),
             batch_size=256, shuffle=True,
             num_workers=1, pin_memory=True)

test_dataset = torch.utils.data.ConcatDataset([test_dataset_cub, test_dataset_dog])

test_loader = torch.utils.data.DataLoader(
             torch.utils.data.ConcatDataset([test_dataset_cub, test_dataset_dog]),
             batch_size=256, shuffle=True,
             num_workers=1, pin_memory=True)


# ## Prepare ViT for transfer learning

# In[10]:


# like we discussed in part1, we will use only the last class token (produced by the last block) for transfer learning
model = deit_small_patch16_224(pretrained=True, use_top_n_heads=6,use_patch_outputs=False).cuda()
# freeze backbone and add linear classifier on top that is for 120 classes
for name,param in model.named_parameters():
    param.requires_grad = False
model.head = torch.nn.Linear(in_features=model.head.in_features, out_features=320) # dogs dataset has 120 classes


# In[12]:


model.head.apply(model._init_weights)
for param in model.head.parameters():
    param.requires_grad = True

model = model.to(device)


# In[13]:


criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))


# ## Training

# In[14]:

losses = []
epochs = 50

print('Training....')
for epoch in range(epochs):
    l=0
    count = 0
    with tqdm(train_loader) as p_bar:
        for samples, targets in p_bar:
            
            samples = samples.to(device)
            targets = targets.to(device)
            
            outputs = model(samples, fine_tune=True)
            loss = criterion(outputs, targets)

            loss_value = loss.item()
            l=l+loss_value

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count = count + 1

        losses.append(l/count)

plt.plot(losses)
plt.savefig('Deit-combined_6heads_centercrop.png')
# ## Testing

# In[15]:


print('Testing....')
acc=0
with tqdm(test_loader) as p_bar:
    for samples, targets in p_bar:
        samples = samples.to(device)
        targets = targets.to(device)
        
        outputs = model(samples, fine_tune=True)
        acc+=torch.sum(outputs.argmax(dim=-1) == targets).item()

print('Accuracy:{0:.3%}'.format(acc/len(test_dataset)))
torch.save(model.state_dict(),'Deit-combined_6heads_centercrop.ckpt')


# In[ ]:


# 20 epochs 6 use_top_n_heads lr=0.01 - 76.8%
# 20 epochs 4 use_top_n_heads lr=0.01 - 76.71%
# 40 epochs 6 use_top_n_heads lr=0.01 - 77.26%

# 20 epochs 4 heads lr =0.01 cutmix - 65.14%
# 20 epochs 6 heads lr =0.01 cutmix -
# lr = 0.001