
from torch.utils.data import Dataset
import glob, random, os
from PIL import Image


class TrainTinyImageNetDataset(Dataset):
    def __init__(self, transform, root='./data/tiny-imagenet-200',
                 sd_aug_image_pth='/DATA/nakul/aml/SS-GEAR/esrgan/aug_images', **kwargs):
        self.filenames = glob.glob(f"{root}/train/*/*/*.JPEG")

        file_paths = {x.split('/')[-1]: x for x in self.filenames}
        sr_filenames = set([x.replace('_out', '') for x in glob.glob('/DATA/nakul/aml/SS-GEAR/esrgan/results/*.JPEG')])
        sd_images = glob.glob(f'{sd_aug_image_pth}/*.JPEG')

        self.sd_images_pth = {}

        for i, im in enumerate(sd_images):
            # breakpoint()
            nme = ''.join(im.split('/')[-1].split('_')[:2]) + '.JPEG'
            if nme not in self.sd_images_pth:
                self.sd_images_pth[nme] = [im]
            else:
                self.sd_images_pth[nme].append(im)
        
        # print(self.sd_images_pth)

        self.filenames = [file_paths[x.split('/')[-1]] for x in sr_filenames if ('test' not in x and 'val' not in x)]
        self.transform = transform
        
        self.id_dict = {}

        for i, line in enumerate(open(f'{root}/wnids.txt', 'r')):
            self.id_dict[line.replace('\n', '')] = i

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        nme = img_path.split('/')[-1]
        img = Image.open(os.path.join('/DATA/nakul/aml/SS-GEAR/esrgan/results/', nme.replace('.JPEG', '_out.JPEG'))).convert('RGB')
        # print(img_path)
        label = self.id_dict[img_path.split('/')[4]]

        # pick a random variation from self.sd_images_pth[img]
        try:
            aug_im = random.choice(self.sd_images_pth[nme])
            img2 = Image.open(aug_im).convert('RGB')
        except Exception as e:
            # print(e)
            img2 = img

        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.transform(img2)

        return (img1, img2), label

class TestTinyImageNetDataset(Dataset):
    def __init__(self, transform, root='./data/tiny-imagenet-200', **kwargs):
        self.filenames = glob.glob(f"{root}/val/images/*.JPEG")
        self.id_dict = {}

        self.transform = transform
        

        for i, line in enumerate(open(f'{root}/wnids.txt', 'r')):
            self.id_dict[line.replace('\n', '')] = i

        self.cls_dic = {}
        for i, line in enumerate(open(f'{root}/val/val_annotations.txt', 'r')):
            a = line.split('\t')
            img, cls_id = a[0],a[1]
            self.cls_dic[img] = self.id_dict[cls_id]
 

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        img = Image.open(img_path).convert('RGB')
        

        label = self.cls_dic[img_path.split('/')[-1]]

        if self.transform:
            img1 = self.transform(img)
        return img1, label

# transform = transforms.Normalize((122.4786, 114.2755, 101.3963), (70.4924, 68.5679, 71.8127))

# trainset = TrainTinyImageNetDataset(id=id_dict, transform = transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

# testset = TestTinyImageNetDataset(id=id_dict, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)