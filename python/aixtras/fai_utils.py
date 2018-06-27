from fastai.dataset import BaseDataset, open_image
import glob


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.tiff','.TIFF'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_img_file_list(globstr):
    image_files = glob.glob(globstr)
    return [img_fn for img_fn in image_files if is_image_file(img_fn)]

def default_image_loader(path):
    #return PIL.Image.open(path).convert('RGB')
    return open_image(path)

class ImageAutoencoderFolder(BaseDataset):
    def __init__(self, file_list, transform=None, loader=default_image_loader):
        self.imgs = file_list
        self.transform = transform
        self.loader = loader
        self.n = len(self.imgs)
        super().__init__(transform)

    def get1item(self, idx):
        path = self.imgs[idx]
        img = self.loader(path)
        x, _ = self.get(self.transform, img, img)
        return x, x

    def get_c(self):
        return None

    def get_n(self):
        return self.n

    def get_sz(self):
        return 0
