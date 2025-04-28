import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.stats import wasserstein_distance
from pytorch_fid import fid_score
import torchvision
from torchvision import transforms
from pathlib import Path
import glob
from PIL import Image

IMAGE_EXTENSIONS = ['.bmp', '.jpg', '.jpeg', '.pgm', '.png', '.ppm', '.tif',
                    '.tiff', '.webp', ".svs"]


def check_path(path: str):
    """
    Checks if a path exists.

    :param path: Path to be checked.
    """
    if not Path(path).is_dir():
        raise NotADirectoryError(f"Path {path} doesn't exist")
    
def get_img_files(path: str):
    return sorted([str(file) for file in Path(path).glob('*') if
                   file.suffix in IMAGE_EXTENSIONS])
    
def SSIM(path1, path2) -> np.float64:
    """
    Calcualte SSIM between two images. Convert to grayscale before.

    :param img1: Image 1 in RGB color space
    :param img2: Image 2 in RGB color space
    :return: Computed SSIM
    """

    ssims = []

    for img2 in sorted(glob.glob(path2 + "/*.*")):
        name = img2.split("/")[-1]
        img2 = cv2.imread(img2)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        img1 = cv2.imread(path1 + name)
        # img1 = cv2.imread(path1 + name)
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        result = ssim(img1, img2, channel_axis=False)
        ssims.append(result)
    return ssims

def av_wasserstein(hist1: np.ndarray, hist2: np.ndarray) -> np.float64:
    return np.mean([wasserstein_distance(hist1[i], hist2[i]) for i in range(3)])

def get_hist(img: np.ndarray) -> np.ndarray:
    """
    Get the histogram of an image.

    :param img: image whose histogram is to be computed.
    :return: image histogram.
    """
    hist = [np.histogram(img[..., j].flatten(), bins=256, range=[0, 256],
                         density=True)[0] for j in range(3)]

    return np.array(hist)

def WD(path1: str, path2: str):
    """
    Computes WD for unpaired ditributions.
    The average histogram for each distribution is first calculated and then WD
    from these two histograms is computed.

    :param path1: Directory containing first distribution images.
    :param path2: Directory containing second distribution images.
    """

    path1, path2 = Path(path1), Path(path2)
    av_hists = []
    for p in [path1, path2]:
        check_path(p)
        files = get_img_files(p)
        hist = np.zeros((3, 256))
        for f in files:
            img = cv2.imread(f, 1)
            hist += get_hist(img)
        av_hists.append(hist / len(files))

    return av_wasserstein(av_hists[0], av_hists[1])

def FID(path1, path2):
    """Calculates the FID of two paths.

    :param path: Path containing images.
    :return: FID scores.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


    fid_result = fid_score.calculate_fid_given_paths([path1, path2], batch_size=2, device="cuda", dims=2048)


    return fid_result

if __name__ == "__main__":
    root_Camelyon = '../Camleyon/'
    Camelyon_src = root_Camelyon + "trainA/"
    Camelyon_dst = root_Camelyon + "trainB/"

    extra_mcn_cyto = '../../git/multistain_cyclegan_normalization/outputs/Cyto/'
    extra_mcn_camelyon = '../../git/multistain_cyclegan_normalization/outputs/Camelyon/'

    extra_methods = [extra_mcn_camelyon]

    for method in extra_methods:
        result_ssim_src = SSIM(Camelyon_src, method)
        result_ssim_src = np.mean(result_ssim_src)

        result_fid = FID(Camelyon_dst, method)
        result_wd = WD(Camelyon_dst, method)
        name = method.split("/")[-2]
        print(f"{name}:\t" + str(result_fid) + "\t" + str(result_wd)+"\t" + str(result_ssim_src) + "\t") 
    