from glob import glob
import os
from skimage.metrics import peak_signal_noise_ratio
from PIL import Image
import numpy as np
import csv

dirs = [os.path.dirname(d) for d in glob("deeplearning/*/GT.png")]

for d in dirs:
    gt = np.array(Image.open(os.path.join(d, "GT.png")))
    p = np.array(Image.open(os.path.join(d, "11000.png")))
    dp = np.array(Image.open(os.path.join(d, "deep_prior.png")))

    with open(os.path.join(d, "psnr.csv"),'w') as out:
        writer = csv.writer(out)
        writer.writerow(['method', 'psnr'])
        writer.writerow(['ours', peak_signal_noise_ratio(gt,p)])
        writer.writerow(['deep-image-prior', peak_signal_noise_ratio(gt,dp)])
