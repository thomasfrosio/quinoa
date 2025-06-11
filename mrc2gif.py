import numpy as np
import mrcfile
from PIL import Image

stack = mrcfile.read('/Users/cix56657/Projects/synthetic_tomogram/tilt_series_aligned_masked.mrc')

imgs = []
for i in range(19, 19+22):
    imgs.append(Image.fromarray(stack[i,...]))
for i in range(40, -1, -1):
    imgs.append(Image.fromarray(stack[i,...]))
for i in range(0, 19):
    imgs.append(Image.fromarray(stack[i,...]))

imgs = [img.resize((256, 256), Image.LANCZOS) for img in imgs]
imgs[0].save("/Users/cix56657/Projects/synthetic_tomogram/tilt_series_aligned_masked.gif",
             save_all=True, append_images=imgs[1:], duration=150, loop=0)

