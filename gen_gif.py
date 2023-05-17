import imageio as imio
from PIL import Image
import os


imgs = []
pos_encoding = "sin_cos"# 'raw'


sorted_file_names = sorted(os.listdir(f'outputs/{pos_encoding}'))
for path in sorted_file_names:
    # read image with PIl
    im = Image.open(f'outputs/{pos_encoding}/{path}')
    imgs.append(im)


animation_filename = f'{pos_encoding}_animation.gif'
imio.mimsave(animation_filename, imgs, duration=120)