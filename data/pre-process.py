import os
import random

from os.path import isdir
from PIL import Image, ImageFile, ImageEnhance

PATH = './test/' # also ./test/
SIZE = (64, 32)

def save(img, path, ind):
    img.save(path + '/' + str(ind) + '.png')

def process_img(path):
    ind = 1
    files = os.listdir(path)
    for file in files:
        if file == '.DS_Store':
            continue
        # gray
        img = Image.open(path + '/' + file)
        gray = img.convert('L')
        # resize
        resized = gray.resize(SIZE, Image.ANTIALIAS)
        save(resized, path, ind)
        ind += 1
        # random crop
        offset = random.uniform(4, 16)
        cropped = gray.crop((offset, offset, gray.size[0], gray.size[1]))
        cropped = cropped.resize(SIZE, Image.ANTIALIAS)
        save(cropped, path, ind)
        ind += 1
        cropped = gray.crop((offset, 0, gray.size[0], gray.size[1] - offset))
        cropped = cropped.resize(SIZE, Image.ANTIALIAS)
        save(cropped, path, ind)
        ind += 1
        cropped = gray.crop((0, offset, gray.size[0] - offset, gray.size[1]))
        cropped = cropped.resize(SIZE, Image.ANTIALIAS)
        save(cropped, path, ind)
        ind += 1
        cropped = gray.crop((0, 0, gray.size[0] - offset, gray.size[1] - offset))
        cropped = cropped.resize(SIZE, Image.ANTIALIAS)
        save(cropped, path, ind)
        ind += 1
        # rotate
        angle = random.randint(12, 18)
        rotated = gray.rotate(angle, resample=Image.BICUBIC)
        rotated = rotated.resize(SIZE, Image.ANTIALIAS)
        save(rotated, path, ind)
        ind += 1
        angle = random.randint(-18, -12)
        rotated = gray.rotate(angle, resample=Image.BICUBIC)
        rotated = rotated.resize(SIZE, Image.ANTIALIAS)
        save(rotated, path, ind)
        ind += 1
        # horizontal flip
        flipped = gray.transpose(Image.FLIP_LEFT_RIGHT)
        flipped = flipped.resize(SIZE, Image.ANTIALIAS)
        save(flipped, path, ind)
        ind += 1
        # enhance
        enhanced = ImageEnhance.Color(gray).enhance(random.uniform(0.5, 1.5))
        enhanced = ImageEnhance.Brightness(enhanced).enhance(random.uniform(0.5, 1.5))
        enhanced = ImageEnhance.Contrast(enhanced).enhance(random.uniform(0.5, 1.5))
        enhanced = ImageEnhance.Sharpness(enhanced).enhance(random.uniform(0.5, 1.5))
        enhanced = enhanced.resize(SIZE, Image.ANTIALIAS)
        save(enhanced, path, ind)
        ind += 1
        enhanced = ImageEnhance.Color(gray).enhance(random.uniform(0.5, 1.5))
        enhanced = ImageEnhance.Brightness(enhanced).enhance(random.uniform(0.5, 1.5))
        enhanced = ImageEnhance.Contrast(enhanced).enhance(random.uniform(0.5, 1.5))
        enhanced = ImageEnhance.Sharpness(enhanced).enhance(random.uniform(0.5, 1.5))
        enhanced = enhanced.resize(SIZE, Image.ANTIALIAS)
        save(enhanced, path, ind)
        ind += 1

if __name__ == '__main__':
    dirs = os.listdir(PATH)
    for dir in dirs:
        dir_path = PATH + dir
        if isdir(dir_path):
            process_img(dir_path)

