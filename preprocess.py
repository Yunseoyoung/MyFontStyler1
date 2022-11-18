import sys, io, os
from glob import glob
import numpy as np
from PIL import ImageFont
import matplotlib.pyplot as plt

from get_data.font2img import draw_example
from get_data.package import pickle_examples
from common.dataset import save_fixed_sample
import pandas as pd
from scipy.misc import imresize

batch_size = 16
IMG_SIZE = 128
EMBEDDING_DIM = 128
FONTS_NUM = 25
EMBEDDING_NUM = 100

save_pngs = 'static/preprocessed_pngs'

def preprocess():


    imgs = glob(os.path.join('static', 'uploads', '*'))
    imgs.sort()

    img1 = plt.imread(imgs[0])
    img2 = plt.imread(imgs[1])


    os.makedirs(save_pngs, exist_ok=True)
    img_crop = img1[132:-182, 132:-132, :]
    plt.figure(figsize=(12, 15))
    idx = 1
    for row in range(13):
        row1 = img_crop[60 + row * 213:213 * (row + 1):, :]
        for col in range(10):
            col1 = row1[5:, 3 + 191 * col:191 * (col + 1), :]
            plt.subplot(13, 10, idx)
            plt.axis('off')
            plt.imshow(col1[:, 5:, :])
            plt.imsave(f'{save_pngs}/{idx}.png', col1[:, 5:, :])
            idx += 1
    plt.tight_layout()
    print(idx)

    idx = 131
    img_crop2 = img2[132:-182, 132:-132, :]
    plt.figure(figsize=(12, 15))
    idx2 = 1
    for row in range(13):
        row1 = img_crop2[60 + row * 213:213 * (row + 1):, :]
        for col in range(10):
            col1 = row1[5:, 3 + 191 * col:191 * (col + 1), :]
            plt.subplot(13, 10, idx2)
            plt.axis('off')
            plt.imshow(col1[:, 5:, :])
            plt.imsave(f'{save_pngs}/{idx}.png', col1[:, 5:, :])
            idx2 += 1
            idx += 1

def link_src_trg():
    preprocess()
    preprocessed_pngs = save_pngs

    label_info = pd.read_csv('dataset/256.txt', sep='\t', header=None)
    label_info = label_info.iloc[::2].reset_index(drop=True)

    SRC_PATH = './get_data/fonts/source/'
    TRG_PATH = './get_data/fonts/handwriting_fonts/'
    os.makedirs(TRG_PATH, exist_ok=True)
    OUTPUT_PATH = './dataset-11172/'

    src_font = glob(os.path.join(SRC_PATH, '*.ttf'))[0]
    print('source font:', src_font)

    trg_fonts = glob(os.path.join('./get_data/fonts/target', '*.ttf'))
    trg_fonts = trg_fonts[46:49]
    trg_fonts.sort()
    print('target fonts:', len(trg_fonts), '개')

    labels = []
    for i in range(len(label_info)):
        labels.append(label_info.iloc[i].values)

    labels = np.concatenate(labels)
    charset = labels
    charset = charset[:-4]

    count = 0
    font_label = 0
    canvas_size = 128
    font_count = 0
    src_char_size = 90
    trg_char_size = 115
    OUTPUT_PATH = './static/handwritings/img_with_srcfont/'
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    src_font = ImageFont.truetype(src_font, size=src_char_size)

    font = trg_fonts[0]
    font = ImageFont.truetype(font, size=trg_char_size)
    character_count = 0
    for c in charset:
        if c == None:
            break
        e = draw_example(c, src_font, font, canvas_size)
        if e:
            e.save(os.path.join(OUTPUT_PATH, "%d_%04d.png" % (font_label, character_count)))
            character_count += 1
            count += 1
            if count % 1000 == 0:
                print("processed %d chars" % count)
    font_label += 1
    print("processed %d chars, end" % count)

    OUTPUT_PATH2 = './static/handwritings/realimg_with_srcfont'
    os.makedirs(OUTPUT_PATH2, exist_ok=True)
    src_imgs = sorted(glob(f'{OUTPUT_PATH}/*.png'),
                      key=lambda x: int(os.path.basename(x)[2:6]))
    tar_imgs = sorted(glob(f'{preprocessed_pngs}/*.png'), key=lambda x: int(os.path.basename(x)[:-4]))
    for srcfile, tarfile in zip(src_imgs, tar_imgs):
        img = plt.imread(srcfile)
        img2 = plt.imread(tarfile)
        basename = os.path.basename(srcfile)
        basename2 = os.path.basename(tarfile)
        assert int(basename[2:6]) + 1 == int(basename2[:-4])
        h, w = img2.shape[:-1]
        img2 = img2[:, (w - h) // 2:-(w - h) // 2]
        img2 = imresize(img2, (128, 128))
        img3 = img.copy()
        img3[:, :128] = img2[:, :, 0] / 255
        plt.imsave(os.path.join(OUTPUT_PATH2, basename), img3, cmap='gray')

    from_dir = OUTPUT_PATH2
    save_dir = OUTPUT_PATH2
    os.makedirs(save_dir, exist_ok=True)
    train_path = os.path.join(save_dir, "train.obj")
    val_path = os.path.join(save_dir, "val.obj")

    pickle_examples(from_dir, train_path=train_path, val_path=val_path,
                    train_val_split=0, with_charid=True)

    sample_size = 24
    img_size = 128
    fontid = 0
    data_dir = save_dir
    save_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)

    save_fixed_sample(sample_size, img_size, data_dir, save_dir, val=False, with_charid=True, resize_fix=90)


