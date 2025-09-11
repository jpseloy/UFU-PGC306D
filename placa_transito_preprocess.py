#!/usr/bin/env python3

"""Atividade 3
Grupo
Augusto Carvalho
Igor Gonçalves
João Eloy
"""

# Importação das bibliotecas
import os
import sys
import shutil
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib
import zipfile
from skimage.transform import resize
from skimage.util import crop
from imageio import imread
from PIL import Image

print(f"Pytorch version: {torch.__version__}")

def imshow(image, ax=None, title=None, normalize=True):
  """Imshow for Tensor."""
  if ax is None:
    fig, ax = plt.subplots()
  image = image.numpy()

  if normalize:
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    image = std * image + mean
    image = np.clip(image, 0, 1)

  ax.imshow(image)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['left'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.tick_params(axis='both', length=0)
  ax.set_xticklabels('')
  ax.set_yticklabels('')

  return ax

# Especificando o transformador das imagens para o formato específico do pytorch (média = 0.5 e desvio padrão 0.5)
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# baixando base de dados de placas de transito
print("baixando base de dados de placas de transito... ", end='', flush=True)
if not os.path.exists("GTSRB_Final_Training_Images.zip"):
  url = ("https://sid.erda.dk/public/archives/"
  + "daaeac0d7ce1152aea9b61d9f1e19370/"
  + "GTSRB_Final_Training_Images.zip")
  filename = "./GTSRB_Final_Training_Images.zip"
  urllib.request.urlretrieve(url, filename)
print("pronto")

IMG_SIZE = 28
TEST_SIZE = 0.2

# lê o arquivo e separa os datasets de treinamento e teste
print("lê o arquivo e separa os datasets de treinamento e teste... ", end='', flush=True)

X, Xt, y, yt = list(), list(), list(), list()

archive = zipfile.ZipFile(
                          'GTSRB_Final_Training_Images.zip', 'r')
file_paths = [file for file in archive.namelist()
              if '.ppm' in file]

file_class_describe = [file for file in archive.namelist()
              if '.csv' in file]

pasta = './GTSRB_novo/Final_Training/Images'
os.makedirs(pasta, exist_ok=True)

all_infos = pd.DataFrame()
for class_info in file_class_describe:
  with archive.open(class_info) as class_file:
    subpasta = class_file.name.split('/')[-2]
    caminho = pasta+'/'+subpasta+'/'
    os.makedirs(caminho, exist_ok=True)
    shutil.copy(class_file.name, caminho)

    df = pd.read_csv(class_file, sep=';', engine='python')
    all_infos = pd.concat([all_infos, df])

for filename in file_paths:

  with archive.open(filename) as img_file:
    img = Image.open(img_file).convert('RGB')
    #img = Image.open(img_file)

  img = np.array(img, dtype=np.uint8)
  #print(img.shape)

  img_class = int(filename.split('/')[-2])
  filtered_multiple_conditions = all_infos[(all_infos['ClassId'] == img_class) & (all_infos['Filename'] == filename.split('/')[-1])]
  #print(filtered_multiple_conditions)
  RoiX1 = int(filtered_multiple_conditions['Roi.X1'].iloc[0])
  RoiY1 = int(filtered_multiple_conditions['Roi.Y1'].iloc[0])
  RoiX2 = int(filtered_multiple_conditions['Roi.X2'].iloc[0])
  RoiY2 = int(filtered_multiple_conditions['Roi.Y2'].iloc[0])
  #print (f"Region of interess: x1={RoiX1} y1={RoiY1} x2={RoiX2} y2={RoiY2}")

  # Crop a region: [row_start:row_end, col_start:col_end]
  cropped_image = img[RoiY1:RoiY2, RoiX1:RoiX2]
  #print(cropped_image.shape)

  img2 = resize(cropped_image,
               output_shape=(IMG_SIZE, IMG_SIZE),
               mode='reflect', anti_aliasing=True)
  img2 = (img2 * 255).astype(np.uint8)
  #print(img2.shape)

  subpasta = str(img_class).zfill(5)
  caminho = pasta+'/'+subpasta+'/'
  #print(caminho)
  os.makedirs(caminho, exist_ok=True)
  new_img_file = caminho+filename.split('/')[-1]
  #print(new_img_file)

  bgr_image = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
  cv2.imwrite(new_img_file, bgr_image)

archive.close()
print("pronto")

sys.exit(0)
