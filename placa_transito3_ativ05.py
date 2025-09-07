#!/usr/bin/env python3

"""Atividade 5
Grupo:
Augusto Carvalho
Igor Gonçalves
João Eloy
"""

"""Código base
https://colab.research.google.com/drive/1LP3dg48-xWF4W_AXfPJNgLlwzlGfBVYM
"""

"""
Realize o treino de uma CNN com Pytorch/TensorFlow.
Construa a sua rede convolucional para classificar as
imagens de ants e bees; conforme os seus parâmetros.

Verifiquem as especificações abaixo:
  ➢ Não use o mesmo modelo criado em aula.
  ➢ Teste os hiperparâmetros para melhorar o seu resultado.
  ➢ Descreva a estrutura final da sua CNN.
  ➢ Use as células do colab descrevendo como foi processo de
    treino e quais mudanças tiveram mais efeito nos resultados.
  ➢ Comente sobre as dificuldades encontradas e o que pode
    ser feito para melhorar os resultados. Bom trabalho a todos!
"""

import os
import sys
import time
import copy
import shutil
import random

# Os imports necessários
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2
from imageio import imread
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from torchinfo import summary
import urllib
import zipfile
from skimage.transform import resize
from skimage.util import crop

#########################
arquivozipado = 'GTSRB_Final_Training_Images.zip'
# baixando base de dados de placas de transito
print("Baixando base de dados de placas de transito... ", end='', flush=True)
if not os.path.exists(arquivozipado):
  url = ("https://sid.erda.dk/public/archives/"
  + "daaeac0d7ce1152aea9b61d9f1e19370/"
  + "GTSRB_Final_Training_Images.zip")
  filename = "./GTSRB_Final_Training_Images.zip"
  urllib.request.urlretrieve(url, filename)
print("pronto")

print(f"Extraindo os arquivos... ", end='', flush=True)
if not os.path.exists(os.path.splitext(arquivozipado)[0]):
  # Pasta onde os arquivos serão extraídos
  extract_to = os.path.splitext(arquivozipado)[0]+'/'
  # Abre e extrai todos os arquivos
  with zipfile.ZipFile(arquivozipado, 'r') as zip_ref:
    zip_ref.extractall(extract_to)
  print(f"Arquivos extraídos para: {extract_to} ", end='', flush=True)
print("pronto")
print("")

data_transforms = {
  # dados de treinamento
  'train': transforms.Compose([
    transforms.Resize((224,224)), # Pré-processamento para formatar as imagens em tamanho (224x224)
    #transforms.RandomHorizontalFlip(), # Rotaciona a imagem de baixo para cima
    transforms.ToTensor(), # Transforma no tipo de dado do pytorch
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalização contém a média e o desvio padrão para cada um dos canais RGB
  ]),

  # Podemos fazer cortes diferentes, mas a imagem final tem que ser 224x224
  # dados de validação
  'val': transforms.Compose([
    transforms.Resize(256), # Pré-processamento para formatar as imagens em tamanho (256x256)
    transforms.CenterCrop(224), # Corte na imagem para (224x224)
    transforms.ToTensor(),  # Transforma no tipo de dado do pytorch
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalização contém a média e o desvio padrão para cada um dos canais RGB
  ]),
}

# Carrega dataset completo
dataset = datasets.ImageFolder(root='GTSRB_Final_Training_Images/GTSRB/Final_Training/Images', transform=data_transforms['train'])

# Define proporção treino/teste
train_size = int(0.8 * len(dataset))  # 80% treino
test_size = len(dataset) - train_size

# Divide de forma aleatória
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Cria dataloaders
train_loader = DataLoader(train_dataset.dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset.dataset, batch_size=8, shuffle=False)

dataset_sizes = { 'train': train_size, 'val': test_size}
dataloaders = { 'train': train_loader, 'val': test_loader}

print("Dataset Sumary:")
print(f'Dataset sizes: {dataset_sizes}')
print(f'class names: {dataset.classes}')
print("")

print("Ambiente de execução:")
# Utiliza GPU ou CPU, verificar no colab (Ambiente de execução -> Alterar o tipo de ambiente de execução)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Pytorch version: {torch.__version__}")
print(f"Device used: {device}")
print(f"Current GPU device name: {torch.cuda.get_device_name(0)}") # Assuming at least one GPU
print("")

# Funções auxiliares para a classificação das imagens usando CNN

# Função auxiliar para impimir um batch de imagens
def imshow(inp, title=None):
  """Imshow for Tensor."""

  # Reorganizar o dado para o formato adequado do matplotlib
  inp = inp.numpy().transpose((1, 2, 0))
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  inp = std * inp + mean
  inp = np.clip(inp, 0, 1) # limita os dados da imagem entre [0,1]
  # plota a imagem
  plt.imshow(inp)
  if title is not None:
    plt.title(title)
  plt.pause(0.01)  # pause a bit so that plots are updated

def imshow2(inp, title=None):
  """Imshow for Tensor."""

  # Reorganizar o dado para o formato adequado do matplotlib
  inp = inp.numpy().transpose((1, 2, 0))
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  inp = std * inp + mean
  inp = np.clip(inp, 0, 1) # limita os dados da imagem entre [0,1]
  # plota a imagem
  plt.imshow(inp)
  if title is not None:
    plt.title(title)
  plt.pause(0.01)  # pause a bit so that plots are updated
  plt.show()

# Apresenta a dimensão de saída da imagem após a convolução (qual o tamanho do feature map?)
def out_conv2d(dim_input, kernel_size, padding=0, dilation=1, stride=1):
  dim_output = ((dim_input + 2 * padding - dilation * (kernel_size-1) - 1)/stride) + 1
  return dim_output

# Get a batch of training data
#inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
#out = torchvision.utils.make_grid(inputs)

# Plota o batch de imagens (correspondendo a imagem e o rótulo)
#imshow2(out, title=[class_names[x] for x in classes])

# função de treino
def train_model(model, criterion, optimizer, num_epochs=10):

    # Calcula o tempo do treinamento
    since = time.time()

    # Faz uma cópia do modelo, e a medida que vai melhorando o modelo; atualiza a CNN com os pesos ajustados
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # iterar até a quantidade finita de épocas
    for epoch in range(num_epochs):
        print('Epoch \033[0;32m{}\033[0m/{}'.format(epoch +1, num_epochs ))
        print('-' * 12)
        sinceepoch = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #print("fase train...")
                model.train()  # Set model to training mode
            else:
                #print("fase test...")
                model.eval()   # Set model to evaluate mode

            # calular o erro da época
            running_loss = 0.0

            # calcular a acurácia
            running_corrects = 0

            # Iterate over data.
            #for inputs, labels in dataloaders[phase]:
            #print("Iterating over data...")

            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zera o parâmetro do gradiente para não obter os gradientes das épocas passadas
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs) # passa os dados de treino para o modelo
                    _, preds = torch.max(outputs, 1) # o resultado da predição
                    loss = criterion(outputs, labels) # o cálculo do erro

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward() # calcula a retropropagação do erro
                        optimizer.step() # atualiza os pesos da CNN com base na retropropagação do erro

                # statistics
                running_loss += loss.item() * inputs.size(0) # calcula o erro das amostras
                running_corrects += torch.sum(preds == labels.data) # calcula a acurácia das amostras

            epoch_loss = running_loss / dataset_sizes[phase] # calcula o erro geral da época atual
            epoch_acc = running_corrects.double() / dataset_sizes[phase] # calcula a acurácia geral da época atual

            # Média por batch
            avg_loss = epoch_loss
            avg_acc = epoch_acc.item()

            if phase == 'train':
                loss_train.append((epoch+1, avg_loss))
                acc_train.append((epoch+1, avg_acc))
            else:
                loss_test.append((epoch+1, avg_loss))
                acc_test.append((epoch+1, avg_acc))

            #return avg_loss, avg_acc

            # Mostra a fase, o erro, e acurácia da respectiva fase (treinamento ou validação)
            print('{:>6} Loss: {:>6.4f} Acc: \033[1;32m{:>6.4f}\033[0m'.format(
                phase, epoch_loss, epoch_acc))

            # Verifica se a acurácia atual é melhor do que a acurácia que está salva.
            # Se for melhor, então atualiza o modelo e os pesos da CNN
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


        # verifica o tempo de treinamento desta epoca
        time_elapsedepoch = time.time() - sinceepoch
        print('       Epoch training complete in {:.0f}m {:.0f}s'.format(
              time_elapsedepoch // 60, time_elapsedepoch % 60))
        print("")

    # verifica o tempo final do treinamento
    time_elapsed = time.time() - since
    print('Total Training complete in {:.0f}m {:.0f}s'.format(
          time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: \033[0;32m{:4f}\033[0m'.format(best_acc))

    # Ler o melhor modelo da fase de treinamento
    model.load_state_dict(best_model_wts)
    # retorna o melhor modelo de treino
    return model

class_names = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons"
}

# Função para visualizar os dados de validação por batch (número de imagens = 6)
def visualize_model(model, num_images=6):
    # Salva se o modelo estava em treino ou eval
    was_training = model.training
    model.eval()

    fig = plt.figure(figsize=(10,10))
    images_so_far = 0

    # Sorteia índices aleatórios do dataset de validação
    val_dataset = dataloaders['val'].dataset
    indices = random.sample(range(len(val_dataset)), num_images)

    with torch.no_grad():
        for idx in indices:
            # Pega uma imagem e label aleatórios
            img, label = val_dataset[idx]

            # Prepara entrada para o modelo (adiciona dimensão do batch)
            inputs = img.unsqueeze(0).to(device)
            label = torch.tensor(label).to(device)

            # Passa pelo modelo
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)

            # Plota imagem com predição e rótulo real
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis("off")
            ax.set_title(f"pred: {class_names[pred.item()]}\ntrue: {class_names[label.item()]}")
            imshow(img)

            if images_so_far == num_images:
                break

    plt.tight_layout()
    filename='./predict_'+str(epocas)+'epocas'+'.png'
    plt.savefig(filename, dpi=150)
    plt.show()
    # Res

# testando as saidas da camadas conv2d
# Use esta opção para clacular corretamente as dimensões

# Insira a dimensão de entrada (111x111) e o tamanho do filtro (3x3)
# Gera como resultado o tamanho do feature map
print("tamanho do feature map:")
print(out_conv2d(111, 3))
print("")

# Criando uma rede convolucional (topologia da CNN)
class classificador(nn.Module):

  def __init__(self):
    super().__init__()

    # Criação das camadas convolucionais
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3)) # saida 222x222 (no pooling cai pela metade -> 111x111) que vai gerar 64 feature maps. Definimos esses parâmetros para a nossa CNN, 64 feature maps e filtro/kernel 3x3
    self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3)) # saida 109x109 (no pooling cai pela metade -> 54x54) que vai gerar 128 feature maps. O kernel eh definido pela própria CNN, mas podemos acessar os valores do kernel criado pela CNN

    # Criação da função relu
    self.activation = nn.ReLU()

    # Criação do pooling
    self.pool = nn.MaxPool2d(kernel_size = (2,2))

    # Criação do flatten para vetorizar a imagem ao final das camadas totalmente conectadas
    self.flatten = nn.Flatten() # 54*54 * 128 channels

    # Camadas lineares da rede neural (os neurônios totalmente conectados)
    self.linear1 = nn.Linear(in_features=128 * 54*54, out_features=128)
    self.linear2 = nn.Linear(128, 64)
    self.output = nn.Linear(64, 43) # gera para as duas classes

    # Dropout para diminuir overfitting (desativa 20% dos neurônios de uma camada oculta)
    self.dropout = nn.Dropout(p = 0.2)

  # Fluxo da passagem da imagem na rede
  def forward(self, X):

    # X = Imagem de entrada 224x224
    # Ao passar a imagem de entrada para a primeira convolução 3x3 a saída eh 222x222
    # Ao passar o pooling na imagem 222x222, então a imagem cai para 111x111
    X = self.pool(self.activation(self.conv1(X)))

    # X = Feature map 111x111
    # Ao passar o feature map para a segunda convolução 3x3 a saída eh 109x109
    # Ao passar o pooling na imagem 109x109, então a imagem cai para 54x54
    X = self.pool(self.activation(self.conv2(X)))

    # 54x54 um único vetor de features
    X = self.flatten(X)

    # desativa 20% dos neurônios da primeira camada oculta
    X = self.dropout(self.activation(self.linear1(X)))

    # desativa 20% dos neurônios da segunda camada oculta
    X = self.dropout(self.activation(self.linear2(X)))

    # passar o dado pela camada de saída
    X = self.output(X)

    # retorna o vetor de ativação do neurônio (valor ponto flutuante, indica a intensidade de ativação do neurônio)
    return X



# Criando objeto da estrutura da rede (topologia da rede)
print("topologia da rede:")
net = classificador()
print(net)
summary(net)
print("")

# Acesso aos pesos da camada especificada pelo nome (primeira camada convolucional)
print("pesos da primeira camada convolucional:")
#print(net.conv1.weight)
print("")

# Acesso ao bias da camada especificada pelo nome (primeira camada convolucional)
print("bias da primeira camada convolucional:")
#print(net.conv1.bias)
print("")

# Definindo os parâmetros importantes do treinamento

# Função de custo e função de otimização dos parâmetros
criterion = nn.CrossEntropyLoss() # define o critério do erro (função de perda eh entropia)
optimizer = optim.SGD(net.parameters(), lr=0.01) # define a taxa de aprendizado e o otimizador SGD (Stochastic gradient descent)

# Colocar a rede na GPU
net.to(device)

loss_train = []
loss_test = []
acc_train = []
acc_test = []
# treinar o modelo. Apresentar os dados de treino para treinar o modelo
epocas = 5
trained_model = train_model(net, criterion, optimizer, num_epochs=epocas)

# Separando épocas e valores
epochs_loss_train, values_loss_train = zip(*loss_train)
epochs_acc_train, values_acc_train   = zip(*acc_train)
epochs_loss_test, values_loss_test   = zip(*loss_test)
epochs_acc_test, values_acc_test     = zip(*acc_test)

plt.figure(figsize=(12,5))

# --- Gráfico de Loss ---
plt.subplot(1,2,1)
plt.plot(epochs_loss_train, values_loss_train, label="Train Loss", marker='o')
plt.plot(epochs_loss_test, values_loss_test, label="Test Loss", marker='o')
plt.xlabel("Época")
plt.ylabel("Loss")
plt.title("Evolução da Loss")
plt.legend()
plt.grid(True)

# --- Gráfico de Acurácia ---
plt.subplot(1,2,2)
plt.plot(epochs_acc_train, values_acc_train, label="Train Accuracy", marker='o')
plt.plot(epochs_acc_test, values_acc_test, label="Test Accuracy", marker='o')
plt.xlabel("Época")
plt.ylabel("Acurácia")
plt.title("Evolução da Acurácia")
plt.legend()
plt.grid(True)

plt.tight_layout()
filename='./historico_aprendizagem_'+str(epocas)+'epocas'+'.png'
plt.savefig(filename, dpi=150)
plt.show()

# Visualizar a predição de alguns dados de teste (6 imagens)
visualize_model(trained_model)
print("")

print(f"\033[0;31mFim da rede otimizada\033[0m")
sys.exit(0)

"""Redes Neurais convolucionais pré treinadas e Transfer Learning

Existem modelos pré treinados disponíveis no módulo models do pytorch.

Estas redes foram treinadas e validadas em competições e são capzaes de classificar várias coisas: 1000 classes.

Podemos usar estas redes como base para os nossos problemas e retreiná-las com os nossos dados.
"""

# Carregando o modelo pré-treinado VGG16 (CNN) do pytorch
weights=models.VGG16_Weights.DEFAULT
model_ft = models.vgg16(weights=weights)

# Mostra a topologia/estrutura da CNN VGG16
print("topologia da rede VGG16:")
print(model_ft)
summary(model_ft)
print("@@@@@@@@@@@@@@@")

# Acessando a quantidade de neurônios da última camada oculta (4096)
num_ftrs = model_ft.classifier[6].in_features

# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
# Foi removido a camada de 1000 classes e substituído por 2 classes de saída
# E criado uma nova camada (4096, 2) para substituir a camada oculta existente
# Assim, usamos o modelo pré treinado da VGG16 para o nosso problema em questão
model_ft.classifier[6] = nn.Linear(num_ftrs, 2)

# Apresenta a rede para GPU
model_ft = model_ft.to(device)

# Define os critérios
criterion = nn.CrossEntropyLoss()

# Define o otimizador e a taxa de aprendizado
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001)

# treinar a última camada da rede CNN, e apresentar o que a rede extraiu de features do VGG16
epocas = 4
trained_model = train_model(model_ft, criterion, optimizer_ft, num_epochs=epocas)
# Visualiza a predição de alguns dados de teste (6 imagens)
visualize_model(trained_model)

print(f"\033[0;31mFim do programa\033[0m")
sys.exit(0)
