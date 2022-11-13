# IMPORTS
from tensorflow.keras.models import Sequential # modelo sequencial
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # camadas convulação, pooling, flattening, rede neural densa
from tensorflow.python.keras.layers import BatchNormalization # normalização do mapa de características
from tensorflow.keras.preprocessing.image import ImageDataGenerator # gerar imagens adicionais
import numpy as np
from tensorflow.keras.preprocessing import image

 
# PRÉ PROCESSAMENTO DE IMAGEM 
classificador = Sequential() 
classificador.add(Conv2D(64, (3,3), input_shape = (64, 64, 3), activation = 'relu')) # primeira camada de convolução
# 64 filtros (o recomendado é que seja 64), (3,3) -> dimensão do detector de características
# input_shape -> altura, largura e canais da imagem (3 canais = RGB)
# activation = função de ativação
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2))) # camada de pooling matriz 2x2

classificador.add(Conv2D(64, (3,3), input_shape = (64, 64, 3), activation = 'relu')) # segunda camada de convolução
# 32 filtros (o recomendado é que seja 64), (3,3) -> dimensão do detector de características
# input_shape -> altura, largura e canais da imagem (3 canais = RGB)
# activation = função de ativação
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2))) # camada de pooling matriz 2x2

classificador.add(Flatten()) # flattening --> transforma a matriz em vetor


# REDE NEURAL DENSA
classificador.add(Dense(units = 128, activation = 'relu')) # primeira camada oculta ; units --> qtd de neuronios
classificador.add(Dropout(0.2)) 
classificador.add(Dense(units = 128, activation = 'relu')) # segunda camada oculta
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 1, activation = 'sigmoid')) # camada de saída ; units = 1 --> classificação binária
classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                      metrics = ['accuracy'])
# para mais de duas classes -> loss = 'categorical_crossentropy'

# geração de imagens --> alteração nas imagens para melhorar o treinamento da rn
gerador_treinamento = ImageDataGenerator(rescale = 1./255, # normalização dos dados
                                         rotation_range = 7, # grau de rotação
                                         horizontal_flip = True, # giros horizontais
                                         shear_range = 0.2, # mudanças dos pixels para outra direção
                                         height_shift_range = 0.07, # faixa de mudança da altura
                                         zoom_range = 0.2) # zoom
gerador_teste = ImageDataGenerator(rescale = 1./255) # normalização dos dados 

# base de dados para treinamento 
base_treinamento = gerador_treinamento.flow_from_directory('dataset/training_set',
                                                           target_size = (64, 64),
                                                           batch_size = 32,
                                                           class_mode = 'binary')

# base de dados para teste
base_teste = gerador_teste.flow_from_directory('dataset/test_set',
                                               target_size = (64, 64),
                                               batch_size = 32,
                                               class_mode = 'binary')
                                               # para classificações não binárias é necessário alterar o class_mode

# TREINAMENTO DO CLASSIFICADOR
classificador.fit_generator(base_treinamento, steps_per_epoch = 1732/32, 
                            epochs = 10, validation_data = base_teste, 
                            validation_steps = 742/32)
                            # steps_per_epoch = quantidade de imagens que temos na base de dados de treinamento (base_treinamento)
                            # validation_steps = quantidade de registros (base_teste)
                            
# TESTE DO CLASSIFICADOR - única imagem -> usuário
imagem_teste = image.load_img('dataset/test_set/Fire/imagemTeste4.jpg',
                              target_size = (64,64)) #imagem escolhida e dimensões
imagem_teste = image.img_to_array(imagem_teste)
imagem_teste /= 255 #normalização
imagem_teste = np.expand_dims(imagem_teste, axis = 0) #expandir dimensões // formato para o tensor flow
previsao = classificador.predict(imagem_teste)

# resultado do classificador 
base_treinamento.class_indices # saber qual valor (0 / 1) é cada classe (verde / madura)

    #previsao = (previsao > 0.5) #retorna true or false
if (previsao > 0.5):
    print("Verde")
else:
    print("Madura")


