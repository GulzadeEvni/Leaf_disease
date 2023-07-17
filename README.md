# Leaf_disease

Patates yapraklarında oluşan Early Blight (Erken Yanıklık) ve Late Blight (Geç Yanıklık) hastalıklarını tespit etmek için oluşturulan bir modeldir.

**Örnek:**

![tahmin](https://github.com/GulzadeEvni/Leaf_disease/assets/111283320/f0a87c4a-532f-4399-aedf-c8f6bbe32f34)



Model, gerekli parametreler ayarlanarak CNN (Convolutional neural network) 'Evrişimli Sinir Ağları' ile eğitilmiştir.

```shell

from keras.api._v2.keras import activations
input_shape=(BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,CHANNELS)
number_classes=3
model=models.Sequential([

    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(32,(3,3),activation='relu',input_shape=input_shape), #convolutional layer
    layers.MaxPooling2D((2,2)),#pooling layer
    layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(number_classes,activation='softmax'),
])
model.build(input_shape=input_shape)
```

Fastapi oluşturmak için gereklilikler
```shell
from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
```

**Fastapi Arayüzü**

![fastapi](https://github.com/GulzadeEvni/Leaf_disease/assets/111283320/1b7e2e94-6a8e-44c9-9bf7-c518d751a065)

**Resim Yükleme**

![fastapi2](https://github.com/GulzadeEvni/Leaf_disease/assets/111283320/69e76c09-ae02-44de-b086-e39f38feacf8)

**Tahmin**

![astapi3](https://github.com/GulzadeEvni/Leaf_disease/assets/111283320/ce5af0c9-0888-4ac2-abbc-f791e7045c02)

**Postman Arayüzü**

![postman](https://github.com/GulzadeEvni/Leaf_disease/assets/111283320/bfff3c6f-5021-4d83-b033-804f04d79d03)
