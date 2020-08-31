import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)), #layer 1
    keras.layers.Dense(128, activation = 'relu'), #layer 2
    keras.layers.Dense(10, activation = 'softmax') #layere 3
])

model.compile(optimizer = 'adam',
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy'])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 1)
print('Test accuracy:', test_acc)


COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def predict(model, image, correct_label):
    prediction = model.predict(np.array([image]))
    predicted_class = class_names[np.argmax(prediction)]
    show_image(image, class_names[correct_label], predicted_class)

def show_image(img, label, guess):
    plt.figure()
    plt.imshow(img, cmap = plt.cm.binary)
    print("expected: " + label)
    print("guess: " + guess)
    plt.colorbar()
    plt.grid(False)
    plt.legend()
    plt.show()

def get_number():
    while True:
        num = input("pick a number : ")
        if num.isdigit():
            num = int(num)
            if 0 <= num <= 1000:
                return int(num)
            else:
                print("try again ")

num = get_number()
image = test_images[num]
label = test_labels[num]
predict(model, image, label)
