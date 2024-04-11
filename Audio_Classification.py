# %%
import numpy as np
import librosa.display, os
import matplotlib.pyplot as plt
import warnings
from tensorflow.keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from gtts import gTTS
import os




warnings.filterwarnings('ignore')


# %matplotlib inline

def create_spectrogram(audio_file, image_file):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    y, sr = librosa.load(audio_file)
    ms = librosa.feature.melspectrogram(y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr)

    fig.savefig(image_file)
    plt.close(fig)

# Function to preprocess spectrogram image
def preprocess_spectrogram(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def text_to_speech(text, filename):
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    os.system(f'start {filename}')

def create_pngs_from_wavs(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    dir = os.listdir(input_path)

    for i, file in enumerate(dir):
        input_file = os.path.join(input_path, file)
        print('input_file', input_file)
        output_file = os.path.join(output_path, file.replace('.wav', '.png'))
        create_spectrogram(input_file, output_file)


# %%
"""
Create PNG files containing spectrograms from all the WAV files in the "Sounds/background" directory.
"""

# %%
# create_pngs_from_wavs('Sounds/Aslan', 'Spectrograms/Aslan')

# %%
"""
Create PNG files containing spectrograms from all the WAV files in the "Sounds/chainsaw" directory.
"""

# %%
# create_pngs_from_wavs('Sounds/Esek', 'Spectrograms/Esek')

# %%
"""
Create PNG files containing spectrograms from all the WAV files in the "Sounds/engine" directory.
"""

# %%
# create_pngs_from_wavs('Sounds/Inek', 'Spectrograms/Inek')

# %%
"""
Create PNG files containing spectrograms from all the WAV files in the "Sounds/storm" directory.
"""

# %%

#create_pngs_from_wavs('audio1/fake1', 'spectrogram1/fake1')

#create_pngs_from_wavs('audio1/real1', 'spectrogram1/real1')

# %%
"""
Define two new helper functions for loading and displaying spectrograms and declare two Python lists — one to store spectrogram images, and another to store class labels.
"""

# %%
from keras.preprocessing import image


def load_images_from_path(path, label):
    images = []
    labels = []

    for file in os.listdir(path):
        images.append(image.img_to_array(image.load_img(os.path.join(path, file), target_size=(224, 224, 3))))
        labels.append((label))

    return images, labels


def show_images(images):
    fig, axes = plt.subplots(1, 8, figsize=(20, 20), subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i] / 255)


x = []
y = []

# %%
"""
Load the background spectrogram images, add them to the list named `x`, and label them with 0s.
"""

# %%
images, labels = load_images_from_path("C:/Users/kmgee/OneDrive/Desktop/Real_Fake_Main/spectrogram1/fake1", 0)
x += images
y += labels

images, labels = load_images_from_path("C:/Users/kmgee/OneDrive/Desktop/Real_Fake_Main/spectrogram1/real1", 1)
x += images
y += labels


# %%
"""
Split the images and labels into two datasets — one for training, and one for testing. Then divide the pixel values by 255 and one-hot-encode the labels using Keras's [to_categorical](https://keras.io/api/utils/python_utils/#to_categorical-function) function.
"""

# %%
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.3, random_state=0)

x_train_norm = np.array(x_train) / 255
x_test_norm = np.array(x_test) / 255

y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

# %%
"""
## Build and train a CNN

State-of-the-art image classification typically isn't done with traditional neural networks. Rather, it is performed with convolutional neural networks that use [convolution layers](https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/) to extract features from images and [pooling layers](https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/) to downsize images so features can be detected at various resolutions. The next task is to build a CNN containing a series of convolution and pooling layers for feature extraction, a pair of fully connected layers for classification, and a `softmax` layer that outputs probabilities for each class, and to train it with spectrogram images and labels. Start by defining the CNN.
"""

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# %%
"""
Train the CNN and save the `history` object returned by `fit` in a local variable.
"""

# %%
hist = model.fit(x_train_norm, y_train_encoded, validation_data=(x_test_norm, y_test_encoded), batch_size=10, epochs=20)
model.save('audio_classification_model7.h5')
# %%
"""
Plot the training and validation accuracy.
"""

# Plot accuracy and loss
def plot_history(history):
    plt.figure(figsize=(12, 6))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.show()


plot_history(hist)

from sklearn.metrics import classification_report, confusion_matrix

# Evaluate model on test data
y_pred = model.predict(x_test_norm)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test_encoded, axis=1)

# Classification Report
print("Classification Report:")
print(classification_report(y_true_classes, y_pred_classes))

# Confusion Matrix
print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
print(conf_matrix)


# %%
"""
The accuracy is decent given that the network was trained with just 280 images, but it might be possible to achieve higher accuracy by employing transfer learning.
"""

# %%
"""
## Use transfer learning to improve accuracy

[Transfer learning](https://towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a) is a powerful technique that allows sophisticated CNNs trained by Google, Microsoft, and others on GPUs to be repurposed and used to solve domain-specific problems. Many pretrained CNNs are available in the public domain, and several are included with Keras. Let's use [`MobileNetV2`](https://keras.io/api/applications/mobilenet/), a pretrained CNN from Google that is optimized for mobile devices, to extract features from spectrogram images.

> `MobileNetV2` requires less processing power and has a smaller memory footprint than CNNs such as `ResNet50V2`. That's why it is ideal for mobile devices. You can learn more about it in the [Google AI blog](https://ai.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html).

Start by calling Keras's [MobileNetV2](https://keras.io/api/applications/mobilenet/) function to instantiate `MobileNetV2` without the classification layers. Use the [preprocess_input](https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet/preprocess_input) function for `MobileNet` networks to preprocess the training and testing images. Then run both datasets through `MobileNetV2` to extract features.
"""

# %%

# %%
"""
Run the test images through the network and use a confusion matrix to assess the results.
"""


# %%
'''
from sklearn.metrics import confusion_matrix
import seaborn as sns

sns.set()
model = load_model('audio_classification_model6.h5')
#create_spectrogram("C:/Users/USER/Desktop/Project4U-Intern/KAGGLE/audio1/fake1/biden-to-linus.wav", 'spectrogram1/new_sample1.png')
create_spectrogram("C:/Users/USER/Desktop/Project4U-Intern/KAGGLE/audio1/fake1/biden-to-taylor.wav", 'spectrogram1/new_sample10.png')

# Preprocess and predict
#preprocessed_image1 = preprocess_spectrogram('spectrogram1/new_sample1.png')
preprocessed_image2 = preprocess_spectrogram('spectrogram1/new_sample10.png')

#predictions1 = model.predict(preprocessed_image1)
predictions2 = model.predict(preprocessed_image2)

class_labels = ['fake1', 'real1']
# Print predictions for the first audio file
# Print predictions for the first audio file
# Get the label with the maximum average value for the first audio file
# Get the label with the maximum average value for the first audio file
#max_label1 = class_labels[np.argmax(np.mean(predictions1, axis=0))]

# Get the label with the maximum average value for the second audio file
max_label2 = class_labels[np.argmax(np.mean(predictions2, axis=0))]

# Print the results
print("First audio file")
print(f'Max Average Label: {max_label1}')

# ... (Previous code remains unchanged)

# Check the max average label for the first audio file
if max_label1 == 'fake1':
    speech_text = "this audio is fake"
    print(speech_text)
    text_to_speech(speech_text, 'speech1.mp3')
elif max_label1 == 'real1':
    speech_text = "this audio is real"
    print(speech_text)
    text_to_speech(speech_text, 'speech1.mp3')


os.system('start speech1.mp3')

print("\nSecond audio file")
print(f'Max Average Label: {max_label2}')

# Check the max average label for the second audio file
if max_label2 == 'fake1':
    speech_text = "this audio is fake"
    print(speech_text)
    text_to_speech(speech_text, 'speech2.mp3')
elif max_label2 == 'real1':
    speech_text = "this audio is real"
    print(speech_text)
    text_to_speech(speech_text, 'speech2.mp3')


os.system('start speech2.mp3')
'''