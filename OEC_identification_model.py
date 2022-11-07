import tensorflow as tf
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import skimage
import tiffile as tiff
# Don't modify this file
# DATA PREPARATION

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

# Requires Images to train on, (different) images to test on, and masks that match the training images.
TRAIN_PATH = "LOECs/Training_images/"
TEST_PATH = "LOECs/Testing_images/"
TRAIN_MASKS = "LOECs/Masks/"

seed = 42
np.random.seed = seed

# Train data

train_images = []

for directory_path in glob.glob(TRAIN_PATH):
    for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
        print(img_path)
        img = cv2.imread(img_path, 1)
        img = skimage.transform.resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        train_images.append(img)

# Convert list to array for machine learning processing
train_images = np.array(train_images)
X_train = train_images.astype('uint8')

# Capture mask/label info as a list

train_masks = []
for directory_path in glob.glob(TRAIN_MASKS):
    for mask_path in glob.glob(os.path.join(directory_path, "*.tif")):
        print(mask_path)
        mask = cv2.imread(mask_path, 0)
        mask = skimage.transform.resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        train_masks.append(mask)

# Convert list to array for machine learning processing
train_masks = np.array(train_masks)
Y_train = train_masks.astype('bool')

# Test data

test_images = []
X_test = []
for directory_path in glob.glob(TEST_PATH):
    for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
        print(img_path)
        img = cv2.imread(img_path, 1)
        img = skimage.transform.resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        test_images.append(img)
test_images = np.array(test_images)
X_test = test_images.astype('uint8')

print(X_test.dtype)
print(X_train.dtype)

print(X_test)
print(X_train)

print(X_test.shape)
print(X_train.shape)

print(X_test.size)
print(X_train.size)

print('Done!')

# UNET Model below


# Build the model
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

# Contraction path
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

# Expansive path
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.h5', verbose=1, save_best_only=True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs')
]

results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=100,
                    callbacks=callbacks)
print("Display some examples")
idx = random.randint(0, len(X_train))

preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_test = model.predict(X_test, verbose=1)

preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

ix = random.randint(0, len(preds_test_t)-1)
# imshow(X_train[ix])
# plt.show()
# imshow(np.squeeze(Y_train[ix]))
# plt.show()
# imshow(np.squeeze(preds_train_t[ix]))
# plt.show()

fig, (ax0, ax1, ax2, ax3) = plt.subplots(ncols=4, figsize=(12, 4))

img0 = ax0.imshow(X_train[ix])
ax0.set_title("Image_raw")
ax0.axis("off")

img1 = ax1.imshow(np.squeeze(Y_train[ix]))
ax1.set_title("Train_mask")
ax1.axis("off")

img2 = ax2.imshow(np.squeeze(preds_test_t[ix]))
ax2.set_title("Mask_predicted")
ax2.axis("off")

img3 = ax3.imshow(np.squeeze(X_test[ix]))
ax3.set_title("test image")
ax3.axis("off")

fig.tight_layout()

plt.show()

output_list = range(len(preds_test_t))
print(output_list)

for n in output_list:
    tiff.imwrite("LOECs/Output_folder/Mask" + '_' + str(n) + '.tif', np.squeeze(preds_test_t[n]))
    tiff.imwrite("LOECs/Output_folder/Sample" + '_' + str(n) + '.tif', np.squeeze(X_test[n]))
