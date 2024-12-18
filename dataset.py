import os
import numpy as np
from PIL import Image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.client import device_lib
from sklearn.utils.class_weight import compute_class_weight

print(f"TensorFlow version: {tf.__version__}")

# Configure TensorFlow for DirectML
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True  # Show device placement info

# Create session with the config
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

# Check available devices
def get_available_devices():
    devices = device_lib.list_local_devices()
    print("\nAvailable devices:")
    for device in devices:
        if "DML" in device.physical_device_desc:
            print(f"Found DirectML device: {device.physical_device_desc}")
        else:
            print(f"Found device: {device.physical_device_desc}")
    return devices

devices = get_available_devices()

# Set up parameters
IMG_SIZE = 224
NUM_CHANNELS = 1
BATCH_SIZE = 16
EPOCHS = 50

def load_dataset(dataset_path, images_per_class=500):
    images = []
    labels = []
    label_mapping = {}
    
    # Add data augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    emotion_categories = [d for d in os.listdir(dataset_path) 
                         if os.path.isdir(os.path.join(dataset_path, d))]
    
    for i, emotion in enumerate(emotion_categories):
        label_mapping[emotion] = i
    
    for emotion in emotion_categories:
        emotion_path = os.path.join(dataset_path, emotion)
        img_files = [f for f in os.listdir(emotion_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Limit to images_per_class images
        img_files = img_files[:images_per_class]
        
        print(f"\nProcessing {emotion} class: {len(img_files)} images")
        for i, img_name in enumerate(img_files):
            if i % 100 == 0:
                print(f"Processed {i}/{len(img_files)} images")
                
            img_path = os.path.join(emotion_path, img_name)
            
            try:
                img = Image.open(img_path).convert('L')
                img = img.resize((IMG_SIZE, IMG_SIZE))
                img_array = np.array(img, dtype='float32')
                img_array = img_array / 255.0
                img_array = img_array.reshape(IMG_SIZE, IMG_SIZE, 1)
                
                # Add original image
                images.append(img_array)
                labels.append(label_mapping[emotion])
                
                # Add augmented images if we don't have enough real images
                if len(img_files) < images_per_class:
                    num_augmented = images_per_class - len(img_files)
                    if i < num_augmented:
                        aug_img = datagen.random_transform(img_array)
                        images.append(aug_img)
                        labels.append(label_mapping[emotion])
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    # Convert to numpy arrays
    images = np.array(images, dtype='float32')
    labels = np.array(labels)
    
    print(f"\nTotal images loaded: {len(images)}")
    print(f"Images per class: {images_per_class}")
    print(f"Number of classes: {len(emotion_categories)}")
    
    return images, labels, len(emotion_categories)

def create_model(num_classes):
    print("\nCreating model...")
    
    with tf.device('/device:DML:0'):
        model = Sequential([
            # First block - more filters
            Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
            BatchNormalization(),
            Conv2D(64, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Second block
            Conv2D(128, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            Conv2D(128, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Third block
            Conv2D(256, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            Conv2D(256, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Dense layers
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("Model created successfully")
    return model

def main():
    print("\nLoading dataset...")
    dataset_path = 'dataset'
    
    # Load exactly 500 images per class
    X, y, num_classes = load_dataset(dataset_path, images_per_class=500)
    
    # Convert labels to categorical
    y = to_categorical(y, num_classes=num_classes)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("\nInitializing CNN model...")
    model = create_model(num_classes)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.00001
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]

    print("\nStarting training on DirectML device...")
    with tf.device('/device:DML:0'):
        history = model.fit(
            X_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )

    # Evaluate
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {test_accuracy*100:.2f}%")

    # Save model
    model.save('dog_emotion_model.h5')
    print("\nModel saved as 'dog_emotion_model.h5'")

if __name__ == "__main__":
    main()
