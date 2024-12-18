import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from collections import Counter

# Set up parameters (same as in dataset.py)
IMG_SIZE = 224
NUM_CHANNELS = 3
BATCH_SIZE = 16

# Configure TensorFlow for DirectML
physical_devices = tf.config.list_physical_devices('DirectML')
if physical_devices:
    print("\nDirectML devices found:", physical_devices)
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except Exception as e:
        print(f"Error configuring DirectML: {e}")
else:
    print("\nNo DirectML devices found, using CPU")

def preprocess_frame(frame):
    """Preprocess frame for model input"""
    try:
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to grayscale
        gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        # Resize
        resized = cv2.resize(gray_frame, (IMG_SIZE, IMG_SIZE))
        # Normalize
        normalized = resized.astype('float32') / 255.0
        # Convert grayscale to RGB by repeating the channel
        rgb_image = np.stack([normalized] * 3, axis=-1)
        # Add batch dimension
        processed = np.expand_dims(rgb_image, axis=0)
        return processed
    except Exception as e:
        print(f"Error preprocessing frame: {e}")
        return None

def create_model(num_classes):
    model = tf.keras.Sequential([
        # First block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                              input_shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Second block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Third block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def load_model_and_labels():
    """Load the trained model and emotion labels"""
    try:
        # Get emotion categories first
        dataset_dir = 'dataset'
        emotion_categories = sorted([d for d in os.listdir(dataset_dir) 
                                  if os.path.isdir(os.path.join(dataset_dir, d))])
        
        # Try to load saved model
        try:
            model = tf.keras.models.load_model('model2.h5')
            print("Loaded saved model successfully")
        except Exception as e:
            print(f"Could not load saved model ({e}), creating new one...")
            model = create_model(len(emotion_categories))
            
        print("Model and labels loaded successfully")
        return model, emotion_categories
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def analyze_frame(frame, model, emotion_categories):
    """Analyze frame using our trained model"""
    try:
        # Preprocess the frame
        processed_frame = preprocess_frame(frame)
        if processed_frame is None:
            return None, 0
        
        # Make prediction
        predictions = model.predict(processed_frame, verbose=0)
        emotion_idx = np.argmax(predictions[0])
        confidence = predictions[0][emotion_idx] * 100
        
        return emotion_categories[emotion_idx], confidence
            
    except Exception as e:
        print(f"Error analyzing frame: {e}")
        return None, 0

def extract_frames(video_path, num_frames=30):
    """Extract frames from video at regular intervals"""
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video file")
            return frames

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, total_frames // num_frames)

        for frame_idx in range(0, total_frames, interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)

        cap.release()
        print(f"Extracted {len(frames)} frames from video")
    except Exception as e:
        print(f"Error extracting frames: {e}")
    
    return frames

def main():
    # Load model and labels
    model, emotion_categories = load_model_and_labels()
    if model is None or emotion_categories is None:
        return

    while True:
        # Get video path from user
        video_path = input("\nEnter path to video file (or 'q' to quit): ")
        if video_path.lower() == 'q':
            break

        if not os.path.exists(video_path):
            print("Error: Video file not found")
            continue

        # Extract frames
        print("\nExtracting frames from video...")
        frames = extract_frames(video_path)
        if not frames:
            print("No frames extracted. Please try another video.")
            continue

        # Analyze frames
        print("\nAnalyzing frames...")
        emotions = []
        for i, frame in enumerate(frames):
            emotion, confidence = analyze_frame(frame, model, emotion_categories)
            if emotion:
                emotions.append(emotion)
                print(f"Frame {i+1}/{len(frames)}: {emotion} ({confidence:.2f}% confidence)")

        # Get majority emotion
        if emotions:
            emotion_counts = Counter(emotions)
            majority_emotion = emotion_counts.most_common(1)[0][0]
            percentage = (emotion_counts[majority_emotion] / len(emotions)) * 100

            print("\nResults:")
            print(f"Total frames analyzed: {len(frames)}")
            print(f"Emotion distribution:")
            for emotion, count in emotion_counts.items():
                print(f"- {emotion}: {count} frames ({(count/len(emotions))*100:.1f}%)")
            print(f"\nFinal prediction: This dog appears to be {majority_emotion}")
            print(f"Confidence: {percentage:.1f}% of frames showed this emotion")
        else:
            print("\nCould not determine dog emotion")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"\nAn error occurred: {e}") 