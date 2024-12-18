import os
import cv2
import numpy as np
import tensorflow as tf
from transformers import pipeline
from PIL import Image
from collections import Counter

IMG_SIZE = 224
NUM_CHANNELS = 3

def preprocess_frame(frame):
    """Preprocess frame for model input"""
    try:
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize
        resized = cv2.resize(rgb_frame, (IMG_SIZE, IMG_SIZE))
        # Normalize pixel values to [0, 1]
        normalized = resized.astype('float32') / 255.0
        # Add batch dimension
        processed = np.expand_dims(normalized, axis=0)
        return processed
    except Exception as e:
        print(f"Error preprocessing frame: {e}")
        return None

def test_video(video_path):
    # Load the emotion detection model using transformers
    print("\nLoading emotion detection model...")
    model = pipeline("image-classification", model="Dewa/dog_emotion_v2")

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_to_skip = 10 # Number of frames to skip for 0.5 second intervals
    
    frame_count = 0
    processed_count = 0
    emotions = []
    confidences = []
    print("\nProcessing video...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # Only process frames at 0.5 second intervals
        if frame_count % frames_to_skip != 0:
            continue
            
        processed_count += 1
        if processed_count % 10 == 0:  # Print progress every 10 processed frames
            print(f"Processed {processed_count} frames")

        # Preprocess the frame
        processed_frame = preprocess_frame(frame)
        if processed_frame is None:
            continue

        # Convert processed frame to PIL image
        pil_image = Image.fromarray((processed_frame[0] * 255).astype(np.uint8))

        # Get predictions using the pipeline
        predictions = model(pil_image)  # Pass the PIL image directly
        emotions.append(predictions[0]['label'])
        confidences.append(predictions[0]['score'])

    # Clean up
    cap.release()

    # Calculate overall results
    if emotions:  # Check if any frames were processed
        most_common_emotion = Counter(emotions).most_common(1)[0][0]
        average_confidence = sum(confidences) / len(confidences)

        print(f"\nProcessing complete!")
        print(f"Total frames in video: {frame_count}")
        print(f"Frames processed: {processed_count}")
        print(f"Overall emotion: {most_common_emotion}")
        print(f"Average confidence: {average_confidence:.2f}")
    else:
        print("\nNo frames were processed successfully")

if __name__ == "__main__":
    video_path = input("Enter the path to your video file: ")
    test_video(video_path)
    