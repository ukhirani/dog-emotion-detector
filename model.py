import pandas as pd
import os
import cv2
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from skimage.feature import hog
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

class EmotionDetectionModel:
    def __init__(self):
        self.classes = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt']
        self.image_size = (96, 96)
        self.scaler = StandardScaler()
        self.pca = None
        self.lda = None
        self.svm = None
        
    def load_and_preprocess_data(self, csv_path, image_folder, samples_per_class=2000):
        """Load and preprocess the dataset with balanced class distribution"""
        data = pd.read_csv(csv_path)
        
        # Balance dataset
        balanced_data = []
        for emotion in self.classes:
            class_data = data[data['label'] == emotion]
            if len(class_data) < samples_per_class:
                print(f"Class '{emotion}' has {len(class_data)} samples")
                balanced_data.append(class_data)  # Use all available samples
            else:
                sampled_data = class_data.sample(samples_per_class, random_state=42)
                balanced_data.append(sampled_data)
        
        balanced_data = pd.concat(balanced_data)
        
        # Extract features
        X = []
        y = []
        for _, row in balanced_data.iterrows():
            image_path = os.path.join(image_folder, row['pth'])
            if os.path.exists(image_path):
                # Read and preprocess image
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, self.image_size)
                image = cv2.equalizeHist(image)  # Improve contrast
                
                # Extract HOG features with optimized parameters
                features, _ = hog(image,
                                orientations=12,
                                pixels_per_cell=(6, 6),
                                cells_per_block=(3, 3),
                                visualize=True,
                                channel_axis=None)
                X.append(features)
                y.append(row['label'])
                
        return np.array(X), np.array(y)
    
    def train(self, X, y):
        """Train the model with optimized parameters"""
        # Normalize features
        X = self.scaler.fit_transform(X)
        
        # Apply PCA
        self.pca = PCA(n_components=0.95)
        X_pca = self.pca.fit_transform(X)
        
        # Apply LDA
        self.lda = LDA(n_components=len(self.classes)-1)
        X_lda = self.lda.fit_transform(X_pca, y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_lda, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        
        # Define SVM with balanced class weights
        base_svm = SVC(
            probability=True,
            class_weight='balanced',
            random_state=42
        )
        
        # Grid search for optimal parameters
        param_grid = {
            'C': [1, 10, 100],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'linear']
        }
        
        grid_search = GridSearchCV(
            base_svm,
            param_grid,
            cv=5,
            scoring='balanced_accuracy',
            n_jobs=-1
        )
        
        # Train the model
        grid_search.fit(X_train, y_train)
        self.svm = grid_search.best_estimator_
        
        # Evaluate the model
        y_pred = self.svm.predict(X_test)
        print("\nBest parameters:", grid_search.best_params_)
        print("\nAccuracy:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.classes))
        
        return X_test, y_test
    
    def predict(self, image_path):
        """Predict emotion for a new image with confidence scores"""
        # Read and preprocess image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, self.image_size)
        image = cv2.equalizeHist(image)
        
        # Extract features
        features, _ = hog(image,
                         orientations=12,
                         pixels_per_cell=(6, 6),
                         cells_per_block=(3, 3),
                         visualize=True,
                         channel_axis=None)
        
        # Transform features
        features = self.scaler.transform([features])
        features = self.pca.transform(features)
        features = self.lda.transform(features)
        
        # Get prediction and probabilities
        prediction = self.svm.predict(features)[0]
        probabilities = self.svm.predict_proba(features)[0]
        
        # Get top 3 predictions with probabilities
        top_3_idx = np.argsort(probabilities)[-3:][::-1]
        results = []
        for idx in top_3_idx:
            emotion = self.classes[idx]
            prob = probabilities[idx]
            results.append((emotion, prob))
        
        return prediction, results

def main():
    # Initialize and train the model
    model = EmotionDetectionModel()
    
    # Load and preprocess data
    csv_path = './Dataset/labels.csv'
    image_folder = './Dataset/'
    X, y = model.load_and_preprocess_data(csv_path, image_folder)
    
    # Train the model
    X_test, y_test = model.train(X, y)
    
    # Test prediction on a sample image
    test_image_path = './Dataset/anger/image0000006.jpg'
    prediction, top_3_results = model.predict(test_image_path)
    
    print(f"\nTest prediction for {test_image_path}:")
    print(f"Predicted emotion: {prediction}")
    print("Top 3 predictions with confidence:")
    for emotion, confidence in top_3_results:
        print(f"{emotion}: {confidence:.2%}")

if __name__ == "__main__":
    main()
