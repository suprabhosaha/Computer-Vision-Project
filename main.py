import sys
import cv2
import numpy as np
from ColumnPooling import ColumnPooling
from ConnectedComponentExtraction import ConnectedComponentsExtractor


def main(image_path: str, output_vis_path: str = None):
    print(f"Processing image: {image_path}")

    # Step 1: Extract connected components and filtered bounding boxes
    cc_extractor = ConnectedComponentsExtractor()
    bboxes, label_image = cc_extractor.extract(image_path, output_path=output_vis_path, draw=True)

    print(f"Total filtered components: {len(bboxes)}")

    # Step 2: Load grayscale image (used for feature extraction)
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Step 3: Initialize column pooling
    pooling = ColumnPooling(num_columns=15)

    # Step 4: Extract features from components
    component_features = pooling.extract_component_features(gray, [(x1, y1, x2 - x1, y2 - y1) for (x1, y1, x2, y2) in bboxes])

    # Step 5: Pool features column-wise
    pooled_features = pooling.pool([(x1, y1, x2 - x1, y2 - y1) for (x1, y1, x2, y2) in bboxes], component_features)

    # Step 6: Print output
    print(f"Pooled features extracted from {len(pooled_features)} columns.")
    for col, feat in pooled_features.items():
        print(f"Column {col}: Feature vector shape = {feat.shape}")


import sys
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from ColumnPooling import ColumnPooling
from ConnectedComponentExtraction import ConnectedComponentsExtractor
from sklearn.model_selection import train_test_split

class Classifier:
    def __init__(self, num_columns=15, classes=4):
        self.num_columns = num_columns
        self.classes = classes
        self.svms = [SVC(kernel='linear', probability=True) for _ in range(self.num_columns)]  # 15 SVMs
        self.scaler = StandardScaler()

    def train(self, X, y):
        # Standardizing the features for training
        X_scaled = self.scaler.fit_transform(X)
        
        # Train each SVM on the respective column data
        for col_idx in range(self.num_columns):
            # Extract the features for the current column
            column_features = X_scaled[:, col_idx].reshape(-1, 1)  # Features for column 'col_idx'
            
            # Train the SVM for this column
            self.svms[col_idx].fit(column_features, y)

    def predict(self, X):
        # Standardize the test data
        X_scaled = self.scaler.transform(X)

        # Predict using each column's SVM and aggregate the results
        predictions = []
        for col_idx in range(self.num_columns):
            column_features = X_scaled[:, col_idx].reshape(-1, 1)
            pred = self.svms[col_idx].predict(column_features)
            predictions.append(pred)

        # Aggregate predictions (use majority vote here)
        final_predictions = []
        for i in range(len(X)):
            # Get the most common class predicted across all columns
            column_preds = [predictions[col_idx][i] for col_idx in range(self.num_columns)]
            final_predictions.append(np.bincount(column_preds).argmax())  # Majority vote

        return np.array(final_predictions)

def extract_training_data(image_paths, labels, num_columns=15):
    pooled_features = []
    for image_path in image_paths:
        cc_extractor = ConnectedComponentsExtractor()
        bboxes, _ = cc_extractor.extract(image_path, draw=False)  # Extract components without drawing
        
        # Initialize ColumnPooling
        pooling = ColumnPooling(num_columns)
        
        # Extract features for each component in each column
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        component_features = pooling.extract_component_features(gray, [(x1, y1, x2 - x1, y2 - y1) for (x1, y1, x2, y2) in bboxes])
        
        # Pool features per column
        pooled_features_for_image = pooling.pool([(x1, y1, x2 - x1, y2 - y1) for (x1, y1, x2, y2) in bboxes], component_features)
        
        # Convert pooled features to a single vector (flatten for each column)
        feature_vector = []
        for col in range(num_columns):
            feature_vector.append(pooled_features_for_image.get(col, np.zeros_like(component_features[0])))

        pooled_features.append(np.concatenate(feature_vector))  # Concatenate features of all columns

    return np.array(pooled_features), np.array(labels)


def main(image_paths, labels, test_image_paths, test_labels):
    # Prepare training data
    X, y = extract_training_data(image_paths, labels)
    
    # Initialize classifier
    classifier = Classifier(num_columns=15, classes=4)

    # Train the classifier
    classifier.train(X, y)

    # Prepare test data
    X_test, y_test = extract_training_data(test_image_paths, test_labels)

    # Make predictions
    predictions = classifier.predict(X_test)

    # Print classification results
    print("Test accuracy:", np.mean(predictions == y_test))

