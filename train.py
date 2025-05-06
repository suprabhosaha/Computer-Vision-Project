import cv2
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pooling import ColumnPooling
from ConnectedComponentExtraction import ConnectedComponentsExtractor

class PrinterClassifier:
    def __init__(self, num_columns=15, num_classes=3):
        self.num_columns = num_columns
        self.num_classes = num_classes
        self.svms = [SVC(kernel='linear', probability=True) for _ in range(num_columns)]
        self.scaler = StandardScaler()

    def extract_training_data(self, image_paths, labels, num_columns=15):
        pooled_features = []
        for img_path, label in zip(image_paths, labels):
            try:
                cc_extractor = ConnectedComponentsExtractor()
                bboxes, _ = cc_extractor.extract(img_path, draw=False)

                if not bboxes:
                    print(f"Skipping {img_path}: No bounding boxes found.")
                    continue

                gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    print(f"Failed to load {img_path}. Skipping.")
                    continue

                pooling = ColumnPooling(num_columns)
                pooled = pooling.pool(gray, [(x1, y1, x2 - x1, y2 - y1) for (x1, y1, x2, y2) in bboxes])

                # Concatenate column features into a single vector
                feature_vector = []
                for col in range(num_columns):
                    feat = pooled.get(col, np.zeros_like(next(iter(pooled.values()), np.zeros(1))))
                    feature_vector.append(feat)

                pooled_features.append(np.concatenate(feature_vector))
            except Exception as e:
                print(f"Failed to process {img_path}: {e}")

        return np.array(pooled_features), np.array(labels)

    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)

        for col_idx in range(self.num_columns):
            column_features = X_scaled[:, col_idx].reshape(-1, 1)
            self.svms[col_idx].fit(column_features, y)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)

        predictions = []
        for col_idx in range(self.num_columns):
            column_features = X_scaled[:, col_idx].reshape(-1, 1)
            pred = self.svms[col_idx].predict(column_features)
            predictions.append(pred)

        final_predictions = []
        for i in range(len(X)):
            column_preds = [predictions[col_idx][i] for col_idx in range(self.num_columns)]
            final_predictions.append(np.argmax(np.bincount(column_preds)))

        return np.array(final_predictions)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        print(f"Accuracy: {accuracy:.2f}")
        return accuracy

if __name__ == "__main__":
    classifier = PrinterClassifier(num_columns=15, num_classes=3)

    data_dir = "data"
    image_paths = []
    labels = []

    for class_id in range(3):
        printer_dir = os.path.join(data_dir, f"{class_id}")
        if not os.path.exists(printer_dir):
            continue

        for fname in os.listdir(printer_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                image_paths.append(os.path.join(printer_dir, fname))
                labels.append(class_id)
    

    print("Extracting features...")
    X, y = classifier.extract_training_data(image_paths, labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training classifier...")
    classifier.train(X_train, y_train)

    print("Evaluating...")
    classifier.evaluate(X_test, y_test)

    import joblib
    joblib.dump(classifier, 'model/printer_classifier.pkl')