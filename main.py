import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ColumnPooling import ColumnPooling
from ConnectedComponentExtraction import ConnectedComponentsExtractor


class Classifier:
    def __init__(self, num_columns=15, classes=4):
        self.num_columns = num_columns
        self.classes = classes
        self.svms = [SVC(kernel='linear', probability=True) for _ in range(self.num_columns)]
        self.scaler = StandardScaler()

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
            final_predictions.append(np.bincount(column_preds).argmax())
        return np.array(final_predictions)


def load_image_paths_from_directory(data_dir):
    image_paths = []
    labels = []
    for class_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for filename in os.listdir(class_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                image_paths.append(os.path.join(class_path, filename))
                labels.append(int(class_name))
    return image_paths, labels


def extract_training_data(image_paths, labels, num_columns=15):
    pooled_features = []
    for idx, image_path in enumerate(image_paths):
        try:
            cc_extractor = ConnectedComponentsExtractor()
            bboxes, _ = cc_extractor.extract(image_path, draw=False)
                       
            pooling = ColumnPooling(num_columns)
            gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            component_features = pooling.extract_component_features(
                gray, bboxes
            )
            print(f"Extracted {len(component_features)} features from {image_path}")
            pooled = pooling.pool(
                [(x1, y1, x2 - x1, y2 - y1) for (x1, y1, x2, y2) in bboxes],
                component_features
            )

            feature_vector = []
            for col in range(num_columns):
                feature_vector.append(pooled.get(col, np.zeros_like(component_features[0])))
            pooled_features.append(np.concatenate(feature_vector))
        except Exception as e:
            print(f"Skipping {image_path} due to error: {e}")
    return np.array(pooled_features), np.array(labels[:len(pooled_features)])


def main(data_dir, test_ratio=0.2, num_columns=15):
    image_paths, labels = load_image_paths_from_directory(data_dir)
    X_train_paths, X_test_paths, y_train, y_test = train_test_split(image_paths, labels, test_size=test_ratio, stratify=labels, random_state=42)

    print(f"Extracting training features from {len(X_train_paths)} images...")
    X_train, y_train = extract_training_data(X_train_paths, y_train, num_columns)

    print(f"Extracting test features from {len(X_test_paths)} images...")
    X_test, y_test = extract_training_data(X_test_paths, y_test, num_columns)

    classifier = Classifier(num_columns=num_columns)
    classifier.train(X_train, y_train)

    predictions = classifier.predict(X_test)

    acc = np.mean(predictions == y_test)
    print(f"Test Accuracy: {acc * 100:.2f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory with class folders (0, 1, 2...)')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='Test set split ratio')
    args = parser.parse_args()

    main(args.data_dir, args.test_ratio)
