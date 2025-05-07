import cv2
import numpy as np
import joblib
from ConnectedComponentExtraction import ConnectedComponentsExtractor


class CharacterClassifier:
    def __init__(self, num_svms=15, num_classes=3):
        self.num_svms = num_svms
        self.num_classes = num_classes
        self.svms = [SVC(kernel='linear', probability=True) for _ in range(num_svms)]
        self.scaler = StandardScaler()
        self.feature_size = None  # Will be determined during training

    def extract_features(self, image, bbox):
        """Extract features for a single character component"""
        x1, y1, x2, y2 = bbox
        char_region = image[y1:y2, x1:x2]
        
        # Basic features - you can expand these
        if char_region.size == 0:
            return np.zeros(10)  # Fallback for empty regions
            
        char_region = cv2.resize(char_region, (20, 20))
        hist = cv2.calcHist([char_region], [0], None, [8], [0, 256]).flatten()
        moments = cv2.moments(char_region)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        features = np.concatenate([
            hist,
            hu_moments,
            [moments['m00'], moments['m10'], moments['m01']]
        ])
        return features

    def prepare_training_data(self, data_dir):
        """Prepare training data from directory structure: data/class_0/, data/class_1/, etc."""
        X = []
        y = []
        
        extractor = ConnectedComponentsExtractor()
        
        for class_id in range(self.num_classes):
            class_dir = os.path.join(data_dir, f"class_{class_id}")
            if not os.path.exists(class_dir):
                continue
                
            for img_file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_file)
                try:
                    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if gray is None:
                        continue
                        
                    bboxes, _ = extractor.extract(img_path, draw=False)
                    
                    for bbox in bboxes:
                        features = self.extract_features(gray, bbox)
                        X.append(features)
                        y.append(class_id)
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
        
        return np.array(X), np.array(y)

    def train(self, X, y):
        """Train the ensemble of SVMs"""
        if len(X) == 0:
            raise ValueError("No training data provided")
            
        X_scaled = self.scaler.fit_transform(X)
        self.feature_size = X_scaled.shape[1]
        
        feature_groups = np.array_split(X_scaled, self.num_svms, axis=1)
        
        for i in range(self.num_svms):
            if feature_groups[i].shape[1] > 0:  # Check if group has features
                self.svms[i].fit(feature_groups[i], y)

    def predict(self, X):
        """Make predictions using the ensemble"""
        if self.feature_size is None:
            raise RuntimeError("Classifier has not been trained yet")
            
        X_scaled = self.scaler.transform(X)
        feature_groups = np.array_split(X_scaled, self.num_svms, axis=1)
        
        predictions = []
        for i in range(self.num_svms):
            if feature_groups[i].shape[1] > 0:
                pred = self.svms[i].predict(feature_groups[i])
                predictions.append(pred)
            else:
                predictions.append(np.zeros(len(X)))  # Fallback
                
        final_predictions = []
        for sample_idx in range(len(X)):
            votes = [pred[sample_idx] for pred in predictions]
            final_predictions.append(np.argmax(np.bincount(votes)))
            
        return np.array(final_predictions)

    def evaluate(self, X_test, y_test):
        """Evaluate classifier performance"""
        y_pred = self.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        print(f"Accuracy: {accuracy:.2f}")
        return accuracy

class PrinterImageClassifier:
    def __init__(self, model_path='model/character_classifier.pkl'):
        self.classifier = joblib.load(model_path)
        self.extractor = ConnectedComponentsExtractor()
        
        self.class_colors = {
            0: (0, 255, 0),   
            1: (0, 0, 255),   
            2: (255, 0, 0)    
        }

    def classify_image(self, image_path, output_path=None):
        """Classify all components in a printer image"""
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Image not found at {image_path}")
        
        bboxes, _ = self.extractor.extract(image_path, draw=False)
        if not bboxes:
            print("No components found in the image")
            return []

        features = []
        for bbox in bboxes:
            features.append(self.classifier.extract_features(gray, bbox))
        features = np.array(features)
        predictions = self.classifier.predict(features)
        
        if output_path:
            self._visualize_results(image_path, bboxes, predictions, output_path)
        
        return list(zip(bboxes, predictions))

    def _visualize_results(self, image_path, bboxes, predictions, output_path):
        """Draw bounding boxes with class labels"""
        image = cv2.imread(image_path)
        
        for (x1, y1, x2, y2), pred in zip(bboxes, predictions):
            color = self.class_colors.get(pred, (255, 255, 0))  # Default yellow for unknown classes
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"Class {pred}", (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imwrite(output_path, image)
        print(f"Classification results saved to {output_path}")

if __name__ == "__main__":
    classifier = PrinterImageClassifier(model_path='model/character_classifier.pkl')    
    input_image = "data/2/Versalink_page (Copy 5).jpg"  # Change to your image path
    output_image = "classified_output.jpg"
    results = classifier.classify_image(input_image, output_image)
    overall_max = np.max([np.max(pred) for _, pred in results])
    # Print results
    print("\nClassification Results:")
    # print("----------------------")
    # for i, ((x1, y1, x2, y2), pred) in enumerate(results):
    #     print(f"Component {i+1}:")
    #     print(f"  Position: ({x1}, {y1}) to ({x2}, {y2})")
    #     print(f"  Class: {pred}")
    #     print(f"  Size: {x2-x1}x{y2-y1} pixels")
    #     print()
    print(overall_max)