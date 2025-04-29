import numpy as np
import cv2
from typing import List, Tuple, Dict
from ComponentFeatureExtractor import ComponentFeatureExtractor

class ColumnPooling:
    def __init__(self, num_columns: int = 15):
        self.Nc = num_columns  # Number of columns

    def estimate_printed_text_bounds(self, bboxes: List[Tuple[int, int, int, int]]) -> Tuple[int, int]:
        # Extract row coordinates (x_min and x_max)
        x_mins = sorted([bbox[0] for bbox in bboxes])
        x_maxs = sorted([bbox[0] + bbox[2] for bbox in bboxes])

        # Get 1% index
        k = max(1, int(0.01 * len(bboxes)))

        # Estimate P1 and P2 using median of 1% left-most and right-most coordinates
        P1 = int(np.median(x_mins[:k]))
        P2 = int(np.median(x_maxs[-k:]))

        return P1, P2

    def assign_components_to_columns(self, bboxes: List[Tuple[int, int, int, int]], P1: int, P2: int) -> Dict[int, List[int]]:
        column_assignments = {i: [] for i in range(self.Nc)}
        column_width = (P2 - P1) / self.Nc

        for idx, (x, y, w, h) in enumerate(bboxes):
            x_start = x
            x_end = x + w

            # Calculate column indices overlapped
            start_col = int((x_start - P1) / column_width)
            end_col = int((x_end - P1) / column_width)

            # Clamp within bounds
            start_col = max(0, min(start_col, self.Nc - 1))
            end_col = max(0, min(end_col, self.Nc - 1))

            if start_col == end_col:
                column_assignments[start_col].append(idx)
            else:
                # Compute overlap area in both columns
                start_col_boundary = P1 + start_col * column_width
                end_col_boundary = P1 + (end_col + 1) * column_width

                overlap_start = min(x_end, start_col_boundary + column_width) - max(x_start, start_col_boundary)
                overlap_end = min(x_end, end_col_boundary) - max(x_start, end_col_boundary - column_width)

                if overlap_start > overlap_end:
                    column_assignments[start_col].append(idx)
                elif overlap_end > overlap_start:
                    column_assignments[end_col].append(idx)
                else:
                    column_assignments[start_col].append(idx)  # Tie-break to left column

        return column_assignments
    
    def extract_component_features(self, image: np.ndarray, bboxes: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        extractor = ComponentFeatureExtractor()
        features = []

        for (x, y, w, h) in bboxes:
            component_img = image[y:y+h, x:x+w]
            f1, f_bmpv = extractor.extract_features(component_img)
            combined = np.concatenate([f1, f_bmpv])
            features.append(combined)

        return features

    def average_pool_features(self, component_features: List[np.ndarray], column_assignments: Dict[int, List[int]]) -> Dict[int, np.ndarray]:
        pooled_features = {}

        for col_idx, indices in column_assignments.items():
            if not indices:
                continue
            pooled = np.mean([component_features[i] for i in indices], axis=0)
            pooled_features[col_idx] = pooled

        return pooled_features


    def pool(self, image: np.ndarray, bboxes: List[Tuple[int, int, int, int]]) -> Dict[int, np.ndarray]:
        # Step 1: Estimate P1 and P2 (start and end of printed text region)
        P1, P2 = self.estimate_printed_text_bounds(bboxes)

        # Step 2: Assign components to columns
        column_assignments = self.assign_components_to_columns(bboxes, P1, P2)

        # Step 3: Extract features
        component_features = self.extract_component_features(image, bboxes)

        # Step 4: Average pool features per column
        pooled_features = self.average_pool_features(component_features, column_assignments)

        return pooled_features

