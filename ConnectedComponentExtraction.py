import cv2
import numpy as np
import os

class ConnectedComponentsExtractor:
    def __init__(self, area_filter=(0.4, 5.0), width_range=(5, 95), height_range=(16, 110)):
        self.area_filter = area_filter
        self.width_range = width_range
        self.height_range = height_range

    def _binarize_image(self, gray_image):
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    def _filter_components(self, stats, median_area):
        valid_indices = []
        for i in range(1, stats.shape[0]):
            x, y, w, h, area = stats[i]

            if not (self.area_filter[0] * median_area <= area <= self.area_filter[1] * median_area):
                continue
            if not (self.width_range[0] <= w <= self.width_range[1]):
                continue
            if not (self.height_range[0] <= h <= self.height_range[1]):
                continue

            valid_indices.append(i)
        return valid_indices

    def extract(self, image_path, output_path=None, draw=True):
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Image not found at {image_path}")
        original = cv2.imread(image_path)

        binary = self._binarize_image(gray)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        areas = stats[1:, cv2.CC_STAT_AREA]
        median_area = np.median(areas)

        valid_indices = self._filter_components(stats, median_area)

        if draw:
            for idx in valid_indices:
                x, y, w, h, _ = stats[idx]
                cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if output_path:
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                cv2.imwrite(output_path, original)

        bboxes = []
        for idx in valid_indices:
            x, y, w, h, _ = stats[idx]
            bboxes.append((x, y, x + w, y + h))

        return bboxes, labels



# # Initialize extractor
# extractor = ConnectedComponentsExtractor()

# # Run extraction
# bboxes, labels = extractor.extract(
#     image_path="data/2/Versalink_page.jpg",
#     output_path="data/output_components.jpg"
# )
# # print(bboxes,labels)
# print(f"Extracted {len(bboxes)} character components")
