import cv2
import numpy as np

class VisualAnalyzer:
    def analyze_frame(self, frame: np.ndarray) -> dict:
        return {
            "brightness": self._calculate_brightness(frame),
            "contrast": self._calculate_contrast(frame),
            "saturation": self._calculate_saturation(frame),
            "dominant_colors": self._extract_dominant_colors(frame)
        }

    def _calculate_brightness(self, frame: np.ndarray ) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        brightness = np.mean(gray) / 255.0

        return round(float(brightness), 3)

    def _calculate_contrast(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        contrast = np.std(gray) / 255.0

        return round(float(contrast), 3)

    def _calculate_saturation(self, frame: np.ndarray) -> float:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        saturation_channel = hsv[:, :, 1]

        saturation = np.mean(saturation_channel)/255.0
        return round(float(saturation), 3)

    def _extract_dominant_colors(self, frame: np.ndarray, color_count: int = 3) -> list[dict]:
        resized = cv2.resize(frame, (100, 100))

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        pixels = rgb.reshape(-1, 3).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)

        _, labels, centers = cv2.kmeans(pixels, color_count, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        labels = labels.flatten()

        counts = np.bincount(labels, minlength=color_count)

        order = np.argsort(counts)[::-1]

        colors = []

        for index in order:
            rgb_color = centers[index].astype(int)
            red, green, blue = rgb_color

            hex_color = (f"#{red:02X}{green:02X}{blue:02X}")

            proportion = counts[index] / len(labels)

            colors.append({
                "hex": hex_color,
                "rgb": [int(red), int(green), int(blue)],
                "proportion": round(float(proportion), 3)
            })

        return colors