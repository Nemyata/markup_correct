import os
import io
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


class CutMouthModel:
    def __init__(self):
        self.model = None

    def set_model(self, model_path: str):
        if not os.path.exists(model_path):
            raise ValueError(f"Can't get model with path {model_path}")
        self.model = YOLO(model_path)
        self.model.to("cpu")
        self.names = self.model.names
        print(self.names)

    @staticmethod
    def open_image_from_bytes(file: io.BytesIO):
        img = Image.open(file)
        img = img.convert("RGB")
        return img

    def analyze(self, photo_bytes: io.BytesIO, save_image_path: Path, save_label_path: Path, label_path: Path, confidence_threshold: float = 0.6):
        try:
            photo_bytes.seek(0)
            photo = self.open_image_from_bytes(photo_bytes)
            image = np.array(photo)
            h, w = image.shape[:2]

            print(f"Original image size: {w}x{h}")

            result = self.model(photo)
            print(f"Result: {result}")

            if len(result[0].boxes) == 0:
                return False, "Mouth not detected", None

            elem = result[0].boxes.data[0]
            confidence = elem[4].item()

            if confidence < confidence_threshold:
                return False, f"Confidence below threshold: {confidence:.2f}", None

            n = 25  # Ареол вокруг обнаруженного объекта
            x_min = max(int(elem[0]) - n, 0)
            y_min = max(int(elem[1]) - n, 0)
            x_max = min(int(elem[2]) + n, w)
            y_max = min(int(elem[3]) + n, h)

            print(f"Crop size: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")

            # Чтение разметки и её преобразование в пиксели для оригинала
            with open(label_path, "r") as f:
                labels = f.readlines()

            # Обрезка изображения
            cropped_image = image[y_min:y_max, x_min:x_max]

            # Преобразование разметки для обрезанного изображения
            updated_labels = []
            for label in labels:
                parts = label.split()
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])

                # Преобразование в пиксели для обрезанного изображения
                x_center_px = x_center * w
                y_center_px = y_center * h
                width_px = width * w
                height_px = height * h

                # Перевод в координаты для обрезанного изображения
                new_x_center = (x_center_px - x_min) / (x_max - x_min)
                new_y_center = (y_center_px - y_min) / (y_max - y_min)
                new_width = width_px / (x_max - x_min)
                new_height = height_px / (y_max - y_min)

                # Только разметка для обрезанного изображения
                updated_labels.append(f"{class_id} {new_x_center:.6f} {new_y_center:.6f} {new_width:.6f} {new_height:.6f}")

            # Сохранение разметки для обрезанного изображения
            with open(save_label_path, "w") as label_file:
                label_file.write("\n".join(updated_labels))

            # Сохранение обрезанного изображения
            cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(str(save_image_path), cropped_image_rgb)

            return True, cropped_image, "Success"

        except Exception as e:
            print(f"Error during analysis: {e}")
            return False, None, f"Error during analysis: {e}"


def process_folder(cut_mouth_model, folder_path: Path, save_directory: Path):
    images_path = folder_path / "images"
    labels_path = folder_path / "labels"

    # Создаем папки для сохранения обрезанных изображений и новых разметок
    cropped_images_path = save_directory / "image"
    cropped_labels_path = save_directory / "labels"
    cropped_images_path.mkdir(parents=True, exist_ok=True)
    cropped_labels_path.mkdir(parents=True, exist_ok=True)

    # Проходим по изображениям и их разметкам
    for image_file in images_path.glob("*.jpg"):
        image_name = image_file.stem
        label_file = labels_path / f"{image_name}.txt"

        if not label_file.exists():
            print(f"Label file for {image_file} not found.")
            continue

        print(f"Processing {image_file} with {label_file}")

        # Считываем изображение
        with open(image_file, "rb") as image_f:
            photo_bytes = io.BytesIO(image_f.read())
            photo_bytes.name = str(image_file)

        # Пути для сохранения обрезанных изображений и разметки
        save_image_path = cropped_images_path / f"{image_name}.jpg"
        save_label_path = cropped_labels_path / f"{image_name}.txt"

        # Выполняем анализ изображения
        result, cropped_image, message = cut_mouth_model.analyze(photo_bytes, save_image_path, save_label_path, label_file)

        if result:
            print(f"Processing completed for {image_name}")
        else:
            print(f"Error during processing {image_name}: {message}")


# Пример использования
if __name__ == "__main__":
    model_path = "TEMP_VAR.pt"
    folder_path = Path(r"D:\DATASET\Dental\YOLO_data\test")
    save_directory = Path(r"D:\DATASET\Dental\YOLO_data_result")

    cut_mouth_model = CutMouthModel()
    cut_mouth_model.set_model(model_path)

    process_folder(cut_mouth_model, folder_path, save_directory)
