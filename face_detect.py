import os
import shutil
import torch
import face_recognition
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def copy_images_with_face(input_face_path, dataset_folder, output_folder, tolerance=0.6, use_cuda=True):
   
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    print(f"Using device: {device}")

    input_image = face_recognition.load_image_file('D:\\ALL_pictures\\traditional day\\IMG-20250124-WA004.jpg')
    input_encodings = face_recognition.face_encodings(input_image)
    if not input_encodings:
        print("No face found in the input image.")
        return
    input_encoding = torch.tensor(input_encodings[0], device=device)

    os.makedirs('D:\\otp', exist_ok=True)

    for filename in os.listdir(dataset_folder):
        file_path = os.path.join(dataset_folder, filename)
        if not os.path.isfile(file_path):
            continue
        try:
            image = face_recognition.load_image_file(file_path)
            encodings = face_recognition.face_encodings(image)
            if not encodings:
                continue
   
            encodings_array = np.array(encodings)
            encodings_tensor = torch.tensor(encodings_array, device=device)

            distances = torch.norm(encodings_tensor - input_encoding, dim=1)
            if torch.any(distances <= tolerance):
                shutil.copy(file_path, os.path.join(output_folder, filename))
                logging.info(f"Copied {filename} to output folder")
        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")

copy_images_with_face(
    'D:\\ALL_pictures\\traditional day\\IMG-20250124-WA004.jpg',
    'D:\\ALL_pictures\\traditional day',
    'D:\\otp'
)