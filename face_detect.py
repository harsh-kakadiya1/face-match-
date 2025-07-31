#!/usr/bin/env python3
"""
Face Recognition Image Sorter

This script identifies and copies images containing a specific person's face
from a dataset folder to an output folder using face recognition technology.

Usage:
    1. Update the file paths in the configuration section below
    2. Run the script: python face_recognition_image_sorter.py
"""

import os
import shutil
import torch
import face_recognition
import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_recognition.log'),
        logging.StreamHandler()
    ]
)

def copy_images_with_face(input_face_path, dataset_folder, output_folder, tolerance=0.6, use_cuda=True):
    """
    Copy images containing a specific person's face from dataset to output folder.
    
    Args:
        input_face_path (str): Path to reference image containing the target face
        dataset_folder (str): Path to folder containing images to search through
        output_folder (str): Path to folder where matching images will be copied
        tolerance (float): Face matching sensitivity (0.4=strict, 0.6=balanced, 0.8=lenient)
        use_cuda (bool): Enable GPU acceleration if available
    
    Returns:
        dict: Processing statistics
    """
    
    # Initialize device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Validate input paths
    if not os.path.exists(input_face_path):
        logging.error(f"Reference image not found: {input_face_path}")
        return {"error": "Reference image not found"}
    
    if not os.path.exists(dataset_folder):
        logging.error(f"Dataset folder not found: {dataset_folder}")
        return {"error": "Dataset folder not found"}
    
    # Load and encode the reference face
    print("Loading reference image...")
    try:
        input_image = face_recognition.load_image_file(input_face_path)
        input_encodings = face_recognition.face_encodings(input_image)
        
        if not input_encodings:
            logging.error("No face found in the reference image.")
            return {"error": "No face found in reference image"}
        
        if len(input_encodings) > 1:
            logging.warning("Multiple faces found in reference image. Using the first one.")
        
        input_encoding = torch.tensor(input_encodings[0], device=device)
        print("Reference face loaded successfully")
        
    except Exception as e:
        logging.error(f"Error loading reference image: {e}")
        return {"error": f"Error loading reference image: {e}"}
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder: {output_folder}")
    
    # Get list of image files
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}
    image_files = []
    
    for filename in os.listdir(dataset_folder):
        file_path = os.path.join(dataset_folder, filename)
        if (os.path.isfile(file_path) and 
            Path(filename).suffix.lower() in supported_extensions):
            image_files.append((filename, file_path))
    
    if not image_files:
        logging.warning("No supported image files found in dataset folder")
        return {"error": "No supported image files found"}
    
    print(f"Found {len(image_files)} images to process")
    
    # Process images
    processed_count = 0
    copied_count = 0
    error_count = 0
    
    print("Starting face recognition processing...")
    
    for filename, file_path in image_files:
        processed_count += 1
        
        # Show progress every 50 images
        if processed_count % 50 == 0:
            print(f"Progress: {processed_count}/{len(image_files)} images processed")
        
        try:
            # Load and process current image
            image = face_recognition.load_image_file(file_path)
            encodings = face_recognition.face_encodings(image)
            
            if not encodings:
                continue
            
            # Calculate face distances
            encodings_array = np.array(encodings)
            encodings_tensor = torch.tensor(encodings_array, device=device)
            distances = torch.norm(encodings_tensor - input_encoding, dim=1)
            
            # Check if any face matches within tolerance
            if torch.any(distances <= tolerance):
                shutil.copy(file_path, os.path.join(output_folder, filename))
                copied_count += 1
                logging.info(f"Copied {filename} to output folder")
                
        except Exception as e:
            error_count += 1
            logging.error(f"Error processing {filename}: {e}")
    
    # Calculate and display results
    success_rate = ((processed_count - error_count) / processed_count * 100) if processed_count > 0 else 0
    
    results = {
        "processed": processed_count,
        "copied": copied_count,
        "errors": error_count,
        "success_rate": f"{success_rate:.1f}%"
    }
    
    print("\n" + "="*50)
    print("PROCESSING COMPLETE!")
    print("="*50)
    print(f"Images processed: {results['processed']}")
    print(f"Images copied: {results['copied']}")
    print(f"Errors encountered: {results['errors']}")
    print(f"Success rate: {results['success_rate']}")
    print("="*50)
    
    return results


def main():
    """
    Main function - Configure your paths here
    """
    
    print("Face Recognition Image Sorter")
    print("="*50)
    
    # ====================================================================
    # CONFIGURATION - UPDATE THESE PATHS FOR YOUR USE CASE
    # ====================================================================
    
    # Path to reference image containing the target face
    REFERENCE_IMAGE_PATH = "path/to/reference/person.jpg"
    
    # Path to folder containing images to search through
    DATASET_FOLDER_PATH = "path/to/photo/collection"
    
    # Path to folder where matching images will be copied
    OUTPUT_FOLDER_PATH = "path/to/output/folder"
    
    # Face matching sensitivity (0.4=very strict, 0.6=balanced, 0.8=lenient)
    TOLERANCE = 0.6
    
    # Enable GPU acceleration (set to False if you don't have CUDA)
    USE_GPU = True
    
    # ====================================================================
    # IMPORTANT: Update the paths above before running!
    # ====================================================================
    
    # Validate configuration
    if any(path.startswith("path/to/") for path in [REFERENCE_IMAGE_PATH, DATASET_FOLDER_PATH, OUTPUT_FOLDER_PATH]):
        print("ERROR: Please update the file paths in the configuration section!")
        print("Open this script and modify the paths in the main() function.")
        print("\nExample paths:")
        print("  REFERENCE_IMAGE_PATH = 'C:/Users/YourName/Pictures/reference_face.jpg'")
        print("  DATASET_FOLDER_PATH = 'C:/Users/YourName/Pictures/vacation_photos'")
        print("  OUTPUT_FOLDER_PATH = 'C:/Users/YourName/Pictures/sorted_photos'")
        return
    
    # Run the face recognition process
    results = copy_images_with_face(
        input_face_path=REFERENCE_IMAGE_PATH,
        dataset_folder=DATASET_FOLDER_PATH,
        output_folder=OUTPUT_FOLDER_PATH,
        tolerance=TOLERANCE,
        use_cuda=USE_GPU
    )
    
    # Handle results
    if "error" in results:
        print(f"Processing failed: {results['error']}")
    else:
        print(f"Successfully processed {results['processed']} images!")
        print(f"Found and copied {results['copied']} matching images!")


if __name__ == "__main__":
    main()
