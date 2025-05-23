# Face Recognition Image Sorter

A Python tool that automatically identifies and copies images containing a specific person's face from a large dataset. This project uses advanced face recognition technology with GPU acceleration support to efficiently process thousands of images.

## 🌟 Features

- **Face Recognition**: Uses state-of-the-art face recognition algorithms to identify matching faces
- **GPU Acceleration**: Optional CUDA support for faster processing of large image datasets
- **Batch Processing**: Efficiently processes entire folders of images
- **Flexible Tolerance**: Adjustable similarity threshold for face matching
- **Robust Error Handling**: Continues processing even if individual images fail
- **Detailed Logging**: Comprehensive logging for monitoring progress and debugging

## 🚀 Use Cases

- **Event Photography**: Sort through hundreds of event photos to find images of specific people
- **Family Photo Organization**: Organize family albums by identifying specific family members
- **Security Applications**: Filter surveillance footage or images for specific individuals
- **Social Media Management**: Quickly find photos containing specific people from large collections
- **Professional Photography**: Sort client photos efficiently

## 📋 Requirements

### System Requirements
- Python 3.7 or higher
- Windows/Linux/macOS
- Optional: NVIDIA GPU with CUDA support for acceleration

### Python Dependencies
See `requirements.txt`:
```
torch>=1.9.0
face-recognition>=1.3.0
numpy>=1.21.0
opencv-python>=4.5.0
dlib>=19.22.0
cmake>=3.21.0
```

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/face-recognition-sorter.git
cd face-recognition-sorter
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv face_recognition_env
source face_recognition_env/bin/activate  # On Windows: face_recognition_env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Additional Setup for Windows Users
If you encounter issues with dlib installation on Windows:
```bash
pip install cmake
pip install dlib
```

## 📖 Usage

### Basic Usage
```python
from face_recognition_sorter import copy_images_with_face

# Sort images based on a reference face
copy_images_with_face(
    input_face_path='path/to/reference/image.jpg',
    dataset_folder='path/to/image/dataset',
    output_folder='path/to/output/folder',
    tolerance=0.6,
    use_cuda=True
)
```

### Command Line Usage
```python
python face_recognition_sorter.py
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_face_path` | str | Required | Path to the reference image containing the target face |
| `dataset_folder` | str | Required | Path to folder containing images to search through |
| `output_folder` | str | Required | Path where matching images will be copied |
| `tolerance` | float | 0.6 | Face matching sensitivity (lower = more strict) |
| `use_cuda` | bool | True | Enable GPU acceleration if available |

### Tolerance Settings Guide
- **0.4**: Very strict matching (identical twins might not match)
- **0.6**: Default - good balance of accuracy and recall
- **0.8**: More lenient (may include similar-looking people)

## 🔧 Configuration

### GPU Configuration
The script automatically detects CUDA availability:
- **GPU Available**: Uses CUDA for faster processing
- **GPU Unavailable**: Falls back to CPU processing

### Logging Configuration
Modify logging level in the script:
```python
logging.basicConfig(level=logging.INFO)  # Change to DEBUG for verbose output
```

## 📁 Project Structure
```
face-recognition-sorter/
├── face_recognition_sorter.py    # Main script
├── requirements.txt              # Python dependencies
├── README.md                    # Project documentation
├── examples/                    # Example usage scripts
│   └── example_usage.py
└── tests/                       # Unit tests
    └── test_face_recognition.py
```

## 🚨 Important Notes

### Privacy and Ethics
- **Consent**: Always ensure you have permission to process images of people
- **Data Protection**: Handle personal images responsibly and securely
- **Legal Compliance**: Follow local privacy laws and regulations

### Performance Tips
- **GPU Usage**: Enable CUDA for processing large datasets (100+ images)
- **Image Quality**: Higher resolution images provide better face recognition accuracy
- **Batch Size**: Process images in smaller batches for better memory management

### Limitations
- Requires clear, front-facing photos for best results
- Performance depends on image quality and lighting conditions
- May struggle with heavily edited or filtered images

## 🔍 Troubleshooting

### Common Issues

**1. "No module named 'face_recognition'"**
```bash
pip install face-recognition
```

**2. "No face found in the input image"**
- Ensure the reference image contains a clear, visible face
- Try using a different reference image
- Check image file format (supports JPG, PNG, etc.)

**3. CUDA Out of Memory**
```python
# Disable CUDA and use CPU
copy_images_with_face(..., use_cuda=False)
```

**4. Permission Denied Errors**
- Ensure you have read/write permissions for input and output folders
- Run with administrator privileges if necessary

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [face_recognition](https://github.com/ageitgey/face_recognition) library by Adam Geitgey
- [dlib](http://dlib.net/) library for face detection algorithms
- PyTorch team for GPU acceleration support

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/yourusername/face-recognition-sorter/issues) page
2. Create a new issue with detailed information
3. Include error messages and system information

---

**Made with ❤️ for efficient photo organization**
