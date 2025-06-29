# 🚲 Bike Detection System

A comprehensive bike detection and tracking system using YOLO and DeepSORT, built with Streamlit for an interactive web interface.

## 📋 Table of Contents

- [Features](#-features)
- [Demo](#-demo)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Model Information](#-model-information)
- [Configuration](#-configuration)
- [API Reference](#-api-reference)
- [Contributing](#-contributing)
- [License](#-license)
- [Troubleshooting](#-troubleshooting)

## ✨ Features

### 🎯 Core Detection Capabilities
- **Multi-brand bike detection**: Supports Didi, HelloRide, and Meituan bikes
- **Real-time object tracking**: Uses DeepSORT algorithm for consistent tracking across frames
- **Confidence-based filtering**: Adjustable confidence thresholds for detection accuracy
- **ROI (Region of Interest) support**: Focus detection on specific areas of the video

### 📊 Analytics & Visualization
- **Live statistics**: Real-time metrics during video processing
- **Brand distribution charts**: Interactive pie and bar charts using Plotly
- **Detection history**: Comprehensive tracking of all detected objects
- **Performance metrics**: Processing speed and efficiency monitoring

### 🎮 Interactive Interface
- **Streamlit web app**: User-friendly interface for video upload and processing
- **Customizable parameters**: Adjust confidence thresholds, tracking settings, and alerts
- **Progress tracking**: Real-time progress bars and status updates
- **Report generation**: Downloadable detailed analysis reports

### 🚨 Smart Features
- **Alert system**: Notifications when bike counts exceed thresholds
- **Video export**: Processed videos with detection overlays
- **Batch processing**: Handle multiple video formats (MP4, MOV, AVI, MKV)

## 🎬 Demo

### Input Video Processing
1. Upload your video file through the web interface
2. Adjust detection parameters in the sidebar
3. Click "Start Detection" to begin processing
4. Monitor real-time progress and statistics

### Output Features
- Annotated video with bounding boxes and tracking IDs
- Color-coded detection by bike brand:
  - 🟠 **Didi**: Orange boxes
  - 🟢 **HelloRide**: Green boxes
  - 🟡 **Meituan**: Yellow boxes
- Frame-by-frame statistics overlay
- Downloadable analysis report

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster processing)
- FFmpeg (for video processing)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Msumali/china-shared-bike-detection-and-tracking-application.git
cd bike-detection-system
```

### Step 2: Create Virtual Environment
```bash
python -m venv bike_env
source bike_env/bin/activate  # On Windows: bike_env\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Model
Place your trained YOLO model (`best.pt`) in the `models/` directory:
```
models/
└── best.pt
```

### Step 5: Create Required Directories
```bash
mkdir -p outputs temp
```

## 📦 Requirements

Create a `requirements.txt` file with the following dependencies:

```txt
altair>=5.5.0
attrs>=25.3.0
blinker>=1.9.0
cachetools>=6.1.0
certifi>=2025.6.15
charset-normalizer>=3.4.2
choreographer>=1.0.9
click>=8.2.1
colorama>=0.4.6
contourpy>=1.3.2
cycler>=0.12.1
deep-sort-realtime>=1.3.2
filelock>=3.18.0
fonttools>=4.58.4
fsspec>=2025.5.1
gitdb>=4.0.12
GitPython>=3.1.44
idna>=3.10
Jinja2>=3.1.6
jsonschema>=4.24.0
jsonschema-specifications>=2025.4.1
kaleido>=1.0.0
kiwisolver>=1.4.8
logistro>=1.1.0
MarkupSafe>=3.0.2
matplotlib>=3.10.3
mpmath>=1.3.0
narwhals>=1.44.0
networkx>=3.5
numpy>=2.3.1
opencv-python>=4.11.0.86
orjson>=3.10.18
packaging>=25.0
pandas>=2.3.0
pillow>=11.2.1
plotly>=6.2.0
protobuf>=6.31.1
psutil>=7.0.0
py-cpuinfo>=9.0.0
pyarrow>=20.0.0
pydeck>=0.9.1
pyparsing>=3.2.3
python-dateutil>=2.9.0.post0
pytz>=2025.2
PyYAML>=6.0.2
referencing>=0.36.2
reportlab>=4.4.2
requests>=2.32.4
rpds-py>=0.25.1
scipy>=1.16.0
setuptools>=80.9.0
simplejson>=3.20.1
six>=1.17.0
smmap>=5.0.2
streamlit>=1.46.1
sympy>=1.14.0
tenacity>=9.1.2
toml>=0.10.2
torch>=2.7.1
torchvision>=0.22.1
tornado>=6.5.1
tqdm>=4.67.1
typing_extensions>=4.14.0
tzdata>=2025.2
ultralytics>=8.3.160
ultralytics-thop>=2.0.14
urllib3>=2.5.0
watchdog>=6.0.0

```

## 💻 Usage

### Web Interface (Recommended)
1. **Start the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Access the interface:**
   Open your browser and navigate to `http://localhost:8501`

3. **Process a video:**
   - Upload a video file (MP4, MOV, AVI, MKV)
   - Adjust settings in the sidebar:
     - Confidence threshold (0.1 - 1.0)
     - Enable/disable object tracking
     - Set alert thresholds
     - Configure ROI if needed
   - Click "Start Detection"
   - Monitor progress and view results

### Command Line Interface
```python
from detect import detect_bikes_from_video

# Basic usage
results = detect_bikes_from_video(
    video_path="input_video.mp4",
    output_path="outputs/processed_video.mp4",
    confidence_threshold=0.3,
    use_tracking=True
)

# With progress callback
def progress_callback(current_frame, total_frames):
    progress = (current_frame / total_frames) * 100
    print(f"Progress: {progress:.1f}%")

results = detect_bikes_from_video(
    "input_video.mp4",
    progress_callback=progress_callback
)
```

## 📁 Project Structure

```
bike-detection-system/
├── app.py                  # Streamlit web application
├── detect.py              # Core detection and tracking logic
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .gitignore           # Git ignore rules
├── models/              # Model files
│   └── best.pt         # YOLO model weights
├── outputs/            # Processed videos
├── temp/              # Temporary uploaded files
├── sample_videos/     # Example input videos

```

## 🤖 Model Information

### YOLO Configuration
- **Model Type**: YOLOv8 (Ultralytics)
- **Classes**: 3 bike brands (Didi, HelloRide, Meituan)
- **Input Resolution**: Configurable (default: 640x640)
- **Model File**: `models/best.pt`

### Tracking Algorithm
- **Algorithm**: DeepSORT
- **Features**: 
  - Persistent tracking across frames
  - Handling of occlusions
  - Unique ID assignment
  - Track lifecycle management

### Performance Metrics
- **Detection Accuracy**: Depends on model training quality
- **Processing Speed**: ~15-30 FPS on GPU, ~5-10 FPS on CPU
- **Memory Usage**: ~2-4GB GPU memory for typical videos

## ⚙️ Configuration

### Detection Parameters
```python
# Confidence threshold for detections
confidence_threshold = 0.3  # Range: 0.1 - 1.0

# Enable object tracking
use_tracking = True

# Process every nth frame (for speed optimization)
process_every_nth_frame = 1

# Alert threshold
alert_threshold = 10
```

### Video Processing Settings
```python
# Output video codec options
codecs = ['avc1', 'mp4v', 'XVID', 'MJPG']

# ROI coordinates (x1, y1, x2, y2)
roi_coordinates = (0, 0, 640, 480)
```

### Color Mapping
```python
BRAND_COLORS = {
    'Didi': (255, 165, 0),      # Orange
    'HelloRide': (0, 255, 0),   # Green  
    'Meituan': (255, 255, 0)    # Yellow
}
```

## 📊 API Reference

### BikeDetector Class

```python
class BikeDetector:
    def __init__(self, confidence_threshold=0.3, use_tracking=True):
        """Initialize the bike detector"""
        
    def detect_and_track(self, frame, frame_number):
        """Detect bikes and track them across frames"""
        
    def draw_detections(self, frame, detections):
        """Draw bounding boxes with enhanced information"""
        
    def get_statistics(self):
        """Get current detection statistics"""
```

### Main Detection Function

```python
def detect_bikes_from_video(
    video_path,
    output_path="outputs/processed_video.mp4",
    confidence_threshold=0.3,
    use_tracking=True,
    progress_callback=None
):
    """
    Process video and detect bikes
    
    Args:
        video_path (str): Path to input video
        output_path (str): Path for output video
        confidence_threshold (float): Detection confidence threshold
        use_tracking (bool): Enable object tracking
        progress_callback (function): Callback for progress updates
    
    Returns:
        dict: Detection results and statistics
    """
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Model Not Found
```
Error: [Errno 2] No such file or directory: 'models/best.pt'
```
**Solution**: Ensure the YOLO model file is placed in the `models/` directory.

#### 2. DeepSORT Import Error
```
ImportError: No module named 'deep_sort_realtime'
```
**Solution**: Install the tracking dependency:
```bash
pip install deep-sort-realtime
```

#### 3. Video Codec Issues
```
Error: Could not initialize video writer
```
**Solution**: The system will automatically try different codecs. Ensure FFmpeg is installed.

#### 4. CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**: 
- Reduce video resolution
- Process fewer frames simultaneously
- Use CPU processing: `device='cpu'`

#### 5. Slow Processing
**Solutions**:
- Use GPU acceleration
- Reduce confidence threshold
- Process every nth frame
- Reduce video resolution

### Performance Optimization

1. **GPU Acceleration**:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Batch Processing**:
   - Process multiple videos sequentially
   - Use multiprocessing for parallel processing

3. **Memory Management**:
   - Clear cache between videos
   - Monitor memory usage

## 🤝 Contributing

Contributions are welcome! If you'd like to contribute to this project:

Fork the repository
Create a feature branch for your changes
Make your changes and test them
Submit a pull request with a clear description

Please ensure your code follows the existing style and includes appropriate documentation.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 .
black .
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints where applicable
- Add docstrings for functions and classes
- Keep functions focused and modular

## 📄 License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics** for the YOLO implementation
- **DeepSORT** for object tracking capabilities
- **Streamlit** for the web interface framework
- **OpenCV** for computer vision operations
- **Plotly** for interactive visualizations

## 📞 Support

For support and questions:

- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions for general questions
- **Email**: pmsumali@gmail.com

## 🔮 Future Enhancements

- [ ] Real-time camera feed processing
- [ ] Mobile app integration
- [ ] Advanced analytics dashboard
- [ ] Multi-camera support
- [ ] Cloud deployment options
- [ ] API endpoints for integration
- [ ] Enhanced model training pipeline
- [ ] Database integration for historical data

---

**Made with ❤️ for bike sharing analytics**
