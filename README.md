# Face Alignment Node for ComfyUI

This custom node for ComfyUI uses InsightFace to align the facial features in your end image to match those in your start image. It's particularly useful for workflows like Wan 2.1 Fun InP Video where you need to align faces between frames.

## Installation

### Option 1: Using ComfyUI Manager (Recommended)

1. Open ComfyUI and make sure you have [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) installed
2. Go to the "Manager" tab in ComfyUI
3. Click on "Install Custom Nodes"
4. Search for "Face Alignment" or "InsightFace"
5. Click Install
6. Restart ComfyUI when prompted

### Option 2: Manual Installation

1. Place these files in a directory within your ComfyUI installation
2. Run the installation script:
   ```
   python install_face_alignment.py
   ```
3. Restart ComfyUI

## Usage

After installation, a new node called "Face Alignment (InsightFace)" will be available in the "image/processing" category of the node menu.

### Node Inputs

- **source_image**: The reference image (usually your start frame)
- **target_image**: The image to be aligned (usually your end frame)
- **alignment_strength**: Controls how strongly to align (0.0 = no alignment, 1.0 = full alignment)
- **detection_threshold**: Confidence threshold for face detection
- **use_all_landmarks** (optional): When checked, uses all facial landmarks for alignment instead of just eyes

### Node Output

- **IMAGE**: The aligned target image

## Integration with Wan 2.1 Fun InP Video workflow

1. Load your workflow
2. Add the Face Alignment node
3. Connect your Start_image to the "source_image" input
4. Connect your End_image to the "target_image" input
5. Connect the output of the Face Alignment node to the "end_image" input of the WanFunInpaintToVideo node
6. Adjust the alignment_strength parameter as needed

## Compatibility

This node is designed to work with the latest version of InsightFace and includes multiple fallback mechanisms to handle API changes. It supports:

- Both 5-point and 106-point landmark detection
- Multiple initialization methods for different InsightFace versions
- Automatic adaptation to available face analysis models

## Requirements

- InsightFace (latest version)
- OpenCV
- PyTorch
- NumPy

These dependencies will be installed automatically by ComfyUI Manager or the installation script. 