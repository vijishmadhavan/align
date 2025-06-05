import os
import numpy as np
import cv2
import torch
from PIL import Image
import insightface
from insightface.app import FaceAnalysis
import warnings

class FaceAlignmentNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "target_image": ("IMAGE",),
                "alignment_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "detection_threshold": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 0.9, "step": 0.05}),
            },
            "optional": {
                "use_all_landmarks": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "align_faces"
    CATEGORY = "image/processing"

    def __init__(self):
        self.face_analyzer = None
        
    def initialize_face_analyzer(self):
        if self.face_analyzer is None:
            try:
                # Try to initialize with current API
                self.face_analyzer = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], 
                                                allowed_modules=['detection', 'landmark_2d_106'])
                self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
            except Exception as e:
                # Fallback to older or newer API if needed
                try:
                    print(f"First initialization attempt failed: {e}. Trying alternate initialization...")
                    self.face_analyzer = FaceAnalysis(name="buffalo_l")
                    self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
                except Exception as e2:
                    print(f"Failed to initialize face analyzer: {e2}")
                    self.face_analyzer = None
                    raise RuntimeError(f"Could not initialize InsightFace: {e2}")
        
    def align_faces(self, source_image, target_image, alignment_strength=1.0, detection_threshold=0.5, use_all_landmarks=False):
        self.initialize_face_analyzer()
        
        # Convert from ComfyUI tensor format to numpy images
        source_np = source_image[0].cpu().numpy() * 255
        source_np = source_np.astype(np.uint8)
        source_np = cv2.cvtColor(source_np, cv2.COLOR_RGB2BGR)
        
        target_np = target_image[0].cpu().numpy() * 255
        target_np = target_np.astype(np.uint8)
        target_np = cv2.cvtColor(target_np, cv2.COLOR_RGB2BGR)
        
        # Detect faces in both images
        try:
            source_faces = self.face_analyzer.get(source_np)
            target_faces = self.face_analyzer.get(target_np)
        except Exception as e:
            print(f"Face detection failed: {e}")
            return target_image
        
        # If no faces found in either image, return the original target image
        if len(source_faces) == 0 or len(target_faces) == 0:
            print("No faces detected in one or both images")
            return target_image
        
        # Get the largest face from each image (by bounding box area)
        source_face = max(source_faces, key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))
        target_face = max(target_faces, key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))
        
        # Get landmarks for alignment
        try:
            # Try to access landmarks with different attribute names (API may vary)
            if hasattr(source_face, 'landmark_2d_106'):
                source_landmarks = source_face.landmark_2d_106
                target_landmarks = target_face.landmark_2d_106
            elif hasattr(source_face, 'landmark_2d'):
                source_landmarks = source_face.landmark_2d
                target_landmarks = target_face.landmark_2d
            elif hasattr(source_face, 'kps'):
                # Some versions only provide 5-point landmarks
                source_landmarks = source_face.kps
                target_landmarks = target_face.kps
                use_all_landmarks = False  # Force use of all landmarks since we only have 5 points
            else:
                raise AttributeError("Could not find landmarks attribute in face detection")
        except Exception as e:
            print(f"Failed to extract landmarks: {e}")
            return target_image
        
        # Check if we're using 5-point or 106-point landmarks
        if len(source_landmarks) < 10:  # We have 5-point landmarks
            # Use standard 5 points (eyes, nose, mouth corners)
            left_eye_source = source_landmarks[0]
            right_eye_source = source_landmarks[1]
            left_eye_target = target_landmarks[0]
            right_eye_target = target_landmarks[1]
        else:  # We have many landmarks (68 or 106)
            if use_all_landmarks:
                # Use all landmarks for a more comprehensive alignment
                source_center = np.mean(source_landmarks, axis=0)
                target_center = np.mean(target_landmarks, axis=0)
                
                # Get all distances from center for scaling
                source_dists = np.linalg.norm(source_landmarks - source_center, axis=1)
                target_dists = np.linalg.norm(target_landmarks - target_center, axis=1)
                
                # Average scale ratio
                scale = np.mean(source_dists) / np.mean(target_dists) if np.mean(target_dists) > 0 else 1.0
                
                # Create transformation matrix based on centers
                rotation_matrix = cv2.getRotationMatrix2D(
                    tuple(target_center), 0, 1 + (scale - 1) * alignment_strength
                )
                
                # Add translation to match face positions
                translation = source_center - target_center
                rotation_matrix[0, 2] += translation[0] * alignment_strength
                rotation_matrix[1, 2] += translation[1] * alignment_strength
            else:
                # Use eye landmarks for alignment (we take the eye center points)
                try:
                    # Try to get eye landmarks (index may vary between API versions)
                    left_eye_source = np.mean(source_landmarks[60:68], axis=0)
                    right_eye_source = np.mean(source_landmarks[68:76], axis=0)
                    left_eye_target = np.mean(target_landmarks[60:68], axis=0)
                    right_eye_target = np.mean(target_landmarks[68:76], axis=0)
                except IndexError:
                    # Fallback to likely eye positions if the indices don't match
                    eye_indices = np.arange(len(source_landmarks) // 3, len(source_landmarks) // 2)
                    left_eye_source = np.mean(source_landmarks[eye_indices[:len(eye_indices)//2]], axis=0)
                    right_eye_source = np.mean(source_landmarks[eye_indices[len(eye_indices)//2:]], axis=0)
                    left_eye_target = np.mean(target_landmarks[eye_indices[:len(eye_indices)//2]], axis=0)
                    right_eye_target = np.mean(target_landmarks[eye_indices[len(eye_indices)//2:]], axis=0)
        
        # Only compute angles and rotation if we're not using all landmarks
        if not use_all_landmarks or len(source_landmarks) < 10:
            # Calculate angle and scale
            source_eye_vector = right_eye_source - left_eye_source
            target_eye_vector = right_eye_target - left_eye_target
            
            source_eye_distance = np.linalg.norm(source_eye_vector)
            target_eye_distance = np.linalg.norm(target_eye_vector)
            
            # Calculate angle
            source_angle = np.arctan2(source_eye_vector[1], source_eye_vector[0])
            target_angle = np.arctan2(target_eye_vector[1], target_eye_vector[0])
            angle_diff = (target_angle - source_angle) * 180 / np.pi
            
            # Calculate scale
            scale = source_eye_distance / target_eye_distance if target_eye_distance > 0 else 1.0
            
            # Calculate center points
            source_center = (left_eye_source + right_eye_source) / 2
            target_center = (left_eye_target + right_eye_target) / 2
            
            # Create transformation matrix
            rotation_matrix = cv2.getRotationMatrix2D(
                tuple(target_center), angle_diff * alignment_strength, 1 + (scale - 1) * alignment_strength
            )
            
            # Add translation to match face positions
            translation = source_center - target_center
            rotation_matrix[0, 2] += translation[0] * alignment_strength
            rotation_matrix[1, 2] += translation[1] * alignment_strength
        
        # Apply transformation
        h, w = target_np.shape[:2]
        aligned_target = cv2.warpAffine(target_np, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC)
        
        # Convert back to ComfyUI format
        aligned_target_rgb = cv2.cvtColor(aligned_target, cv2.COLOR_BGR2RGB)
        aligned_target_pil = Image.fromarray(aligned_target_rgb)
        
        # Convert to tensor
        aligned_tensor = torch.from_numpy(np.array(aligned_target_pil).astype(np.float32) / 255.0).unsqueeze(0)
        
        return (aligned_tensor,)

# Register the node in ComfyUI
NODE_CLASS_MAPPINGS = {
    "FaceAlignment": FaceAlignmentNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceAlignment": "Face Alignment (InsightFace)"
} 