"""
MRI Scan Validator

Validates whether an uploaded image is actually an MRI scan before classification.
Uses multiple heuristics to determine if an image is medical imaging.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2


class MRIValidator:
    """
    Validates if an image is likely to be an MRI scan using multiple criteria:
    1. Grayscale/low color variance (MRI scans are typically grayscale)
    2. Specific contrast patterns
    3. Absence of complex textures (natural images have more complex textures)
    4. Edge density analysis
    5. Confidence threshold from the model
    """
    
    def __init__(self, confidence_threshold=0.85, strict_mode=True):
        """
        Initialize the MRI validator
        
        Args:
            confidence_threshold: Minimum confidence to accept prediction (0-1)
            strict_mode: If True, apply stricter validation criteria
        """
        self.confidence_threshold = confidence_threshold
        self.strict_mode = strict_mode
    
    def is_grayscale_like(self, image):
        """
        Check if image is grayscale or has very low color variance
        MRI scans are typically grayscale or near-grayscale
        """
        img_array = np.array(image)
        
        # If already grayscale
        if len(img_array.shape) == 2:
            return True
        
        # Check if RGB channels are similar (grayscale-like)
        if img_array.shape[2] == 3:
            r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
            
            # Calculate variance between channels
            rg_diff = np.abs(r.astype(float) - g.astype(float)).mean()
            rb_diff = np.abs(r.astype(float) - b.astype(float)).mean()
            gb_diff = np.abs(g.astype(float) - b.astype(float)).mean()
            
            avg_diff = (rg_diff + rb_diff + gb_diff) / 3
            
            # MRI scans should have very low color variance
            # Threshold: < 5 for grayscale, < 15 for near-grayscale
            threshold = 5 if self.strict_mode else 15
            return avg_diff < threshold
        
        return False
    
    def check_medical_imaging_characteristics(self, image):
        """
        Check for characteristics typical of medical imaging:
        - Relatively uniform background (dark)
        - Central bright region (brain tissue)
        - Limited color palette
        - Specific intensity distribution
        """
        img_array = np.array(image.convert('L'))  # Convert to grayscale
        
        # Normalize
        img_array = img_array.astype(float) / 255.0
        
        # Check intensity distribution
        hist, bins = np.histogram(img_array.flatten(), bins=50, range=(0, 1))
        
        # MRI scans typically have:
        # 1. High concentration of dark pixels (background)
        # 2. Some bright pixels (brain tissue)
        # 3. Bimodal or specific distribution
        
        dark_pixels = np.sum(hist[:10]) / np.sum(hist)  # First 20% of histogram
        bright_pixels = np.sum(hist[20:]) / np.sum(hist)  # Bright regions
        
        # Medical images typically have 40-70% dark background
        has_dark_background = 0.30 < dark_pixels < 0.80
        
        # Check for central brightness (brain region)
        h, w = img_array.shape
        center_region = img_array[h//4:3*h//4, w//4:3*w//4]
        center_brightness = center_region.mean()
        edge_brightness = (
            img_array[:h//4, :].mean() + 
            img_array[3*h//4:, :].mean() + 
            img_array[:, :w//4].mean() + 
            img_array[:, 3*w//4:].mean()
        ) / 4
        
        # Center should be significantly brighter than edges
        center_is_brighter = center_brightness > edge_brightness * 1.2
        
        return has_dark_background and center_is_brighter
    
    def check_edge_density(self, image):
        """
        Check edge density - medical images have specific edge patterns
        """
        img_array = np.array(image.convert('L'))
        
        # Apply Canny edge detection
        edges = cv2.Canny(img_array, 50, 150)
        
        # Calculate edge density
        edge_density = np.sum(edges > 0) / edges.size
        
        # Medical images typically have 0.05-0.20 edge density
        # Natural images often have higher edge density
        is_valid = 0.03 < edge_density < 0.25
        
        return is_valid, edge_density
    
    def check_image_size_ratio(self, image):
        """
        Check if image has reasonable aspect ratio for MRI scans
        """
        width, height = image.size
        ratio = max(width, height) / min(width, height)
        
        # MRI scans are typically square or near-square
        # Allow up to 1.5:1 ratio
        return ratio < 1.8
    
    def check_image_quality(self, image):
        """
        Check if image has sufficient quality (not too small/blurry)
        """
        width, height = image.size
        
        # Minimum size check (at least 100x100)
        if width < 100 or height < 100:
            return False, "Image too small (minimum 100x100 pixels)"
        
        # Check for blur (using Laplacian variance)
        img_array = np.array(image.convert('L'))
        laplacian_var = cv2.Laplacian(img_array, cv2.CV_64F).var()
        
        # Very low variance indicates severe blur
        if laplacian_var < 10:
            return False, "Image appears too blurry"
        
        return True, "OK"
    
    def validate_with_model_confidence(self, probabilities, predicted_class):
        """
        Validate based on model's confidence
        
        If model is very uncertain (low max probability), it might not be an MRI
        """
        max_prob = np.max(probabilities)
        
        # If confidence is very low, likely not an MRI
        if max_prob < self.confidence_threshold:
            return False, f"Low confidence ({max_prob*100:.1f}%). Image may not be an MRI scan."
        
        return True, f"Confidence: {max_prob*100:.1f}%"
    
    def validate_image(self, image, model_output=None):
        """
        Main validation function that runs all checks
        
        Args:
            image: PIL Image object
            model_output: Optional tuple of (probabilities, predicted_class) from model
        
        Returns:
            is_valid: Boolean indicating if image appears to be an MRI
            confidence_score: Float 0-1 indicating confidence in validation
            messages: List of validation messages
        """
        messages = []
        scores = []
        
        # 1. Check if grayscale-like
        is_grayscale = self.is_grayscale_like(image)
        if is_grayscale:
            messages.append("✓ Image is grayscale/near-grayscale (typical for MRI)")
            scores.append(1.0)
        else:
            messages.append("✗ Image has high color variance (unusual for MRI)")
            scores.append(0.0)
        
        # 2. Check medical imaging characteristics
        has_medical_chars = self.check_medical_imaging_characteristics(image)
        if has_medical_chars:
            messages.append("✓ Image has medical imaging characteristics")
            scores.append(1.0)
        else:
            messages.append("✗ Image lacks typical medical imaging patterns")
            scores.append(0.0)
        
        # 3. Check edge density
        valid_edges, edge_density = self.check_edge_density(image)
        if valid_edges:
            messages.append(f"✓ Edge density appropriate ({edge_density:.3f})")
            scores.append(1.0)
        else:
            messages.append(f"✗ Unusual edge density ({edge_density:.3f})")
            scores.append(0.3)  # Not a complete fail
        
        # 4. Check aspect ratio
        valid_ratio = self.check_image_size_ratio(image)
        if valid_ratio:
            messages.append("✓ Aspect ratio appropriate for MRI")
            scores.append(1.0)
        else:
            messages.append("⚠ Unusual aspect ratio for MRI scan")
            scores.append(0.5)
        
        # 5. Check image quality
        quality_ok, quality_msg = self.check_image_quality(image)
        if quality_ok:
            messages.append(f"✓ Image quality sufficient")
            scores.append(1.0)
        else:
            messages.append(f"✗ {quality_msg}")
            scores.append(0.0)
        
        # 6. Check model confidence (if provided)
        if model_output is not None:
            probabilities, predicted_class = model_output
            conf_valid, conf_msg = self.validate_with_model_confidence(
                probabilities, predicted_class
            )
            if conf_valid:
                messages.append(f"✓ {conf_msg}")
                scores.append(1.0)
            else:
                messages.append(f"✗ {conf_msg}")
                scores.append(0.0)
        
        # Calculate overall confidence
        avg_score = np.mean(scores)
        
        # Determine if valid based on score
        if self.strict_mode:
            # In strict mode, require 80% of checks to pass
            is_valid = avg_score >= 0.80
        else:
            # In relaxed mode, require 60% of checks to pass
            is_valid = avg_score >= 0.60
        
        return is_valid, avg_score, messages
    
    def get_validation_summary(self, is_valid, confidence_score, messages):
        """
        Generate a human-readable validation summary
        """
        status = "✅ VALID MRI SCAN" if is_valid else "❌ NOT AN MRI SCAN"
        confidence_pct = confidence_score * 100
        
        summary = f"{status} (Confidence: {confidence_pct:.1f}%)\n\n"
        summary += "Validation Checks:\n"
        summary += "\n".join(f"  {msg}" for msg in messages)
        
        if not is_valid:
            summary += "\n\n⚠️  WARNING: This image does not appear to be an MRI scan."
            summary += "\n   Please upload a brain MRI scan image for accurate classification."
        
        return summary


# Quick validation function for easy integration
def quick_validate_mri(image, confidence_threshold=0.85, strict=True):
    """
    Quick validation function
    
    Args:
        image: PIL Image
        confidence_threshold: Minimum confidence threshold
        strict: Use strict validation mode
    
    Returns:
        is_valid, confidence_score, summary_message
    """
    validator = MRIValidator(confidence_threshold=confidence_threshold, strict_mode=strict)
    is_valid, score, messages = validator.validate_image(image)
    summary = validator.get_validation_summary(is_valid, score, messages)
    
    return is_valid, score, summary


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python mri_validator.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    image = Image.open(image_path).convert('RGB')
    
    print(f"Validating image: {image_path}")
    print("=" * 60)
    
    is_valid, score, summary = quick_validate_mri(image, strict=True)
    print(summary)
    print("=" * 60)
    
    if is_valid:
        print("\n✅ Image validated. Safe to proceed with classification.")
    else:
        print("\n❌ Image validation failed. Do not proceed with classification.")

