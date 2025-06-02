#!/usr/bin/env python3
"""
Sticky Note Detection and OCR Script
Detects colored sticky notes, extracts text, and creates overlays
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import os
import json
from datetime import datetime
import argparse
import base64
import requests
from typing import Optional, Tuple

class StickyNoteDetector:
    def __init__(self, image_path, debug=False):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.original_image = self.image.copy()
        self.height, self.width = self.image.shape[:2]
        self.debug = debug
        
        # Improved color ranges for sticky notes (HSV) - more precise
        self.color_ranges = {
            'yellow': {
                'lower': np.array([22, 100, 120]),  # More saturated yellow only
                'upper': np.array([30, 255, 255]),
                'color': (0, 255, 255)  # Yellow in BGR
            },
            'light_yellow': {
                'lower': np.array([20, 30, 150]),   # Low saturation yellow
                'upper': np.array([30, 100, 255]),  # Cap at 100 saturation to avoid overlap with yellow
                'color': (100, 255, 255)  # Light Yellow in BGR
            },
            'orange': {
                'lower': np.array([8, 120, 100]),   # More saturated orange, narrower range
                'upper': np.array([18, 255, 255]),  # Reduced upper bound to avoid red overlap
                'color': (0, 165, 255)  # Orange in BGR
            },
            'red': {
                'lower': np.array([0, 140, 50]),    # Higher saturation to avoid pink overlap
                'upper': np.array([6, 255, 255]),   # Reduced upper bound to avoid pink
                'color': (0, 0, 255)  # Red in BGR
            },
            'red_high': {
                'lower': np.array([174, 140, 50]),  # Higher saturation, higher hue start
                'upper': np.array([180, 255, 255]),
                'color': (0, 0, 255)  # Red in BGR
            },
            'pink': {
                'lower': np.array([155, 60, 80]),   # Higher hue start, higher saturation and value
                'upper': np.array([175, 255, 255]), # Broader range for pink detection
                'color': (255, 0, 255)  # Magenta in BGR
            },
            'green': {
                'lower': np.array([45, 60, 50]),    # Narrower green range, higher saturation
                'upper': np.array([75, 255, 255]),  # Reduced upper bound to avoid cyan
                'color': (0, 255, 0)  # Green in BGR
            },
            'light_green': {
                'lower': np.array([40, 30, 150]),   # Low saturation green
                'upper': np.array([75, 60, 255]),   # Cap at 60 saturation, match green hue range
                'color': (150, 255, 150)  # Light green in BGR
            },
            'cyan': {
                'lower': np.array([80, 60, 50]),    # Higher hue start, higher saturation
                'upper': np.array([100, 255, 255]), # Broader range for cyan detection
                'color': (255, 255, 0)  # Cyan in BGR
            },
            'blue': {
                'lower': np.array([105, 100, 50]),  # Higher hue start to avoid cyan overlap
                'upper': np.array([130, 255, 255]),
                'color': (255, 100, 0)  # Blue in BGR
            },
            'light_blue': {
                'lower': np.array([100, 30, 150]),  # Low saturation blue
                'upper': np.array([130, 100, 255]), # Cap at 100 saturation
                'color': (255, 200, 150)  # Light blue in BGR
            },
            'purple': {
                'lower': np.array([130, 50, 50]),   # Purple/violet range
                'upper': np.array([150, 255, 255]),
                'color': (255, 0, 128)  # Purple in BGR
            },
            'light_pink': {
                'lower': np.array([150, 20, 150]),  # Very light pink, different hue from pink
                'upper': np.array([160, 50, 255]),  # Lower saturation than regular pink
                'color': (255, 150, 255)  # Light pink in BGR
            }
        }
        
        self.detected_notes = []
        self.debug_info = {
            'total_contours': 0,
            'filtered_by_area': 0,
            'filtered_by_shape': 0,
            'filtered_by_confidence': 0,
            'pre_merge_count': 0,
            'post_merge_count': 0
        }
        
    def preprocess_image(self):
        """Apply preprocessing to improve detection"""
        # Convert to HSV for better color detection
        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        # Apply bilateral filter to reduce noise while preserving edges
        self.blurred = cv2.bilateralFilter(self.image, 9, 75, 75)
        
        # Enhance contrast
        lab = cv2.cvtColor(self.blurred, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        self.enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Edge detection for shape validation
        gray = cv2.cvtColor(self.enhanced, cv2.COLOR_BGR2GRAY)
        self.edges = cv2.Canny(gray, 50, 150)
        
    def validate_sticky_note_shape(self, contour, bbox):
        """Validate if a contour is likely a sticky note based on shape characteristics"""
        x, y, w, h = bbox
        
        # Calculate shape metrics
        area = cv2.contourArea(contour)
        rect_area = w * h
        extent = area / rect_area if rect_area > 0 else 0
        
        # Aspect ratio check (sticky notes are roughly square)
        aspect_ratio = w / h if h > 0 else 0
        
        # Perimeter check
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Approximate polygon
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Validation criteria
        is_valid = True
        validation_scores = {}
        rejection_reasons = []
        
        # 1. Minimum area (adaptive based on image size) - MUCH MORE RELAXED
        min_area_ratio = 0.00002  # 0.002% of image area (was 0.01%)
        min_area = self.width * self.height * min_area_ratio
        validation_scores['area'] = area >= min_area
        if area < min_area:
            is_valid = False
            rejection_reasons.append(f"area too small: {area:.0f} < {min_area:.0f}")
            
        # 2. Maximum area (to avoid detecting entire image sections)
        max_area_ratio = 0.25  # 25% of image area (was 20%)
        max_area = self.width * self.height * max_area_ratio
        validation_scores['max_area'] = area <= max_area
        if area > max_area:
            is_valid = False
            rejection_reasons.append(f"area too large: {area:.0f} > {max_area:.0f}")
            
        # 3. Aspect ratio (sticky notes are somewhat square) - MUCH MORE RELAXED
        validation_scores['aspect_ratio'] = 0.2 < aspect_ratio < 5.0  # (was 0.3-3.5)
        if not (0.2 < aspect_ratio < 5.0):
            is_valid = False
            rejection_reasons.append(f"bad aspect ratio: {aspect_ratio:.2f}")
            
        # 4. Extent (how much the contour fills its bounding box) - MUCH MORE RELAXED
        validation_scores['extent'] = extent > 0.3  # (was 0.5)
        if extent < 0.3:
            is_valid = False
            rejection_reasons.append(f"low extent: {extent:.2f}")
            
        # 5. Rectangularity (4-6 vertices for a rectangle-like shape) - MORE RELAXED
        validation_scores['vertices'] = 3 <= len(approx) <= 20  # (was 3-12)
        if not (3 <= len(approx) <= 20):
            is_valid = False
            rejection_reasons.append(f"bad vertex count: {len(approx)}")
            
        # 6. Minimum dimensions - MUCH MORE RELAXED
        min_dimension = 20  # pixels (was 30)
        validation_scores['min_dimension'] = w >= min_dimension and h >= min_dimension
        if w < min_dimension or h < min_dimension:
            is_valid = False
            rejection_reasons.append(f"too small: {w}x{h}")
        
        if self.debug and not is_valid:
            self.debug_info['filtered_by_shape'] += 1
            if self.debug and len(rejection_reasons) > 0:
                print(f"  Rejected: {', '.join(rejection_reasons)}")
            
        return is_valid, validation_scores
        
    def detect_colored_regions(self, color_name):
        """Detect regions of specific color with improved filtering"""
        hsv_enhanced = cv2.cvtColor(self.enhanced, cv2.COLOR_BGR2HSV)
        
        # Special handling for red which wraps around in HSV
        if color_name == 'red':
            # Red wraps around in HSV, so we need to check both low and high hue ranges
            mask_low = cv2.inRange(hsv_enhanced, 
                                  self.color_ranges['red']['lower'],
                                  self.color_ranges['red']['upper'])
            mask_high = cv2.inRange(hsv_enhanced, 
                                   self.color_ranges['red_high']['lower'],
                                   self.color_ranges['red_high']['upper'])
            mask = cv2.bitwise_or(mask_low, mask_high)
        elif color_name == 'red_high':
            # Skip red_high as it's handled by red
            return []
        else:
            # Create mask for the color
            mask = cv2.inRange(hsv_enhanced, 
                              self.color_ranges[color_name]['lower'],
                              self.color_ranges[color_name]['upper'])
        
        # Morphological operations to clean up the mask
        kernel_small = np.ones((3,3), np.uint8)
        kernel_large = np.ones((5,5), np.uint8)  # Reduced from 7x7
        
        # Remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        # Fill small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)
        
        # Apply edge information to refine mask
        mask_edges = cv2.bitwise_and(mask, self.edges)
        mask = cv2.bitwise_or(mask, mask_edges)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if self.debug:
            print(f"\nColor {color_name}: found {len(contours)} contours")
            self.debug_info['total_contours'] += len(contours)
        
        regions = []
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Validate shape
            is_valid, scores = self.validate_sticky_note_shape(contour, (x, y, w, h))
            
            if is_valid:
                regions.append({
                    'bbox': (x, y, w, h),
                    'area': cv2.contourArea(contour),
                    'contour': contour,
                    'color': color_name,
                    'validation_scores': scores
                })
        
        if self.debug:
            print(f"  Valid regions: {len(regions)}")
        
        return regions
    
    def merge_overlapping_regions(self, regions, overlap_threshold=0.3):
        """Merge overlapping or nearby regions (for stacked notes)"""
        if not regions:
            return regions
        
        # Define which colors are similar and should be merged
        similar_colors = {
            'yellow': ['light_yellow'],
            'light_yellow': ['yellow'],
            'blue': ['light_blue'],
            'light_blue': ['blue'],
            'green': ['light_green'],
            'light_green': ['green'],
            'pink': ['light_pink'],  # Removed red overlap since ranges are now separated
            'light_pink': ['pink'],
            'red': ['red_high'],  # Only merge red variants, not pink
            'red_high': ['red'],
            'orange': [],  # Orange is distinct
            'purple': [],  # Purple is distinct
            'cyan': []     # Cyan is distinct
        }
            
        merged = []
        used = set()
        
        for i, region1 in enumerate(regions):
            if i in used:
                continue
                
            x1, y1, w1, h1 = region1['bbox']
            merged_bbox = [x1, y1, x1 + w1, y1 + h1]
            group = [region1]
            used.add(i)
            
            for j, region2 in enumerate(regions[i+1:], i+1):
                if j in used:
                    continue
                    
                x2, y2, w2, h2 = region2['bbox']
                
                # Calculate IoU (Intersection over Union)
                x_left = max(x1, x2)
                y_top = max(y1, y2)
                x_right = min(x1 + w1, x2 + w2)
                y_bottom = min(y1 + h1, y2 + h2)
                
                if x_right > x_left and y_bottom > y_top:
                    # Calculate intersection area
                    intersection_area = (x_right - x_left) * (y_bottom - y_top)
                    
                    # Calculate union area
                    union_area = region1['area'] + region2['area'] - intersection_area
                    
                    # IoU ratio
                    iou = intersection_area / union_area if union_area > 0 else 0
                    
                    # Also check if one region is mostly contained in another
                    containment1 = intersection_area / region1['area'] if region1['area'] > 0 else 0
                    containment2 = intersection_area / region2['area'] if region2['area'] > 0 else 0
                    
                    # Check if colors are the same or similar
                    color1 = region1['color']
                    color2 = region2['color']
                    colors_match = (color1 == color2) or (color2 in similar_colors.get(color1, []))
                    
                    # Only merge if high overlap or one is contained in the other
                    # AND colors are the same or similar
                    if colors_match and (iou > overlap_threshold or containment1 > 0.9 or containment2 > 0.9):
                        # Merge regions
                        merged_bbox[0] = min(merged_bbox[0], x2)
                        merged_bbox[1] = min(merged_bbox[1], y2)
                        merged_bbox[2] = max(merged_bbox[2], x2 + w2)
                        merged_bbox[3] = max(merged_bbox[3], y2 + h2)
                        group.append(region2)
                        used.add(j)
            
            # Create merged region - use the primary (non-light) color if available
            primary_color = group[0]['color']
            for region in group:
                if not region['color'].startswith('light_') and region['color'] != 'red_high':
                    primary_color = region['color']
                    break
            
            x, y, x2, y2 = merged_bbox
            merged_region = {
                'bbox': (x, y, x2 - x, y2 - y),
                'area': (x2 - x) * (y2 - y),
                'color': primary_color,
                'sub_regions': len(group),
                'confidence': self.calculate_region_confidence(group)
            }
            merged.append(merged_region)
            
        return merged
    
    def calculate_region_confidence(self, regions):
        """Calculate confidence score for a region based on validation scores"""
        if not regions:
            return 0
            
        # Average the validation scores from all sub-regions
        total_score = 0
        score_count = 0
        
        for region in regions:
            if 'validation_scores' in region:
                scores = region['validation_scores']
                # Weight different criteria
                weights = {
                    'area': 1.0,
                    'max_area': 1.0,
                    'aspect_ratio': 2.0,  # More important
                    'extent': 1.5,
                    'vertices': 1.5,
                    'min_dimension': 1.0
                }
                
                for criterion, passed in scores.items():
                    weight = weights.get(criterion, 1.0)
                    total_score += (1.0 if passed else 0.0) * weight
                    score_count += weight
                    
        return (total_score / score_count * 100) if score_count > 0 else 0
    
    def filter_by_confidence(self, regions, min_confidence=60):
        """Filter regions by confidence score"""
        filtered = []
        for region in regions:
            if 'confidence' in region:
                if region['confidence'] >= min_confidence:
                    filtered.append(region)
                elif self.debug:
                    self.debug_info['filtered_by_confidence'] += 1
                    print(f"  Filtered out region with confidence {region['confidence']:.1f}%")
        return filtered
    
    def filter_by_relative_size(self, regions, size_ratio_threshold=0.3):
        """Filter out regions that are too small compared to the median size"""
        if not regions:
            return regions
            
        # Calculate areas and find median
        areas = [r['area'] for r in regions]
        areas.sort()
        median_area = areas[len(areas) // 2] if areas else 0
        
        # Calculate the 30th percentile for a more robust lower bound
        percentile_30_idx = int(len(areas) * 0.3)
        percentile_30_area = areas[percentile_30_idx] if percentile_30_idx < len(areas) else median_area
        
        # Use a combination of percentile and ratio thresholds
        # Minimum should be at least 10% of median or 70% of 30th percentile
        min_area_threshold = max(
            median_area * size_ratio_threshold,
            percentile_30_area * 0.7
        )
        
        # Also set an absolute minimum based on image size
        # A real sticky note should be at least 0.001% of the image (10x higher than shape validation)
        absolute_min = self.width * self.height * 0.001
        min_area_threshold = max(min_area_threshold, absolute_min)
        
        filtered = []
        for region in regions:
            if region['area'] >= min_area_threshold:
                filtered.append(region)
            elif self.debug:
                print(f"  Filtered by relative size: area {region['area']:.0f} < threshold {min_area_threshold:.0f} (median: {median_area:.0f})")
                
        if self.debug and len(filtered) < len(regions):
            print(f"\nSize filtering: removed {len(regions) - len(filtered)} tiny regions")
            print(f"  Median area: {median_area:.0f}")
            print(f"  30th percentile: {percentile_30_area:.0f}")
            print(f"  Min area threshold: {min_area_threshold:.0f}")
            print(f"  Absolute minimum: {absolute_min:.0f}")
            
        return filtered
    
    def detect_all_notes(self):
        """Detect all sticky notes of all colors"""
        self.preprocess_image()
        
        if self.debug:
            print("\n=== Debug Mode: Detection Process ===")
        
        all_regions = []
        for color_name in self.color_ranges.keys():
            regions = self.detect_colored_regions(color_name)
            all_regions.extend(regions)
        
        if self.debug:
            print(f"\nTotal regions before merging: {len(all_regions)}")
            self.debug_info['pre_merge_count'] = len(all_regions)
        
        # Sort by area (largest first) and merge overlapping
        all_regions.sort(key=lambda x: x['area'], reverse=True)
        # Higher threshold = less aggressive merging
        merged_regions = self.merge_overlapping_regions(all_regions, overlap_threshold=0.5)  # Was 0.15
        
        if self.debug:
            print(f"Total regions after merging: {len(merged_regions)}")
            self.debug_info['post_merge_count'] = len(merged_regions)
        
        # Filter by confidence - MUCH LOWER THRESHOLD
        filtered_regions = self.filter_by_confidence(merged_regions, min_confidence=40)  # Was 50
        
        if self.debug:
            print(f"Total regions after confidence filter: {len(filtered_regions)}")
        
        # NEW: Filter by relative size to remove tiny false positives
        size_filtered_regions = self.filter_by_relative_size(filtered_regions, size_ratio_threshold=0.1)
        
        if self.debug:
            print(f"Total regions after size filter: {len(size_filtered_regions)}")
            print("\n=== Debug Summary ===")
            print(f"Total contours found: {self.debug_info['total_contours']}")
            print(f"Filtered by shape validation: {self.debug_info['filtered_by_shape']}")
            print(f"Regions before merging: {self.debug_info['pre_merge_count']}")
            print(f"Regions after merging: {self.debug_info['post_merge_count']}")
            print(f"Filtered by low confidence: {self.debug_info['filtered_by_confidence']}")
            print(f"Filtered by relative size: {len(filtered_regions) - len(size_filtered_regions)}")
            print(f"Final detection count: {len(size_filtered_regions)}")
            print("=" * 40)
        
        # Sort by position (top to bottom, left to right)
        size_filtered_regions.sort(key=lambda x: (x['bbox'][1], x['bbox'][0]))
        
        self.detected_notes = size_filtered_regions
        return size_filtered_regions
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """Encode image to base64 for API calls"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def transcribe_with_claude(self, image_path: str, api_key: str) -> Tuple[str, float]:
        """Transcribe text using Claude API"""
        try:
            base64_image = self.encode_image_to_base64(image_path)
            
            headers = {
                "anthropic-version": "2023-06-01",
                "x-api-key": api_key,
                "content-type": "application/json"
            }
            
            data = {
                "model": "claude-3-haiku-20240307",  # Fast and cost-effective
                "max_tokens": 1024,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please transcribe any text you see in this sticky note image. If the text is handwritten, do your best to interpret it. Only return the transcribed text, nothing else."
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image
                            }
                        }
                    ]
                }]
            }
            
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result['content'][0]['text'].strip()
                # Confidence is not directly provided by Claude, so we'll use a high value for successful transcriptions
                confidence = 95.0 if text else 0.0
                return text, confidence
            else:
                print(f"Claude API error: {response.status_code} - {response.text}")
                return "", 0.0
                
        except Exception as e:
            print(f"Error transcribing with Claude: {e}")
            return "", 0.0
    
    def transcribe_with_chatgpt(self, image_path: str, api_key: str) -> Tuple[str, float]:
        """Transcribe text using ChatGPT API"""
        try:
            base64_image = self.encode_image_to_base64(image_path)
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "gpt-4o-mini",  # More cost-effective vision model
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please transcribe any text you see in this sticky note image. If the text is handwritten, do your best to interpret it. Only return the transcribed text, nothing else."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }],
                "max_tokens": 300
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result['choices'][0]['message']['content'].strip()
                # Confidence is not directly provided by ChatGPT, so we'll use a high value for successful transcriptions
                confidence = 95.0 if text else 0.0
                return text, confidence
            else:
                print(f"ChatGPT API error: {response.status_code} - {response.text}")
                return "", 0.0
                
        except Exception as e:
            print(f"Error transcribing with ChatGPT: {e}")
            return "", 0.0
    
    def extract_text_with_llm(self, bbox: Tuple[int, int, int, int], note_id: int, llm_provider: str, api_key: str) -> Tuple[str, float]:
        """Extract text using LLM instead of local OCR"""
        x, y, w, h = bbox
        
        # Add padding around the region
        padding = 15
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(self.width, x + w + padding)
        y_end = min(self.height, y + h + padding)
        
        # Extract region
        region = self.original_image[y_start:y_end, x_start:x_end]
        
        # Save cropped image
        cropped_path = f'cropped_note_{note_id}.jpg'
        cv2.imwrite(cropped_path, region)
        
        # Transcribe using selected LLM
        if llm_provider == 'claude':
            return self.transcribe_with_claude(cropped_path, api_key)
        elif llm_provider == 'chatgpt':
            return self.transcribe_with_chatgpt(cropped_path, api_key)
        else:
            return "", 0.0
    
    def extract_text_from_region(self, bbox, note_id):
        """Extract text from a specific region using improved OCR preprocessing"""
        x, y, w, h = bbox
        
        # Add padding around the region
        padding = 15
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(self.width, x + w + padding)
        y_end = min(self.height, y + h + padding)
        
        # Extract region
        region = self.original_image[y_start:y_end, x_start:x_end]
        
        # Enhanced preprocessing for better OCR
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Apply different preprocessing techniques
        techniques = []
        
        # 1. Adaptive threshold with different parameters
        adaptive1 = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
        techniques.append(('adaptive_gaussian', adaptive1))
        
        # 2. Otsu's threshold
        _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        techniques.append(('otsu', otsu))
        
        # 3. Morphological operations to clean text
        kernel = np.ones((2,2), np.uint8)
        morph = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
        techniques.append(('morphological', morph))
        
        # 4. Invert if dark text on light background
        inverted = cv2.bitwise_not(otsu)
        techniques.append(('inverted', inverted))
        
        best_text = ""
        best_confidence = 0
        
        for technique_name, processed in techniques:
            # Scale up for better OCR
            scale_factor = 3
            scaled = cv2.resize(processed, None, fx=scale_factor, fy=scale_factor, 
                               interpolation=cv2.INTER_CUBIC)
            
            # Apply slight blur to smooth edges
            scaled = cv2.GaussianBlur(scaled, (3, 3), 0)
            
            # OCR with different configurations
            configs = [
                '--psm 6 --oem 3',  # Uniform block of text
                '--psm 11 --oem 3',  # Sparse text
                '--psm 3 --oem 3',   # Fully automatic page segmentation
            ]
            
            for config in configs:
                try:
                    # Get detailed data
                    data = pytesseract.image_to_data(scaled, config=config, output_type=pytesseract.Output.DICT)
                    
                    # Filter out low confidence words
                    words = []
                    confidences = []
                    for i in range(len(data['text'])):
                        conf = int(data['conf'][i])
                        text = data['text'][i].strip()
                        if conf > 30 and text:  # Lower threshold for better detection
                            words.append(text)
                            confidences.append(conf)
                    
                    if words:
                        text = ' '.join(words)
                        avg_confidence = sum(confidences) / len(confidences)
                        
                        if avg_confidence > best_confidence:
                            best_confidence = avg_confidence
                            best_text = text
                            
                except Exception as e:
                    continue
        
        # Save cropped image with preprocessing preview
        preview = np.hstack([gray, techniques[0][1], techniques[1][1]])
        cv2.imwrite(f'cropped_note_{note_id}.jpg', region)
        cv2.imwrite(f'cropped_note_{note_id}_preprocessed.jpg', preview)
        
        return best_text, best_confidence
    
    def create_overlay_image(self, output_path='overlay_result.jpg'):
        """Create an overlay image showing all detected sticky notes"""
        overlay = self.original_image.copy()
        
        for i, note in enumerate(self.detected_notes):
            x, y, w, h = note['bbox']
            color = self.color_ranges[note['color']]['color']
            
            # Draw bounding box
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 8)
            
            # Add note number
            font_scale = max(1, min(w, h) // 100)
            cv2.putText(overlay, str(i + 1), (x + 10, y + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 3)
            
            # Add color label
            cv2.putText(overlay, note['color'], (x + 10, y + h - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, color, 2)
        
        cv2.imwrite(output_path, overlay)
        return output_path
    
    def process_all_notes(self, transcribe=False, llm_provider=None, api_key=None):
        """Complete processing pipeline"""
        print("Detecting sticky notes...")
        notes = self.detect_all_notes()
        print(f"Found {len(notes)} sticky notes")
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"sticky_notes_output_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        
        if transcribe:
            if llm_provider:
                print(f"Extracting text from each note using {llm_provider.upper()}...")
                print(f"Note: This will make {len(notes)} API calls. Costs may apply.")
            else:
                print("Extracting text from each note using local OCR...")
        
        for i, note in enumerate(notes):
            if transcribe:
                print(f"Processing note {i + 1}/{len(notes)}")
                # Extract text using LLM or local OCR
                if llm_provider and api_key:
                    text, ocr_confidence = self.extract_text_with_llm(note['bbox'], i + 1, llm_provider, api_key)
                else:
                    text, ocr_confidence = self.extract_text_from_region(note['bbox'], i + 1)
            else:
                # No transcription - just save the detection
                text = ""
                ocr_confidence = 0
            
            result = {
                'note_id': i + 1,
                'color': note['color'],
                'bbox': note['bbox'],
                'area': note['area'],
                'text': text,
                'ocr_confidence': ocr_confidence,
                'detection_confidence': note.get('confidence', 0),
                'sub_regions': note.get('sub_regions', 1),
                'transcription_method': llm_provider.upper() if llm_provider else 'OCR' if transcribe else 'None'
            }
            results.append(result)
            
            if transcribe:
                # Move cropped images to output directory
                if os.path.exists(f'cropped_note_{i + 1}.jpg'):
                    os.rename(f'cropped_note_{i + 1}.jpg', 
                             os.path.join(output_dir, f'cropped_note_{i + 1}.jpg'))
                if os.path.exists(f'cropped_note_{i + 1}_preprocessed.jpg') and not llm_provider:
                    os.rename(f'cropped_note_{i + 1}_preprocessed.jpg', 
                             os.path.join(output_dir, f'cropped_note_{i + 1}_preprocessed.jpg'))
        
        # Create overlay
        overlay_path = os.path.join(output_dir, 'overlay_result.jpg')
        self.create_overlay_image(overlay_path)
        
        # Save results as JSON
        json_path = os.path.join(output_dir, 'detection_results.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create readable summary
        summary_path = os.path.join(output_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Sticky Note Detection Results\n")
            f.write(f"============================\n\n")
            f.write(f"Total notes detected: {len(results)}\n")
            f.write(f"Image: {self.image_path}\n")
            f.write(f"Processed: {datetime.now()}\n")
            f.write(f"Transcription: {'Enabled' if transcribe else 'Disabled'}\n")
            if transcribe and llm_provider:
                f.write(f"Transcription method: {llm_provider.upper()}\n")
            f.write("\n")
            
            for result in results:
                f.write(f"Note {result['note_id']} ({result['color']}):\n")
                f.write(f"  Bounding box: {result['bbox']}\n")
                f.write(f"  Detection confidence: {result['detection_confidence']:.1f}%\n")
                if transcribe:
                    f.write(f"  OCR confidence: {result['ocr_confidence']:.1f}%\n")
                    f.write(f"  Text: {result['text']}\n")
                f.write(f"  Sub-regions: {result['sub_regions']}\n")
                f.write("-" * 50 + "\n\n")
        
        print(f"\nResults saved to: {output_dir}/")
        print(f"- Overlay image: overlay_result.jpg")
        if transcribe:
            print(f"- Individual crops: cropped_note_X.jpg")
            if not llm_provider:
                print(f"- Preprocessed images: cropped_note_X_preprocessed.jpg")
        print(f"- JSON data: detection_results.json")
        print(f"- Summary: summary.txt")
        
        return results, output_dir

def main():
    parser = argparse.ArgumentParser(description='Detect and transcribe sticky notes')
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('--tesseract-path', help='Path to tesseract executable (if needed)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to see filtering details')
    parser.add_argument('--transcribe', action='store_true', help='Enable OCR transcription of detected notes (default: detection only)')
    parser.add_argument('--llm', choices=['claude', 'chatgpt'], help='Use LLM for transcription instead of local OCR')
    parser.add_argument('--api-key', help='API key for the selected LLM provider')
    parser.add_argument('--api-key-env', help='Environment variable name containing the API key (default: ANTHROPIC_API_KEY for Claude, OPENAI_API_KEY for ChatGPT)')
    
    args = parser.parse_args()
    
    # Handle API key
    api_key = None
    if args.llm:
        if args.api_key:
            api_key = args.api_key
        else:
            # Try environment variable
            env_var = args.api_key_env
            if not env_var:
                env_var = 'ANTHROPIC_API_KEY' if args.llm == 'claude' else 'OPENAI_API_KEY'
            
            api_key = os.environ.get(env_var)
            if not api_key:
                print(f"Error: No API key provided. Use --api-key or set {env_var} environment variable")
                return
        
        # Enable transcription if LLM is selected
        args.transcribe = True
    
    # Set tesseract path if provided
    if args.tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_path
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found")
        return
    
    # Process the image
    detector = StickyNoteDetector(args.image_path, debug=args.debug)
    results, output_dir = detector.process_all_notes(
        transcribe=args.transcribe,
        llm_provider=args.llm,
        api_key=api_key
    )
    
    print(f"\nProcessing complete! Check '{output_dir}' for results.")

if __name__ == "__main__":
    main()
