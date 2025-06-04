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
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

class StickyNoteDetector:
    def __init__(self, image_path, debug=None, saturation_boost=None):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.original_image = self.image.copy()
        self.height, self.width = self.image.shape[:2]
        
        # Load debug mode from environment or parameter
        self.debug = debug if debug is not None else os.getenv('DEBUG_MODE', 'false').lower() == 'true'
        
        # Load saturation boost from environment or parameter
        self.saturation_boost = saturation_boost if saturation_boost is not None else float(os.getenv('SATURATION_BOOST', '1.0'))
        
        # Load detection parameters from environment variables
        self.min_confidence = float(os.getenv('MIN_CONFIDENCE', '40'))
        self.overlap_threshold = float(os.getenv('OVERLAP_THRESHOLD', '0.3'))
        self.size_ratio_threshold = float(os.getenv('SIZE_RATIO_THRESHOLD', '0.1'))
        
        # Load shape validation parameters
        self.min_area_ratio = float(os.getenv('MIN_AREA_RATIO', '0.00002'))
        self.max_area_ratio = float(os.getenv('MAX_AREA_RATIO', '0.25'))
        self.min_aspect_ratio = float(os.getenv('MIN_ASPECT_RATIO', '0.2'))
        self.max_aspect_ratio = float(os.getenv('MAX_ASPECT_RATIO', '5.0'))
        self.min_extent = float(os.getenv('MIN_EXTENT', '0.3'))
        self.min_vertices = int(os.getenv('MIN_VERTICES', '3'))
        self.max_vertices = int(os.getenv('MAX_VERTICES', '20'))
        self.min_dimension = int(os.getenv('MIN_DIMENSION', '20'))
        self.absolute_min_area_ratio = float(os.getenv('ABSOLUTE_MIN_AREA_RATIO', '0.001'))
        
        # Helper function to parse HSV values from environment
        def parse_hsv(env_var, default):
            value = os.getenv(env_var, default)
            return np.array([int(x) for x in value.split(',')])
        
        # Load color ranges from environment variables with fallback defaults
        self.color_ranges = {
            'yellow': {
                'lower': parse_hsv('YELLOW_LOWER', '22,50,120'),
                'upper': parse_hsv('YELLOW_UPPER', '60,255,255'),
                'color': (0, 255, 255)  # Yellow in BGR
            },
            'greenish_yellow': {
                'lower': parse_hsv('GREENISH_YELLOW_LOWER', '100,15,80'),
                'upper': parse_hsv('GREENISH_YELLOW_UPPER', '120,120,255'),
                'color': (0, 255, 200)  # Greenish Yellow in BGR
            },
            'warm_yellow': {
                'lower': parse_hsv('WARM_YELLOW_LOWER', '155,100,110'),
                'upper': parse_hsv('WARM_YELLOW_UPPER', '165,120,130'),
                'color': (0, 200, 255)  # Warm Yellow in BGR
            },
            'white': {
                'lower': parse_hsv('WHITE_LOWER', '0,0,180'),
                'upper': parse_hsv('WHITE_UPPER', '180,30,255'),
                'color': (255, 255, 255)  # White in BGR
            },
            'orange': {
                'lower': parse_hsv('ORANGE_LOWER', '8,120,100'),
                'upper': parse_hsv('ORANGE_UPPER', '18,255,255'),
                'color': (0, 165, 255)  # Orange in BGR
            },
            'red': {
                'lower': parse_hsv('RED_LOWER', '0,50,50'),
                'upper': parse_hsv('RED_UPPER', '10,255,255'),
                'color': (0, 0, 255)  # Red in BGR
            },
            'red_high': {
                'lower': parse_hsv('RED_HIGH_LOWER', '150,50,50'),
                'upper': parse_hsv('RED_HIGH_UPPER', '180,255,255'),
                'color': (0, 0, 255)  # Red in BGR
            },
            'green': {
                'lower': parse_hsv('GREEN_LOWER', '45,60,50'),
                'upper': parse_hsv('GREEN_UPPER', '75,255,255'),
                'color': (0, 255, 0)  # Green in BGR
            },
            'light_green': {
                'lower': parse_hsv('LIGHT_GREEN_LOWER', '40,30,150'),
                'upper': parse_hsv('LIGHT_GREEN_UPPER', '75,60,255'),
                'color': (150, 255, 150)  # Light green in BGR
            },
            'cyan': {
                'lower': parse_hsv('CYAN_LOWER', '80,30,50'),
                'upper': parse_hsv('CYAN_UPPER', '100,255,255'),
                'color': (255, 255, 0)  # Cyan in BGR
            },
            'blue': {
                'lower': parse_hsv('BLUE_LOWER', '105,100,50'),
                'upper': parse_hsv('BLUE_UPPER', '130,255,255'),
                'color': (255, 100, 0)  # Blue in BGR
            },
            'light_blue': {
                'lower': parse_hsv('LIGHT_BLUE_LOWER', '100,30,150'),
                'upper': parse_hsv('LIGHT_BLUE_UPPER', '130,100,255'),
                'color': (255, 200, 150)  # Light blue in BGR
            },
            'purple': {
                'lower': parse_hsv('PURPLE_LOWER', '130,50,50'),
                'upper': parse_hsv('PURPLE_UPPER', '150,255,255'),
                'color': (255, 0, 128)  # Purple in BGR
            },
            'light_red': {
                'lower': parse_hsv('LIGHT_RED_LOWER', '0,30,120'),
                'upper': parse_hsv('LIGHT_RED_UPPER', '10,140,255'),
                'color': (100, 100, 255)  # Light red in BGR
            },
            'light_red_high': {
                'lower': parse_hsv('LIGHT_RED_HIGH_LOWER', '150,30,120'),
                'upper': parse_hsv('LIGHT_RED_HIGH_UPPER', '180,140,255'),
                'color': (100, 100, 255)  # Light red in BGR
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
        
    def enhance_color_saturation(self, image, boost_factor=1.5):
        """Enhance color saturation to improve color detection"""
        if boost_factor == 1.0:
            return image
            
        # Convert to HSV for saturation enhancement
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Boost saturation
        s = s.astype(np.float32)
        s = s * boost_factor
        s = np.clip(s, 0, 255)  # Ensure values stay within valid range
        s = s.astype(np.uint8)
        
        # Merge back and convert to BGR
        enhanced_hsv = cv2.merge([h, s, v])
        enhanced_bgr = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
        
        if self.debug:
            print(f"Applied saturation boost: {boost_factor}x")
            
        return enhanced_bgr
        
    def preprocess_image(self):
        """Apply preprocessing to improve detection"""
        # Apply saturation enhancement first if enabled
        if self.saturation_boost != 1.0:
            self.image = self.enhance_color_saturation(self.image, self.saturation_boost)
            if self.debug:
                print(f"Enhanced image saturation by {self.saturation_boost}x")
        
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
        min_area_ratio = self.min_area_ratio
        min_area = self.width * self.height * min_area_ratio
        validation_scores['area'] = area >= min_area
        if area < min_area:
            is_valid = False
            rejection_reasons.append(f"area too small: {area:.0f} < {min_area:.0f}")
            
        # 2. Maximum area (to avoid detecting entire image sections)
        max_area_ratio = self.max_area_ratio
        max_area = self.width * self.height * max_area_ratio
        validation_scores['max_area'] = area <= max_area
        if area > max_area:
            is_valid = False
            rejection_reasons.append(f"area too large: {area:.0f} > {max_area:.0f}")
            
        # 3. Aspect ratio (sticky notes are somewhat square) - MUCH MORE RELAXED
        validation_scores['aspect_ratio'] = self.min_aspect_ratio < aspect_ratio < self.max_aspect_ratio
        if not (self.min_aspect_ratio < aspect_ratio < self.max_aspect_ratio):
            is_valid = False
            rejection_reasons.append(f"bad aspect ratio: {aspect_ratio:.2f}")
            
        # 4. Extent (how much the contour fills its bounding box) - MUCH MORE RELAXED
        validation_scores['extent'] = extent > self.min_extent
        if extent < self.min_extent:
            is_valid = False
            rejection_reasons.append(f"low extent: {extent:.2f}")
            
        # 5. Rectangularity (4-6 vertices for a rectangle-like shape) - MORE RELAXED
        validation_scores['vertices'] = self.min_vertices <= len(approx) <= self.max_vertices
        if not (self.min_vertices <= len(approx) <= self.max_vertices):
            is_valid = False
            rejection_reasons.append(f"bad vertex count: {len(approx)}")
            
        # 6. Minimum dimensions - MUCH MORE RELAXED
        min_dimension = self.min_dimension
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
        elif color_name == 'light_red':
            # Light red also wraps around in HSV
            mask_low = cv2.inRange(hsv_enhanced, 
                                  self.color_ranges['light_red']['lower'],
                                  self.color_ranges['light_red']['upper'])
            mask_high = cv2.inRange(hsv_enhanced, 
                                   self.color_ranges['light_red_high']['lower'],
                                   self.color_ranges['light_red_high']['upper'])
            mask = cv2.bitwise_or(mask_low, mask_high)
        elif color_name in ['red_high', 'light_red_high']:
            # Skip these as they're handled by red and light_red
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
            'yellow': ['greenish_yellow', 'warm_yellow', 'white'],
            'greenish_yellow': ['yellow', 'warm_yellow', 'white'],
            'warm_yellow': ['yellow', 'greenish_yellow', 'white'],
            'white': ['yellow', 'greenish_yellow', 'warm_yellow'],
            'blue': ['light_blue'],
            'light_blue': ['blue'],
            'green': ['light_green'],
            'light_green': ['green'],
            'red': ['red_high', 'light_red', 'light_red_high'],  # Include all red variants (including former pink)
            'red_high': ['red', 'light_red', 'light_red_high'],
            'light_red': ['red', 'red_high', 'light_red_high'],
            'light_red_high': ['red', 'red_high', 'light_red'],
            'orange': [],  # Orange is distinct
            'purple': [],  # Purple is distinct
            'cyan': []     # Cyan is now a single broad range
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
        absolute_min = self.width * self.height * self.absolute_min_area_ratio
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
        merged_regions = self.merge_overlapping_regions(all_regions, overlap_threshold=self.overlap_threshold)
        
        if self.debug:
            print(f"Total regions after merging: {len(merged_regions)}")
            self.debug_info['post_merge_count'] = len(merged_regions)
        
        # Filter by confidence - MUCH LOWER THRESHOLD
        filtered_regions = self.filter_by_confidence(merged_regions, min_confidence=self.min_confidence)
        
        if self.debug:
            print(f"Total regions after confidence filter: {len(filtered_regions)}")
        
        # NEW: Filter by relative size to remove tiny false positives
        size_filtered_regions = self.filter_by_relative_size(filtered_regions, size_ratio_threshold=self.size_ratio_threshold)
        
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
        """Transcribe text using ChatGPT API with retry logic"""
        max_retries = 3
        base_delay = 1  # Start with 1 second delay
        
        for attempt in range(max_retries):
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
                elif response.status_code == 429:  # Rate limit error
                    if attempt < max_retries - 1:
                        # Extract wait time from error message if available
                        try:
                            error_data = response.json()
                            error_message = error_data.get('error', {}).get('message', '')
                            if 'Please try again in' in error_message:
                                # Extract wait time (e.g., "242ms" or "1.5s")
                                import re
                                wait_match = re.search(r'try again in (\d+(?:\.\d+)?)(ms|s)', error_message)
                                if wait_match:
                                    wait_time = float(wait_match.group(1))
                                    unit = wait_match.group(2)
                                    if unit == 'ms':
                                        wait_time = wait_time / 1000
                                    delay = max(wait_time, base_delay * (2 ** attempt))
                                else:
                                    delay = base_delay * (2 ** attempt)
                            else:
                                delay = base_delay * (2 ** attempt)
                        except:
                            delay = base_delay * (2 ** attempt)
                        
                        print(f"Rate limit hit. Waiting {delay:.1f} seconds before retry {attempt + 1}/{max_retries}...")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"ChatGPT API rate limit exceeded after {max_retries} attempts: {response.text}")
                        return "", 0.0
                else:
                    print(f"ChatGPT API error: {response.status_code} - {response.text}")
                    return "", 0.0
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"Error transcribing with ChatGPT (attempt {attempt + 1}): {e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"Error transcribing with ChatGPT after {max_retries} attempts: {e}")
                    return "", 0.0
        
        return "", 0.0
    
    def transcribe_with_ollama(self, image_path: str, model_name: str = "llava") -> Tuple[str, float]:
        """Transcribe text using Ollama local LLM"""
        try:
            import base64
            
            # Encode image to base64
            base64_image = self.encode_image_to_base64(image_path)
            
            # Prepare the request payload for Ollama
            payload = {
                "model": model_name,
                "prompt": "Please transcribe any text you see in this sticky note image. If the text is handwritten, do your best to interpret it. Only return the transcribed text, nothing else.",
                "images": [base64_image],
                "stream": False
            }
            
            # Make request to Ollama API (default port 11434)
            ollama_url = os.getenv('OLLAMA_URL', 'http://localhost:11434')
            response = requests.post(
                f"{ollama_url}/api/generate",
                json=payload,
                timeout=60  # Longer timeout for local processing
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get('response', '').strip()
                # Confidence is not directly provided by Ollama, so we'll use a high value for successful transcriptions
                confidence = 95.0 if text else 0.0
                return text, confidence
            else:
                print(f"Ollama API error: {response.status_code} - {response.text}")
                return "", 0.0
                
        except ImportError:
            print("Error: requests library is required for Ollama integration")
            return "", 0.0
        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to Ollama. Make sure Ollama is running on localhost:11434")
            print("Install Ollama from https://ollama.ai and run: ollama pull llava")
            return "", 0.0
        except Exception as e:
            print(f"Error transcribing with Ollama: {e}")
            return "", 0.0
    
    def extract_text_with_llm(self, bbox: Tuple[int, int, int, int], note_id: int, llm_provider: str, api_key: str = None, ollama_model: str = 'llava') -> Tuple[str, float]:
        """Extract text using LLM instead of local OCR"""
        x, y, w, h = bbox
        
        # Add padding around the region
        padding = 15
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(self.width, x + w + padding)
        y_end = min(self.height, y + h + padding)
        
        # Extract region from the saturated image if saturation boost was applied, otherwise use original
        source_image = self.image if self.saturation_boost != 1.0 else self.original_image
        region = source_image[y_start:y_end, x_start:x_end]
        
        # Save cropped image
        cropped_path = f'cropped_note_{note_id}.jpg'
        cv2.imwrite(cropped_path, region)
        
        # Transcribe using selected LLM
        if llm_provider == 'claude':
            return self.transcribe_with_claude(cropped_path, api_key)
        elif llm_provider == 'chatgpt':
            return self.transcribe_with_chatgpt(cropped_path, api_key)
        elif llm_provider == 'ollama':
            return self.transcribe_with_ollama(cropped_path, ollama_model)
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
        
        # Extract region from the saturated image if saturation boost was applied, otherwise use original
        source_image = self.image if self.saturation_boost != 1.0 else self.original_image
        region = source_image[y_start:y_end, x_start:x_end]
        
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
        # Use the saturated image if saturation boost was applied, otherwise use original
        source_image = self.image if self.saturation_boost != 1.0 else self.original_image
        overlay = source_image.copy()
        
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
    
    def process_all_notes(self, transcribe=False, llm_provider=None, api_key=None, ollama_model='llava'):
        """Complete processing pipeline"""
        print("Detecting sticky notes...")
        notes = self.detect_all_notes()
        print(f"Found {len(notes)} sticky notes")
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"sticky_notes_output_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save enhanced image for debugging if saturation was boosted
        if self.saturation_boost != 1.0:
            enhanced_path = os.path.join(output_dir, 'enhanced_saturation.jpg')
            cv2.imwrite(enhanced_path, self.image)
            if self.debug:
                print(f"Saved saturation-enhanced image: {enhanced_path}")
        
        results = []
        
        if transcribe:
            if llm_provider and llm_provider != 'ollama':
                print(f"Extracting text from each note using {llm_provider.upper()}...")
                print(f"Note: This will make {len(notes)} API calls. Costs may apply.")
                print("Adding delays between calls to respect rate limits...")
            elif llm_provider == 'ollama':
                print(f"Extracting text from each note using Ollama ({ollama_model})...")
                print("Note: This will use your local Ollama installation.")
            else:
                print("Extracting text from each note using local OCR...")
        
        for i, note in enumerate(notes):
            if transcribe:
                print(f"Processing note {i + 1}/{len(notes)}")
                # Extract text using LLM or local OCR
                if llm_provider and llm_provider != 'ollama' and api_key:
                    text, ocr_confidence = self.extract_text_with_llm(note['bbox'], i + 1, llm_provider, api_key, ollama_model)
                    # Add delay between API calls to respect rate limits
                    if i < len(notes) - 1:  # Don't delay after the last call
                        time.sleep(0.5)  # 500ms delay between calls
                elif llm_provider == 'ollama':
                    text, ocr_confidence = self.extract_text_with_llm(note['bbox'], i + 1, llm_provider, None, ollama_model)
                    # Add delay between local LLM calls to prevent overload
                    if i < len(notes) - 1:  # Don't delay after the last call
                        time.sleep(0.2)  # Shorter delay for local processing
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
            f.write(f"Saturation boost: {self.saturation_boost}x\n")
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
        
        # Create HTML report
        html_path = self.create_html_report(results, output_dir)
        
        print(f"\nResults saved to: {output_dir}/")
        print(f"- Overlay image: overlay_result.jpg")
        if transcribe:
            print(f"- Individual crops: cropped_note_X.jpg")
            if not llm_provider:
                print(f"- Preprocessed images: cropped_note_X_preprocessed.jpg")
        print(f"- JSON data: detection_results.json")
        print(f"- Summary: summary.txt")
        print(f"- HTML report: detection_report.html")
        
        return results, output_dir
    
    def create_html_report(self, results, output_dir):
        """Create an HTML file with a table showing all detected notes"""
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sticky Note Detection Results</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }}
        .summary {{
            background-color: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .controls {{
            background-color: #f0f8ff;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            text-align: center;
        }}
        .btn {{
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin: 0 5px;
            font-size: 14px;
        }}
        .btn:hover {{
            background-color: #45a049;
        }}
        .btn-secondary {{
            background-color: #2196F3;
        }}
        .btn-secondary:hover {{
            background-color: #1976D2;
        }}
        .btn-warning {{
            background-color: #ff9800;
        }}
        .btn-warning:hover {{
            background-color: #f57c00;
        }}
        .btn-danger {{
            background-color: #f44336;
        }}
        .btn-danger:hover {{
            background-color: #d32f2f;
        }}
        .btn-small {{
            padding: 5px 10px;
            font-size: 12px;
            margin: 2px;
        }}
        .btn-purple {{
            background-color: #9c27b0;
        }}
        .btn-purple:hover {{
            background-color: #7b1fa2;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        tr:hover {{
            background-color: #e8f4f8;
        }}
        .note-image {{
            max-width: 250px;
            max-height: 250px;
            border: 2px solid #ddd;
            border-radius: 4px;
            cursor: pointer;
        }}
        .note-image:hover {{
            border-color: #4CAF50;
            transform: scale(1.02);
            transition: all 0.2s ease;
        }}
        .details-toggle {{
            background: none;
            border: none;
            color: #666;
            cursor: pointer;
            font-size: 12px;
            padding: 2px 5px;
            border-radius: 3px;
        }}
        .details-toggle:hover {{
            background-color: #f0f0f0;
        }}
        .technical-details {{
            display: none;
            margin-top: 10px;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 4px;
            font-size: 12px;
            color: #666;
        }}
        .technical-details.expanded {{
            display: block;
        }}
        .break-note-btn {{
            background-color: #17a2b8;
            color: white;
            border: none;
            padding: 3px 8px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
            margin: 2px;
        }}
        .break-note-btn:hover {{
            background-color: #138496;
        }}
        .sub-row {{
            background-color: #f0f8ff !important;
            border-left: 4px solid #17a2b8;
        }}
        .sub-row td {{
            padding-left: 20px;
            font-style: italic;
        }}
        .dismiss-btn {{
            background-color: #6c757d;
            color: white;
            border: none;
            padding: 2px 6px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 10px;
        }}
        .dismiss-btn:hover {{
            background-color: #545b62;
        }}
        .overlay-modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.8);
        }}
        .overlay-content {{
            position: relative;
            margin: 2% auto;
            width: 90%;
            max-width: 1200px;
            text-align: center;
        }}
        .overlay-image {{
            max-width: 100%;
            max-height: 90vh;
            border-radius: 8px;
            cursor: crosshair;
        }}
        .close-overlay {{
            position: absolute;
            top: 10px;
            right: 25px;
            color: white;
            font-size: 35px;
            font-weight: bold;
            cursor: pointer;
        }}
        .close-overlay:hover {{
            color: #ccc;
        }}
        .view-overlay-btn {{
            background-color: #6f42c1;
            color: white;
            border: none;
            padding: 8px 15px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            margin: 5px;
        }}
        .view-overlay-btn:hover {{
            background-color: #5a32a3;
        }}
        .color-badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 12px;
            color: white;
            font-size: 12px;
            font-weight: bold;
            text-transform: uppercase;
        }}
        .confidence-bar {{
            width: 100px;
            height: 20px;
            background-color: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
            display: inline-block;
            vertical-align: middle;
        }}
        .confidence-fill {{
            height: 100%;
            border-radius: 10px;
            transition: width 0.3s ease;
        }}
        .bbox {{
            font-family: monospace;
            font-size: 12px;
            color: #666;
        }}
        .text-content {{
            max-width: 300px;
            word-wrap: break-word;
        }}
        .text-input {{
            width: 100%;
            min-height: 60px;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 8px;
            font-family: inherit;
            font-size: 14px;
            resize: vertical;
        }}
        .text-input:focus {{
            border-color: #4CAF50;
            outline: none;
            box-shadow: 0 0 5px rgba(76, 175, 80, 0.3);
        }}
        .no-text {{
            color: #999;
            font-style: italic;
        }}
        .edit-mode .text-display {{
            display: none;
        }}
        .edit-mode .text-input {{
            display: block;
        }}
        .text-input {{
            display: none;
        }}
        .status-message {{
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
            display: none;
        }}
        .status-success {{
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }}
        .status-error {{
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }}
        .row-deleted {{
            opacity: 0.5;
            background-color: #ffebee !important;
            text-decoration: line-through;
        }}
        .row-deleted:hover {{
            background-color: #ffcdd2 !important;
        }}
        .delete-column {{
            width: 120px;
            text-align: center;
        }}
        .custom-row {{
            background-color: #e8f5e9 !important;
            border-left: 4px solid #4CAF50;
        }}
        .drawing-box {{
            position: absolute;
            border: 3px solid #4CAF50;
            background-color: rgba(76, 175, 80, 0.2);
            pointer-events: none;
            display: none;
        }}
        .drawing-instructions {{
            background-color: #fff3cd;
            color: #856404;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
            display: none;
        }}
        .color-selector {{
            margin: 10px 0;
            display: none;
        }}
        .color-option {{
            display: inline-block;
            width: 30px;
            height: 30px;
            margin: 5px;
            border: 2px solid #ddd;
            border-radius: 4px;
            cursor: pointer;
        }}
        .color-option:hover {{
            border-color: #333;
            transform: scale(1.1);
        }}
        .color-option.selected {{
            border-color: #4CAF50;
            border-width: 3px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🗒️ Sticky Note Detection Results</h1>
        
        <div class="summary">
            <h3>Summary</h3>
            <p><strong>Image:</strong> {image_name}</p>
            <p><strong>Total Notes Detected:</strong> {total_notes}</p>
            <p><strong>Processed:</strong> {timestamp}</p>
            <p><strong>Transcription:</strong> {transcription_method}</p>
        </div>

        <div class="controls">
            <button class="btn" onclick="toggleEditMode()">📝 Toggle Edit Mode</button>
            <button class="btn btn-secondary" onclick="saveEdits()">💾 Save Edits</button>
            <button class="btn btn-warning" onclick="downloadMarkdown()">📄 Download as Markdown</button>
            <button class="btn view-overlay-btn" onclick="showOverlay()">🖼️ View Full Image</button>
            <button class="btn btn-purple" onclick="toggleDrawMode()">✏️ Draw New Note</button>
            <div id="statusMessage" class="status-message"></div>
        </div>

        <div id="drawingInstructions" class="drawing-instructions">
            <strong>Drawing Mode Active!</strong> Click and drag on the full image to draw a bounding box for a new note.
            <br>Press ESC or click "Draw New Note" again to cancel.
        </div>

        <div id="colorSelector" class="color-selector">
            <strong>Select color for new note:</strong>
            <div class="color-option" style="background-color: #FFD700;" data-color="yellow" title="Yellow"></div>
            <div class="color-option" style="background-color: #FF8C00;" data-color="orange" title="Orange"></div>
            <div class="color-option" style="background-color: #DC143C;" data-color="red" title="Red"></div>
            <div class="color-option" style="background-color: #32CD32;" data-color="green" title="Green"></div>
            <div class="color-option" style="background-color: #4169E1;" data-color="blue" title="Blue"></div>
            <div class="color-option" style="background-color: #00CED1;" data-color="cyan" title="Cyan"></div>
            <div class="color-option" style="background-color: #8A2BE2;" data-color="purple" title="Purple"></div>
            <div class="color-option selected" style="background-color: #FFFFFF; border-color: #333;" data-color="white" title="White"></div>
        </div>

        <table id="notesTable">
            <thead>
                <tr>
                    <th>Note #</th>
                    <th>Image</th>
                    <th>Transcribed Text</th>
                    <th class="delete-column">Actions</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    </div>

    <!-- Overlay Modal -->
    <div id="overlayModal" class="overlay-modal">
        <div class="overlay-content">
            <span class="close-overlay" onclick="hideOverlay()">&times;</span>
            <div style="position: relative; display: inline-block;">
                <img id="overlayImage" class="overlay-image" src="overlay_result.jpg" alt="Full Detection Results">
                <div id="drawingBox" class="drawing-box"></div>
            </div>
        </div>
    </div>

    <script>
        let editMode = false;
        let drawMode = false;
        let isDrawing = false;
        let startX, startY;
        let selectedColor = 'white';
        let originalData = {original_data};
        let deletedRows = new Set(); // Track deleted row indices
        let subNoteCounter = 0; // Counter for sub-notes
        let customNoteCounter = 0; // Counter for custom notes
        let imageScale = 1; // Scale factor for the displayed image

        // Initialize color selector
        document.querySelectorAll('.color-option').forEach(option => {{
            option.addEventListener('click', function() {{
                document.querySelectorAll('.color-option').forEach(o => o.classList.remove('selected'));
                this.classList.add('selected');
                selectedColor = this.getAttribute('data-color');
            }});
        }});

        function toggleEditMode() {{
            editMode = !editMode;
            const table = document.getElementById('notesTable');
            const button = document.querySelector('button[onclick="toggleEditMode()"]');
            
            if (editMode) {{
                table.classList.add('edit-mode');
                button.textContent = '👁️ View Mode';
                button.style.backgroundColor = '#ff9800';
                showStatus('Edit mode enabled. Click on text fields to edit.', 'success');
            }} else {{
                table.classList.remove('edit-mode');
                button.textContent = '📝 Toggle Edit Mode';
                button.style.backgroundColor = '#4CAF50';
                showStatus('View mode enabled.', 'success');
            }}
        }}

        function toggleDrawMode() {{
            drawMode = !drawMode;
            const button = document.querySelector('button[onclick="toggleDrawMode()"]');
            const instructions = document.getElementById('drawingInstructions');
            const colorSelector = document.getElementById('colorSelector');
            
            if (drawMode) {{
                button.textContent = '❌ Cancel Drawing';
                button.style.backgroundColor = '#f44336';
                instructions.style.display = 'block';
                colorSelector.style.display = 'block';
                showStatus('Drawing mode enabled. Open the full image to draw a bounding box.', 'success');
                // Automatically open the overlay if not already open
                if (document.getElementById('overlayModal').style.display !== 'block') {{
                    showOverlay();
                }}
            }} else {{
                button.textContent = '✏️ Draw New Note';
                button.style.backgroundColor = '#9c27b0';
                instructions.style.display = 'none';
                colorSelector.style.display = 'none';
                showStatus('Drawing mode disabled.', 'success');
            }}
        }}

        function toggleDetails(index) {{
            const detailsDiv = document.getElementById(`details-${{index}}`);
            const button = document.querySelector(`button[onclick="toggleDetails(${{index}})"]`);
            
            if (detailsDiv.classList.contains('expanded')) {{
                detailsDiv.classList.remove('expanded');
                button.textContent = '📊 Technical Details';
            }} else {{
                detailsDiv.classList.add('expanded');
                button.textContent = '📊 Hide Details';
            }}
        }}

        function breakNote(index) {{
            const row = document.querySelector(`tr[data-note-id="${{index + 1}}"]`);
            subNoteCounter++;
            
            // Create new sub-row
            const newRow = document.createElement('tr');
            newRow.className = 'sub-row';
            newRow.setAttribute('data-parent-id', index + 1);
            newRow.setAttribute('data-sub-id', subNoteCounter);
            
            newRow.innerHTML = `
                <td><strong>#${{index + 1}}.sub${{subNoteCounter}}</strong></td>
                <td style="text-align: center; color: #666;">
                    <em>Sub-note of #${{index + 1}}</em>
                </td>
                <td class="text-content">
                    <div class="text-display"><span class="no-text">Enter text for this sub-note</span></div>
                    <textarea class="text-input" placeholder="Enter text for this sub-note..."></textarea>
                </td>
                <td class="delete-column">
                    <button class="dismiss-btn" onclick="dismissSubNote(this)" title="Remove this sub-note">
                        ✖️ Dismiss
                    </button>
                </td>
            `;
            
            // Insert after the main row
            row.parentNode.insertBefore(newRow, row.nextSibling);
            showStatus(`Sub-note created for Note #${{index + 1}}`, 'success');
        }}

        function dismissSubNote(button) {{
            const row = button.closest('tr');
            const parentId = row.getAttribute('data-parent-id');
            row.remove();
            showStatus(`Sub-note removed from Note #${{parentId}}`, 'success');
        }}

        function showOverlay() {{
            const modal = document.getElementById('overlayModal');
            modal.style.display = 'block';
            document.body.style.overflow = 'hidden'; // Prevent background scrolling
            
            // Calculate image scale when overlay is shown
            setTimeout(() => {{
                const img = document.getElementById('overlayImage');
                const naturalWidth = img.naturalWidth;
                const displayedWidth = img.width;
                imageScale = displayedWidth / naturalWidth;
            }}, 100);
        }}

        function hideOverlay() {{
            const modal = document.getElementById('overlayModal');
            modal.style.display = 'none';
            document.body.style.overflow = 'auto'; // Restore scrolling
            
            // Reset drawing if in progress
            if (isDrawing) {{
                isDrawing = false;
                document.getElementById('drawingBox').style.display = 'none';
            }}
        }}

        // Drawing functionality
        const overlayImage = document.getElementById('overlayImage');
        const drawingBox = document.getElementById('drawingBox');

        overlayImage.addEventListener('mousedown', function(e) {{
            if (!drawMode) return;
            
            const rect = this.getBoundingClientRect();
            startX = e.clientX - rect.left;
            startY = e.clientY - rect.top;
            isDrawing = true;
            
            drawingBox.style.left = startX + 'px';
            drawingBox.style.top = startY + 'px';
            drawingBox.style.width = '0px';
            drawingBox.style.height = '0px';
            drawingBox.style.display = 'block';
        }});

        overlayImage.addEventListener('mousemove', function(e) {{
            if (!isDrawing || !drawMode) return;
            
            const rect = this.getBoundingClientRect();
            const currentX = e.clientX - rect.left;
            const currentY = e.clientY - rect.top;
            
            const width = Math.abs(currentX - startX);
            const height = Math.abs(currentY - startY);
            const left = Math.min(startX, currentX);
            const top = Math.min(startY, currentY);
            
            drawingBox.style.left = left + 'px';
            drawingBox.style.top = top + 'px';
            drawingBox.style.width = width + 'px';
            drawingBox.style.height = height + 'px';
        }});

        overlayImage.addEventListener('mouseup', function(e) {{
            if (!isDrawing || !drawMode) return;
            
            isDrawing = false;
            const rect = this.getBoundingClientRect();
            const endX = e.clientX - rect.left;
            const endY = e.clientY - rect.top;
            
            const width = Math.abs(endX - startX);
            const height = Math.abs(endY - startY);
            
            // Minimum size check
            if (width < 20 || height < 20) {{
                drawingBox.style.display = 'none';
                showStatus('Bounding box too small. Please draw a larger box.', 'error');
                return;
            }}
            
            // Convert to original image coordinates
            const x = Math.min(startX, endX) / imageScale;
            const y = Math.min(startY, endY) / imageScale;
            const w = width / imageScale;
            const h = height / imageScale;
            
            // Create new note
            createCustomNote(Math.round(x), Math.round(y), Math.round(w), Math.round(h));
            
            // Hide drawing box and exit draw mode
            drawingBox.style.display = 'none';
            toggleDrawMode();
        }});

        function createCustomNote(x, y, w, h) {{
            customNoteCounter++;
            const noteId = `custom_${{customNoteCounter}}`;
            const tbody = document.querySelector('#notesTable tbody');
            
            // Create new row
            const newRow = document.createElement('tr');
            newRow.className = 'custom-row';
            newRow.setAttribute('data-note-id', noteId);
            newRow.setAttribute('data-custom', 'true');
            
            // Create a simple colored rectangle as placeholder image
            const canvas = document.createElement('canvas');
            canvas.width = 100;
            canvas.height = 100;
            const ctx = canvas.getContext('2d');
            
            // Fill with selected color
            const colorMap = {{
                'yellow': '#FFD700',
                'orange': '#FF8C00',
                'red': '#DC143C',
                'green': '#32CD32',
                'blue': '#4169E1',
                'cyan': '#00CED1',
                'purple': '#8A2BE2',
                'white': '#FFFFFF'
            }};
            
            ctx.fillStyle = colorMap[selectedColor] || '#CCCCCC';
            ctx.fillRect(0, 0, 100, 100);
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 2;
            ctx.strokeRect(0, 0, 100, 100);
            
            const imageDataUrl = canvas.toDataURL();
            
            newRow.innerHTML = `
                <td><strong>#${{noteId}}</strong></td>
                <td>
                    <img src="${{imageDataUrl}}" alt="Custom Note" class="note-image" style="max-width: 100px; max-height: 100px;">
                    <br>
                    <span style="color: #4CAF50; font-weight: bold;">✏️ Custom Note</span>
                    <br>
                    <button class="details-toggle" onclick="toggleDetails('${{noteId}}')">
                        📊 Technical Details
                    </button>
                    <div class="technical-details" id="details-${{noteId}}">
                        <strong>Color:</strong> <span class="color-badge" style="background-color: ${{colorMap[selectedColor]}};">${{selectedColor}}</span><br>
                        <strong>Bounding Box:</strong> <span class="bbox">x:${{x}}, y:${{y}}, w:${{w}}, h:${{h}}</span><br>
                        <strong>Area:</strong> ${{(w * h).toLocaleString()}} px²<br>
                        <strong>Type:</strong> User-drawn<br>
                        <strong>Created:</strong> ${{new Date().toLocaleString()}}
                    </div>
                </td>
                <td class="text-content">
                    <div class="text-display"><span class="no-text">Enter text for this custom note</span></div>
                    <textarea class="text-input" placeholder="Enter text for this custom note..."></textarea>
                </td>
                <td class="delete-column">
                    <button class="btn btn-danger btn-small" onclick="removeCustomNote('${{noteId}}')" title="Remove this custom note">
                        🗑️ Delete
                    </button>
                </td>
            `;
            
            tbody.appendChild(newRow);
            
            // Store custom note data
            if (!window.customNotes) window.customNotes = [];
            window.customNotes.push({{
                note_id: noteId,
                bbox: [x, y, w, h],
                color: selectedColor,
                area: w * h,
                is_custom: true,
                created_at: new Date().toISOString()
            }});
            
            showStatus(`Custom note created at position (${{x}}, ${{y}}) with size ${{w}}x${{h}}`, 'success');
            updateSummary();
            
            // Auto-enable edit mode for the new note
            if (!editMode) {{
                toggleEditMode();
            }}
        }}

        function removeCustomNote(noteId) {{
            const row = document.querySelector(`tr[data-note-id="${{noteId}}"]`);
            if (row) {{
                row.remove();
                // Remove from custom notes array
                if (window.customNotes) {{
                    window.customNotes = window.customNotes.filter(note => note.note_id !== noteId);
                }}
                showStatus(`Custom note removed`, 'success');
                updateSummary();
            }}
        }}

        // Close overlay with Escape key
        document.addEventListener('keydown', function(e) {{
            if (e.key === 'Escape') {{
                if (drawMode) {{
                    toggleDrawMode();
                }} else {{
                    hideOverlay();
                }}
            }}
        }});

        function toggleRowDelete(index) {{
            const row = document.querySelector(`tr[data-note-id="${{index + 1}}"]`);
            const button = row.querySelector('.btn-danger');
            
            if (deletedRows.has(index)) {{
                // Undelete row
                deletedRows.delete(index);
                row.classList.remove('row-deleted');
                button.textContent = '🗑️ Delete';
                button.title = 'Mark as false positive';
                showStatus(`Note #${{index + 1}} restored`, 'success');
            }} else {{
                // Delete row
                deletedRows.add(index);
                row.classList.add('row-deleted');
                button.textContent = '↩️ Restore';
                button.title = 'Restore this note';
                showStatus(`Note #${{index + 1}} marked for deletion`, 'success');
            }}
            
            updateSummary();
        }}

        function updateSummary() {{
            const totalNotes = originalData.length;
            const customCount = window.customNotes ? window.customNotes.length : 0;
            const deletedCount = deletedRows.size;
            const activeCount = totalNotes - deletedCount + customCount;
            
            // Update summary in the page
            const summaryDiv = document.querySelector('.summary');
            const totalNotesP = summaryDiv.querySelector('p:nth-child(2)');
            let summaryText = `<strong>Total Notes Detected:</strong> ${{totalNotes}}`;
            if (customCount > 0 || deletedCount > 0) {{
                summaryText += ` (Active: ${{activeCount}}`;
                if (deletedCount > 0) summaryText += `, Deleted: ${{deletedCount}}`;
                if (customCount > 0) summaryText += `, Custom: ${{customCount}}`;
                summaryText += `)`;
            }}
            totalNotesP.innerHTML = summaryText;
        }}

        function showStatus(message, type) {{
            const statusDiv = document.getElementById('statusMessage');
            statusDiv.textContent = message;
            statusDiv.className = `status-message status-${{type}}`;
            statusDiv.style.display = 'block';
            setTimeout(() => {{
                statusDiv.style.display = 'none';
            }}, 3000);
        }}

        function saveEdits() {{
            const allTextInputs = document.querySelectorAll('.text-input');
            const editedData = [];
            
            // Process main notes (non-deleted)
            originalData.forEach((item, index) => {{
                if (!deletedRows.has(index)) {{
                    const editedItem = JSON.parse(JSON.stringify(item));
                    const textInput = allTextInputs[index];
                    if (textInput) {{
                        editedItem.text = textInput.value;
                        editedItem.edited = true;
                        editedItem.edit_timestamp = new Date().toISOString();
                    }}
                    editedData.push(editedItem);
                    
                    // Check for sub-notes
                    const subRows = document.querySelectorAll(`tr[data-parent-id="${{index + 1}}"]`);
                    subRows.forEach(subRow => {{
                        const subTextInput = subRow.querySelector('.text-input');
                        if (subTextInput && subTextInput.value.trim()) {{
                            const subItem = JSON.parse(JSON.stringify(item));
                            subItem.note_id = `${{item.note_id}}.sub${{subRow.getAttribute('data-sub-id')}}`;
                            subItem.text = subTextInput.value;
                            subItem.is_sub_note = true;
                            subItem.parent_note_id = item.note_id;
                            subItem.edited = true;
                            subItem.edit_timestamp = new Date().toISOString();
                            editedData.push(subItem);
                        }}
                    }});
                }}
            }});

            // Add custom notes
            if (window.customNotes) {{
                window.customNotes.forEach(customNote => {{
                    const row = document.querySelector(`tr[data-note-id="${{customNote.note_id}}"]`);
                    if (row) {{
                        const textInput = row.querySelector('.text-input');
                        const noteData = {{
                            note_id: customNote.note_id,
                            color: customNote.color,
                            bbox: customNote.bbox,
                            area: customNote.area,
                            text: textInput ? textInput.value : '',
                            ocr_confidence: 0,
                            detection_confidence: 100,
                            sub_regions: 1,
                            transcription_method: 'Manual',
                            is_custom: true,
                            created_at: customNote.created_at,
                            edited: true,
                            edit_timestamp: new Date().toISOString()
                        }};
                        editedData.push(noteData);
                    }}
                }});
            }}

            // Add metadata about deletions, sub-notes, and custom notes
            const subNoteCount = document.querySelectorAll('.sub-row').length;
            const customNoteCount = window.customNotes ? window.customNotes.length : 0;
            const metadata = {{
                original_count: originalData.length,
                final_count: editedData.length,
                deleted_count: deletedRows.size,
                sub_note_count: subNoteCount,
                custom_note_count: customNoteCount,
                deleted_note_ids: Array.from(deletedRows).map(i => originalData[i].note_id),
                export_timestamp: new Date().toISOString()
            }};

            const finalData = {{
                metadata: metadata,
                notes: editedData
            }};

            // Create and download the edited JSON
            const blob = new Blob([JSON.stringify(finalData, null, 2)], {{
                type: 'application/json'
            }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'edited_detection_results.json';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            showStatus(`Edits saved! ${{editedData.length}} notes included (${{customNoteCount}} custom), ${{deletedRows.size}} deleted, ${{subNoteCount}} sub-notes.`, 'success');
        }}

        function downloadMarkdown() {{
            let markdown = '# Sticky Note Transcriptions\\n\\n';
            let noteNumber = 1;
            
            // Process main notes (non-deleted)
            originalData.forEach((item, index) => {{
                if (!deletedRows.has(index)) {{
                    const textInput = document.querySelectorAll('.text-input')[index];
                    const text = textInput ? textInput.value.trim() : '';
                    if (text) {{
                        markdown += `${{noteNumber}}. ${{text}}\\n`;
                        noteNumber++;
                    }}
                    
                    // Add sub-notes
                    const subRows = document.querySelectorAll(`tr[data-parent-id="${{index + 1}}"]`);
                    subRows.forEach(subRow => {{
                        const subTextInput = subRow.querySelector('.text-input');
                        const subText = subTextInput ? subTextInput.value.trim() : '';
                        if (subText) {{
                            markdown += `${{noteNumber}}. ${{subText}}\\n`;
                            noteNumber++;
                        }}
                    }});
                }}
            }});

            // Add custom notes
            if (window.customNotes) {{
                window.customNotes.forEach(customNote => {{
                    const row = document.querySelector(`tr[data-note-id="${{customNote.note_id}}"]`);
                    if (row) {{
                        const textInput = row.querySelector('.text-input');
                        const text = textInput ? textInput.value.trim() : '';
                        if (text) {{
                            markdown += `${{noteNumber}}. ${{text}} *(custom)*\\n`;
                            noteNumber++;
                        }}
                    }}
                }});
            }}

            if (noteNumber === 1) {{
                markdown += 'No text content found.\\n';
            }}

            // Add metadata
            const subNoteCount = document.querySelectorAll('.sub-row').length;
            const customNoteCount = window.customNotes ? window.customNotes.length : 0;
            markdown += `\\n---\\n`;
            markdown += `*Exported: ${{new Date().toLocaleString()}}*\\n`;
            markdown += `*Total notes: ${{noteNumber - 1}} (from ${{originalData.length}} detected)*\\n`;
            if (deletedRows.size > 0) {{
                markdown += `*Excluded ${{deletedRows.size}} false positives*\\n`;
            }}
            if (subNoteCount > 0) {{
                markdown += `*Includes ${{subNoteCount}} sub-notes from broken notes*\\n`;
            }}
            if (customNoteCount > 0) {{
                markdown += `*Includes ${{customNoteCount}} custom user-drawn notes*\\n`;
            }}

            const blob = new Blob([markdown], {{
                type: 'text/markdown'
            }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'sticky_notes_transcription.md';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            showStatus(`Markdown downloaded! ${{noteNumber - 1}} notes included.`, 'success');
        }}

        // Update text displays when inputs change
        document.addEventListener('input', function(e) {{
            if (e.target.classList.contains('text-input')) {{
                const row = e.target.closest('tr');
                const textDisplay = row.querySelector('.text-display');
                if (textDisplay) {{
                    if (e.target.value.trim()) {{
                        textDisplay.innerHTML = e.target.value;
                    }} else {{
                        textDisplay.innerHTML = '<span class="no-text">No text detected</span>';
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
        
        # Color mapping for badges
        color_styles = {
            'yellow': 'background-color: #FFD700;',
            'light_yellow': 'background-color: #FFFFE0; color: #333;',
            'orange': 'background-color: #FF8C00;',
            'red': 'background-color: #DC143C;',
            'light_red': 'background-color: #FFB6C1; color: #333;',
            'green': 'background-color: #32CD32;',
            'light_green': 'background-color: #90EE90; color: #333;',
            'blue': 'background-color: #4169E1;',
            'light_blue': 'background-color: #ADD8E6; color: #333;',
            'cyan': 'background-color: #00CED1;',
            'purple': 'background-color: #8A2BE2;'
        }
        
        # Generate table rows
        table_rows = []
        for result in results:
            note_id = result['note_id']
            color = result['color']
            bbox = result['bbox']
            area = result['area']
            detection_conf = result['detection_confidence']
            ocr_conf = result['ocr_confidence']
            text = result['text']
            
            # Color badge style
            color_style = color_styles.get(color, 'background-color: #666;')
            
            # Confidence bar color
            if detection_conf >= 80:
                conf_color = '#4CAF50'  # Green
            elif detection_conf >= 60:
                conf_color = '#FF9800'  # Orange
            else:
                conf_color = '#F44336'  # Red
                
            # OCR confidence bar color
            if ocr_conf >= 80:
                ocr_color = '#4CAF50'  # Green
            elif ocr_conf >= 60:
                ocr_color = '#FF9800'  # Orange
            else:
                ocr_color = '#F44336'  # Red
            
            # Image path
            image_filename = f"cropped_note_{note_id}.jpg"
            
            # Text content
            text_display = text if text.strip() else '<span class="no-text">No text detected</span>'
            
            row = f"""
                <tr data-note-id="{note_id}">
                    <td><strong>#{note_id}</strong></td>
                    <td>
                        <img src="{image_filename}" alt="Note {note_id}" class="note-image" 
                             onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                        <div style="display:none; padding: 20px; background: #f0f0f0; text-align: center; border-radius: 4px;">
                            Image not found
                        </div>
                        <br>
                        <button class="details-toggle" onclick="toggleDetails({note_id - 1})">
                            📊 Technical Details
                        </button>
                        <div class="technical-details" id="details-{note_id - 1}">
                            <strong>Color:</strong> <span class="color-badge" style="{color_style}">{color}</span><br>
                            <strong>Bounding Box:</strong> <span class="bbox">x:{bbox[0]}, y:{bbox[1]}, w:{bbox[2]}, h:{bbox[3]}</span><br>
                            <strong>Area:</strong> {area:,.0f} px²<br>
                            <strong>Detection Confidence:</strong> 
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: {detection_conf}%; background-color: {conf_color};"></div>
                            </div>
                            <small>{detection_conf:.1f}%</small><br>
                            <strong>OCR Confidence:</strong> 
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: {ocr_conf}%; background-color: {ocr_color};"></div>
                            </div>
                            <small>{ocr_conf:.1f}%</small>
                        </div>
                    </td>
                    <td class="text-content">
                        <div class="text-display">{text_display}</div>
                        <textarea class="text-input" placeholder="Enter transcribed text...">{text}</textarea>
                    </td>
                    <td class="delete-column">
                        <button class="btn btn-danger btn-small" onclick="toggleRowDelete({note_id - 1})" title="Mark as false positive">
                            🗑️ Delete
                        </button>
                        <br>
                        <button class="break-note-btn" onclick="breakNote({note_id - 1})" title="Split into multiple notes">
                            ✂️ Break Note
                        </button>
                    </td>
                </tr>
            """
            table_rows.append(row)
        
        # Fill in the template
        html_filled = html_content.format(
            image_name=os.path.basename(self.image_path),
            total_notes=len(results),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            transcription_method=results[0]['transcription_method'] if results else 'None',
            table_rows=''.join(table_rows),
            original_data=json.dumps(results)
        )
        
        # Save HTML file
        html_path = os.path.join(output_dir, 'detection_report.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_filled)
        
        return html_path

def main():
    parser = argparse.ArgumentParser(description='Detect and transcribe sticky notes')
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('--tesseract-path', help='Path to tesseract executable (overrides TESSERACT_PATH env var)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (overrides DEBUG_MODE env var)')
    parser.add_argument('--transcribe', action='store_true', help='Enable OCR transcription of detected notes (default: detection only)')
    parser.add_argument('--llm', choices=['claude', 'chatgpt', 'ollama'], help='Use LLM for transcription instead of local OCR')
    parser.add_argument('--api-key', help='API key for the selected LLM provider (overrides env vars)')
    parser.add_argument('--api-key-env', help='Environment variable name containing the API key')
    parser.add_argument('--ollama-model', default='llava', help='Ollama model to use for transcription (default: llava)')
    parser.add_argument('--saturation-boost', type=float, help='Color saturation boost factor (1.0=no change, 1.5=50%% boost, 2.0=double, etc.)')
    
    args = parser.parse_args()
    
    # Handle API key - check command line first, then environment variables
    api_key = None
    if args.llm and args.llm != 'ollama':
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
    elif args.llm == 'ollama':
        # Ollama doesn't need an API key
        args.transcribe = True
        print("Using Ollama for local LLM transcription. Make sure Ollama is running with a vision model.")
        print(f"Model: {args.ollama_model}")
        print("If you haven't installed Ollama yet, visit: https://ollama.ai")
        print(f"To install the model, run: ollama pull {args.ollama_model}")
    
    # Set tesseract path - command line argument takes precedence over environment variable
    tesseract_path = args.tesseract_path or os.getenv('TESSERACT_PATH')
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found")
        return
    
    # Process the image - debug parameter from command line takes precedence over env var
    detector = StickyNoteDetector(args.image_path, debug=args.debug if args.debug else None, saturation_boost=args.saturation_boost)
    results, output_dir = detector.process_all_notes(
        transcribe=args.transcribe,
        llm_provider=args.llm,
        api_key=api_key,
        ollama_model=args.ollama_model
    )
    
    print(f"\nProcessing complete! Check '{output_dir}' for results.")

if __name__ == "__main__":
    main()
