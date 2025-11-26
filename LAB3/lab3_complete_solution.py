"""
LAB3: Complete Solution for Feature Detection and Extraction Tasks
This script implements solutions for 5 main tasks in computer vision:
1. Feature Detection using ORB/SIFT
2. HOG (Histogram of Oriented Gradients) Features
3. LBP (Local Binary Patterns) Features  
4. Image Matching and Correspondence
5. Object Detection and Recognition

Author: Solution for AVPR LAB3
Date: November 2024
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, io, exposure
from skimage.feature import hog
import os
from scipy.spatial.distance import euclidean

class LAB3FeatureExtraction:
    def __init__(self, images_path="Materials and codes for features detection and extraction/code/images"):
        self.images_path = images_path
        self.setup_matplotlib()
    
    def setup_matplotlib(self):
        """Setup matplotlib for better visualization"""
        plt.rcParams['figure.figsize'] = (15, 10)
        plt.rcParams['font.size'] = 12
    
    def load_image(self, filename, color_mode=cv2.IMREAD_GRAYSCALE):
        """Load image with error handling"""
        image_path = os.path.join(self.images_path, filename)
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found")
            return None
        
        image = cv2.imread(image_path, color_mode)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return None
        
        return image
    
    def task1_feature_detection(self):
        """
        Task 1: Feature Detection using ORB and SIFT
        Compare different feature detectors and their characteristics
        """
        print("\n" + "="*60)
        print("TASK 1: FEATURE DETECTION USING ORB AND SIFT")
        print("="*60)
        
        # Load test image
        image = self.load_image('01.png')
        if image is None:
            image = self.load_image('Picture1.jpg')
        if image is None:
            print("No suitable image found for Task 1")
            return
        
        # Initialize feature detectors
        orb = cv2.ORB_create(nfeatures=500)
        sift = cv2.SIFT_create()
        
        # Detect keypoints and compute descriptors
        kp_orb, desc_orb = orb.getectAndCompute(image, None)
        kp_sift, desc_sift = sift.detectAndCompute(image, None)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # ORB keypoints
        img_orb = cv2.drawKeypoints(image, kp_orb, None, 
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        axes[0, 1].imshow(img_orb)
        axes[0, 1].set_title(f'ORB Keypoints (Count: {len(kp_orb)})')
        axes[0, 1].axis('off')
        
        # SIFT keypoints
        img_sift = cv2.drawKeypoints(image, kp_sift, None,
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        axes[1, 0].imshow(img_sift)
        axes[1, 0].set_title(f'SIFT Keypoints (Count: {len(kp_sift)})')
        axes[1, 0].axis('off')
        
        # Comparison of keypoint distributions
        orb_coords = np.array([kp.pt for kp in kp_orb])
        sift_coords = np.array([kp.pt for kp in kp_sift])
        
        axes[1, 1].scatter(orb_coords[:, 0], orb_coords[:, 1], 
                          alpha=0.6, c='red', s=10, label='ORB')
        axes[1, 1].scatter(sift_coords[:, 0], sift_coords[:, 1], 
                          alpha=0.6, c='blue', s=10, label='SIFT')
        axes[1, 1].set_title('Keypoint Distribution Comparison')
        axes[1, 1].legend()
        axes[1, 1].set_xlim(0, image.shape[1])
        axes[1, 1].set_ylim(image.shape[0], 0)
        
        plt.tight_layout()
        plt.savefig('task1_feature_detection.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Analysis
        print(f"\nFEATURE DETECTOR ANALYSIS:")
        print(f"ORB Keypoints: {len(kp_orb)}")
        print(f"SIFT Keypoints: {len(kp_sift)}")
        print(f"ORB Descriptor shape: {desc_orb.shape if desc_orb is not None else 'None'}")
        print(f"SIFT Descriptor shape: {desc_sift.shape if desc_sift is not None else 'None'}")
        
        return {'orb': (kp_orb, desc_orb), 'sift': (kp_sift, desc_sift)}
    
    def task2_hog_features(self):
        """
        Task 2: HOG (Histogram of Oriented Gradients) Features
        Extract and visualize HOG features from images
        """
        print("\n" + "="*60)
        print("TASK 2: HOG (HISTOGRAM OF ORIENTED GRADIENTS) FEATURES")
        print("="*60)
        
        # Load test image
        image = self.load_image('Picture1.jpg')
        if image is None:
            image = self.load_image('i1.jpg')
        if image is None:
            print("No suitable image found for Task 2")
            return
        
        # Resize image for consistent processing
        image = cv2.resize(image, (128, 128))
        
        # Extract HOG features with visualization
        features, hog_image = hog(image, 
                                 orientations=9,
                                 pixels_per_cell=(8, 8),
                                 cells_per_block=(2, 2),
                                 block_norm='L2-Hys',
                                 visualize=True,
                                 feature_vector=True)
        
        # Compute gradients for analysis
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        magnitude, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Gradient magnitude
        axes[0, 1].imshow(magnitude, cmap='hot')
        axes[0, 1].set_title('Gradient Magnitude')
        axes[0, 1].axis('off')
        
        # Gradient direction
        axes[0, 2].imshow(angle, cmap='hsv')
        axes[0, 2].set_title('Gradient Direction')
        axes[0, 2].axis('off')
        
        # HOG visualization
        axes[1, 0].imshow(hog_image, cmap='gray')
        axes[1, 0].set_title('HOG Visualization')
        axes[1, 0].axis('off')
        
        # HOG feature histogram
        axes[1, 1].hist(features, bins=50, alpha=0.7)
        axes[1, 1].set_title('HOG Feature Distribution')
        axes[1, 1].set_xlabel('Feature Value')
        axes[1, 1].set_ylabel('Frequency')
        
        # Feature vector visualization
        axes[1, 2].plot(features[:100])
        axes[1, 2].set_title('HOG Feature Vector (First 100 components)')
        axes[1, 2].set_xlabel('Feature Index')
        axes[1, 2].set_ylabel('Feature Value')
        
        plt.tight_layout()
        plt.savefig('task2_hog_features.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nHOG FEATURE ANALYSIS:")
        print(f"Feature vector length: {len(features)}")
        print(f"Feature value range: [{features.min():.4f}, {features.max():.4f}]")
        print(f"Feature mean: {features.mean():.4f}")
        print(f"Feature std: {features.std():.4f}")
        
        return features
    
    def task3_lbp_features(self):
        """
        Task 3: LBP (Local Binary Patterns) Features
        Extract LBP features with different parameters and compare them
        """
        print("\n" + "="*60)
        print("TASK 3: LBP (LOCAL BINARY PATTERNS) FEATURES")
        print("="*60)
        
        # Load test image
        image = self.load_image('2.jpg')
        if image is None:
            image = self.load_image('Picture2.jpg')
        if image is None:
            print("No suitable image found for Task 3")
            return
        
        # Normalize image
        if image.max() > 1:
            image = image.astype(np.float64) / 255.0
        
        # Convert to uint8 for LBP computation
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Compute LBP with different parameters
        lbp_8_1 = feature.local_binary_pattern(image_uint8, P=8, R=1, method='uniform')
        lbp_16_2 = feature.local_binary_pattern(image_uint8, P=16, R=2, method='uniform')
        lbp_24_3 = feature.local_binary_pattern(image_uint8, P=24, R=3, method='uniform')
        
        # Compute histograms
        def compute_lbp_histogram(lbp_image, n_points):
            n_bins = n_points + 2
            hist, _ = np.histogram(lbp_image.ravel(), bins=n_bins, 
                                 range=(0, n_bins), density=True)
            return hist
        
        hist_8_1 = compute_lbp_histogram(lbp_8_1, 8)
        hist_16_2 = compute_lbp_histogram(lbp_16_2, 16)
        hist_24_3 = compute_lbp_histogram(lbp_24_3, 24)
        
        # Create visualization
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # LBP images
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(lbp_8_1, cmap='gray')
        axes[0, 1].set_title('LBP (8, 1)')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(lbp_16_2, cmap='gray')
        axes[0, 2].set_title('LBP (16, 2)')
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(lbp_24_3, cmap='gray')
        axes[0, 3].set_title('LBP (24, 3)')
        axes[0, 3].axis('off')
        
        # Histograms
        axes[1, 0].bar(range(len(hist_8_1)), hist_8_1)
        axes[1, 0].set_title('LBP (8,1) Histogram')
        axes[1, 0].set_xlabel('LBP Code')
        axes[1, 0].set_ylabel('Frequency')
        
        axes[1, 1].bar(range(len(hist_16_2)), hist_16_2)
        axes[1, 1].set_title('LBP (16,2) Histogram')
        axes[1, 1].set_xlabel('LBP Code')
        axes[1, 1].set_ylabel('Frequency')
        
        axes[1, 2].bar(range(len(hist_24_3)), hist_24_3)
        axes[1, 2].set_title('LBP (24,3) Histogram')
        axes[1, 2].set_xlabel('LBP Code')
        axes[1, 2].set_ylabel('Frequency')
        
        # Comparison of histograms
        axes[1, 3].plot(hist_8_1[:10], 'r-', label='LBP (8,1)', linewidth=2)
        axes[1, 3].plot(hist_16_2[:10], 'g-', label='LBP (16,2)', linewidth=2)
        axes[1, 3].plot(hist_24_3[:10], 'b-', label='LBP (24,3)', linewidth=2)
        axes[1, 3].set_title('LBP Histogram Comparison')
        axes[1, 3].legend()
        axes[1, 3].set_xlabel('First 10 LBP Codes')
        axes[1, 3].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('task3_lbp_features.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nLBP FEATURE ANALYSIS:")
        print(f"LBP (8,1) - Unique values: {len(np.unique(lbp_8_1))}")
        print(f"LBP (16,2) - Unique values: {len(np.unique(lbp_16_2))}")
        print(f"LBP (24,3) - Unique values: {len(np.unique(lbp_24_3))}")
        
        return {'lbp_8_1': lbp_8_1, 'lbp_16_2': lbp_16_2, 'lbp_24_3': lbp_24_3,
                'hist_8_1': hist_8_1, 'hist_16_2': hist_16_2, 'hist_24_3': hist_24_3}
    
    def task4_image_matching(self):
        """
        Task 4: Image Matching and Correspondence
        Match features between two images using different descriptors
        """
        print("\n" + "="*60)
        print("TASK 4: IMAGE MATCHING AND CORRESPONDENCE")
        print("="*60)
        
        # Load two images for matching
        image1 = self.load_image('i1.jpg')
        image2 = self.load_image('i2.jpg')
        
        if image1 is None or image2 is None:
            # Use alternative images
            image1 = self.load_image('Picture1.jpg')
            image2 = self.load_image('Picture2.jpg')
        
        if image1 is None or image2 is None:
            print("No suitable image pair found for Task 4")
            return
        
        # Initialize feature detectors
        sift = cv2.SIFT_create()
        orb = cv2.ORB_create(nfeatures=500)
        
        # Detect keypoints and compute descriptors for both methods
        kp1_sift, desc1_sift = sift.detectAndCompute(image1, None)
        kp2_sift, desc2_sift = sift.detectAndCompute(image2, None)
        
        kp1_orb, desc1_orb = orb.detectAndCompute(image1, None)
        kp2_orb, desc2_orb = orb.detectAndCompute(image2, None)
        
        # SIFT matching
        if desc1_sift is not None and desc2_sift is not None:
            bf_sift = cv2.BFMatcher()
            matches_sift = bf_sift.knnMatch(desc1_sift, desc2_sift, k=2)
            
            good_matches_sift = []
            for match_pair in matches_sift:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches_sift.append(m)
        else:
            good_matches_sift = []
        
        # ORB matching
        if desc1_orb is not None and desc2_orb is not None:
            bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches_orb = bf_orb.match(desc1_orb, desc2_orb)
            good_matches_orb = sorted(matches_orb, key=lambda x: x.distance)[:50]
        else:
            good_matches_orb = []
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # SIFT matches
        if good_matches_sift:
            img_matches_sift = cv2.drawMatches(image1, kp1_sift, image2, kp2_sift,
                                             good_matches_sift, None,
                                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            axes[0, 0].imshow(img_matches_sift)
            axes[0, 0].set_title(f'SIFT Matches (Count: {len(good_matches_sift)})')
        else:
            axes[0, 0].text(0.5, 0.5, 'No SIFT matches found', 
                           transform=axes[0, 0].transAxes, ha='center', va='center')
            axes[0, 0].set_title('SIFT Matches')
        axes[0, 0].axis('off')
        
        # ORB matches
        if good_matches_orb:
            img_matches_orb = cv2.drawMatches(image1, kp1_orb, image2, kp2_orb,
                                            good_matches_orb, None,
                                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            axes[0, 1].imshow(img_matches_orb)
            axes[0, 1].set_title(f'ORB Matches (Count: {len(good_matches_orb)})')
        else:
            axes[0, 1].text(0.5, 0.5, 'No ORB matches found', 
                           transform=axes[0, 1].transAxes, ha='center', va='center')
            axes[0, 1].set_title('ORB Matches')
        axes[0, 1].axis('off')
        
        # Match distance distributions
        if good_matches_sift:
            sift_distances = [m.distance for m in good_matches_sift]
            axes[1, 0].hist(sift_distances, bins=20, alpha=0.7, color='blue')
            axes[1, 0].set_title('SIFT Match Distance Distribution')
            axes[1, 0].set_xlabel('Distance')
            axes[1, 0].set_ylabel('Frequency')
        
        if good_matches_orb:
            orb_distances = [m.distance for m in good_matches_orb]
            axes[1, 1].hist(orb_distances, bins=20, alpha=0.7, color='red')
            axes[1, 1].set_title('ORB Match Distance Distribution')
            axes[1, 1].set_xlabel('Distance')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('task4_image_matching.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nIMAGE MATCHING ANALYSIS:")
        print(f"SIFT keypoints: Image1={len(kp1_sift)}, Image2={len(kp2_sift)}")
        print(f"ORB keypoints: Image1={len(kp1_orb)}, Image2={len(kp2_orb)}")
        print(f"Good SIFT matches: {len(good_matches_sift)}")
        print(f"Good ORB matches: {len(good_matches_orb)}")
        
        return {'sift_matches': good_matches_sift, 'orb_matches': good_matches_orb}
    
    def task5_object_detection(self):
        """
        Task 5: Object Detection and Recognition
        Implement circle detection using HoughCircles and template matching
        """
        print("\n" + "="*60)
        print("TASK 5: OBJECT DETECTION AND RECOGNITION")
        print("="*60)
        
        # Load image for circle detection
        image = self.load_image('finding_circles.jpg', cv2.IMREAD_COLOR)
        if image is None:
            # Create synthetic image with circles for demonstration
            image = np.zeros((400, 400, 3), dtype=np.uint8)
            cv2.circle(image, (100, 100), 30, (255, 255, 255), -1)
            cv2.circle(image, (300, 150), 25, (255, 255, 255), -1)
            cv2.circle(image, (200, 300), 35, (255, 255, 255), -1)
            print("Using synthetic image for circle detection")
        
        original_image = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply preprocessing
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Circle detection using HoughCircles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=100
        )
        
        detected_image = original_image.copy()
        circle_count = 0
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            circle_count = len(circles)
            
            for (x, y, r) in circles:
                # Draw outer circle
                cv2.circle(detected_image, (x, y), r, (0, 255, 0), 2)
                # Draw center
                cv2.circle(detected_image, (x, y), 2, (0, 0, 255), 3)
        
        # Edge detection for additional analysis
        edges = cv2.Canny(blurred, 50, 150)
        
        # Template matching (if we have a suitable template)
        template_match_result = None
        if circle_count > 0 and len(circles) > 0:
            # Create a simple circle template
            template_size = 50
            template = np.zeros((template_size, template_size), dtype=np.uint8)
            cv2.circle(template, (template_size//2, template_size//2), 
                      template_size//4, 255, -1)
            
            # Perform template matching
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            template_match_result = result
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Preprocessed image
        axes[0, 1].imshow(blurred, cmap='gray')
        axes[0, 1].set_title('Preprocessed (Blurred)')
        axes[0, 1].axis('off')
        
        # Edge detection
        axes[0, 2].imshow(edges, cmap='gray')
        axes[0, 2].set_title('Edge Detection (Canny)')
        axes[0, 2].axis('off')
        
        # Circle detection result
        axes[1, 0].imshow(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title(f'Circle Detection (Found: {circle_count})')
        axes[1, 0].axis('off')
        
        # Template matching result
        if template_match_result is not None:
            axes[1, 1].imshow(template_match_result, cmap='hot')
            axes[1, 1].set_title('Template Matching Result')
            axes[1, 1].axis('off')
        else:
            axes[1, 1].text(0.5, 0.5, 'No template matching performed', 
                           transform=axes[1, 1].transAxes, ha='center', va='center')
            axes[1, 1].set_title('Template Matching')
            axes[1, 1].axis('off')
        
        # Analysis plot
        if circles is not None and len(circles) > 0:
            radii = circles[:, 2]
            axes[1, 2].hist(radii, bins=10, alpha=0.7, color='green')
            axes[1, 2].set_title('Detected Circle Radii Distribution')
            axes[1, 2].set_xlabel('Radius')
            axes[1, 2].set_ylabel('Count')
        else:
            axes[1, 2].text(0.5, 0.5, 'No circles detected', 
                           transform=axes[1, 2].transAxes, ha='center', va='center')
            axes[1, 2].set_title('Circle Analysis')
        
        plt.tight_layout()
        plt.savefig('task5_object_detection.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nOBJECT DETECTION ANALYSIS:")
        print(f"Detected circles: {circle_count}")
        if circles is not None and len(circles) > 0:
            print(f"Circle positions and radii:")
            for i, (x, y, r) in enumerate(circles):
                print(f"  Circle {i+1}: Center=({x}, {y}), Radius={r}")
        
        return {'circles': circles, 'circle_count': circle_count}
    
    def run_all_tasks(self):
        """Run all 5 tasks in sequence"""
        print("LAB3: FEATURE DETECTION AND EXTRACTION - COMPLETE SOLUTION")
        print("="*70)
        
        results = {}
        
        # Run all tasks
        results['task1'] = self.task1_feature_detection()
        results['task2'] = self.task2_hog_features()
        results['task3'] = self.task3_lbp_features()
        results['task4'] = self.task4_image_matching()
        results['task5'] = self.task5_object_detection()
        
        # Summary
        print("\n" + "="*70)
        print("LAB3 SUMMARY - ALL TASKS COMPLETED")
        print("="*70)
        print("Generated output files:")
        print("  - task1_feature_detection.png")
        print("  - task2_hog_features.png")
        print("  - task3_lbp_features.png")
        print("  - task4_image_matching.png")
        print("  - task5_object_detection.png")
        print("\nAll tasks have been successfully completed!")
        
        return results

def main():
    """Main function to run LAB3 solution"""
    # Initialize the LAB3 solution
    lab3 = LAB3FeatureExtraction()
    
    # Run all tasks
    results = lab3.run_all_tasks()
    
    return results

if __name__ == "__main__":
    main()
