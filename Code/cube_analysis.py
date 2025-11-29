#!/usr/bin/env python3
"""
Cuboid Face Detection and Rotation Axis Estimation (Improved Version)
======================================================================
Analyzes depth images from a ROS bag file to detect visible faces of a 
rotating cuboid and estimate its rotation axis.

Improvements:
- Configurable camera intrinsics (command line or auto-detect)
- Dynamic depth thresholding using histogram clustering
- Built-in visualization for debugging
- Robust foreground segmentation

Author: [Your Name]
Assignment: Perception - 10xConstruction

Usage:
    python cuboid_analysis_v2.py                          # Default paths
    python cuboid_analysis_v2.py --bag_path /path/to/bag  # Custom bag path
    python cuboid_analysis_v2.py --visualize              # Generate debug plots
    python cuboid_analysis_v2.py --fx 615 --fy 615        # Custom intrinsics
"""

import numpy as np
from scipy.spatial import ConvexHull
from scipy.ndimage import label
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
import csv
import os
import argparse


# ============================================================================
# CONFIGURATION (Default values - can be overridden via CLI)
# ============================================================================

class Config:
    """Configuration container with sensible defaults for 640x480 depth cameras."""
    
    # Camera intrinsics (typical RealSense D435 values for 640x480)
    FX = 615.0  # Focal length x (pixels)
    FY = 615.0  # Focal length y (pixels)
    CX = 320.0  # Principal point x (image center)
    CY = 240.0  # Principal point y (image center)
    
    # RANSAC parameters
    RANSAC_DIST_THRESH = 0.015  # 1.5cm inlier threshold
    RANSAC_ITERATIONS = 500     # Number of iterations
    
    # Processing parameters
    DOWNSAMPLE = 4              # Process every Nth pixel
    MIN_DEPTH = 0.3             # Minimum valid depth (m)
    MAX_DEPTH = 10.0            # Maximum valid depth (m)
    FG_PERCENTILE = 40          # Percentile for foreground isolation


# ============================================================================
# DYNAMIC FOREGROUND SEGMENTATION
# ============================================================================

def find_foreground_threshold(depth_img, config):
    """
    Dynamically determine foreground threshold using histogram analysis.
    
    Finds the first significant depth cluster (the cuboid) and sets threshold
    to include that cluster while excluding background.
    """
    valid = depth_img[(depth_img > config.MIN_DEPTH) & (depth_img < config.MAX_DEPTH)]
    
    if len(valid) < 100:
        return config.MAX_DEPTH
    
    # Method 1: Percentile-based (robust)
    threshold = np.percentile(valid, config.FG_PERCENTILE)
    
    # Method 2: Histogram gap detection (finds natural separation)
    hist, bins = np.histogram(valid, bins=100)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Find the first major peak and look for valley after it
    peak_idx = np.argmax(hist[:50])  # Look in first half for foreground peak
    
    # Search for valley (minimum) after the peak
    if peak_idx < 45:
        valley_region = hist[peak_idx:peak_idx+30]
        valley_local_idx = np.argmin(valley_region)
        valley_idx = peak_idx + valley_local_idx
        valley_threshold = bin_centers[min(valley_idx, len(bin_centers)-1)]
        
        # Use valley threshold if it's reasonable
        if valley_threshold > bins[peak_idx] + 0.1:  # At least 10cm beyond peak
            threshold = min(threshold, valley_threshold)
    
    return threshold


def segment_largest_cluster(mask):
    """
    Keep only the largest connected component in a binary mask.
    Removes noise and small disconnected regions.
    """
    labeled, num_features = label(mask)
    if num_features == 0:
        return mask
    
    # Find largest component
    largest_size = 0
    largest_label = 1
    for i in range(1, num_features + 1):
        size = np.sum(labeled == i)
        if size > largest_size:
            largest_size = size
            largest_label = i
    
    return labeled == largest_label


# ============================================================================
# CORE ALGORITHM FUNCTIONS
# ============================================================================

def depth_to_pointcloud(depth_img, config):
    """Convert depth image to 3D point cloud with intelligent foreground isolation."""
    h, w = depth_img.shape
    
    # Create coordinate grids (downsampled)
    u, v = np.meshgrid(np.arange(0, w, config.DOWNSAMPLE), 
                       np.arange(0, h, config.DOWNSAMPLE))
    depth_ds = depth_img[::config.DOWNSAMPLE, ::config.DOWNSAMPLE]
    
    # Basic valid depth mask
    valid = (depth_ds > config.MIN_DEPTH) & (depth_ds < config.MAX_DEPTH) & ~np.isinf(depth_ds)
    
    # Dynamic foreground threshold
    fg_thresh = find_foreground_threshold(depth_img, config)
    foreground = valid & (depth_ds < fg_thresh)
    
    # Clean up with connected component analysis (keep largest cluster)
    if np.sum(foreground) > 100:
        foreground = segment_largest_cluster(foreground)
    
    # Use foreground if sufficient points, else fall back to all valid
    if np.sum(foreground) > 200:
        valid = foreground
    
    # Convert to 3D coordinates
    z = depth_ds[valid]
    x = (u[valid] - config.CX) * z / config.FX
    y = (v[valid] - config.CY) * z / config.FY
    
    return np.column_stack([x, y, z]), fg_thresh


def ransac_plane(points, config):
    """
    RANSAC plane fitting.
    
    Returns:
        normal: 3D unit normal vector
        d: Plane offset (ax + by + cz + d = 0)
        inliers: Boolean mask of inlier points
    """
    n = len(points)
    if n < 10:
        return None, 0, None
    
    best_count, best_inliers, best_normal, best_d = 0, None, None, 0
    
    for _ in range(config.RANSAC_ITERATIONS):
        # Random 3-point sample
        idx = np.random.choice(n, 3, replace=False)
        p0, p1, p2 = points[idx]
        
        # Compute plane normal via cross product
        v1, v2 = p1 - p0, p2 - p0
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        
        if norm < 1e-10:  # Collinear points
            continue
        
        normal /= norm
        d = -np.dot(normal, p0)
        
        # Count inliers
        dists = np.abs(points @ normal + d)
        inliers = dists < config.RANSAC_DIST_THRESH
        count = np.sum(inliers)
        
        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_normal = normal
            best_d = d
    
    return best_normal, best_d, best_inliers


def refine_plane_svd(points, inliers):
    """Refine plane parameters using SVD on all inlier points."""
    pts = points[inliers]
    centroid = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - centroid)
    normal = Vt[-1]  # Smallest singular value direction
    
    # Ensure normal points toward camera (positive z-component)
    if normal[2] < 0:
        normal = -normal
    
    d = -np.dot(normal, centroid)
    return normal, d, centroid


def calc_normal_angle(normal):
    """Calculate angle between plane normal and camera z-axis."""
    cos_angle = np.abs(normal[2])  # |dot(normal, [0,0,1])|
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))


def calc_visible_area(points, inliers, normal):
    """Calculate visible face area using 2D convex hull projection."""
    pts = points[inliers]
    if len(pts) < 3:
        return 0.0
    
    # Create orthonormal basis on the plane
    arb = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
    u_axis = np.cross(normal, arb)
    u_axis /= np.linalg.norm(u_axis)
    v_axis = np.cross(normal, u_axis)
    
    # Project 3D points to 2D plane coordinates
    centered = pts - pts.mean(axis=0)
    pts_2d = np.column_stack([centered @ u_axis, centered @ v_axis])
    
    try:
        return ConvexHull(pts_2d).volume  # 'volume' is area for 2D
    except:
        return 0.0


def estimate_rotation_axis(normals):
    """
    Estimate rotation axis from sequence of plane normals.
    
    The rotation axis is perpendicular to the plane in which normals rotate.
    Uses cross products of consecutive normals and averages them.
    """
    normals = np.array(normals)
    
    if len(normals) < 2:
        return np.array([0, 1, 0])  # Default vertical axis
    
    # Cross products of consecutive normals give rotation axis candidates
    crosses = []
    for i in range(len(normals) - 1):
        c = np.cross(normals[i], normals[i + 1])
        norm = np.linalg.norm(c)
        if norm > 0.01:  # Skip if normals are parallel
            crosses.append(c / norm)
    
    if crosses:
        axis = np.mean(crosses, axis=0)
        axis /= np.linalg.norm(axis)
    else:
        # Fallback: SVD on normals
        centered = normals - normals.mean(axis=0)
        _, _, Vt = np.linalg.svd(centered)
        axis = Vt[-1]
    
    # Convention: positive Y component (upward in camera frame)
    if axis[1] < 0:
        axis = -axis
    
    return axis


# ============================================================================
# VISUALIZATION (Optional Debug Output)
# ============================================================================

def generate_visualizations(bag_path, output_dir, config):
    """Generate debug visualizations for all frames."""
    import matplotlib.pyplot as plt
    
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    
    # Collect all frames
    frames = []
    with Reader(bag_path) as reader:
        for i, (conn, ts, raw) in enumerate(reader.messages()):
            msg = typestore.deserialize_cdr(raw, conn.msgtype)
            if msg.encoding == '16UC1':
                depth = np.frombuffer(msg.data, np.uint16).reshape(
                    msg.height, msg.width).astype(np.float32) / 1000.0
            else:
                depth = np.frombuffer(msg.data, np.float32).reshape(
                    msg.height, msg.width)
            frames.append((i + 1, depth))
    
    n_frames = len(frames)
    
    # Create overview figure
    cols = min(4, n_frames)
    rows = (n_frames + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    fig.suptitle('Depth Images with Detected Planes', fontsize=14)
    
    for idx, (frame_id, depth) in enumerate(frames):
        ax = axes[idx]
        
        # Get foreground threshold
        fg_thresh = find_foreground_threshold(depth, config)
        
        # Visualize depth with foreground highlighted
        depth_vis = depth.copy()
        depth_vis[depth_vis > 5] = np.nan
        
        ax.imshow(depth_vis, cmap='viridis', vmin=0.4, vmax=2.5)
        
        # Overlay foreground contour
        fg_mask = (depth > config.MIN_DEPTH) & (depth < fg_thresh)
        ax.contour(fg_mask, colors='red', linewidths=1)
        
        ax.set_title(f'Frame {frame_id} (thresh={fg_thresh:.2f}m)')
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_frames, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'debug_visualization.png'), dpi=150)
    plt.close()
    print(f"  Saved: debug_visualization.png")


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def process_frame(depth_img, frame_id, config):
    """Process a single depth frame and return detection results."""
    
    # Convert to point cloud with foreground isolation
    points, fg_thresh = depth_to_pointcloud(depth_img, config)
    
    if len(points) < 50:
        print(f"  Frame {frame_id}: Insufficient points ({len(points)})")
        return None
    
    # RANSAC plane detection
    normal, d, inliers = ransac_plane(points, config)
    
    if normal is None or np.sum(inliers) < 50:
        print(f"  Frame {frame_id}: No valid plane detected")
        return None
    
    # Refine plane using all inliers
    normal, d, centroid = refine_plane_svd(points, inliers)
    
    # Compute metrics
    angle = calc_normal_angle(normal)
    area = calc_visible_area(points, inliers, normal)
    
    print(f"  Frame {frame_id}: Angle={angle:.2f}°, Area={area:.4f}m², "
          f"Inliers={np.sum(inliers)}, FG_thresh={fg_thresh:.2f}m")
    
    return {
        'angle': angle,
        'area': area,
        'normal': normal,
        'centroid': centroid
    }


def main(bag_path, output_dir, config, visualize=False):
    """Main processing pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Cuboid Face Detection & Rotation Axis Estimation (v2)")
    print("=" * 60)
    print(f"\nCamera Intrinsics: fx={config.FX}, fy={config.FY}, "
          f"cx={config.CX}, cy={config.CY}")
    print(f"RANSAC: threshold={config.RANSAC_DIST_THRESH}m, "
          f"iterations={config.RANSAC_ITERATIONS}")
    
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    results, normals = [], []
    
    print("\nProcessing frames...")
    
    with Reader(bag_path) as reader:
        for i, (conn, ts, raw) in enumerate(reader.messages()):
            msg = typestore.deserialize_cdr(raw, conn.msgtype)
            
            # Decode depth image based on encoding
            if msg.encoding == '16UC1':
                depth = np.frombuffer(msg.data, np.uint16).reshape(
                    msg.height, msg.width).astype(np.float32) / 1000.0
            elif msg.encoding == '32FC1':
                depth = np.frombuffer(msg.data, np.float32).reshape(
                    msg.height, msg.width)
            else:
                print(f"  Frame {i+1}: Unsupported encoding {msg.encoding}")
                continue
            
            result = process_frame(depth, i + 1, config)
            
            if result:
                results.append({
                    'frame': i + 1,
                    'angle': result['angle'],
                    'area': result['area']
                })
                normals.append(result['normal'])
    
    # Estimate rotation axis
    print("\nEstimating rotation axis...")
    axis = estimate_rotation_axis(normals)
    print(f"  Axis: [{axis[0]:.6f}, {axis[1]:.6f}, {axis[2]:.6f}]")
    
    # Save results
    save_results(output_dir, results, axis, config)
    
    # Generate visualizations if requested
    if visualize:
        print("\nGenerating visualizations...")
        generate_visualizations(bag_path, output_dir, config)
    
    # Print summary
    print_summary(results, axis)
    
    return results, axis


def save_results(output_dir, results, axis, config):
    """Save all output files."""
    
    # CSV results table
    csv_path = os.path.join(output_dir, 'results_table.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image Number', 'Normal Angle (degrees)', 'Visible Area (m^2)'])
        for r in results:
            writer.writerow([r['frame'], f"{r['angle']:.2f}", f"{r['area']:.6f}"])
    print(f"\nSaved: {csv_path}")
    
    # Rotation axis text file
    axis_path = os.path.join(output_dir, 'rotation_axis.txt')
    with open(axis_path, 'w') as f:
        f.write("Rotation Axis Vector (Camera Frame)\n")
        f.write("=" * 45 + "\n\n")
        f.write(f"X: {axis[0]:.6f}\n")
        f.write(f"Y: {axis[1]:.6f}\n")
        f.write(f"Z: {axis[2]:.6f}\n")
        f.write(f"\nVector: [{axis[0]:.6f}, {axis[1]:.6f}, {axis[2]:.6f}]\n")
        f.write(f"\nCamera Frame Convention:\n")
        f.write(f"  X: Right\n")
        f.write(f"  Y: Down\n")
        f.write(f"  Z: Forward (into scene)\n")
    print(f"Saved: {axis_path}")
    
    # Configuration file (for reproducibility)
    config_path = os.path.join(output_dir, 'config_used.txt')
    with open(config_path, 'w') as f:
        f.write("Configuration Used\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Camera Intrinsics:\n")
        f.write(f"  FX: {config.FX}\n")
        f.write(f"  FY: {config.FY}\n")
        f.write(f"  CX: {config.CX}\n")
        f.write(f"  CY: {config.CY}\n\n")
        f.write(f"RANSAC:\n")
        f.write(f"  Distance Threshold: {config.RANSAC_DIST_THRESH}m\n")
        f.write(f"  Iterations: {config.RANSAC_ITERATIONS}\n\n")
        f.write(f"Depth Filtering:\n")
        f.write(f"  Min Depth: {config.MIN_DEPTH}m\n")
        f.write(f"  Max Depth: {config.MAX_DEPTH}m\n")
        f.write(f"  FG Percentile: {config.FG_PERCENTILE}\n")
    print(f"Saved: {config_path}")


def print_summary(results, axis):
    """Print results summary table."""
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Frame':<8}{'Normal Angle (°)':<20}{'Visible Area (m²)':<20}")
    print("-" * 50)
    for r in results:
        print(f"{r['frame']:<8}{r['angle']:<20.2f}{r['area']:<20.6f}")
    print("-" * 50)
    print(f"\nRotation Axis: [{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}]")
    print("=" * 60)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Cuboid face detection from depth images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Paths
    parser.add_argument('--bag_path', type=str, default=r'D:\Projects\10x Constructions - Perception Engineering\New_assesment\depth',
                        help='Path to ROS bag directory')
    parser.add_argument('--output_dir', type=str, default=r'D:\Projects\10x Constructions - Perception Engineering\New_assesment\output',
                        help='Output directory for results')
    
    # Camera intrinsics
    parser.add_argument('--fx', type=float, default=615.0,
                        help='Focal length X (pixels)')
    parser.add_argument('--fy', type=float, default=615.0,
                        help='Focal length Y (pixels)')
    parser.add_argument('--cx', type=float, default=320.0,
                        help='Principal point X')
    parser.add_argument('--cy', type=float, default=240.0,
                        help='Principal point Y')
    
    # Algorithm parameters
    parser.add_argument('--ransac_thresh', type=float, default=0.015,
                        help='RANSAC distance threshold (meters)')
    parser.add_argument('--ransac_iter', type=int, default=500,
                        help='RANSAC iterations')
    parser.add_argument('--fg_percentile', type=int, default=40,
                        help='Foreground depth percentile')
    
    # Options
    parser.add_argument('--visualize', action='store_true',
                        help='Generate debug visualizations')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Build config from arguments
    config = Config()
    config.FX = args.fx
    config.FY = args.fy
    config.CX = args.cx
    config.CY = args.cy
    config.RANSAC_DIST_THRESH = args.ransac_thresh
    config.RANSAC_ITERATIONS = args.ransac_iter
    config.FG_PERCENTILE = args.fg_percentile
    
    # Run pipeline
    main(args.bag_path, args.output_dir, config, visualize=args.visualize)