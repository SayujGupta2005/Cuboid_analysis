#!/usr/bin/env python3
"""
Cuboid Face Detection and Rotation Axis Estimation
===================================================
This script analyzes depth images from a ROS bag file to detect visible faces
of a rotating cuboid and estimate its rotation axis.

Author: [Your Name]
Assignment: Perception - 10xConstruction
"""

import numpy as np
from scipy.spatial import ConvexHull
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
import csv
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

# Camera intrinsic parameters (typical 640x480 depth sensor values)
FX = 600.0  # Focal length in x (pixels)
FY = 600.0  # Focal length in y (pixels)
CX = 320.0  # Principal point x
CY = 240.0  # Principal point y

# RANSAC parameters
RANSAC_DIST_THRESH = 0.015  # 1.5cm tolerance
RANSAC_ITERATIONS = 500
DOWNSAMPLE = 4  # Process every Nth pixel for efficiency

# Depth filtering
MIN_DEPTH, MAX_DEPTH = 0.3, 3.0  # meters


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def depth_to_pointcloud(depth_img):
    """Convert depth image to 3D point cloud with downsampling and foreground isolation."""
    h, w = depth_img.shape
    
    # Downsample and create coordinate grids
    u, v = np.meshgrid(np.arange(0, w, DOWNSAMPLE), np.arange(0, h, DOWNSAMPLE))
    depth_ds = depth_img[::DOWNSAMPLE, ::DOWNSAMPLE]
    
    # Valid depth mask
    valid = (depth_ds > MIN_DEPTH) & (depth_ds < MAX_DEPTH) & ~np.isinf(depth_ds)
    valid_depths = depth_ds[valid]
    
    # Focus on foreground (cuboid) - find the depth cluster in the nearest 30%
    if len(valid_depths) > 100:
        depth_range = valid_depths.max() - valid_depths.min()
        foreground_thresh = valid_depths.min() + 0.35 * depth_range
        foreground_mask = valid & (depth_ds < foreground_thresh)
        
        # Use foreground if it has enough points
        if np.sum(foreground_mask) > 200:
            valid = foreground_mask
    
    z = depth_ds[valid]
    x = (u[valid] - CX) * z / FX
    y = (v[valid] - CY) * z / FY
    
    return np.column_stack([x, y, z])


def ransac_plane(points):
    """RANSAC plane fitting - returns (normal, d, inlier_mask)."""
    n = len(points)
    best_count, best_inliers, best_normal, best_d = 0, None, None, 0
    
    for _ in range(RANSAC_ITERATIONS):
        # Random 3-point sample
        idx = np.random.choice(n, 3, replace=False)
        p0, p1, p2 = points[idx]
        
        # Compute plane normal via cross product
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal /= norm
        d = -np.dot(normal, p0)
        
        # Count inliers
        dists = np.abs(points @ normal + d)
        inliers = dists < RANSAC_DIST_THRESH
        count = np.sum(inliers)
        
        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_normal = normal
            best_d = d
    
    return best_normal, best_d, best_inliers


def refine_plane(points, inliers):
    """Refine plane using SVD on inlier points."""
    pts = points[inliers]
    centroid = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - centroid)
    normal = Vt[-1]
    
    # Ensure normal points toward camera (positive z-component)
    if normal[2] < 0:
        normal = -normal
    
    d = -np.dot(normal, centroid)
    return normal, d


def calc_normal_angle(normal):
    """Angle between plane normal and camera z-axis (degrees)."""
    cos_angle = np.abs(normal[2])  # |dot(normal, [0,0,1])|
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))


def calc_area(points, inliers, normal):
    """Calculate visible face area using convex hull."""
    pts = points[inliers]
    if len(pts) < 3:
        return 0.0
    
    # Create orthonormal basis on plane
    arb = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
    u = np.cross(normal, arb)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    
    # Project to 2D
    centered = pts - pts.mean(axis=0)
    pts_2d = np.column_stack([centered @ u, centered @ v])
    
    try:
        return ConvexHull(pts_2d).volume  # 2D 'volume' = area
    except:
        return 0.0


def estimate_rotation_axis(normals):
    """Estimate rotation axis from sequence of plane normals."""
    normals = np.array(normals)
    
    # Cross products of consecutive normals give rotation axis candidates
    crosses = []
    for i in range(len(normals) - 1):
        c = np.cross(normals[i], normals[i+1])
        norm = np.linalg.norm(c)
        if norm > 0.01:
            crosses.append(c / norm)
    
    if crosses:
        axis = np.mean(crosses, axis=0)
        axis /= np.linalg.norm(axis)
    else:
        # Fallback: SVD
        centered = normals - normals.mean(axis=0)
        _, _, Vt = np.linalg.svd(centered)
        axis = Vt[-1]
    
    # Convention: positive Y component
    if axis[1] < 0:
        axis = -axis
    
    return axis


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def process_frame(depth_img, frame_id):
    """Process single depth frame, return results dict."""
    points = depth_to_pointcloud(depth_img)
    
    if len(points) < 50:
        return None
    
    # RANSAC plane detection
    normal, d, inliers = ransac_plane(points)
    if normal is None or np.sum(inliers) < 50:
        return None
    
    # Refine and compute metrics
    normal, d = refine_plane(points, inliers)
    angle = calc_normal_angle(normal)
    area = calc_area(points, inliers, normal)
    
    # Scale area back (compensate for downsampling)
    # Note: Area calculation on downsampled points is approximate
    
    print(f"  Frame {frame_id}: Angle={angle:.2f}°, Area={area:.4f} m², Inliers={np.sum(inliers)}")
    
    return {'angle': angle, 'area': area, 'normal': normal}


def main(bag_path, output_dir):
    """Main processing function."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 55)
    print("Cuboid Face Detection & Rotation Axis Estimation")
    print("=" * 55)
    
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    results, normals = [], []
    
    print("\nProcessing frames...")
    
    with Reader(bag_path) as reader:
        for i, (conn, ts, raw) in enumerate(reader.messages()):
            msg = typestore.deserialize_cdr(raw, conn.msgtype)
            
            # Decode depth image
            if msg.encoding == '16UC1':
                depth = np.frombuffer(msg.data, np.uint16).reshape(
                    msg.height, msg.width).astype(np.float32) / 1000.0
            else:
                depth = np.frombuffer(msg.data, np.float32).reshape(
                    msg.height, msg.width)
            
            res = process_frame(depth, i + 1)
            if res:
                results.append({'frame': i+1, 'angle': res['angle'], 'area': res['area']})
                normals.append(res['normal'])
    
    # Rotation axis estimation
    print("\nEstimating rotation axis...")
    axis = estimate_rotation_axis(normals) if len(normals) >= 2 else np.array([0, 1, 0])
    print(f"  Axis: [{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}]")
    
    # Save CSV results
    csv_path = os.path.join(output_dir, 'results_table.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Image Number', 'Normal Angle (degrees)', 'Visible Area (m^2)'])
        for r in results:
            w.writerow([r['frame'], f"{r['angle']:.2f}", f"{r['area']:.6f}"])
    
    # Save rotation axis
    axis_path = os.path.join(output_dir, 'rotation_axis.txt')
    with open(axis_path, 'w') as f:
        f.write("Rotation Axis Vector (Camera Frame)\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"X: {axis[0]:.6f}\n")
        f.write(f"Y: {axis[1]:.6f}\n")
        f.write(f"Z: {axis[2]:.6f}\n")
        f.write(f"\nVector: [{axis[0]:.6f}, {axis[1]:.6f}, {axis[2]:.6f}]\n")
    
    # Print summary
    print("\n" + "=" * 55)
    print(f"{'Frame':<8}{'Normal Angle (°)':<18}{'Visible Area (m²)'}")
    print("-" * 45)
    for r in results:
        print(f"{r['frame']:<8}{r['angle']:<18.2f}{r['area']:.6f}")
    print("-" * 45)
    print(f"Rotation Axis: [{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}]")
    print("=" * 55)
    
    print(f"\nOutputs saved to: {output_dir}")
    return results, axis


if __name__ == "__main__":
    main(r'D:\Projects\10x Constructions - Perception Engineering\New_assesment\depth', 'D:\Projects\10x Constructions - Perception Engineering\New_assesment\output')