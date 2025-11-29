"""
Cuboid Face Detection and Rotation Axis Estimation
===================================================
Detects the largest visible face of a rotating cuboid from depth images
and estimates its rotation axis.

Author: Sayuj Gupta - 3rd year CSE student at IIT Jammu
Assignment: Perception - 10xConstruction
Github repo - https://github.com/SayujGupta2005/Cuboid_analysis

"""

import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial import ConvexHull
import csv

# =============================================================================
# CONFIGURATION - Update these as needed
# =============================================================================

BAG_PATH = r'D:\Projects\10x Constructions - Perception Engineering\New_assesment\depth'  # Path to your ROS bag folder containing depth.db3

# Camera intrinsics for 640x480 depth camera (typical RealSense D435 values)
# Assumed values
FX, FY = 615.0, 615.0   # Focal lengths in pixels
CX, CY = 320.0, 240.0   # Principal point (image center)

# Depth filtering - only consider points within this range
MIN_DEPTH = 0.1  # meters
MAX_DEPTH = 3  # meters (adjust if cuboid is further away)
# Change in MAX_DEPTH will affect the area calculated as it will affect the number of points considered by the RANSAC

# RANSAC parameters
RANSAC_ITERATIONS = 1000 # Enough to get a good fit
RANSAC_THRESHOLD = 0.015  # 1.5 cm tolerance for plane inliers


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def ransac_plane_fit(points):
    """
    Find the dominant plane in a point cloud using RANSAC.
    
    RANSAC works by:
    1. Randomly picking 3 points to define a plane
    2. Counting how many other points lie close to that plane (inliers)
    3. Repeating many times and keeping the plane with most inliers
    
    Returns:
        best_inliers: Boolean mask indicating which points belong to the plane
    """
    n_points = len(points)
    best_inliers = np.zeros(n_points, dtype=bool)
    
    # Subsample for speed if too many points (check all points for inliers though)
    if n_points > 5000:
        sample_points = points[np.random.choice(n_points, 5000, replace=False)]
    else:
        sample_points = points

    for _ in range(RANSAC_ITERATIONS):
        # Pick 3 random points
        idx = np.random.choice(len(sample_points), 3, replace=False)
        p1, p2, p3 = sample_points[idx]
        
        # Compute plane normal using cross product of two edge vectors
        # A plane through 3 points has normal = (p2-p1) × (p3-p1)
        normal = np.cross(p2 - p1, p3 - p1)
        
        # Skip degenerate case (collinear points)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal = normal / norm  # Normalize to unit vector
        
        # Plane equation: ax + by + cz + d = 0, where (a,b,c) = normal
        d = -np.dot(normal, p1)
        
        # Compute distance of ALL points to this plane
        distances = np.abs(np.dot(points, normal) + d)
        
        # Points within threshold are inliers
        inliers = distances < RANSAC_THRESHOLD
        
        # Keep this plane if it has more inliers than previous best
        if np.sum(inliers) > np.sum(best_inliers):
            best_inliers = inliers

    return best_inliers


def refine_plane_with_svd(points):
    """
    Refine plane normal using SVD (Singular Value Decomposition).
    
    SVD finds the principal directions of the point cloud.
    For points on a plane, the smallest principal direction = plane normal.
    
    Uses covariance matrix (3x3) instead of data matrix (Nx3) for efficiency.
    """
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    
    # Covariance matrix captures the spread of points in each direction
    cov_matrix = np.dot(centered.T, centered)
    
    # SVD decomposes into U, S, Vt where rows of Vt are principal directions
    _, _, vt = np.linalg.svd(cov_matrix)
    
    # Last row of Vt = direction of least variance = plane normal
    normal = vt[2, :]
    
    # Ensure normal points toward camera (positive Z component)
    if normal[2] < 0:
        normal = -normal
    
    return normal


def compute_area(points, normal):
    """
    Compute visible area by projecting 3D points onto the plane
    and finding the convex hull area in 2D.
    """
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    
    # Create 2D coordinate system on the plane
    # Need two orthogonal vectors that lie in the plane (perpendicular to normal)
    arbitrary = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
    u_axis = np.cross(normal, arbitrary)
    u_axis = u_axis / np.linalg.norm(u_axis)
    v_axis = np.cross(normal, u_axis)
    
    # Project 3D points onto 2D plane coordinates
    pts_2d = np.column_stack([np.dot(centered, u_axis), np.dot(centered, v_axis)])
    
    # Convex hull gives the outer boundary; its "volume" in 2D = area
    try:
        return ConvexHull(pts_2d).volume
    except:
        return 0.0


def estimate_rotation_axis(normals):
    """
    Estimate the axis of rotation from a sequence of plane normals.
    
    As the cuboid rotates, its face normals trace out a path.
    The rotation axis is perpendicular to the plane containing these normals,
    i.e., the direction of LEAST variance among the normals.
    """
    normals = np.array(normals)
    centered = normals - np.mean(normals, axis=0)
    
    # SVD on covariance matrix - smallest eigenvector = rotation axis
    cov = np.dot(centered.T, centered)
    _, _, vt = np.linalg.svd(cov)
    axis = vt[2, :]
    
    # Normalize sign convention (positive along dominant component)
    max_idx = np.argmax(np.abs(axis))
    if axis[max_idx] < 0:
        axis = -axis
    
    return axis


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main processing pipeline."""
    
    print("=" * 55)
    print("Cuboid Face Detection & Rotation Axis Estimation")
    print("=" * 55)
    
    # Initialize ROS bag reader
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    
    try:
        reader = Reader(BAG_PATH)
        reader.open()
    except Exception as e:
        print(f"Error: Could not open bag at '{BAG_PATH}'")
        print(f"Details: {e}")
        return
    
    results = []
    normals = []
    
    print(f"\n{'Frame':<8}{'Angle (°)':<15}{'Area (m²)':<15}")
    print("-" * 40)
    
    # Process each depth image in the bag
    for i, (connection, timestamp, rawdata) in enumerate(reader.messages()):
        msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
        
        # --- Step 1: Decode depth image ---
        # 16UC1 = 16-bit unsigned, single channel, values in millimeters
        depth = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
        depth = depth.astype(np.float32) / 1000.0  # Convert mm to meters
        
        # --- Step 2: Convert to 3D point cloud ---
        # Using pinhole camera model: X = (u - cx) * Z / fx
        v, u = np.indices(depth.shape)  # v = row (y), u = column (x)
        
        # Filter valid depth values
        mask = (depth > MIN_DEPTH) & (depth < MAX_DEPTH)
        
        z = depth[mask]
        x = (u[mask] - CX) * z / FX
        y = (v[mask] - CY) * z / FY
        points = np.column_stack((x, y, z))
        
        if len(points) < 100:
            print(f"{i+1:<8}Skipped - insufficient points")
            continue
        
        # --- Step 3: RANSAC plane detection ---
        inliers_mask = ransac_plane_fit(points)
        plane_points = points[inliers_mask]
        
        # --- Step 4: Refine plane normal with SVD ---
        normal = refine_plane_with_svd(plane_points)
        normals.append(normal)
        
        # --- Step 5: Compute normal angle ---
        # Angle between plane normal and camera axis [0, 0, 1]
        # cos(angle) = normal · [0,0,1] = normal[2]
        angle = np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0)))
        
        # --- Step 6: Compute visible area ---
        area = compute_area(plane_points, normal)
        
        results.append([i + 1, angle, area])
        print(f"{i+1:<8}{angle:<15.2f}{area:<15.4f}")
    
    reader.close()
    
    # --- Step 7: Estimate rotation axis ---
    print("-" * 40)
    
    if len(normals) >= 2:
        axis = estimate_rotation_axis(normals)
        print(f"\nRotation Axis: [{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}]")
        
        # Interpret the axis direction
        if np.abs(axis[0]) > 0.9:
            print("Interpretation: Rotation around X-axis (horizontal)")
        elif np.abs(axis[1]) > 0.9:
            print("Interpretation: Rotation around Y-axis (vertical)")
        elif np.abs(axis[2]) > 0.9:
            print("Interpretation: Rotation around Z-axis (depth)")
        
        # Save rotation axis
        with open('rotation_axis.txt', 'w') as f:
            f.write(f"Rotation Axis Vector (Camera Frame)\n")
            f.write(f"X: {axis[0]:.6f}\n")
            f.write(f"Y: {axis[1]:.6f}\n")
            f.write(f"Z: {axis[2]:.6f}\n")
        print("\nSaved: rotation_axis.txt")
    
    # --- Save results table ---
    with open('results_table.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image Number', 'Normal Angle (degrees)', 'Visible Area (m^2)'])
        writer.writerows(results)
    print("Saved: results_table.csv")
    
    print("=" * 55)


if __name__ == "__main__":
    main()