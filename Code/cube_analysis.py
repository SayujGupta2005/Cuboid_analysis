import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial import ConvexHull
import csv

# --- Configuration ---
BAG_PATH = r'D:\Projects\10x Constructions - Perception Engineering\New_assesment\depth'  # UPDATE THIS TO YOUR PATH IF NEEDED
FX, FY = 615.0, 615.0
CX, CY = 320.0, 240.0
MAX_DIST = 1.5

def get_plane_ransac(points, iter=500, thresh=0.015):
    """Finds the best plane using RANSAC."""
    best_inliers = []
    best_normal = None
    n_points = len(points)
    
    # Optimization: Downsample for RANSAC speed if too many points
    if n_points > 5000:
        sample_indices = np.random.choice(n_points, 5000, replace=False)
        sample_points = points[sample_indices]
    else:
        sample_points = points

    for _ in range(iter):
        # Pick 3 random points
        idx = np.random.choice(len(sample_points), 3, replace=False)
        p1, p2, p3 = sample_points[idx]
        
        v1, v2 = p2 - p1, p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm == 0: continue
        normal /= norm
        
        d = -np.dot(normal, p1)
        # Check against ALL points, not just sample
        distances = np.abs(np.dot(points, normal) + d)
        inliers = distances < thresh
        
        if np.sum(inliers) > np.sum(best_inliers):
            best_inliers = inliers
            best_normal = normal

    return best_normal, best_inliers

def main():
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    results = []
    normals = []

    print(f"{'Frame':<10} {'Angle (deg)':<15} {'Area (m^2)':<15}")
    print("-" * 40)

    # Use 'try-except' to handle path errors gracefully
    try:
        reader = Reader(BAG_PATH)
        reader.open()
    except Exception as e:
        print(f"Error opening bag: {e}")
        return

    for i, (connection, timestamp, rawdata) in enumerate(reader.messages()):
        msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
        
        # 1. Decode
        depth = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
        depth = depth.astype(np.float32) / 1000.0

        # 2. Point Cloud
        v, u = np.indices(depth.shape)
        mask = (depth > 0.1) & (depth < MAX_DIST)
        
        z = depth[mask]
        x = (u[mask] - CX) * z / FX
        y = (v[mask] - CY) * z / FY
        points = np.column_stack((x, y, z))

        if len(points) < 100: continue

        # 3. RANSAC
        normal, inliers_mask = get_plane_ransac(points)
        
        # --- FIX IS HERE ---
        # Refine normal using SVD on COVARIANCE matrix (3x3) to avoid MemoryError
        plane_points = points[inliers_mask]
        centroid = np.mean(plane_points, axis=0)
        centered = plane_points - centroid
        
        # Compute 3x3 Covariance Matrix instead of huge NxN matrix
        cov_matrix = np.dot(centered.T, centered)
        u_svd, s, vt = np.linalg.svd(cov_matrix)
        normal = vt[2, :] # Smallest eigenvector
        # -------------------

        if normal[2] < 0: normal = -normal
        normals.append(normal)

        # 4. Angle & Area
        angle = np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0)))

        # Project to 2D for Area
        arb = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
        v1 = np.cross(normal, arb)
        v1 /= np.linalg.norm(v1)
        v2 = np.cross(normal, v1)
        
        pts_2d = np.column_stack((np.dot(centered, v1), np.dot(centered, v2)))
        # ConvexHull needs points to be coplanar-ish
        try:
            area = ConvexHull(pts_2d).volume
        except:
            area = 0.0

        print(f"{i+1:<10} {angle:<15.2f} {area:<15.4f}")
        results.append([i+1, angle, area])
    
    reader.close()

    # 5. Rotation Axis
    if len(normals) > 1:
        normals_centered = np.array(normals) - np.mean(normals, axis=0)
        # Apply same fix here just in case
        cov_normals = np.dot(normals_centered.T, normals_centered)
        u, s, vt = np.linalg.svd(cov_normals)
        axis = vt[2, :]
        if axis[1] < 0: axis = -axis
        print("\nRotation Axis Vector:", np.round(axis, 4))
        
        with open('rotation_axis.txt', 'w') as f:
            f.write(f"Rotation Axis: {axis.tolist()}")

    with open('results_table.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Frame', 'Angle', 'Area'])
        writer.writerows(results)

if __name__ == "__main__":
    main()