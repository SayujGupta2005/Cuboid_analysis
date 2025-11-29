# Cuboid Face Detection & Rotation Analysis

**Author:** Sayuj Gupta  
**Affiliation:** IIT Jammu, 3rd Year CSE  
**Assignment:** Perception Engineering - 10xConstruction

---

## Overview

This project implements a perception pipeline to analyze a rotating 3D cuboid using raw depth data from a ROS bag. Instead of relying on intensity-based feature matching, this approach utilizes **geometric segmentation** to robustly estimate the object's orientation and physical properties.

The system processes depth frames to:

- Reconstruct a 3D point cloud from 2D depth maps
- Detect the largest visible planar face using RANSAC
- Compute the visible surface area (m²) and normal angle relative to the camera
- Estimate the global axis of rotation using SVD on the temporal sequence of normal vectors

---

## Key Algorithms

### 1. Geometric Segmentation (RANSAC)

To isolate the cuboid face from the background and sensor noise, we employ **RANSAC (Random Sample Consensus)**.

- The algorithm iteratively selects random subsets of 3 points to define candidate planes
- It identifies the plane model that maximizes the number of geometric inliers (points within a 1.5cm threshold)
- This ensures robustness against "salt-and-pepper" depth noise common in stereo or ToF sensors

### 2. Area Computation (Convex Hull)

Counting raw pixels is unreliable due to missing depth data (zeros/NaNs). Instead, we:

- Project the geometric inliers onto the detected 2D plane
- Compute the **Convex Hull** of the projected points
- Calculate the area of the resulting polygon, providing a resilient metric even with sparse data

### 3. Axis Estimation (Singular Value Decomposition)

As the cuboid rotates, the normal vector of its face traces a path in 3D space. The axis of rotation is derived by analyzing the covariance of these normal vectors using **SVD**. The eigenvector corresponding to the smallest eigenvalue of the covariance matrix represents the normal to the "plane of normals"—which aligns with the physical axis of rotation.

---

## Installation & Usage

### Prerequisites

The solution is built on the `rosbags` pure Python library (no full ROS 2 installation required) and standard scientific computing stacks.

```bash
pip install numpy scipy rosbags
```

or

```bash
pip install -r requirements.txt
```

### Running the Analysis

1. Place your ROS bag file (e.g., `depth_data.bag` or `depth.db3`) in the project directory
2. Update the `BAG_PATH` variable in `cube_analysis.py` if necessary
3. Run the script:

```bash
python cube_analysis.py
```

---

## Outputs

- **Console:** Real-time logging of frame angles and areas
- **`results_table.csv`:** A CSV file containing Frame ID, Normal Angle, and Visible Area
- **`rotation_axis.txt`:** The final estimated 3D vector for the axis of rotation

---

## License

This project is submitted as part of the perception engineering assessment for 10xConstruction.