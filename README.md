# ENPM661 Project 3 — Phase 2
## A* Path Planning on a Differential Drive Wheelchair Robot in Gazebo

> **Course:** ENPM661 — Planning for Autonomous Robots, Spring 2026  
> **Due Date:** April 18, 2026  
> **ROS Version:** ROS 2 Humble | **Simulator:** Gazebo Classic

---

## 📎 GitHub Repository

🔗 [https://github.com/ninad-iam/ENPM661_Project3](https://github.com/ninad-iam/ENPM661_Project3)


---

## Simulation Videos

| Description | Link |
|-------------|------|
| 2D A* Path Planning Gazebo Simulation Wheelchair Robot | [▶ Click to Watch](https://drive.google.com/drive/folders/16YVOgYryUmOt9y5FmY4bq9KufW_pm6vQ?usp=sharing) |
| FlaconSim Simulation Wheelchair Robot | [▶ Click to Watch](https://drive.google.com/drive/folders/16YVOgYryUmOt9y5FmY4bq9KufW_pm6vQ?usp=sharing) |


---

## Team Members

| Full Name | Directory ID | UID |
|-----------|-------------|-----|
| Rishi Mehta | rishi17 | 122258846 |
| Rahul Ravi VK | rrv | 122054080 |
| Ninad Deshmukh | ninad | 122269453 |
| Pratham Salvi | prat03 | 122021624 |
# A* Navigation with Custom Wheelchair Robot (ROS 2 Humble)

## Overview

This project implements **A'*' path planning and navigation** for a custom **wheelchair robot** in **ROS 2 Humble + Gazebo Classic**.

Instead of using TurtleBot3, a **custom wheelchair model ( URDF/Xacro)** is used as a differential-drive robot. The system:

* Generates a **scaled map (8 m × 16 m)**
* Plans a path using **A***
* Converts path → waypoints
* Drives robot using `/cmd_vel`
* Visualizes **map + path + live odometry**

---

## Custom Wheelchair Robot

* Built using **URDF/Xacro**
* Mesh: `wheelchair.stl` used for reference
* Controller: **Gazebo diff-drive plugin (`libgazebo_ros_diff_drive.so`)**
* Topics:

  * `/cmd_vel` → velocity input
  * `/odom` → feedback

### Robot Parameters

| Parameter        | Value   |
| ---------------- | ------- |
| Wheel Radius     | 0.30 m |
| Wheel Separation | 0.65 m |

The robot behaves as a **non-holonomic differential drive system**. 

---

## A* Planning

* Grid-based A*
* Euclidean heuristic
* 8-connected motion (grid version)

### Differential Drive Action Space (original formulation)

```id="gq7c6o"
[0, RPM1], [RPM1, 0], [RPM1, RPM1],
[0, RPM2], [RPM2, 0], [RPM2, RPM2],
[RPM1, RPM2], [RPM2, RPM1]
```

This approximates realistic motion constraints of the wheelchair. 

---

## Map

* Base map: **2 m × 4 m**
* Scaled map: **8 m × 16 m (4×)**
* Includes:

  * 3 obstacle blocks
  * angled + straight walls
* Map is **inflated using robot radius + clearance**

---

## Dependencies

### ROS 2 + Gazebo

```bash
sudo apt update
sudo apt install -y \
  ros-humble-desktop \
  ros-humble-gazebo-ros-pkgs \
  ros-humble-robot-state-publisher \
  ros-humble-joint-state-publisher \
  ros-humble-xacro
```

### Python

```bash
pip3 install numpy matplotlib
```

---

## Package Structure


diff_bot/
├── scripts/
│   ├── map.py
│   └── nav.py
├── meshes/
│   └── wheelchair.STL
├── urdf/
│   └── wheelchair.urdf.xacro
├── launch/
│   └── gazebo.launch.py
```

---

## How to Run

### 1. Build

```bash
cd ~/enpm661_ws
colcon build --packages-select diff_bot
source install/setup.bash
```

---

### 2. Launch Gazebo

```bash
ros2 launch diff_bot gazebo.launch.py
```

* Spawns **wheelchair robot**
* Loads environment with the scaled obstacle map.

---

### 3. Visualize Map (optional)

```bash
ros2 run diff_bot nav.py --ros-args -p plot_only:=true
```

---

### 4. Run A* Navigation

```bash
ros2 run diff_bot nav.py --ros-args -p plot_only:=false
```

This will:

* compute A* path
* show map + path
* drive robot in Gazebo
* display live trajectory


## Some other goal nodes to try for A* Planning

* Goal -> (1.0, 1.45)
* The goal value is scaled up to the map by the nav.py itsef.

## Summary

* A* implemented from scratch
* Custom **wheelchair robot replaces TurtleBot3**
* Full pipeline:
  **map → planning → control → simulation**

---
