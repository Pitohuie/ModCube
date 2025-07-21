---
layout: default
title: Tutorials
nav_order: 3
has_children: false
permalink: /tutorials/
---

# Tutorials

This section provides comprehensive tutorials for using the RS-ModCubes system, from basic operations to advanced configurations.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Basic Simulation](#basic-simulation)
3. [Control System Tutorial](#control-system-tutorial)
4. [Mission Planning](#mission-planning)
5. [Hardware Integration](#hardware-integration)
6. [Advanced Configuration](#advanced-configuration)
7. [Troubleshooting](#troubleshooting)

## Quick Start

### Launch Your First Simulation

After completing the installation, let's start with a basic simulation:

```bash
# Terminal 1: Launch the system
roslaunch modcube_mission system.launch model_name:=modcube simulated:=true

# Terminal 2: Check system status
rostopic echo /modcube/nav_state

# Terminal 3: Send a simple command
rostopic pub /modcube/controller_command modcube_msgs/ControllerCommand \
  "header:
    stamp: now
    frame_id: 'base_link'
  mode: 1
  setpoint:
    position:
      x: 1.0
      y: 0.0
      z: -1.0
    orientation:
      x: 0.0
      y: 0.0
      z: 0.0
      w: 1.0"
```

This will:
1. Launch the complete ModCube system in simulation mode
2. Display the current navigation state
3. Command the robot to move to position (1, 0, -1)

### Understanding the System Status

```bash
# Check running nodes
rosnode list

# Check available topics
rostopic list

# Check available services
rosservice list

# Monitor system health
rostopic echo /modcube/alarms
```

## Basic Simulation

### Available Simulation Worlds

The ModCube system includes several pre-configured simulation environments:

#### 1. Base Pool Environment

```bash
# Launch base pool simulation
roslaunch modcube_sim_worlds base_pool.launch

# In another terminal, launch the vehicle
roslaunch modcube_mission system.launch model_name:=modcube simulated:=true world:=base_pool
```

#### 2. Transdec Environment

```bash
# Launch Transdec simulation (more complex environment)
roslaunch modcube_sim_worlds transdec.launch

# Launch vehicle in Transdec world
roslaunch modcube_mission system.launch model_name:=modcube simulated:=true world:=transdec
```

#### 3. UMD Test Environment

```bash
# Launch UMD simulation
roslaunch modcube_sim_worlds umd.launch

# Launch vehicle in UMD world
roslaunch modcube_mission system.launch model_name:=modcube simulated:=true world:=umd
```

### Simulation Controls

#### Using Keyboard Teleop

```bash
# Launch teleoperation
roslaunch modcube_common teleop.launch

# Control keys:
# W/S: Forward/Backward
# A/D: Left/Right
# Q/E: Up/Down
# I/K: Pitch up/down
# J/L: Yaw left/right
# U/O: Roll left/right
```

#### Using RQT Control Panel

```bash
# Launch RQT with custom perspective
rqt --perspective-file $(rospack find modcube_common)/config/modcube.perspective
```

## Control System Tutorial

### Understanding the Control Architecture

The ModCube control system uses a hierarchical approach:

1. **Mission Level**: High-level goals and tasks
2. **Guidance Level**: Path planning and trajectory generation
3. **Control Level**: PID controllers and thrust allocation
4. **Actuation Level**: Individual thruster control

### PID Controller Configuration

#### Viewing Current PID Parameters

```bash
# Get current PID parameters
rosservice call /modcube/get_pid_params
```

#### Tuning PID Parameters

```bash
# Set position PID gains
rosservice call /modcube/set_pid_params \
  "position_gains:
    p: [10.0, 10.0, 10.0]
    i: [0.1, 0.1, 0.1]
    d: [5.0, 5.0, 5.0]
  orientation_gains:
    p: [20.0, 20.0, 20.0]
    i: [0.2, 0.2, 0.2]
    d: [8.0, 8.0, 8.0]"
```

#### Real-time PID Tuning with RQT

```bash
# Launch dynamic reconfigure GUI
rosrun rqt_reconfigure rqt_reconfigure

# Select /modcube/pid_controller for real-time tuning
```

### Thruster Management

#### Thruster Configuration

```bash
# View current thruster configuration
rosservice call /modcube/get_thruster_config

# Test individual thrusters
rostopic pub /modcube/thruster_test modcube_msgs/ThrusterCommand \
  "header:
    stamp: now
  thrusters: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]"
```

#### Thrust Allocation Matrix (TAM)

The system uses a 14-thruster configuration. Understanding the TAM:

```python
# Example TAM structure (simplified)
# Each row represents [Fx, Fy, Fz, Tx, Ty, Tz] for each thruster
TAM = [
    [1, 0, 0, 0, 0, 0],  # Thruster 0: Forward thrust
    [0, 1, 0, 0, 0, 0],  # Thruster 1: Lateral thrust
    [0, 0, 1, 0, 0, 0],  # Thruster 2: Vertical thrust
    # ... additional thrusters
]
```

### Control Modes

The system supports multiple control modes:

#### 1. Position Control

```bash
rostopic pub /modcube/controller_command modcube_msgs/ControllerCommand \
  "header:
    stamp: now
  mode: 1  # Position control mode
  setpoint:
    position: {x: 2.0, y: 1.0, z: -2.0}
    orientation: {x: 0, y: 0, z: 0, w: 1}"
```

#### 2. Velocity Control

```bash
rostopic pub /modcube/controller_command modcube_msgs/ControllerCommand \
  "header:
    stamp: now
  mode: 2  # Velocity control mode
  setpoint:
    linear: {x: 0.5, y: 0.0, z: 0.0}
    angular: {x: 0, y: 0, z: 0.1}"
```

#### 3. Direct Thrust Control

```bash
rostopic pub /modcube/thrust_command modcube_msgs/ThrusterCommand \
  "header:
    stamp: now
  thrusters: [0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"
```

## Mission Planning

### Creating Simple Missions

#### Waypoint Navigation

```python
#!/usr/bin/env python
import rospy
from modcube_msgs.msg import ControllerCommand
from geometry_msgs.msg import Point, Quaternion

def navigate_waypoints():
    rospy.init_node('waypoint_navigator')
    pub = rospy.Publisher('/modcube/controller_command', ControllerCommand, queue_size=10)
    
    waypoints = [
        (0, 0, -1),    # Start position
        (5, 0, -1),    # Move forward
        (5, 5, -1),    # Move right
        (0, 5, -1),    # Move back
        (0, 0, -1),    # Return to start
    ]
    
    rate = rospy.Rate(1)  # 1 Hz
    
    for wp in waypoints:
        cmd = ControllerCommand()
        cmd.header.stamp = rospy.Time.now()
        cmd.mode = 1  # Position control
        cmd.setpoint.position = Point(wp[0], wp[1], wp[2])
        cmd.setpoint.orientation = Quaternion(0, 0, 0, 1)
        
        pub.publish(cmd)
        rospy.loginfo(f"Moving to waypoint: {wp}")
        
        # Wait for 10 seconds at each waypoint
        for _ in range(10):
            rate.sleep()

if __name__ == '__main__':
    navigate_waypoints()
```

#### Using the Mission Manager

```bash
# Launch mission manager
roslaunch modcube_mission mission_manager.launch

# Send mission command
rosservice call /modcube/mission_control \
  "command: 'start'
   mission_type: 'waypoint_navigation'
   parameters: ['waypoint_file:=/path/to/waypoints.yaml']"
```

### Advanced Mission Examples

#### Search Pattern Mission

```yaml
# search_pattern.yaml
mission:
  type: "search_pattern"
  parameters:
    search_area:
      min_x: 0
      max_x: 10
      min_y: 0
      max_y: 10
      depth: -2.0
    pattern_type: "lawnmower"
    spacing: 2.0
    speed: 0.5
```

```bash
# Execute search pattern
rosservice call /modcube/mission_control \
  "command: 'start'
   mission_type: 'search_pattern'
   parameters: ['config_file:=search_pattern.yaml']"
```

## Hardware Integration

### IMU Integration

#### Xsens IMU Setup

```bash
# Launch Xsens IMU driver
roslaunch modcube_vehicle xsens_imu.launch

# Check IMU data
rostopic echo /modcube/imu/data

# Calibrate IMU (follow manufacturer instructions)
rosservice call /modcube/imu/calibrate
```

#### IMU Configuration

```yaml
# imu_config.yaml
imu:
  frame_id: "imu_link"
  frequency: 100
  orientation_covariance: [0.01, 0.01, 0.01]
  angular_velocity_covariance: [0.001, 0.001, 0.001]
  linear_acceleration_covariance: [0.01, 0.01, 0.01]
```

### DVL Integration

#### Teledyne DVL Setup

```bash
# Launch DVL driver
roslaunch modcube_vehicle teledyne_dvl.launch port:=/dev/ttyUSB0

# Check DVL data
rostopic echo /modcube/dvl/data

# Test DVL communication
rosservice call /modcube/dvl/test_communication
```

### Thruster Integration

#### Pololu Maestro Setup

```bash
# Launch thruster controller
roslaunch modcube_vehicle actuators.launch

# Test thruster response
rosservice call /modcube/test_thrusters
```

#### Thruster Calibration

```python
#!/usr/bin/env python
# thruster_calibration.py
import rospy
from modcube_msgs.msg import ThrusterCommand

def calibrate_thrusters():
    pub = rospy.Publisher('/modcube/thrust_command', ThrusterCommand, queue_size=10)
    
    # Test each thruster individually
    for i in range(14):
        cmd = ThrusterCommand()
        cmd.header.stamp = rospy.Time.now()
        cmd.thrusters = [0.0] * 14
        cmd.thrusters[i] = 0.1  # 10% thrust
        
        pub.publish(cmd)
        rospy.loginfo(f"Testing thruster {i}")
        rospy.sleep(2)
        
        # Stop thruster
        cmd.thrusters[i] = 0.0
        pub.publish(cmd)
        rospy.sleep(1)
```

## Advanced Configuration

### Custom Vehicle Configuration

#### Creating a New Vehicle Model

1. **Create vehicle description**:

```xml
<!-- my_vehicle.xacro -->
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="my_vehicle">
  
  <!-- Include base ModCube components -->
  <xacro:include filename="$(find modcube_config)/modcube_description/urdf/base.xacro"/>
  
  <!-- Custom thruster configuration -->
  <xacro:include filename="$(find my_vehicle_config)/urdf/thrusters_custom.xacro"/>
  
  <!-- Custom sensor configuration -->
  <xacro:include filename="$(find my_vehicle_config)/urdf/sensors_custom.xacro"/>
  
</robot>
```

2. **Create launch file**:

```xml
<!-- my_vehicle.launch -->
<launch>
  <arg name="namespace" default="my_vehicle"/>
  
  <!-- Load vehicle description -->
  <param name="robot_description" 
         command="$(find xacro)/xacro $(find my_vehicle_config)/urdf/my_vehicle.xacro"/>
  
  <!-- Launch vehicle-specific nodes -->
  <group ns="$(arg namespace)">
    <include file="$(find modcube_vehicle)/launch/base_vehicle.launch">
      <arg name="namespace" value="$(arg namespace)"/>
    </include>
    
    <!-- Custom nodes -->
    <node name="custom_sensor" pkg="my_vehicle_drivers" type="custom_sensor_node"/>
  </group>
  
</launch>
```

### Custom Control Algorithms

#### Implementing a Custom Controller

```python
#!/usr/bin/env python
# custom_controller.py
import rospy
import numpy as np
from modcube_msgs.msg import ControllerCommand, NavState, ThrusterCommand
from modcube_common.controllers.controller import BaseController

class CustomController(BaseController):
    def __init__(self):
        super().__init__()
        
        # Custom parameters
        self.custom_gain = rospy.get_param('~custom_gain', 1.0)
        
        # Publishers and subscribers
        self.thrust_pub = rospy.Publisher('/modcube/thrust_command', ThrusterCommand, queue_size=10)
        self.nav_sub = rospy.Subscriber('/modcube/nav_state', NavState, self.nav_callback)
        self.cmd_sub = rospy.Subscriber('/modcube/controller_command', ControllerCommand, self.cmd_callback)
        
        self.current_state = None
        self.current_command = None
    
    def nav_callback(self, msg):
        self.current_state = msg
        self.update_control()
    
    def cmd_callback(self, msg):
        self.current_command = msg
    
    def update_control(self):
        if self.current_state is None or self.current_command is None:
            return
        
        # Custom control algorithm
        error = self.compute_error(self.current_state, self.current_command)
        wrench = self.compute_wrench(error)
        thrust_cmd = self.allocate_thrust(wrench)
        
        self.thrust_pub.publish(thrust_cmd)
    
    def compute_error(self, state, command):
        # Implement custom error computation
        pass
    
    def compute_wrench(self, error):
        # Implement custom control law
        pass
    
    def allocate_thrust(self, wrench):
        # Use existing thrust allocation
        return self.thrust_allocator.allocate(wrench)

if __name__ == '__main__':
    rospy.init_node('custom_controller')
    controller = CustomController()
    rospy.spin()
```

### Sensor Fusion Configuration

#### Custom State Estimator

```python
#!/usr/bin/env python
# custom_estimator.py
import rospy
import numpy as np
from sensor_msgs.msg import Imu
from modcube_msgs.msg import DVLData, NavState
from geometry_msgs.msg import PoseWithCovariance, TwistWithCovariance

class CustomStateEstimator:
    def __init__(self):
        # Initialize Kalman filter or other estimator
        self.state = np.zeros(12)  # [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
        self.covariance = np.eye(12) * 0.1
        
        # Publishers and subscribers
        self.nav_pub = rospy.Publisher('/modcube/nav_state', NavState, queue_size=10)
        self.imu_sub = rospy.Subscriber('/modcube/imu/data', Imu, self.imu_callback)
        self.dvl_sub = rospy.Subscriber('/modcube/dvl/data', DVLData, self.dvl_callback)
        
        # Timer for state prediction
        self.timer = rospy.Timer(rospy.Duration(0.01), self.predict_step)  # 100 Hz
    
    def predict_step(self, event):
        # Implement prediction step
        dt = 0.01
        # Update state prediction
        self.publish_state()
    
    def imu_callback(self, msg):
        # Update with IMU measurements
        pass
    
    def dvl_callback(self, msg):
        # Update with DVL measurements
        pass
    
    def publish_state(self):
        nav_msg = NavState()
        nav_msg.header.stamp = rospy.Time.now()
        nav_msg.header.frame_id = "odom"
        
        # Fill in state information
        nav_msg.pose.pose.position.x = self.state[0]
        nav_msg.pose.pose.position.y = self.state[1]
        nav_msg.pose.pose.position.z = self.state[2]
        # ... fill in remaining fields
        
        self.nav_pub.publish(nav_msg)
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Simulation Performance Issues

**Problem**: Gazebo runs slowly or crashes

**Solutions**:
```bash
# Reduce simulation complexity
export GAZEBO_MODEL_PATH=/path/to/simple/models

# Use software rendering
export LIBGL_ALWAYS_SOFTWARE=1

# Reduce physics update rate
# Edit world file: <physics><real_time_update_rate>100</real_time_update_rate></physics>
```

#### 2. Control System Issues

**Problem**: Robot doesn't respond to commands

**Debugging steps**:
```bash
# Check if controller is running
rosnode list | grep controller

# Check command topics
rostopic echo /modcube/controller_command

# Check thrust output
rostopic echo /modcube/thrust_command

# Verify thruster allocation
rosservice call /modcube/get_thruster_config
```

#### 3. Hardware Communication Issues

**Problem**: Cannot communicate with hardware

**Solutions**:
```bash
# Check device permissions
ls -l /dev/ttyUSB*
sudo chmod 666 /dev/ttyUSB0

# Check if device is detected
dmesg | grep tty

# Test serial communication
sudo minicom -D /dev/ttyUSB0 -b 115200
```

### Debug Tools

#### ROS Debug Commands

```bash
# Monitor all topics
rostopic list | xargs -I {} rostopic echo {} --noarr

# Check node graph
rqt_graph

# Monitor system resources
htop

# Check ROS logs
roscd && cd ../log
tail -f latest/rosout.log
```

#### Gazebo Debug

```bash
# Verbose Gazebo output
gazebo --verbose

# Check Gazebo plugins
gzserver --verbose

# Monitor Gazebo topics
gz topic -l
gz topic -e /gazebo/default/physics/contacts
```

### Performance Monitoring

```bash
# Monitor ROS node performance
rosrun rqt_top rqt_top

# Check message frequencies
rostopic hz /modcube/nav_state
rostopic hz /modcube/controller_command

# Monitor network usage
iftop
```

## Advanced Topics

### Deep Learning Integration

#### Object Detection with YOLO

```python
#!/usr/bin/env python
# yolo_detector.py
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from modcube_msgs.msg import DetectedObjects, DetectedObject
import torch
from ultralytics import YOLO

class YOLODetector:
    def __init__(self):
        rospy.init_node('yolo_detector')
        
        # Load YOLO model
        self.model = YOLO('yolov8n.pt')  # or custom trained model
        self.bridge = CvBridge()
        
        # Publishers and subscribers
        self.image_sub = rospy.Subscriber('/modcube/camera/image_raw', Image, self.image_callback)
        self.detection_pub = rospy.Publisher('/modcube/detections', DetectedObjects, queue_size=10)
        self.debug_pub = rospy.Publisher('/modcube/debug/detection_image', Image, queue_size=1)
        
        # Detection parameters
        self.confidence_threshold = rospy.get_param('~confidence_threshold', 0.5)
        self.nms_threshold = rospy.get_param('~nms_threshold', 0.4)
        
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            detections = self.detect_objects(cv_image)
            self.publish_detections(detections, msg.header)
            
        except Exception as e:
            rospy.logerr(f"Detection error: {e}")
    
    def detect_objects(self, image):
        # Run YOLO inference
        results = self.model(image, conf=self.confidence_threshold, iou=self.nms_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract detection information
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]
                    
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name
                    }
                    detections.append(detection)
        
        return detections
    
    def publish_detections(self, detections, header):
        msg = DetectedObjects()
        msg.header = header
        
        for det in detections:
            obj = DetectedObject()
            obj.class_name = det['class_name']
            obj.confidence = det['confidence']
            obj.bbox.x = det['bbox'][0]
            obj.bbox.y = det['bbox'][1]
            obj.bbox.width = det['bbox'][2] - det['bbox'][0]
            obj.bbox.height = det['bbox'][3] - det['bbox'][1]
            msg.objects.append(obj)
        
        self.detection_pub.publish(msg)

if __name__ == '__main__':
    detector = YOLODetector()
    rospy.spin()
```

#### Visual SLAM Integration

```python
#!/usr/bin/env python
# visual_slam.py
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import g2o

class VisualSLAM:
    def __init__(self):
        rospy.init_node('visual_slam')
        
        self.bridge = CvBridge()
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Camera parameters
        self.K = None  # Camera intrinsic matrix
        self.prev_frame = None
        self.prev_kp = None
        self.prev_desc = None
        
        # SLAM state
        self.pose = np.eye(4)  # Current pose
        self.trajectory = []
        self.map_points = []
        
        # Publishers and subscribers
        self.image_sub = rospy.Subscriber('/modcube/camera/image_raw', Image, self.image_callback)
        self.info_sub = rospy.Subscriber('/modcube/camera/camera_info', CameraInfo, self.info_callback)
        self.pose_pub = rospy.Publisher('/modcube/slam/pose', PoseStamped, queue_size=10)
        
    def info_callback(self, msg):
        # Extract camera intrinsic parameters
        self.K = np.array(msg.K).reshape(3, 3)
        
    def image_callback(self, msg):
        if self.K is None:
            return
            
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "mono8")
            self.process_frame(cv_image, msg.header)
            
        except Exception as e:
            rospy.logerr(f"SLAM error: {e}")
    
    def process_frame(self, frame, header):
        # Extract ORB features
        kp, desc = self.orb.detectAndCompute(frame, None)
        
        if self.prev_frame is not None and self.prev_desc is not None:
            # Match features
            matches = self.matcher.match(self.prev_desc, desc)
            matches = sorted(matches, key=lambda x: x.distance)
            
            if len(matches) > 50:  # Minimum matches threshold
                # Extract matched points
                pts1 = np.float32([self.prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                pts2 = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                
                # Estimate motion
                E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC)
                _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)
                
                # Update pose
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = t.flatten()
                self.pose = self.pose @ T
                
                # Triangulate 3D points
                self.triangulate_points(pts1, pts2, R, t)
                
                # Publish pose
                self.publish_pose(header)
        
        # Update previous frame
        self.prev_frame = frame
        self.prev_kp = kp
        self.prev_desc = desc
    
    def triangulate_points(self, pts1, pts2, R, t):
        # Triangulate 3D points from stereo correspondences
        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.K @ np.hstack([R, t])
        
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3d = points_4d[:3] / points_4d[3]
        
        # Add to map
        self.map_points.extend(points_3d.T)
    
    def publish_pose(self, header):
        pose_msg = PoseStamped()
        pose_msg.header = header
        
        # Convert pose matrix to ROS message
        pose_msg.pose.position.x = self.pose[0, 3]
        pose_msg.pose.position.y = self.pose[1, 3]
        pose_msg.pose.position.z = self.pose[2, 3]
        
        # Convert rotation matrix to quaternion
        from scipy.spatial.transform import Rotation
        r = Rotation.from_matrix(self.pose[:3, :3])
        q = r.as_quat()
        pose_msg.pose.orientation.x = q[0]
        pose_msg.pose.orientation.y = q[1]
        pose_msg.pose.orientation.z = q[2]
        pose_msg.pose.orientation.w = q[3]
        
        self.pose_pub.publish(pose_msg)

if __name__ == '__main__':
    slam = VisualSLAM()
    rospy.spin()
```

### Multi-Sensor Fusion with Particle Filter

```python
#!/usr/bin/env python
# particle_filter.py
import rospy
import numpy as np
from sensor_msgs.msg import Imu
from modcube_msgs.msg import DVLData, NavState
from geometry_msgs.msg import PoseWithCovariance

class ParticleFilter:
    def __init__(self, num_particles=1000):
        rospy.init_node('particle_filter')
        
        self.num_particles = num_particles
        self.particles = np.zeros((num_particles, 6))  # [x, y, z, roll, pitch, yaw]
        self.weights = np.ones(num_particles) / num_particles
        
        # Initialize particles with random distribution
        self.particles[:, :3] = np.random.normal(0, 1, (num_particles, 3))
        self.particles[:, 3:] = np.random.normal(0, 0.1, (num_particles, 3))
        
        # Process and measurement noise
        self.process_noise = np.diag([0.01, 0.01, 0.01, 0.001, 0.001, 0.001])
        self.imu_noise = np.diag([0.1, 0.1, 0.1])
        self.dvl_noise = np.diag([0.05, 0.05, 0.05])
        
        # Publishers and subscribers
        self.nav_pub = rospy.Publisher('/modcube/nav_state_pf', NavState, queue_size=10)
        self.imu_sub = rospy.Subscriber('/modcube/imu/data', Imu, self.imu_callback)
        self.dvl_sub = rospy.Subscriber('/modcube/dvl/data', DVLData, self.dvl_callback)
        
        # Timer for prediction step
        self.timer = rospy.Timer(rospy.Duration(0.01), self.predict_step)
        
    def predict_step(self, event):
        # Prediction step: add process noise
        noise = np.random.multivariate_normal(np.zeros(6), self.process_noise, self.num_particles)
        self.particles += noise
        
        # Normalize angles
        self.particles[:, 3:] = np.mod(self.particles[:, 3:] + np.pi, 2*np.pi) - np.pi
        
        self.publish_state()
    
    def imu_callback(self, msg):
        # Update step with IMU measurements
        measured_accel = np.array([msg.linear_acceleration.x, 
                                  msg.linear_acceleration.y, 
                                  msg.linear_acceleration.z])
        
        # Compute likelihood for each particle
        for i in range(self.num_particles):
            # Predict acceleration based on particle orientation
            predicted_accel = self.predict_acceleration(self.particles[i])
            
            # Compute likelihood
            diff = measured_accel - predicted_accel
            likelihood = np.exp(-0.5 * diff.T @ np.linalg.inv(self.imu_noise) @ diff)
            self.weights[i] *= likelihood
        
        # Normalize weights
        self.weights /= np.sum(self.weights)
        
        # Resample if effective sample size is low
        if 1.0 / np.sum(self.weights**2) < self.num_particles / 2:
            self.resample()
    
    def dvl_callback(self, msg):
        # Update step with DVL measurements
        measured_velocity = np.array([msg.velocity.x, msg.velocity.y, msg.velocity.z])
        
        # Update particles based on DVL measurement
        for i in range(self.num_particles):
            # Predict velocity based on particle state
            predicted_velocity = self.predict_velocity(self.particles[i])
            
            # Compute likelihood
            diff = measured_velocity - predicted_velocity
            likelihood = np.exp(-0.5 * diff.T @ np.linalg.inv(self.dvl_noise) @ diff)
            self.weights[i] *= likelihood
        
        # Normalize weights
        self.weights /= np.sum(self.weights)
        
        # Resample if needed
        if 1.0 / np.sum(self.weights**2) < self.num_particles / 2:
            self.resample()
    
    def predict_acceleration(self, particle):
        # Predict acceleration based on particle orientation
        roll, pitch, yaw = particle[3:6]
        
        # Gravity vector in body frame
        R = self.euler_to_rotation_matrix(roll, pitch, yaw)
        gravity_world = np.array([0, 0, 9.81])
        gravity_body = R.T @ gravity_world
        
        return gravity_body
    
    def predict_velocity(self, particle):
        # Simple velocity prediction (can be enhanced with motion model)
        return np.zeros(3)
    
    def euler_to_rotation_matrix(self, roll, pitch, yaw):
        # Convert Euler angles to rotation matrix
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])
        
        return R
    
    def resample(self):
        # Systematic resampling
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles
    
    def publish_state(self):
        # Compute weighted mean of particles
        mean_state = np.average(self.particles, weights=self.weights, axis=0)
        
        # Compute covariance
        cov = np.cov(self.particles.T, aweights=self.weights)
        
        # Publish navigation state
        nav_msg = NavState()
        nav_msg.header.stamp = rospy.Time.now()
        nav_msg.header.frame_id = "odom"
        
        nav_msg.pose.pose.position.x = mean_state[0]
        nav_msg.pose.pose.position.y = mean_state[1]
        nav_msg.pose.pose.position.z = mean_state[2]
        
        # Convert Euler to quaternion
        from tf.transformations import quaternion_from_euler
        q = quaternion_from_euler(mean_state[3], mean_state[4], mean_state[5])
        nav_msg.pose.pose.orientation.x = q[0]
        nav_msg.pose.pose.orientation.y = q[1]
        nav_msg.pose.pose.orientation.z = q[2]
        nav_msg.pose.pose.orientation.w = q[3]
        
        # Set covariance
        nav_msg.pose.covariance = cov.flatten().tolist()
        
        self.nav_pub.publish(nav_msg)

if __name__ == '__main__':
    pf = ParticleFilter()
    rospy.spin()
```

### Real-Time Optimization

#### Model Predictive Control (MPC)

```python
#!/usr/bin/env python
# mpc_controller.py
import rospy
import numpy as np
import cvxpy as cp
from modcube_msgs.msg import ControllerCommand, NavState, ThrusterCommand
from geometry_msgs.msg import Wrench

class MPCController:
    def __init__(self):
        rospy.init_node('mpc_controller')
        
        # MPC parameters
        self.horizon = 10  # Prediction horizon
        self.dt = 0.1      # Time step
        
        # State and input dimensions
        self.nx = 12  # [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
        self.nu = 6   # [Fx, Fy, Fz, Tx, Ty, Tz]
        
        # Vehicle parameters
        self.mass = 50.0  # kg
        self.inertia = np.diag([10.0, 10.0, 15.0])  # kg*m^2
        
        # Cost matrices
        self.Q = np.diag([100, 100, 100, 10, 10, 10, 1, 1, 1, 1, 1, 1])  # State cost
        self.R = np.diag([1, 1, 1, 1, 1, 1])  # Input cost
        self.Qf = 10 * self.Q  # Terminal cost
        
        # Constraints
        self.u_max = np.array([200, 200, 200, 50, 50, 50])  # Max forces/torques
        self.u_min = -self.u_max
        
        # Current state and reference
        self.current_state = np.zeros(self.nx)
        self.reference = np.zeros(self.nx)
        
        # Publishers and subscribers
        self.wrench_pub = rospy.Publisher('/modcube/wrench_command', Wrench, queue_size=10)
        self.nav_sub = rospy.Subscriber('/modcube/nav_state', NavState, self.nav_callback)
        self.cmd_sub = rospy.Subscriber('/modcube/controller_command', ControllerCommand, self.cmd_callback)
        
        # Control timer
        self.timer = rospy.Timer(rospy.Duration(self.dt), self.control_step)
        
    def nav_callback(self, msg):
        # Extract state from navigation message
        self.current_state[0] = msg.pose.pose.position.x
        self.current_state[1] = msg.pose.pose.position.y
        self.current_state[2] = msg.pose.pose.position.z
        
        # Convert quaternion to Euler angles
        from tf.transformations import euler_from_quaternion
        q = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
             msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        roll, pitch, yaw = euler_from_quaternion(q)
        self.current_state[3:6] = [roll, pitch, yaw]
        
        self.current_state[6] = msg.twist.twist.linear.x
        self.current_state[7] = msg.twist.twist.linear.y
        self.current_state[8] = msg.twist.twist.linear.z
        self.current_state[9] = msg.twist.twist.angular.x
        self.current_state[10] = msg.twist.twist.angular.y
        self.current_state[11] = msg.twist.twist.angular.z
    
    def cmd_callback(self, msg):
        # Extract reference from command
        self.reference[0] = msg.setpoint.position.x
        self.reference[1] = msg.setpoint.position.y
        self.reference[2] = msg.setpoint.position.z
        
        # Convert quaternion to Euler angles
        from tf.transformations import euler_from_quaternion
        q = [msg.setpoint.orientation.x, msg.setpoint.orientation.y,
             msg.setpoint.orientation.z, msg.setpoint.orientation.w]
        roll, pitch, yaw = euler_from_quaternion(q)
        self.reference[3:6] = [roll, pitch, yaw]
        
        # Zero velocity reference
        self.reference[6:] = 0
    
    def control_step(self, event):
        # Solve MPC optimization problem
        u_opt = self.solve_mpc()
        
        if u_opt is not None:
            # Publish control command
            wrench_msg = Wrench()
            wrench_msg.force.x = u_opt[0]
            wrench_msg.force.y = u_opt[1]
            wrench_msg.force.z = u_opt[2]
            wrench_msg.torque.x = u_opt[3]
            wrench_msg.torque.y = u_opt[4]
            wrench_msg.torque.z = u_opt[5]
            
            self.wrench_pub.publish(wrench_msg)
    
    def solve_mpc(self):
        # Define optimization variables
        x = cp.Variable((self.nx, self.horizon + 1))
        u = cp.Variable((self.nu, self.horizon))
        
        # Cost function
        cost = 0
        constraints = []
        
        # Initial condition
        constraints.append(x[:, 0] == self.current_state)
        
        for k in range(self.horizon):
            # Stage cost
            cost += cp.quad_form(x[:, k] - self.reference, self.Q)
            cost += cp.quad_form(u[:, k], self.R)
            
            # Dynamics constraints
            x_next = self.dynamics(x[:, k], u[:, k])
            constraints.append(x[:, k + 1] == x_next)
            
            # Input constraints
            constraints.append(u[:, k] >= self.u_min)
            constraints.append(u[:, k] <= self.u_max)
        
        # Terminal cost
        cost += cp.quad_form(x[:, self.horizon] - self.reference, self.Qf)
        
        # Solve optimization problem
        problem = cp.Problem(cp.Minimize(cost), constraints)
        
        try:
            problem.solve(solver=cp.OSQP, verbose=False)
            
            if problem.status == cp.OPTIMAL:
                return u[:, 0].value
            else:
                rospy.logwarn(f"MPC solver status: {problem.status}")
                return None
                
        except Exception as e:
            rospy.logerr(f"MPC solver error: {e}")
            return None
    
    def dynamics(self, x, u):
        # Simplified 6-DOF dynamics
        pos = x[:3]
        euler = x[3:6]
        vel_linear = x[6:9]
        vel_angular = x[9:12]
        
        force = u[:3]
        torque = u[3:6]
        
        # Linear acceleration
        accel_linear = force / self.mass
        
        # Angular acceleration
        accel_angular = np.linalg.inv(self.inertia) @ torque
        
        # Euler angle rates (simplified)
        euler_rates = vel_angular
        
        # State derivative
        x_dot = cp.vstack([
            vel_linear,
            euler_rates,
            accel_linear,
            accel_angular
        ])
        
        # Euler integration
        return x + self.dt * x_dot

if __name__ == '__main__':
    mpc = MPCController()
    rospy.spin()
```

## Next Steps

After completing these tutorials:

1. **Explore [Examples](examples.md)** for more complex scenarios
2. **Review [API Documentation](api.md)** for development
3. **Contribute** to the project by submitting improvements
4. **Join the community** for support and collaboration

For more advanced topics, consider:
- Multi-vehicle coordination
- Machine learning integration
- Custom sensor development
- Real-time optimization techniques