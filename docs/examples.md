---
layout: default
title: Examples
nav_order: 5
has_children: false
permalink: /examples/
---

# Examples

This section provides practical examples and use cases for the RS-ModCubes system, demonstrating various capabilities from basic operations to advanced applications.

## Table of Contents

1. [Basic Operations](#basic-operations)
2. [Navigation Examples](#navigation-examples)
3. [Mission Planning](#mission-planning)
4. [Sensor Integration](#sensor-integration)
5. [Multi-Vehicle Coordination](#multi-vehicle-coordination)
6. [Advanced Applications](#advanced-applications)
7. [Custom Development](#custom-development)

## Basic Operations

### Example 1: Simple Position Control

This example demonstrates basic position control of the ModCube vehicle.

```python
#!/usr/bin/env python
"""
Basic Position Control Example

This script demonstrates how to control the ModCube vehicle to move
to specific positions in 3D space.
"""

import rospy
import time
from geometry_msgs.msg import Point, Quaternion
from modcube_msgs.msg import ControllerCommand

class BasicPositionController:
    def __init__(self):
        rospy.init_node('basic_position_controller')
        
        # Publisher for controller commands
        self.cmd_pub = rospy.Publisher('/modcube/controller_command', 
                                     ControllerCommand, queue_size=10)
        
        # Wait for publisher to be ready
        rospy.sleep(1.0)
    
    def goto_position(self, x, y, z, yaw=0.0):
        """Move to specified position with optional yaw angle."""
        cmd = ControllerCommand()
        cmd.header.stamp = rospy.Time.now()
        cmd.header.frame_id = 'odom'
        cmd.mode = 1  # Position control mode
        
        # Set target position
        cmd.setpoint.position = Point(x, y, z)
        
        # Convert yaw to quaternion
        cmd.setpoint.orientation = self.yaw_to_quaternion(yaw)
        
        # Publish command
        self.cmd_pub.publish(cmd)
        rospy.loginfo(f"Moving to position: ({x}, {y}, {z}) with yaw: {yaw}")
    
    def yaw_to_quaternion(self, yaw):
        """Convert yaw angle to quaternion."""
        import math
        return Quaternion(
            x=0.0,
            y=0.0,
            z=math.sin(yaw / 2.0),
            w=math.cos(yaw / 2.0)
        )
    
    def run_demo(self):
        """Run a demonstration sequence."""
        rospy.loginfo("Starting basic position control demo")
        
        # Define waypoints
        waypoints = [
            (0, 0, -1, 0),      # Start position
            (5, 0, -1, 0),      # Move forward
            (5, 5, -1, 1.57),   # Move right and turn
            (0, 5, -1, 3.14),   # Move back and turn
            (0, 0, -1, 0),      # Return to start
        ]
        
        for wp in waypoints:
            self.goto_position(wp[0], wp[1], wp[2], wp[3])
            time.sleep(10)  # Wait 10 seconds at each waypoint
        
        rospy.loginfo("Demo completed")

if __name__ == '__main__':
    try:
        controller = BasicPositionController()
        controller.run_demo()
    except rospy.ROSInterruptException:
        pass
```

### Example 2: Velocity Control

This example shows how to control the vehicle using velocity commands.

```python
#!/usr/bin/env python
"""
Velocity Control Example

Demonstrates velocity-based control for smooth motion patterns.
"""

import rospy
import math
from geometry_msgs.msg import Vector3
from modcube_msgs.msg import ControllerCommand

class VelocityController:
    def __init__(self):
        rospy.init_node('velocity_controller')
        
        self.cmd_pub = rospy.Publisher('/modcube/controller_command',
                                     ControllerCommand, queue_size=10)
        
        rospy.sleep(1.0)
    
    def set_velocity(self, vx, vy, vz, wx=0.0, wy=0.0, wz=0.0):
        """Set linear and angular velocities."""
        cmd = ControllerCommand()
        cmd.header.stamp = rospy.Time.now()
        cmd.header.frame_id = 'base_link'
        cmd.mode = 2  # Velocity control mode
        
        cmd.setpoint.linear = Vector3(vx, vy, vz)
        cmd.setpoint.angular = Vector3(wx, wy, wz)
        
        self.cmd_pub.publish(cmd)
    
    def circle_pattern(self, radius=2.0, speed=0.5, duration=30.0):
        """Execute a circular motion pattern."""
        rospy.loginfo(f"Starting circular pattern: radius={radius}m, speed={speed}m/s")
        
        rate = rospy.Rate(10)  # 10 Hz
        start_time = rospy.Time.now()
        
        while (rospy.Time.now() - start_time).to_sec() < duration:
            # Calculate angular velocity for circular motion
            angular_vel = speed / radius
            
            # Set forward velocity and yaw rate
            self.set_velocity(vx=speed, vy=0.0, vz=0.0, wz=angular_vel)
            
            rate.sleep()
        
        # Stop motion
        self.set_velocity(0, 0, 0, 0, 0, 0)
        rospy.loginfo("Circular pattern completed")
    
    def figure_eight(self, size=3.0, speed=0.3, cycles=2):
        """Execute a figure-eight pattern."""
        rospy.loginfo(f"Starting figure-eight pattern: size={size}m, speed={speed}m/s")
        
        rate = rospy.Rate(20)  # 20 Hz
        
        for cycle in range(cycles):
            # Each cycle takes 2π seconds
            cycle_duration = 2 * math.pi / speed * size
            steps = int(cycle_duration * 20)  # 20 Hz
            
            for step in range(steps):
                t = step / 20.0  # Time in seconds
                
                # Figure-eight parametric equations
                vx = speed * math.cos(t / size)
                vy = speed * math.sin(2 * t / size)
                wz = (-math.sin(t / size) + 2 * math.cos(2 * t / size)) / size
                
                self.set_velocity(vx, vy, 0.0, 0.0, 0.0, wz)
                rate.sleep()
        
        # Stop motion
        self.set_velocity(0, 0, 0, 0, 0, 0)
        rospy.loginfo("Figure-eight pattern completed")

if __name__ == '__main__':
    try:
        controller = VelocityController()
        
        # Execute different patterns
        controller.circle_pattern(radius=3.0, speed=0.5, duration=20.0)
        rospy.sleep(2.0)
        
        controller.figure_eight(size=2.0, speed=0.3, cycles=2)
        
    except rospy.ROSInterruptException:
        pass
```

## Navigation Examples

### Example 3: Waypoint Navigation with Obstacle Avoidance

```python
#!/usr/bin/env python
"""
Waypoint Navigation with Obstacle Avoidance

Demonstrates autonomous navigation between waypoints while avoiding obstacles.
"""

import rospy
import numpy as np
from geometry_msgs.msg import Point, Quaternion
from sensor_msgs.msg import PointCloud2
from modcube_msgs.msg import ControllerCommand, NavState
from modcube_common.motion import MotionClient
from modcube_common.planning import PathPlanner

class WaypointNavigator:
    def __init__(self):
        rospy.init_node('waypoint_navigator')
        
        # Initialize motion client and path planner
        self.motion_client = MotionClient()
        self.path_planner = PathPlanner()
        
        # Subscribers
        self.nav_sub = rospy.Subscriber('/modcube/nav_state', NavState, self.nav_callback)
        self.cloud_sub = rospy.Subscriber('/modcube/pointcloud', PointCloud2, self.cloud_callback)
        
        # Current state
        self.current_pose = None
        self.obstacles = []
        
        rospy.sleep(1.0)
    
    def nav_callback(self, msg):
        """Update current pose."""
        self.current_pose = msg.pose.pose
    
    def cloud_callback(self, msg):
        """Process point cloud for obstacle detection."""
        # Simple obstacle detection (in practice, use more sophisticated methods)
        self.obstacles = self.extract_obstacles(msg)
    
    def extract_obstacles(self, pointcloud):
        """Extract obstacle positions from point cloud."""
        # Simplified obstacle extraction
        # In practice, use clustering, filtering, etc.
        obstacles = []
        
        # Convert point cloud to numpy array (simplified)
        # points = pointcloud_to_array(pointcloud)
        # 
        # for point in points:
        #     if self.is_obstacle(point):
        #         obstacles.append(point)
        
        return obstacles
    
    def navigate_waypoints(self, waypoints, avoid_obstacles=True):
        """Navigate through a list of waypoints."""
        rospy.loginfo(f"Starting navigation through {len(waypoints)} waypoints")
        
        for i, waypoint in enumerate(waypoints):
            rospy.loginfo(f"Navigating to waypoint {i+1}: {waypoint}")
            
            if avoid_obstacles and self.obstacles:
                # Plan path around obstacles
                path = self.path_planner.plan_path(
                    start=self.current_pose,
                    goal=waypoint,
                    obstacles=self.obstacles
                )
                
                if path:
                    success = self.motion_client.follow_trajectory(path)
                else:
                    rospy.logwarn("No valid path found, attempting direct navigation")
                    success = self.motion_client.goto_pose(waypoint)
            else:
                # Direct navigation
                success = self.motion_client.goto_pose(waypoint)
            
            if success:
                rospy.loginfo(f"Reached waypoint {i+1}")
            else:
                rospy.logwarn(f"Failed to reach waypoint {i+1}")
                break
        
        rospy.loginfo("Waypoint navigation completed")
    
    def run_demo(self):
        """Run navigation demo."""
        # Define waypoints
        waypoints = [
            self.create_pose(5, 0, -2, 0),
            self.create_pose(10, 5, -2, 1.57),
            self.create_pose(5, 10, -2, 3.14),
            self.create_pose(0, 5, -2, -1.57),
            self.create_pose(0, 0, -2, 0),
        ]
        
        self.navigate_waypoints(waypoints, avoid_obstacles=True)
    
    def create_pose(self, x, y, z, yaw):
        """Create a pose from position and yaw."""
        from geometry_msgs.msg import Pose
        import math
        
        pose = Pose()
        pose.position = Point(x, y, z)
        pose.orientation = Quaternion(
            x=0.0,
            y=0.0,
            z=math.sin(yaw / 2.0),
            w=math.cos(yaw / 2.0)
        )
        return pose

if __name__ == '__main__':
    try:
        navigator = WaypointNavigator()
        navigator.run_demo()
    except rospy.ROSInterruptException:
        pass
```

### Example 4: Dynamic Target Tracking

```python
#!/usr/bin/env python
"""
Dynamic Target Tracking

Tracks and follows a moving target using visual detection.
"""

import rospy
import math
from geometry_msgs.msg import Point, Twist
from modcube_msgs.msg import AprilTagDetection, ControllerCommand
from modcube_common.vision import TargetTracker

class TargetFollower:
    def __init__(self):
        rospy.init_node('target_follower')
        
        # Target tracking
        self.target_tracker = TargetTracker()
        self.target_position = None
        self.target_velocity = None
        
        # Control parameters
        self.follow_distance = 3.0  # meters
        self.max_speed = 1.0  # m/s
        
        # Publishers and subscribers
        self.cmd_pub = rospy.Publisher('/modcube/controller_command',
                                     ControllerCommand, queue_size=10)
        
        self.detection_sub = rospy.Subscriber('/modcube/apriltag_detections',
                                            AprilTagDetection, self.detection_callback)
        
        # Control timer
        self.control_timer = rospy.Timer(rospy.Duration(0.1), self.control_callback)
        
        rospy.loginfo("Target follower initialized")
    
    def detection_callback(self, msg):
        """Process target detection."""
        if msg.id == 1:  # Follow AprilTag with ID 1
            # Update target position
            self.target_position = msg.pose.pose.position
            
            # Estimate target velocity (simple differentiation)
            if hasattr(self, 'last_target_pos') and hasattr(self, 'last_detection_time'):
                dt = (msg.header.stamp - self.last_detection_time).to_sec()
                if dt > 0:
                    dx = self.target_position.x - self.last_target_pos.x
                    dy = self.target_position.y - self.last_target_pos.y
                    dz = self.target_position.z - self.last_target_pos.z
                    
                    self.target_velocity = Point(
                        x=dx / dt,
                        y=dy / dt,
                        z=dz / dt
                    )
            
            self.last_target_pos = self.target_position
            self.last_detection_time = msg.header.stamp
    
    def control_callback(self, event):
        """Main control loop."""
        if self.target_position is None:
            return
        
        # Predict target future position
        prediction_time = 1.0  # seconds
        predicted_pos = self.predict_target_position(prediction_time)
        
        # Calculate desired following position
        follow_pos = self.calculate_follow_position(predicted_pos)
        
        # Generate control command
        cmd = self.generate_control_command(follow_pos)
        
        # Publish command
        self.cmd_pub.publish(cmd)
    
    def predict_target_position(self, dt):
        """Predict target position after time dt."""
        if self.target_velocity is None:
            return self.target_position
        
        predicted = Point(
            x=self.target_position.x + self.target_velocity.x * dt,
            y=self.target_position.y + self.target_velocity.y * dt,
            z=self.target_position.z + self.target_velocity.z * dt
        )
        
        return predicted
    
    def calculate_follow_position(self, target_pos):
        """Calculate desired following position."""
        # Follow at a fixed distance behind the target
        # Assume target is moving in XY plane
        
        if self.target_velocity and (self.target_velocity.x**2 + self.target_velocity.y**2) > 0.1:
            # Target is moving, follow behind
            vel_mag = math.sqrt(self.target_velocity.x**2 + self.target_velocity.y**2)
            vel_unit_x = -self.target_velocity.x / vel_mag
            vel_unit_y = -self.target_velocity.y / vel_mag
            
            follow_pos = Point(
                x=target_pos.x + vel_unit_x * self.follow_distance,
                y=target_pos.y + vel_unit_y * self.follow_distance,
                z=target_pos.z
            )
        else:
            # Target is stationary, maintain current relative position
            follow_pos = Point(
                x=target_pos.x - self.follow_distance,
                y=target_pos.y,
                z=target_pos.z
            )
        
        return follow_pos
    
    def generate_control_command(self, desired_pos):
        """Generate control command to reach desired position."""
        cmd = ControllerCommand()
        cmd.header.stamp = rospy.Time.now()
        cmd.header.frame_id = 'odom'
        cmd.mode = 1  # Position control
        
        cmd.setpoint.position = desired_pos
        
        # Face towards target
        if self.target_position:
            dx = self.target_position.x - desired_pos.x
            dy = self.target_position.y - desired_pos.y
            yaw = math.atan2(dy, dx)
            
            cmd.setpoint.orientation = self.yaw_to_quaternion(yaw)
        
        return cmd
    
    def yaw_to_quaternion(self, yaw):
        """Convert yaw to quaternion."""
        from geometry_msgs.msg import Quaternion
        return Quaternion(
            x=0.0,
            y=0.0,
            z=math.sin(yaw / 2.0),
            w=math.cos(yaw / 2.0)
        )

if __name__ == '__main__':
    try:
        follower = TargetFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
```

## Mission Planning

### Example 5: Search and Rescue Mission

```python
#!/usr/bin/env python
"""
Search and Rescue Mission

Implements a comprehensive search and rescue operation with
pattern search, target detection, and recovery procedures.
"""

import rospy
import numpy as np
from geometry_msgs.msg import Point, Pose
from modcube_msgs.msg import MissionState, AprilTagDetection
from modcube_mission import BaseMission
from modcube_common.motion import MotionClient
from modcube_common.planning import SearchPlanner

class SearchRescueMission(BaseMission):
    def __init__(self):
        super().__init__()
        
        self.motion_client = MotionClient()
        self.search_planner = SearchPlanner()
        
        # Mission parameters
        self.search_area = {
            'min_x': 0, 'max_x': 20,
            'min_y': 0, 'max_y': 20,
            'depth': -3.0
        }
        
        self.search_pattern = 'lawnmower'
        self.search_spacing = 2.0
        self.search_speed = 0.5
        
        # Target detection
        self.targets_found = []
        self.current_target = None
        
        # Subscribers
        self.detection_sub = rospy.Subscriber('/modcube/apriltag_detections',
                                            AprilTagDetection, self.detection_callback)
        
        rospy.loginfo("Search and Rescue mission initialized")
    
    def detection_callback(self, msg):
        """Handle target detection."""
        if msg.id not in [t['id'] for t in self.targets_found]:
            target = {
                'id': msg.id,
                'position': msg.pose.pose.position,
                'detection_time': rospy.Time.now(),
                'confidence': msg.confidence
            }
            
            self.targets_found.append(target)
            rospy.loginfo(f"New target detected: ID {msg.id} at {msg.pose.pose.position}")
    
    def execute(self):
        """Execute the search and rescue mission."""
        rospy.loginfo("Starting Search and Rescue mission")
        
        try:
            # Phase 1: Search phase
            self.update_state('searching')
            search_success = self.execute_search_phase()
            
            if not search_success:
                self.update_state('failed')
                return False
            
            # Phase 2: Investigation phase
            if self.targets_found:
                self.update_state('investigating')
                investigation_success = self.execute_investigation_phase()
                
                if not investigation_success:
                    self.update_state('failed')
                    return False
            
            # Phase 3: Recovery phase
            if self.targets_found:
                self.update_state('recovering')
                recovery_success = self.execute_recovery_phase()
                
                if not recovery_success:
                    self.update_state('failed')
                    return False
            
            # Mission completed
            self.update_state('completed')
            rospy.loginfo(f"Mission completed. Found {len(self.targets_found)} targets.")
            return True
            
        except Exception as e:
            rospy.logerr(f"Mission failed with error: {e}")
            self.update_state('failed')
            return False
    
    def execute_search_phase(self):
        """Execute systematic search of the area."""
        rospy.loginfo("Starting search phase")
        
        # Generate search pattern
        search_waypoints = self.search_planner.generate_pattern(
            area=self.search_area,
            pattern=self.search_pattern,
            spacing=self.search_spacing
        )
        
        # Execute search pattern
        for waypoint in search_waypoints:
            success = self.motion_client.goto_pose(waypoint)
            if not success:
                rospy.logwarn(f"Failed to reach search waypoint: {waypoint}")
                return False
            
            # Check for early termination if targets found
            if len(self.targets_found) >= 3:  # Stop after finding 3 targets
                rospy.loginfo("Sufficient targets found, ending search phase")
                break
        
        rospy.loginfo(f"Search phase completed. Found {len(self.targets_found)} targets.")
        return True
    
    def execute_investigation_phase(self):
        """Investigate detected targets for detailed analysis."""
        rospy.loginfo("Starting investigation phase")
        
        for target in self.targets_found:
            rospy.loginfo(f"Investigating target {target['id']}")
            
            # Move closer to target for detailed inspection
            investigation_pose = self.calculate_investigation_pose(target['position'])
            success = self.motion_client.goto_pose(investigation_pose)
            
            if success:
                # Perform detailed inspection
                inspection_result = self.perform_inspection(target)
                target['inspection_result'] = inspection_result
                
                rospy.loginfo(f"Target {target['id']} inspection completed")
            else:
                rospy.logwarn(f"Failed to investigate target {target['id']}")
        
        return True
    
    def execute_recovery_phase(self):
        """Execute recovery operations for confirmed targets."""
        rospy.loginfo("Starting recovery phase")
        
        # Prioritize targets based on investigation results
        priority_targets = self.prioritize_targets()
        
        for target in priority_targets:
            if target.get('inspection_result', {}).get('requires_recovery', False):
                rospy.loginfo(f"Executing recovery for target {target['id']}")
                
                recovery_success = self.execute_target_recovery(target)
                
                if recovery_success:
                    target['recovered'] = True
                    rospy.loginfo(f"Target {target['id']} successfully recovered")
                else:
                    rospy.logwarn(f"Failed to recover target {target['id']}")
        
        return True
    
    def calculate_investigation_pose(self, target_pos):
        """Calculate optimal pose for target investigation."""
        from geometry_msgs.msg import Pose
        import math
        
        # Position 2 meters away from target
        investigation_distance = 2.0
        
        pose = Pose()
        pose.position.x = target_pos.x - investigation_distance
        pose.position.y = target_pos.y
        pose.position.z = target_pos.z
        
        # Face towards target
        yaw = math.atan2(target_pos.y - pose.position.y, target_pos.x - pose.position.x)
        pose.orientation.z = math.sin(yaw / 2.0)
        pose.orientation.w = math.cos(yaw / 2.0)
        
        return pose
    
    def perform_inspection(self, target):
        """Perform detailed inspection of target."""
        # Simulate inspection process
        rospy.sleep(5.0)  # Inspection time
        
        # Mock inspection results
        inspection_result = {
            'target_type': 'survivor' if target['id'] % 2 == 0 else 'debris',
            'condition': 'good' if target['confidence'] > 0.8 else 'poor',
            'requires_recovery': target['id'] % 2 == 0,
            'priority': target['confidence']
        }
        
        return inspection_result
    
    def prioritize_targets(self):
        """Prioritize targets based on inspection results."""
        return sorted(self.targets_found, 
                     key=lambda t: t.get('inspection_result', {}).get('priority', 0), 
                     reverse=True)
    
    def execute_target_recovery(self, target):
        """Execute recovery operation for a specific target."""
        # Simulate recovery operation
        rospy.loginfo(f"Deploying recovery mechanism for target {target['id']}")
        
        # Move to recovery position
        recovery_pose = target['position']
        success = self.motion_client.goto_pose(recovery_pose)
        
        if success:
            # Simulate recovery actions
            rospy.sleep(10.0)  # Recovery time
            return True
        
        return False

if __name__ == '__main__':
    try:
        mission = SearchRescueMission()
        mission.execute()
    except rospy.ROSInterruptException:
        pass
```

### Example 6: Autonomous Inspection Mission

```python
#!/usr/bin/env python
"""
Autonomous Inspection Mission

Performs detailed inspection of underwater structures using
computer vision and structured data collection.
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Pose, Point
from modcube_msgs.msg import InspectionReport
from modcube_mission import BaseMission
from modcube_common.vision import DefectDetector
from modcube_common.motion import MotionClient
from cv_bridge import CvBridge

class InspectionMission(BaseMission):
    def __init__(self):
        super().__init__()
        
        self.motion_client = MotionClient()
        self.defect_detector = DefectDetector()
        self.bridge = CvBridge()
        
        # Inspection parameters
        self.inspection_distance = 1.5  # meters from structure
        self.inspection_speed = 0.2     # m/s
        self.image_capture_rate = 2.0   # Hz
        
        # Data collection
        self.inspection_data = []
        self.defects_found = []
        
        # Publishers and subscribers
        self.report_pub = rospy.Publisher('/modcube/inspection_report', 
                                        InspectionReport, queue_size=10)
        
        self.image_sub = rospy.Subscriber('/modcube/camera/image_raw', 
                                        Image, self.image_callback)
        
        self.pointcloud_sub = rospy.Subscriber('/modcube/pointcloud', 
                                             PointCloud2, self.pointcloud_callback)
        
        # Image capture timer
        self.capture_timer = rospy.Timer(rospy.Duration(1.0/self.image_capture_rate), 
                                       self.capture_callback)
        
        rospy.loginfo("Inspection mission initialized")
    
    def image_callback(self, msg):
        """Process incoming images for defect detection."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Detect defects in image
            defects = self.defect_detector.detect_defects(cv_image)
            
            if defects:
                self.process_defects(defects, msg.header)
                
        except Exception as e:
            rospy.logerr(f"Image processing error: {e}")
    
    def pointcloud_callback(self, msg):
        """Process point cloud data for 3D structure analysis."""
        # Extract 3D structure information
        structure_data = self.analyze_structure_3d(msg)
        
        # Store for inspection report
        self.inspection_data.append({
            'timestamp': msg.header.stamp,
            'structure_data': structure_data
        })
    
    def capture_callback(self, event):
        """Periodic data capture during inspection."""
        # Capture current pose and sensor data
        current_pose = self.motion_client.get_current_pose()
        
        if current_pose:
            inspection_point = {
                'timestamp': rospy.Time.now(),
                'pose': current_pose,
                'data_captured': True
            }
            
            self.inspection_data.append(inspection_point)
    
    def execute(self):
        """Execute the inspection mission."""
        rospy.loginfo("Starting autonomous inspection mission")
        
        try:
            # Phase 1: Structure approach
            self.update_state('approaching')
            approach_success = self.approach_structure()
            
            if not approach_success:
                self.update_state('failed')
                return False
            
            # Phase 2: Systematic inspection
            self.update_state('inspecting')
            inspection_success = self.execute_inspection_pattern()
            
            if not inspection_success:
                self.update_state('failed')
                return False
            
            # Phase 3: Detailed defect analysis
            if self.defects_found:
                self.update_state('analyzing')
                analysis_success = self.analyze_defects()
                
                if not analysis_success:
                    self.update_state('failed')
                    return False
            
            # Phase 4: Report generation
            self.update_state('reporting')
            self.generate_inspection_report()
            
            self.update_state('completed')
            rospy.loginfo("Inspection mission completed successfully")
            return True
            
        except Exception as e:
            rospy.logerr(f"Inspection mission failed: {e}")
            self.update_state('failed')
            return False
    
    def approach_structure(self):
        """Approach the structure to be inspected."""
        rospy.loginfo("Approaching inspection structure")
        
        # Define approach waypoints
        approach_poses = self.generate_approach_trajectory()
        
        for pose in approach_poses:
            success = self.motion_client.goto_pose(pose)
            if not success:
                rospy.logwarn("Failed to reach approach waypoint")
                return False
        
        return True
    
    def execute_inspection_pattern(self):
        """Execute systematic inspection pattern."""
        rospy.loginfo("Executing inspection pattern")
        
        # Generate inspection trajectory
        inspection_trajectory = self.generate_inspection_trajectory()
        
        # Follow trajectory at inspection speed
        success = self.motion_client.follow_trajectory(
            trajectory=inspection_trajectory,
            speed=self.inspection_speed
        )
        
        return success
    
    def generate_approach_trajectory(self):
        """Generate trajectory for approaching the structure."""
        # Simplified approach trajectory
        approach_poses = []
        
        # Start position (10m away)
        start_pose = Pose()
        start_pose.position = Point(x=-10.0, y=0.0, z=-2.0)
        start_pose.orientation.w = 1.0
        approach_poses.append(start_pose)
        
        # Intermediate position (5m away)
        mid_pose = Pose()
        mid_pose.position = Point(x=-5.0, y=0.0, z=-2.0)
        mid_pose.orientation.w = 1.0
        approach_poses.append(mid_pose)
        
        # Final approach position
        final_pose = Pose()
        final_pose.position = Point(x=-self.inspection_distance, y=0.0, z=-2.0)
        final_pose.orientation.w = 1.0
        approach_poses.append(final_pose)
        
        return approach_poses
    
    def generate_inspection_trajectory(self):
        """Generate systematic inspection trajectory."""
        trajectory = []
        
        # Vertical scanning pattern
        x_pos = -self.inspection_distance
        z_start = -1.0
        z_end = -4.0
        y_range = [-3.0, 3.0]
        
        # Generate zigzag pattern
        z_positions = np.linspace(z_start, z_end, 10)
        
        for i, z in enumerate(z_positions):
            if i % 2 == 0:  # Even rows: left to right
                y_positions = np.linspace(y_range[0], y_range[1], 5)
            else:  # Odd rows: right to left
                y_positions = np.linspace(y_range[1], y_range[0], 5)
            
            for y in y_positions:
                pose = Pose()
                pose.position = Point(x=x_pos, y=y, z=z)
                pose.orientation.w = 1.0
                trajectory.append(pose)
        
        return trajectory
    
    def process_defects(self, defects, header):
        """Process detected defects."""
        for defect in defects:
            defect_info = {
                'timestamp': header.stamp,
                'type': defect['type'],
                'severity': defect['severity'],
                'location': defect['location'],
                'confidence': defect['confidence'],
                'image_coords': defect['bbox']
            }
            
            self.defects_found.append(defect_info)
            rospy.loginfo(f"Defect detected: {defect['type']} (severity: {defect['severity']})")
    
    def analyze_structure_3d(self, pointcloud):
        """Analyze 3D structure from point cloud."""
        # Simplified 3D analysis
        structure_data = {
            'point_count': pointcloud.width * pointcloud.height,
            'density': 'high',  # Simplified
            'surface_roughness': 0.05,  # Mock value
            'geometric_features': ['planar', 'cylindrical']
        }
        
        return structure_data
    
    def analyze_defects(self):
        """Perform detailed analysis of detected defects."""
        rospy.loginfo(f"Analyzing {len(self.defects_found)} detected defects")
        
        for defect in self.defects_found:
            # Move closer to defect for detailed inspection
            detailed_pose = self.calculate_detailed_inspection_pose(defect)
            
            success = self.motion_client.goto_pose(detailed_pose)
            if success:
                # Perform detailed analysis
                detailed_analysis = self.perform_detailed_defect_analysis(defect)
                defect['detailed_analysis'] = detailed_analysis
        
        return True
    
    def calculate_detailed_inspection_pose(self, defect):
        """Calculate pose for detailed defect inspection."""
        # Move closer to defect location
        pose = Pose()
        pose.position = Point(
            x=-0.5,  # Very close to structure
            y=defect['location']['y'],
            z=defect['location']['z']
        )
        pose.orientation.w = 1.0
        
        return pose
    
    def perform_detailed_defect_analysis(self, defect):
        """Perform detailed analysis of a specific defect."""
        # Simulate detailed analysis
        rospy.sleep(3.0)
        
        analysis = {
            'dimensions': {'length': 0.15, 'width': 0.08, 'depth': 0.02},
            'material_loss': 15.2,  # percentage
            'corrosion_rate': 0.5,  # mm/year
            'repair_urgency': 'medium',
            'recommended_action': 'monitor'
        }
        
        return analysis
    
    def generate_inspection_report(self):
        """Generate comprehensive inspection report."""
        report = InspectionReport()
        report.header.stamp = rospy.Time.now()
        report.header.frame_id = 'inspection_report'
        
        # Summary statistics
        report.total_defects = len(self.defects_found)
        report.inspection_duration = (rospy.Time.now() - self.start_time).to_sec()
        report.area_covered = self.calculate_inspection_area()
        
        # Defect summary
        report.critical_defects = len([d for d in self.defects_found if d['severity'] == 'critical'])
        report.major_defects = len([d for d in self.defects_found if d['severity'] == 'major'])
        report.minor_defects = len([d for d in self.defects_found if d['severity'] == 'minor'])
        
        # Recommendations
        report.recommendations = self.generate_recommendations()
        
        # Publish report
        self.report_pub.publish(report)
        
        # Save detailed report to file
        self.save_detailed_report()
        
        rospy.loginfo("Inspection report generated and published")
    
    def calculate_inspection_area(self):
        """Calculate total area inspected."""
        # Simplified area calculation
        return 25.0  # square meters
    
    def generate_recommendations(self):
        """Generate maintenance recommendations."""
        recommendations = []
        
        critical_defects = [d for d in self.defects_found if d['severity'] == 'critical']
        if critical_defects:
            recommendations.append("Immediate repair required for critical defects")
        
        major_defects = [d for d in self.defects_found if d['severity'] == 'major']
        if major_defects:
            recommendations.append("Schedule maintenance for major defects within 30 days")
        
        if len(self.defects_found) > 10:
            recommendations.append("Consider comprehensive structural assessment")
        
        return recommendations
    
    def save_detailed_report(self):
        """Save detailed inspection report to file."""
        import json
        import os
        
        report_data = {
            'mission_id': self.mission_id,
            'timestamp': rospy.Time.now().to_sec(),
            'defects': self.defects_found,
            'inspection_data': self.inspection_data,
            'summary': {
                'total_defects': len(self.defects_found),
                'area_covered': self.calculate_inspection_area(),
                'recommendations': self.generate_recommendations()
            }
        }
        
        # Save to file
        filename = f"inspection_report_{self.mission_id}_{int(rospy.Time.now().to_sec())}.json"
        filepath = os.path.join('/tmp', filename)
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        rospy.loginfo(f"Detailed report saved to: {filepath}")

if __name__ == '__main__':
    try:
        mission = InspectionMission()
        mission.execute()
    except rospy.ROSInterruptException:
        pass
```

### Example 7: Multi-Robot Coordination

```python
#!/usr/bin/env python
"""
Multi-Robot Coordination Example

Demonstrates coordinated operation of multiple ModCube vehicles
for complex missions requiring teamwork.
"""

import rospy
import numpy as np
from geometry_msgs.msg import Pose, Point, Twist
from modcube_msgs.msg import FleetCommand, VehicleState, FormationConfig
from modcube_common.coordination import FleetManager, FormationController
from modcube_common.communication import InterVehicleComm
from std_msgs.msg import String
import threading
import time

class MultiRobotCoordinator:
    def __init__(self, vehicle_id, fleet_size=3):
        rospy.init_node(f'coordinator_{vehicle_id}')
        
        self.vehicle_id = vehicle_id
        self.fleet_size = fleet_size
        self.is_leader = (vehicle_id == 0)
        
        # Fleet management
        self.fleet_manager = FleetManager(vehicle_id, fleet_size)
        self.formation_controller = FormationController()
        self.inter_vehicle_comm = InterVehicleComm(vehicle_id)
        
        # Vehicle state
        self.current_pose = Pose()
        self.current_velocity = Twist()
        self.vehicle_states = {}  # States of all vehicles
        
        # Mission parameters
        self.formation_type = 'line'  # line, triangle, diamond
        self.formation_spacing = 5.0  # meters
        self.mission_waypoints = []
        self.current_waypoint_idx = 0
        
        # Publishers and subscribers
        self.cmd_pub = rospy.Publisher(f'/modcube_{vehicle_id}/cmd_vel', 
                                     Twist, queue_size=10)
        
        self.state_pub = rospy.Publisher('/fleet/vehicle_states', 
                                       VehicleState, queue_size=10)
        
        self.fleet_cmd_sub = rospy.Subscriber('/fleet/commands', 
                                            FleetCommand, self.fleet_command_callback)
        
        self.vehicle_state_sub = rospy.Subscriber('/fleet/vehicle_states', 
                                                VehicleState, self.vehicle_state_callback)
        
        self.pose_sub = rospy.Subscriber(f'/modcube_{vehicle_id}/nav_state', 
                                       VehicleState, self.pose_callback)
        
        # Inter-vehicle communication
        self.comm_sub = rospy.Subscriber('/fleet/inter_vehicle_comm', 
                                       String, self.inter_vehicle_callback)
        
        self.comm_pub = rospy.Publisher('/fleet/inter_vehicle_comm', 
                                      String, queue_size=10)
        
        # Control loop
        self.control_timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)
        
        # State publishing timer
        self.state_timer = rospy.Timer(rospy.Duration(0.2), self.publish_state)
        
        rospy.loginfo(f"Multi-robot coordinator initialized for vehicle {vehicle_id}")
        
        # Leader initialization
        if self.is_leader:
            self.initialize_mission()
    
    def pose_callback(self, msg):
        """Update current vehicle pose."""
        self.current_pose = msg.pose
        self.current_velocity = msg.velocity
    
    def vehicle_state_callback(self, msg):
        """Update fleet vehicle states."""
        if msg.vehicle_id != self.vehicle_id:
            self.vehicle_states[msg.vehicle_id] = msg
    
    def fleet_command_callback(self, msg):
        """Process fleet-wide commands."""
        if msg.command_type == 'formation_change':
            self.change_formation(msg.formation_config)
        elif msg.command_type == 'mission_waypoints':
            self.update_mission_waypoints(msg.waypoints)
        elif msg.command_type == 'emergency_stop':
            self.emergency_stop()
        elif msg.command_type == 'formation_spacing':
            self.formation_spacing = msg.spacing
    
    def inter_vehicle_callback(self, msg):
        """Process inter-vehicle communication."""
        try:
            comm_data = eval(msg.data)  # Simple parsing, use JSON in production
            
            if comm_data['target_id'] == self.vehicle_id or comm_data['target_id'] == 'all':
                self.process_inter_vehicle_message(comm_data)
                
        except Exception as e:
            rospy.logwarn(f"Failed to process inter-vehicle message: {e}")
    
    def initialize_mission(self):
        """Initialize mission parameters (leader only)."""
        if not self.is_leader:
            return
        
        # Define mission waypoints
        self.mission_waypoints = [
            Point(x=10.0, y=0.0, z=-2.0),
            Point(x=20.0, y=10.0, z=-2.0),
            Point(x=30.0, y=0.0, z=-2.0),
            Point(x=20.0, y=-10.0, z=-2.0),
            Point(x=0.0, y=0.0, z=-2.0)
        ]
        
        # Broadcast initial formation
        self.broadcast_formation_config()
        
        # Start mission after delay
        rospy.Timer(rospy.Duration(5.0), self.start_mission, oneshot=True)
    
    def broadcast_formation_config(self):
        """Broadcast formation configuration to fleet."""
        formation_config = FormationConfig()
        formation_config.formation_type = self.formation_type
        formation_config.spacing = self.formation_spacing
        formation_config.leader_id = self.vehicle_id
        
        fleet_cmd = FleetCommand()
        fleet_cmd.command_type = 'formation_change'
        fleet_cmd.formation_config = formation_config
        
        # Publish via inter-vehicle communication
        comm_msg = {
            'sender_id': self.vehicle_id,
            'target_id': 'all',
            'message_type': 'formation_config',
            'data': {
                'formation_type': self.formation_type,
                'spacing': self.formation_spacing,
                'leader_id': self.vehicle_id
            }
        }
        
        self.comm_pub.publish(String(data=str(comm_msg)))
    
    def start_mission(self, event):
        """Start the coordinated mission."""
        if self.is_leader:
            rospy.loginfo("Starting coordinated mission")
            
            # Broadcast mission start
            comm_msg = {
                'sender_id': self.vehicle_id,
                'target_id': 'all',
                'message_type': 'mission_start',
                'data': {
                    'waypoints': [(wp.x, wp.y, wp.z) for wp in self.mission_waypoints]
                }
            }
            
            self.comm_pub.publish(String(data=str(comm_msg)))
    
    def control_loop(self, event):
        """Main control loop for coordinated movement."""
        if not self.mission_waypoints:
            return
        
        # Calculate desired position based on role
        if self.is_leader:
            desired_pose = self.calculate_leader_position()
        else:
            desired_pose = self.calculate_follower_position()
        
        if desired_pose:
            # Generate control command
            cmd_vel = self.calculate_control_command(desired_pose)
            self.cmd_pub.publish(cmd_vel)
    
    def calculate_leader_position(self):
        """Calculate leader's desired position."""
        if self.current_waypoint_idx >= len(self.mission_waypoints):
            return None
        
        target_waypoint = self.mission_waypoints[self.current_waypoint_idx]
        
        # Check if close enough to current waypoint
        distance = self.calculate_distance(self.current_pose.position, target_waypoint)
        
        if distance < 2.0:  # Within 2 meters
            self.current_waypoint_idx += 1
            
            # Broadcast waypoint update
            if self.current_waypoint_idx < len(self.mission_waypoints):
                comm_msg = {
                    'sender_id': self.vehicle_id,
                    'target_id': 'all',
                    'message_type': 'waypoint_update',
                    'data': {
                        'current_waypoint_idx': self.current_waypoint_idx
                    }
                }
                self.comm_pub.publish(String(data=str(comm_msg)))
            else:
                rospy.loginfo("Mission completed")
                return None
        
        return target_waypoint
    
    def calculate_follower_position(self):
        """Calculate follower's desired position in formation."""
        # Get leader state
        leader_state = self.vehicle_states.get(0)
        if not leader_state:
            return None
        
        # Calculate formation offset
        formation_offset = self.calculate_formation_offset()
        
        # Calculate desired position relative to leader
        desired_pose = Point()
        desired_pose.x = leader_state.pose.position.x + formation_offset[0]
        desired_pose.y = leader_state.pose.position.y + formation_offset[1]
        desired_pose.z = leader_state.pose.position.z + formation_offset[2]
        
        return desired_pose
    
    def calculate_formation_offset(self):
        """Calculate formation offset based on vehicle ID and formation type."""
        if self.formation_type == 'line':
            # Line formation: vehicles arranged in a line behind leader
            offset_x = -self.vehicle_id * self.formation_spacing
            offset_y = 0.0
            offset_z = 0.0
        
        elif self.formation_type == 'triangle':
            # Triangle formation
            if self.vehicle_id == 1:
                offset_x = -self.formation_spacing
                offset_y = -self.formation_spacing / 2
            elif self.vehicle_id == 2:
                offset_x = -self.formation_spacing
                offset_y = self.formation_spacing / 2
            else:
                offset_x = -self.vehicle_id * self.formation_spacing
                offset_y = 0.0
            offset_z = 0.0
        
        elif self.formation_type == 'diamond':
            # Diamond formation
            if self.vehicle_id == 1:
                offset_x = -self.formation_spacing
                offset_y = 0.0
            elif self.vehicle_id == 2:
                offset_x = 0.0
                offset_y = -self.formation_spacing
            elif self.vehicle_id == 3:
                offset_x = 0.0
                offset_y = self.formation_spacing
            else:
                offset_x = -self.vehicle_id * self.formation_spacing
                offset_y = 0.0
            offset_z = 0.0
        
        else:
            # Default: line formation
            offset_x = -self.vehicle_id * self.formation_spacing
            offset_y = 0.0
            offset_z = 0.0
        
        return [offset_x, offset_y, offset_z]
    
    def calculate_control_command(self, target_position):
        """Calculate control command to reach target position."""
        cmd_vel = Twist()
        
        # Position error
        error_x = target_position.x - self.current_pose.position.x
        error_y = target_position.y - self.current_pose.position.y
        error_z = target_position.z - self.current_pose.position.z
        
        # Simple proportional control
        kp = 0.5
        
        cmd_vel.linear.x = kp * error_x
        cmd_vel.linear.y = kp * error_y
        cmd_vel.linear.z = kp * error_z
        
        # Limit velocities
        max_vel = 1.0
        cmd_vel.linear.x = max(-max_vel, min(max_vel, cmd_vel.linear.x))
        cmd_vel.linear.y = max(-max_vel, min(max_vel, cmd_vel.linear.y))
        cmd_vel.linear.z = max(-max_vel, min(max_vel, cmd_vel.linear.z))
        
        return cmd_vel
    
    def calculate_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two positions."""
        dx = pos1.x - pos2.x
        dy = pos1.y - pos2.y
        dz = pos1.z - pos2.z
        return np.sqrt(dx*dx + dy*dy + dz*dz)
    
    def change_formation(self, formation_config):
        """Change formation configuration."""
        self.formation_type = formation_config.formation_type
        self.formation_spacing = formation_config.spacing
        
        rospy.loginfo(f"Formation changed to {self.formation_type} with spacing {self.formation_spacing}")
    
    def update_mission_waypoints(self, waypoints):
        """Update mission waypoints."""
        self.mission_waypoints = waypoints
        self.current_waypoint_idx = 0
        
        rospy.loginfo(f"Mission waypoints updated: {len(waypoints)} waypoints")
    
    def emergency_stop(self):
        """Execute emergency stop."""
        rospy.logwarn("Emergency stop activated")
        
        # Stop vehicle
        stop_cmd = Twist()
        self.cmd_pub.publish(stop_cmd)
        
        # Clear mission waypoints
        self.mission_waypoints = []
    
    def process_inter_vehicle_message(self, comm_data):
        """Process inter-vehicle communication messages."""
        message_type = comm_data['message_type']
        data = comm_data['data']
        
        if message_type == 'formation_config':
            self.formation_type = data['formation_type']
            self.formation_spacing = data['spacing']
            
        elif message_type == 'mission_start':
            waypoints = [Point(x=wp[0], y=wp[1], z=wp[2]) for wp in data['waypoints']]
            self.mission_waypoints = waypoints
            self.current_waypoint_idx = 0
            
        elif message_type == 'waypoint_update':
            self.current_waypoint_idx = data['current_waypoint_idx']
            
        elif message_type == 'obstacle_detected':
            self.handle_obstacle_detection(data)
            
        elif message_type == 'formation_adjustment':
            self.handle_formation_adjustment(data)
    
    def handle_obstacle_detection(self, obstacle_data):
        """Handle obstacle detection from other vehicles."""
        rospy.logwarn(f"Obstacle detected by vehicle {obstacle_data['detector_id']}")
        
        # Implement obstacle avoidance logic
        # This could involve formation adjustment or path replanning
        pass
    
    def handle_formation_adjustment(self, adjustment_data):
        """Handle formation adjustment requests."""
        # Implement dynamic formation adjustment
        pass
    
    def publish_state(self, event):
        """Publish current vehicle state to fleet."""
        state_msg = VehicleState()
        state_msg.vehicle_id = self.vehicle_id
        state_msg.pose = self.current_pose
        state_msg.velocity = self.current_velocity
        state_msg.timestamp = rospy.Time.now()
        
        self.state_pub.publish(state_msg)

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        rospy.logerr("Usage: multi_robot_coordinator.py <vehicle_id>")
        sys.exit(1)
    
    vehicle_id = int(sys.argv[1])
    
    try:
        coordinator = MultiRobotCoordinator(vehicle_id)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
```

### Example 8: Machine Learning Integration

```python
#!/usr/bin/env python
"""
Machine Learning Integration Example

Demonstrates integration of machine learning models for
autonomous decision making and adaptive behavior.
"""

import rospy
import numpy as np
import tensorflow as tf
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Twist, Pose
from modcube_msgs.msg import MLPrediction, BehaviorCommand
from cv_bridge import CvBridge
import cv2
from collections import deque
import pickle
import os

class MLIntegratedController:
    def __init__(self):
        rospy.init_node('ml_controller')
        
        self.bridge = CvBridge()
        
        # Load pre-trained models
        self.load_models()
        
        # Data buffers
        self.image_buffer = deque(maxlen=10)
        self.sensor_buffer = deque(maxlen=50)
        self.action_history = deque(maxlen=100)
        
        # State variables
        self.current_pose = Pose()
        self.current_image = None
        self.current_pointcloud = None
        
        # ML parameters
        self.prediction_confidence_threshold = 0.7
        self.learning_rate = 0.001
        self.exploration_rate = 0.1
        
        # Publishers and subscribers
        self.cmd_pub = rospy.Publisher('/modcube/cmd_vel', Twist, queue_size=10)
        self.prediction_pub = rospy.Publisher('/modcube/ml_prediction', 
                                            MLPrediction, queue_size=10)
        
        self.image_sub = rospy.Subscriber('/modcube/camera/image_raw', 
                                        Image, self.image_callback)
        self.pointcloud_sub = rospy.Subscriber('/modcube/pointcloud', 
                                             PointCloud2, self.pointcloud_callback)
        self.pose_sub = rospy.Subscriber('/modcube/nav_state', 
                                       Pose, self.pose_callback)
        
        # Control loop
        self.control_timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)
        
        # Model update timer
        self.update_timer = rospy.Timer(rospy.Duration(1.0), self.update_models)
        
        rospy.loginfo("ML integrated controller initialized")
    
    def load_models(self):
        """Load pre-trained machine learning models."""
        model_path = rospy.get_param('~model_path', '/tmp/modcube_models')
        
        try:
            # Object detection model
            self.object_detector = tf.keras.models.load_model(
                os.path.join(model_path, 'object_detector.h5')
            )
            
            # Behavior prediction model
            self.behavior_predictor = tf.keras.models.load_model(
                os.path.join(model_path, 'behavior_predictor.h5')
            )
            
            # Reinforcement learning policy
            with open(os.path.join(model_path, 'rl_policy.pkl'), 'rb') as f:
                self.rl_policy = pickle.load(f)
            
            rospy.loginfo("ML models loaded successfully")
            
        except Exception as e:
            rospy.logwarn(f"Failed to load ML models: {e}")
            # Initialize simple models as fallback
            self.initialize_fallback_models()
    
    def initialize_fallback_models(self):
        """Initialize simple fallback models."""
        # Simple CNN for object detection
        self.object_detector = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')  # 10 object classes
        ])
        
        # Simple LSTM for behavior prediction
        self.behavior_predictor = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, input_shape=(10, 6)),  # 10 timesteps, 6 features
            tf.keras.layers.Dense(25, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')  # 3 behaviors
        ])
        
        # Simple Q-learning policy
        self.rl_policy = {
            'q_table': np.random.rand(100, 4),  # 100 states, 4 actions
            'state_discretizer': lambda x: min(99, max(0, int(x * 10)))
        }
        
        rospy.loginfo("Fallback models initialized")
    
    def image_callback(self, msg):
        """Process incoming camera images."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.current_image = cv_image
            
            # Add to buffer
            self.image_buffer.append(cv_image)
            
            # Perform object detection
            self.detect_objects(cv_image)
            
        except Exception as e:
            rospy.logerr(f"Image processing error: {e}")
    
    def pointcloud_callback(self, msg):
        """Process point cloud data."""
        self.current_pointcloud = msg
        
        # Extract features from point cloud
        features = self.extract_pointcloud_features(msg)
        
        # Add to sensor buffer
        sensor_data = {
            'timestamp': msg.header.stamp.to_sec(),
            'pointcloud_features': features
        }
        self.sensor_buffer.append(sensor_data)
    
    def pose_callback(self, msg):
        """Update current pose."""
        self.current_pose = msg
    
    def detect_objects(self, image):
        """Detect objects in the image using ML model."""
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Run inference
            predictions = self.object_detector.predict(processed_image)
            
            # Process predictions
            detected_objects = self.process_object_predictions(predictions)
            
            # Publish predictions
            self.publish_ml_predictions(detected_objects)
            
        except Exception as e:
            rospy.logerr(f"Object detection error: {e}")
    
    def preprocess_image(self, image):
        """Preprocess image for ML model input."""
        # Resize to model input size
        resized = cv2.resize(image, (224, 224))
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)
        
        return batched
    
    def process_object_predictions(self, predictions):
        """Process object detection predictions."""
        detected_objects = []
        
        # Get class with highest confidence
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        
        if confidence > self.prediction_confidence_threshold:
            object_classes = ['obstacle', 'target', 'wall', 'floor', 'ceiling', 
                            'vehicle', 'debris', 'structure', 'marker', 'unknown']
            
            detected_objects.append({
                'class': object_classes[class_idx],
                'confidence': float(confidence),
                'timestamp': rospy.Time.now().to_sec()
            })
        
        return detected_objects
    
    def extract_pointcloud_features(self, pointcloud):
        """Extract features from point cloud data."""
        # Simplified feature extraction
        features = {
            'point_count': pointcloud.width * pointcloud.height,
            'density': 1.0,  # Placeholder
            'mean_distance': 2.0,  # Placeholder
            'variance': 0.5,  # Placeholder
            'obstacle_detected': False  # Placeholder
        }
        
        return features
    
    def control_loop(self, event):
        """Main control loop with ML-based decision making."""
        if not self.current_image or not self.sensor_buffer:
            return
        
        # Get current state
        current_state = self.get_current_state()
        
        # Predict behavior using ML
        predicted_behavior = self.predict_behavior(current_state)
        
        # Generate action using RL policy
        action = self.select_action(current_state, predicted_behavior)
        
        # Convert action to control command
        cmd_vel = self.action_to_command(action)
        
        # Publish command
        self.cmd_pub.publish(cmd_vel)
        
        # Store action for learning
        self.action_history.append({
            'state': current_state,
            'action': action,
            'timestamp': rospy.Time.now().to_sec()
        })
    
    def get_current_state(self):
        """Get current state representation for ML models."""
        state = {
            'pose': {
                'x': self.current_pose.position.x,
                'y': self.current_pose.position.y,
                'z': self.current_pose.position.z
            },
            'sensor_data': list(self.sensor_buffer)[-5:],  # Last 5 sensor readings
            'image_available': self.current_image is not None,
            'pointcloud_available': self.current_pointcloud is not None
        }
        
        return state
    
    def predict_behavior(self, state):
        """Predict optimal behavior using ML model."""
        try:
            # Prepare input for behavior predictor
            behavior_input = self.prepare_behavior_input(state)
            
            # Run prediction
            behavior_probs = self.behavior_predictor.predict(behavior_input)
            
            # Get predicted behavior
            behavior_classes = ['explore', 'approach', 'avoid']
            predicted_class = np.argmax(behavior_probs[0])
            confidence = behavior_probs[0][predicted_class]
            
            return {
                'behavior': behavior_classes[predicted_class],
                'confidence': float(confidence)
            }
            
        except Exception as e:
            rospy.logerr(f"Behavior prediction error: {e}")
            return {'behavior': 'explore', 'confidence': 0.5}
    
    def prepare_behavior_input(self, state):
        """Prepare input for behavior prediction model."""
        # Create feature vector from state
        features = []
        
        # Add pose features
        features.extend([state['pose']['x'], state['pose']['y'], state['pose']['z']])
        
        # Add sensor features (simplified)
        if state['sensor_data']:
            latest_sensor = state['sensor_data'][-1]
            features.extend([
                latest_sensor['pointcloud_features']['point_count'] / 1000.0,
                latest_sensor['pointcloud_features']['density'],
                latest_sensor['pointcloud_features']['mean_distance']
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Create sequence for LSTM (repeat current features for simplicity)
        sequence = np.array([features] * 10)
        
        # Add batch dimension
        batched_sequence = np.expand_dims(sequence, axis=0)
        
        return batched_sequence
    
    def select_action(self, state, predicted_behavior):
        """Select action using RL policy."""
        try:
            # Discretize state for Q-table lookup
            state_idx = self.discretize_state(state)
            
            # Epsilon-greedy action selection
            if np.random.random() < self.exploration_rate:
                # Random action
                action = np.random.randint(0, 4)
            else:
                # Greedy action from Q-table
                q_values = self.rl_policy['q_table'][state_idx]
                action = np.argmax(q_values)
            
            return action
            
        except Exception as e:
            rospy.logerr(f"Action selection error: {e}")
            return 0  # Default action
    
    def discretize_state(self, state):
        """Discretize continuous state for Q-table lookup."""
        # Simple discretization based on position
        x_discrete = min(9, max(0, int((state['pose']['x'] + 10) / 2)))
        y_discrete = min(9, max(0, int((state['pose']['y'] + 10) / 2)))
        
        state_idx = x_discrete * 10 + y_discrete
        return min(99, state_idx)
    
    def action_to_command(self, action):
        """Convert discrete action to velocity command."""
        cmd_vel = Twist()
        
        # Action mapping: 0=forward, 1=backward, 2=left, 3=right
        if action == 0:  # Forward
            cmd_vel.linear.x = 0.5
        elif action == 1:  # Backward
            cmd_vel.linear.x = -0.5
        elif action == 2:  # Left
            cmd_vel.linear.y = 0.5
        elif action == 3:  # Right
            cmd_vel.linear.y = -0.5
        
        return cmd_vel
    
    def update_models(self, event):
        """Periodically update ML models with new data."""
        if len(self.action_history) < 10:
            return
        
        try:
            # Update RL policy with recent experiences
            self.update_rl_policy()
            
            # Optionally retrain other models
            # self.retrain_behavior_predictor()
            
        except Exception as e:
            rospy.logerr(f"Model update error: {e}")
    
    def update_rl_policy(self):
        """Update RL policy using recent experiences."""
        # Simple Q-learning update
        for i in range(len(self.action_history) - 1):
            current_exp = self.action_history[i]
            next_exp = self.action_history[i + 1]
            
            # Calculate reward (simplified)
            reward = self.calculate_reward(current_exp, next_exp)
            
            # Q-learning update
            current_state_idx = self.discretize_state(current_exp['state'])
            next_state_idx = self.discretize_state(next_exp['state'])
            action = current_exp['action']
            
            # Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
            alpha = self.learning_rate
            gamma = 0.9
            
            current_q = self.rl_policy['q_table'][current_state_idx][action]
            max_next_q = np.max(self.rl_policy['q_table'][next_state_idx])
            
            new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)
            self.rl_policy['q_table'][current_state_idx][action] = new_q
    
    def calculate_reward(self, current_exp, next_exp):
        """Calculate reward for RL update."""
        # Simple reward function
        current_pos = current_exp['state']['pose']
        next_pos = next_exp['state']['pose']
        
        # Reward for movement (exploration)
        movement = np.sqrt(
            (next_pos['x'] - current_pos['x'])**2 + 
            (next_pos['y'] - current_pos['y'])**2
        )
        
        reward = movement * 0.1  # Small positive reward for movement
        
        # Penalty for staying in same place
        if movement < 0.1:
            reward -= 0.05
        
        return reward
    
    def publish_ml_predictions(self, predictions):
        """Publish ML predictions."""
        for pred in predictions:
            ml_msg = MLPrediction()
            ml_msg.header.stamp = rospy.Time.now()
            ml_msg.object_class = pred['class']
            ml_msg.confidence = pred['confidence']
            
            self.prediction_pub.publish(ml_msg)

if __name__ == '__main__':
    try:
        controller = MLIntegratedController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
```

### Example 9: Real-Time Optimization and Performance Monitoring

```python
#!/usr/bin/env python
"""
Real-Time Optimization and Performance Monitoring

Demonstrates advanced optimization techniques and comprehensive
performance monitoring for ModCube systems.
"""

import rospy
import numpy as np
import time
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import Imu, FluidPressure
from modcube_msgs.msg import PerformanceMetrics, OptimizationStatus
from std_msgs.msg import Float64MultiArray
import threading
from collections import deque
import psutil
import resource
from scipy.optimize import minimize
import cvxpy as cp

class RealTimeOptimizer:
    def __init__(self):
        rospy.init_node('realtime_optimizer')
        
        # Performance monitoring
        self.cpu_usage_history = deque(maxlen=100)
        self.memory_usage_history = deque(maxlen=100)
        self.control_loop_times = deque(maxlen=1000)
        self.optimization_times = deque(maxlen=100)
        
        # System state
        self.current_pose = Pose()
        self.current_velocity = Twist()
        self.imu_data = None
        self.pressure_data = None
        
        # Optimization parameters
        self.optimization_horizon = 10  # MPC horizon
        self.dt = 0.1  # Time step
        self.max_optimization_time = 0.05  # 50ms max
        
        # Control constraints
        self.max_thrust = 50.0  # N
        self.max_velocity = 2.0  # m/s
        self.max_acceleration = 1.0  # m/s²
        
        # Performance thresholds
        self.cpu_threshold = 80.0  # %
        self.memory_threshold = 85.0  # %
        self.control_frequency_threshold = 8.0  # Hz minimum
        
        # Adaptive parameters
        self.adaptive_horizon = True
        self.dynamic_constraints = True
        self.load_balancing = True
        
        # Publishers and subscribers
        self.cmd_pub = rospy.Publisher('/modcube/cmd_vel', Twist, queue_size=1)
        self.metrics_pub = rospy.Publisher('/modcube/performance_metrics', 
                                         PerformanceMetrics, queue_size=10)
        self.optimization_pub = rospy.Publisher('/modcube/optimization_status', 
                                              OptimizationStatus, queue_size=10)
        
        self.pose_sub = rospy.Subscriber('/modcube/nav_state', Pose, self.pose_callback)
        self.imu_sub = rospy.Subscriber('/modcube/imu', Imu, self.imu_callback)
        self.pressure_sub = rospy.Subscriber('/modcube/pressure', 
                                           FluidPressure, self.pressure_callback)
        
        # Target waypoints
        self.target_waypoints = [
            [10.0, 0.0, -2.0],
            [20.0, 10.0, -3.0],
            [30.0, 0.0, -2.0],
            [20.0, -10.0, -4.0],
            [0.0, 0.0, -2.0]
        ]
        self.current_target_idx = 0
        
        # Control and monitoring timers
        self.control_timer = rospy.Timer(rospy.Duration(self.dt), self.control_loop)
        self.monitoring_timer = rospy.Timer(rospy.Duration(1.0), self.monitor_performance)
        self.optimization_timer = rospy.Timer(rospy.Duration(0.5), self.adaptive_optimization)
        
        rospy.loginfo("Real-time optimizer initialized")
    
    def pose_callback(self, msg):
        """Update current pose."""
        self.current_pose = msg
    
    def imu_callback(self, msg):
        """Update IMU data."""
        self.imu_data = msg
    
    def pressure_callback(self, msg):
        """Update pressure data."""
        self.pressure_data = msg
    
    def control_loop(self, event):
        """Main control loop with real-time optimization."""
        start_time = time.time()
        
        try:
            # Get current state
            current_state = self.get_current_state()
            
            if current_state is None:
                return
            
            # Get current target
            target = self.get_current_target()
            
            if target is None:
                return
            
            # Solve optimization problem
            optimal_control = self.solve_mpc_problem(current_state, target)
            
            # Apply control
            if optimal_control is not None:
                cmd_vel = self.control_to_twist(optimal_control)
                self.cmd_pub.publish(cmd_vel)
            
            # Record timing
            loop_time = time.time() - start_time
            self.control_loop_times.append(loop_time)
            
            # Check for target reached
            self.check_target_reached(current_state, target)
            
        except Exception as e:
            rospy.logerr(f"Control loop error: {e}")
    
    def get_current_state(self):
        """Get current system state."""
        if not self.current_pose:
            return None
        
        state = np.array([
            self.current_pose.position.x,
            self.current_pose.position.y,
            self.current_pose.position.z,
            0.0,  # vx (would come from velocity estimation)
            0.0,  # vy
            0.0   # vz
        ])
        
        return state
    
    def get_current_target(self):
        """Get current target waypoint."""
        if self.current_target_idx >= len(self.target_waypoints):
            return None
        
        return np.array(self.target_waypoints[self.current_target_idx])
    
    def solve_mpc_problem(self, current_state, target):
        """Solve Model Predictive Control optimization problem."""
        start_time = time.time()
        
        try:
            # Adaptive horizon based on system load
            horizon = self.get_adaptive_horizon()
            
            # State and control dimensions
            n_states = 6  # [x, y, z, vx, vy, vz]
            n_controls = 3  # [fx, fy, fz]
            
            # Decision variables
            x = cp.Variable((n_states, horizon + 1))
            u = cp.Variable((n_controls, horizon))
            
            # System dynamics (simplified)
            A = np.eye(n_states)
            A[0:3, 3:6] = np.eye(3) * self.dt  # position integration
            
            B = np.zeros((n_states, n_controls))
            B[3:6, 0:3] = np.eye(3) * self.dt  # acceleration from thrust
            
            # Cost matrices
            Q = np.diag([10, 10, 10, 1, 1, 1])  # State cost
            R = np.diag([0.1, 0.1, 0.1])  # Control cost
            Qf = Q * 10  # Terminal cost
            
            # Objective function
            cost = 0
            constraints = []
            
            # Initial condition
            constraints.append(x[:, 0] == current_state)
            
            for k in range(horizon):
                # Dynamics constraint
                constraints.append(x[:, k+1] == A @ x[:, k] + B @ u[:, k])
                
                # Stage cost
                target_state = np.concatenate([target, np.zeros(3)])
                cost += cp.quad_form(x[:, k] - target_state, Q)
                cost += cp.quad_form(u[:, k], R)
                
                # Control constraints
                constraints.append(cp.norm(u[:, k], 2) <= self.max_thrust)
                
                # Velocity constraints
                constraints.append(cp.norm(x[3:6, k], 2) <= self.max_velocity)
            
            # Terminal cost
            target_state = np.concatenate([target, np.zeros(3)])
            cost += cp.quad_form(x[:, horizon] - target_state, Qf)
            
            # Solve optimization problem
            problem = cp.Problem(cp.Minimize(cost), constraints)
            
            # Set solver parameters for real-time performance
            problem.solve(solver=cp.OSQP, 
                         max_iter=1000, 
                         eps_abs=1e-3, 
                         eps_rel=1e-3,
                         verbose=False)
            
            optimization_time = time.time() - start_time
            self.optimization_times.append(optimization_time)
            
            # Publish optimization status
            self.publish_optimization_status(problem.status, optimization_time)
            
            if problem.status == cp.OPTIMAL:
                return u.value[:, 0]  # Return first control action
            else:
                rospy.logwarn(f"Optimization failed with status: {problem.status}")
                return None
                
        except Exception as e:
            rospy.logerr(f"MPC optimization error: {e}")
            return None
    
    def get_adaptive_horizon(self):
        """Get adaptive optimization horizon based on system load."""
        if not self.adaptive_horizon:
            return self.optimization_horizon
        
        # Get current CPU usage
        current_cpu = psutil.cpu_percent()
        
        # Reduce horizon if CPU usage is high
        if current_cpu > self.cpu_threshold:
            return max(3, self.optimization_horizon // 2)
        elif current_cpu > 60:
            return max(5, int(self.optimization_horizon * 0.75))
        else:
            return self.optimization_horizon
    
    def control_to_twist(self, control):
        """Convert control forces to Twist message."""
        cmd_vel = Twist()
        
        # Simple mapping from forces to velocities
        # In practice, this would involve thruster allocation
        cmd_vel.linear.x = np.clip(control[0] / 10.0, -2.0, 2.0)
        cmd_vel.linear.y = np.clip(control[1] / 10.0, -2.0, 2.0)
        cmd_vel.linear.z = np.clip(control[2] / 10.0, -2.0, 2.0)
        
        return cmd_vel
    
    def check_target_reached(self, current_state, target):
        """Check if current target is reached."""
        distance = np.linalg.norm(current_state[0:3] - target)
        
        if distance < 1.0:  # Within 1 meter
            self.current_target_idx += 1
            if self.current_target_idx < len(self.target_waypoints):
                rospy.loginfo(f"Target {self.current_target_idx - 1} reached. Moving to next target.")
            else:
                rospy.loginfo("All targets reached. Mission complete.")
    
    def monitor_performance(self, event):
        """Monitor system performance metrics."""
        try:
            # CPU and memory usage
            cpu_percent = psutil.cpu_percent()
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent
            
            self.cpu_usage_history.append(cpu_percent)
            self.memory_usage_history.append(memory_percent)
            
            # Control loop frequency
            if len(self.control_loop_times) > 10:
                recent_times = list(self.control_loop_times)[-10:]
                avg_loop_time = np.mean(recent_times)
                control_frequency = 1.0 / avg_loop_time if avg_loop_time > 0 else 0
            else:
                control_frequency = 0
            
            # Optimization performance
            if len(self.optimization_times) > 5:
                recent_opt_times = list(self.optimization_times)[-5:]
                avg_opt_time = np.mean(recent_opt_times)
                max_opt_time = np.max(recent_opt_times)
            else:
                avg_opt_time = 0
                max_opt_time = 0
            
            # Publish performance metrics
            self.publish_performance_metrics(
                cpu_percent, memory_percent, control_frequency,
                avg_opt_time, max_opt_time
            )
            
            # Check for performance issues
            self.check_performance_issues(
                cpu_percent, memory_percent, control_frequency
            )
            
        except Exception as e:
            rospy.logerr(f"Performance monitoring error: {e}")
    
    def publish_performance_metrics(self, cpu_percent, memory_percent, 
                                  control_frequency, avg_opt_time, max_opt_time):
        """Publish performance metrics."""
        metrics = PerformanceMetrics()
        metrics.header.stamp = rospy.Time.now()
        
        metrics.cpu_usage = cpu_percent
        metrics.memory_usage = memory_percent
        metrics.control_frequency = control_frequency
        metrics.avg_optimization_time = avg_opt_time
        metrics.max_optimization_time = max_opt_time
        
        # Calculate additional metrics
        if len(self.cpu_usage_history) > 1:
            metrics.cpu_usage_trend = np.mean(list(self.cpu_usage_history)[-10:])
        
        if len(self.control_loop_times) > 1:
            metrics.control_jitter = np.std(list(self.control_loop_times)[-20:])
        
        self.metrics_pub.publish(metrics)
    
    def publish_optimization_status(self, status, optimization_time):
        """Publish optimization status."""
        opt_status = OptimizationStatus()
        opt_status.header.stamp = rospy.Time.now()
        opt_status.solver_status = str(status)
        opt_status.solve_time = optimization_time
        opt_status.horizon_length = self.get_adaptive_horizon()
        opt_status.target_index = self.current_target_idx
        
        self.optimization_pub.publish(opt_status)
    
    def check_performance_issues(self, cpu_percent, memory_percent, control_frequency):
        """Check for performance issues and take corrective actions."""
        issues = []
        
        # CPU usage check
        if cpu_percent > self.cpu_threshold:
            issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            if self.adaptive_horizon:
                rospy.logwarn("Reducing optimization horizon due to high CPU usage")
        
        # Memory usage check
        if memory_percent > self.memory_threshold:
            issues.append(f"High memory usage: {memory_percent:.1f}%")
            self.cleanup_old_data()
        
        # Control frequency check
        if control_frequency < self.control_frequency_threshold:
            issues.append(f"Low control frequency: {control_frequency:.1f} Hz")
        
        # Optimization time check
        if len(self.optimization_times) > 0:
            recent_opt_time = self.optimization_times[-1]
            if recent_opt_time > self.max_optimization_time:
                issues.append(f"Slow optimization: {recent_opt_time:.3f}s")
        
        if issues:
            rospy.logwarn(f"Performance issues detected: {', '.join(issues)}")
    
    def cleanup_old_data(self):
        """Clean up old data to free memory."""
        # Reduce buffer sizes temporarily
        if len(self.control_loop_times) > 500:
            # Keep only recent data
            recent_data = list(self.control_loop_times)[-500:]
            self.control_loop_times.clear()
            self.control_loop_times.extend(recent_data)
        
        rospy.loginfo("Cleaned up old performance data")
    
    def adaptive_optimization(self, event):
        """Perform adaptive optimization parameter tuning."""
        try:
            # Analyze recent performance
            if len(self.optimization_times) < 5:
                return
            
            recent_opt_times = list(self.optimization_times)[-5:]
            avg_opt_time = np.mean(recent_opt_times)
            
            # Adjust parameters based on performance
            if avg_opt_time > self.max_optimization_time * 0.8:
                # Optimization is taking too long
                if self.optimization_horizon > 3:
                    self.optimization_horizon -= 1
                    rospy.loginfo(f"Reduced optimization horizon to {self.optimization_horizon}")
            
            elif avg_opt_time < self.max_optimization_time * 0.3:
                # Optimization is fast, can increase horizon
                if self.optimization_horizon < 15:
                    self.optimization_horizon += 1
                    rospy.loginfo(f"Increased optimization horizon to {self.optimization_horizon}")
            
            # Adjust control frequency based on system load
            current_cpu = psutil.cpu_percent()
            if current_cpu > self.cpu_threshold and self.dt < 0.2:
                self.dt += 0.01
                # Update timer
                self.control_timer.shutdown()
                self.control_timer = rospy.Timer(rospy.Duration(self.dt), self.control_loop)
                rospy.loginfo(f"Reduced control frequency to {1.0/self.dt:.1f} Hz")
            
            elif current_cpu < 50 and self.dt > 0.05:
                self.dt -= 0.01
                # Update timer
                self.control_timer.shutdown()
                self.control_timer = rospy.Timer(rospy.Duration(self.dt), self.control_loop)
                rospy.loginfo(f"Increased control frequency to {1.0/self.dt:.1f} Hz")
                
        except Exception as e:
            rospy.logerr(f"Adaptive optimization error: {e}")

class PerformanceBenchmark:
    """Performance benchmarking utilities."""
    
    def __init__(self):
        self.benchmark_results = {}
    
    def benchmark_control_loop(self, optimizer, iterations=1000):
        """Benchmark control loop performance."""
        rospy.loginfo(f"Starting control loop benchmark ({iterations} iterations)")
        
        times = []
        for i in range(iterations):
            start_time = time.time()
            
            # Simulate control loop
            current_state = np.random.rand(6)
            target = np.random.rand(3) * 10
            
            optimal_control = optimizer.solve_mpc_problem(current_state, target)
            
            end_time = time.time()
            times.append(end_time - start_time)
            
            if i % 100 == 0:
                rospy.loginfo(f"Benchmark progress: {i}/{iterations}")
        
        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        max_time = np.max(times)
        min_time = np.min(times)
        
        self.benchmark_results['control_loop'] = {
            'avg_time': avg_time,
            'std_time': std_time,
            'max_time': max_time,
            'min_time': min_time,
            'frequency': 1.0 / avg_time
        }
        
        rospy.loginfo(f"Control loop benchmark results:")
        rospy.loginfo(f"  Average time: {avg_time:.4f}s")
        rospy.loginfo(f"  Std deviation: {std_time:.4f}s")
        rospy.loginfo(f"  Max time: {max_time:.4f}s")
        rospy.loginfo(f"  Min time: {min_time:.4f}s")
        rospy.loginfo(f"  Average frequency: {1.0/avg_time:.1f} Hz")
    
    def benchmark_memory_usage(self, duration=60):
        """Benchmark memory usage over time."""
        rospy.loginfo(f"Starting memory benchmark ({duration}s)")
        
        start_time = time.time()
        memory_samples = []
        
        while time.time() - start_time < duration:
            memory_info = psutil.virtual_memory()
            memory_samples.append(memory_info.percent)
            time.sleep(1.0)
        
        # Calculate statistics
        avg_memory = np.mean(memory_samples)
        max_memory = np.max(memory_samples)
        min_memory = np.min(memory_samples)
        
        self.benchmark_results['memory'] = {
            'avg_usage': avg_memory,
            'max_usage': max_memory,
            'min_usage': min_memory,
            'samples': len(memory_samples)
        }
        
        rospy.loginfo(f"Memory benchmark results:")
        rospy.loginfo(f"  Average usage: {avg_memory:.1f}%")
        rospy.loginfo(f"  Max usage: {max_memory:.1f}%")
        rospy.loginfo(f"  Min usage: {min_memory:.1f}%")

if __name__ == '__main__':
    try:
        optimizer = RealTimeOptimizer()
        
        # Optional: Run benchmarks
        if rospy.get_param('~run_benchmark', False):
            benchmark = PerformanceBenchmark()
            benchmark.benchmark_control_loop(optimizer, 100)
            benchmark.benchmark_memory_usage(30)
        
        rospy.spin()
        
    except rospy.ROSInterruptException:
        pass
```

## Advanced Integration Examples

### Example 10: Complete Mission Integration

```python
#!/usr/bin/env python
"""
Complete Mission Integration Example

Demonstrates integration of all advanced features:
- Multi-robot coordination
- Machine learning
- Real-time optimization
- Performance monitoring
"""

import rospy
from multi_robot_coordinator import MultiRobotCoordinator
from ml_integrated_controller import MLIntegratedController
from realtime_optimizer import RealTimeOptimizer
import threading
import time

class IntegratedMissionController:
    def __init__(self, vehicle_id):
        self.vehicle_id = vehicle_id
        
        # Initialize subsystems
        self.coordinator = MultiRobotCoordinator(vehicle_id)
        self.ml_controller = MLIntegratedController()
        self.optimizer = RealTimeOptimizer()
        
        # Mission state
        self.mission_active = False
        self.mission_phase = 'idle'
        
        rospy.loginfo(f"Integrated mission controller initialized for vehicle {vehicle_id}")
    
    def execute_integrated_mission(self):
        """Execute complete integrated mission."""
        rospy.loginfo("Starting integrated mission")
        
        try:
            # Phase 1: Formation and coordination
            self.mission_phase = 'formation'
            self.execute_formation_phase()
            
            # Phase 2: ML-guided exploration
            self.mission_phase = 'exploration'
            self.execute_exploration_phase()
            
            # Phase 3: Optimized task execution
            self.mission_phase = 'execution'
            self.execute_task_phase()
            
            # Phase 4: Return and debrief
            self.mission_phase = 'return'
            self.execute_return_phase()
            
            rospy.loginfo("Integrated mission completed successfully")
            
        except Exception as e:
            rospy.logerr(f"Integrated mission failed: {e}")
    
    def execute_formation_phase(self):
        """Execute formation and coordination phase."""
        rospy.loginfo("Executing formation phase")
        # Formation logic handled by coordinator
        time.sleep(10)  # Simulate formation time
    
    def execute_exploration_phase(self):
        """Execute ML-guided exploration phase."""
        rospy.loginfo("Executing exploration phase")
        # ML exploration logic
        time.sleep(20)  # Simulate exploration time
    
    def execute_task_phase(self):
        """Execute optimized task execution phase."""
        rospy.loginfo("Executing task phase")
        # Optimized task execution
        time.sleep(30)  # Simulate task time
    
    def execute_return_phase(self):
        """Execute return phase."""
        rospy.loginfo("Executing return phase")
        # Return to base
        time.sleep(15)  # Simulate return time

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        rospy.logerr("Usage: integrated_mission.py <vehicle_id>")
        sys.exit(1)
    
    vehicle_id = int(sys.argv[1])
    
    try:
        controller = IntegratedMissionController(vehicle_id)
        
        # Start mission in separate thread
        mission_thread = threading.Thread(target=controller.execute_integrated_mission)
        mission_thread.start()
        
        rospy.spin()
        
    except rospy.ROSInterruptException:
        pass
```

## Summary

These advanced examples demonstrate the full capabilities of the RS-ModCubes system:

1. **Search and Rescue Mission**: Complete autonomous mission with multiple phases
2. **Autonomous Inspection**: Detailed structural inspection with computer vision
3. **Multi-Robot Coordination**: Fleet operations with formation control
4. **Machine Learning Integration**: Adaptive behavior with neural networks
5. **Real-Time Optimization**: MPC with performance monitoring
6. **Integrated Mission**: Complete system integration

Each example showcases different aspects of advanced autonomous underwater vehicle operations, from basic control to sophisticated AI-driven decision making and multi-vehicle coordination.

### Key Features Demonstrated:

- **Modular Architecture**: Each component can be used independently
- **Real-Time Performance**: Optimized for real-time operation
- **Scalability**: Supports single and multi-vehicle operations
- **Adaptability**: Machine learning and adaptive algorithms
- **Robustness**: Error handling and performance monitoring
- **Integration**: Seamless integration of multiple subsystems

These examples provide a comprehensive foundation for developing advanced autonomous underwater vehicle applications using the RS-ModCubes platform.
            area=self.search_area,
            pattern=self.search_pattern,
            spacing=self.search_spacing
        )
        
        rospy.loginfo(f"Generated {len(search_waypoints)} search waypoints")
        
        # Execute search pattern
        for i, waypoint in enumerate(search_waypoints):
            if rospy.is_shutdown() or self.is_aborted():
                return False
            
            rospy.loginfo(f"Moving to search waypoint {i+1}/{len(search_waypoints)}")
            
            success = self.motion_client.goto_position(
                waypoint.x, waypoint.y, waypoint.z,
                timeout=30.0
            )
            
            if not success:
                rospy.logwarn(f"Failed to reach waypoint {i+1}")
                continue
            
            # Pause for detection
            rospy.sleep(2.0)
            
            # Update progress
            progress = (i + 1) / len(search_waypoints) * 100
            self.update_progress(progress)
        
        rospy.loginfo("Search phase completed")
        return True
    
    def execute_investigation_phase(self):
        """Investigate detected targets more closely."""
        rospy.loginfo(f"Starting investigation of {len(self.targets_found)} targets")
        
        for i, target in enumerate(self.targets_found):
            if rospy.is_shutdown() or self.is_aborted():
                return False
            
            rospy.loginfo(f"Investigating target {target['id']}")
            
            # Move closer to target for detailed inspection
            investigation_pos = Point(
                x=target['position'].x,
                y=target['position'].y,
                z=target['position'].z + 1.0  # 1m above target
            )
            
            success = self.motion_client.goto_position(
                investigation_pos.x, investigation_pos.y, investigation_pos.z,
                timeout=20.0
            )
            
            if success:
                # Perform detailed inspection
                self.perform_detailed_inspection(target)
            else:
                rospy.logwarn(f"Failed to reach investigation position for target {target['id']}")
            
            # Update progress
            progress = (i + 1) / len(self.targets_found) * 100
            self.update_progress(progress)
        
        rospy.loginfo("Investigation phase completed")
        return True
    
    def perform_detailed_inspection(self, target):
        """Perform detailed inspection of a target."""
        rospy.loginfo(f"Performing detailed inspection of target {target['id']}")
        
        # Circle around target for multiple viewing angles
        circle_radius = 2.0
        num_positions = 8
        
        for i in range(num_positions):
            angle = 2 * np.pi * i / num_positions
            
            inspect_pos = Point(
                x=target['position'].x + circle_radius * np.cos(angle),
                y=target['position'].y + circle_radius * np.sin(angle),
                z=target['position'].z + 1.0
            )
            
            self.motion_client.goto_position(
                inspect_pos.x, inspect_pos.y, inspect_pos.z,
                timeout=15.0
            )
            
            # Pause for data collection
            rospy.sleep(1.0)
        
        # Update target information
        target['inspected'] = True
        target['inspection_time'] = rospy.Time.now()
    
    def execute_recovery_phase(self):
        """Execute recovery operations for confirmed targets."""
        rospy.loginfo("Starting recovery phase")
        
        # Sort targets by priority (e.g., confidence, detection time)
        sorted_targets = sorted(self.targets_found, 
                              key=lambda t: t['confidence'], reverse=True)
        
        for i, target in enumerate(sorted_targets):
            if rospy.is_shutdown() or self.is_aborted():
                return False
            
            rospy.loginfo(f"Recovering target {target['id']}")
            
            # Move to recovery position
            recovery_pos = Point(
                x=target['position'].x,
                y=target['position'].y,
                z=target['position'].z + 0.5  # 0.5m above target
            )
            
            success = self.motion_client.goto_position(
                recovery_pos.x, recovery_pos.y, recovery_pos.z,
                timeout=20.0
            )
            
            if success:
                # Simulate recovery operation
                self.perform_recovery_operation(target)
            else:
                rospy.logwarn(f"Failed to reach recovery position for target {target['id']}")
            
            # Update progress
            progress = (i + 1) / len(sorted_targets) * 100
            self.update_progress(progress)
        
        rospy.loginfo("Recovery phase completed")
        return True
    
    def perform_recovery_operation(self, target):
        """Perform recovery operation for a target."""
        rospy.loginfo(f"Performing recovery operation for target {target['id']}")
        
        # Simulate manipulator operations
        # In practice, this would involve:
        # - Precise positioning
        # - Manipulator control
        # - Grasping operations
        # - Secure storage
        
        rospy.sleep(5.0)  # Simulate recovery time
        
        target['recovered'] = True
        target['recovery_time'] = rospy.Time.now()
        
        rospy.loginfo(f"Target {target['id']} successfully recovered")

if __name__ == '__main__':
    try:
        rospy.init_node('search_rescue_mission')
        
        mission = SearchRescueMission()
        success = mission.execute()
        
        if success:
            rospy.loginfo("Search and Rescue mission completed successfully")
        else:
            rospy.logwarn("Search and Rescue mission failed")
            
    except rospy.ROSInterruptException:
        pass
```

### Example 6: Inspection Mission

```python
#!/usr/bin/env python
"""
Infrastructure Inspection Mission

Performs detailed inspection of underwater infrastructure
using multiple sensors and systematic coverage patterns.
"""

import rospy
import numpy as np
from geometry_msgs.msg import Point, Pose, Quaternion
from sensor_msgs.msg import Image, PointCloud2
from modcube_msgs.msg import InspectionData
from modcube_mission import BaseMission
from modcube_common.motion import MotionClient
from modcube_common.vision import InspectionAnalyzer

class InspectionMission(BaseMission):
    def __init__(self, structure_model):
        super().__init__()
        
        self.motion_client = MotionClient()
        self.inspection_analyzer = InspectionAnalyzer()
        
        # Structure to inspect
        self.structure_model = structure_model
        self.inspection_points = []
        self.inspection_data = []
        
        # Inspection parameters
        self.inspection_distance = 2.0  # meters from structure
        self.overlap_percentage = 30    # percent overlap between views
        self.inspection_speed = 0.2     # m/s
        
        # Data collection
        self.image_sub = rospy.Subscriber('/modcube/camera/image_raw',
                                        Image, self.image_callback)
        self.cloud_sub = rospy.Subscriber('/modcube/pointcloud',
                                        PointCloud2, self.cloud_callback)
        
        # Data storage
        self.inspection_pub = rospy.Publisher('/modcube/inspection_data',
                                            InspectionData, queue_size=10)
        
        rospy.loginfo("Inspection mission initialized")
    
    def image_callback(self, msg):
        """Process camera images for inspection."""
        if self.get_state() == 'inspecting':
            # Analyze image for defects, corrosion, etc.
            analysis_result = self.inspection_analyzer.analyze_image(msg)
            
            if analysis_result:
                self.store_inspection_data('visual', analysis_result)
    
    def cloud_callback(self, msg):
        """Process point cloud data for 3D inspection."""
        if self.get_state() == 'inspecting':
            # Analyze point cloud for structural deformation
            analysis_result = self.inspection_analyzer.analyze_pointcloud(msg)
            
            if analysis_result:
                self.store_inspection_data('geometric', analysis_result)
    
    def store_inspection_data(self, data_type, analysis_result):
        """Store inspection data with metadata."""
        inspection_data = {
            'timestamp': rospy.Time.now(),
            'position': self.motion_client.get_current_pose(),
            'data_type': data_type,
            'analysis': analysis_result
        }
        
        self.inspection_data.append(inspection_data)
        
        # Publish for real-time monitoring
        msg = InspectionData()
        msg.header.stamp = inspection_data['timestamp']
        msg.data_type = data_type
        msg.position = inspection_data['position']
        # ... fill additional fields
        
        self.inspection_pub.publish(msg)
    
    def execute(self):
        """Execute the inspection mission."""
        rospy.loginfo("Starting infrastructure inspection mission")
        
        try:
            # Phase 1: Generate inspection plan
            self.update_state('planning')
            planning_success = self.generate_inspection_plan()
            
            if not planning_success:
                self.update_state('failed')
                return False
            
            # Phase 2: Execute inspection
            self.update_state('inspecting')
            inspection_success = self.execute_inspection()
            
            if not inspection_success:
                self.update_state('failed')
                return False
            
            # Phase 3: Data analysis and reporting
            self.update_state('analyzing')
            analysis_success = self.analyze_inspection_data()
            
            if not analysis_success:
                self.update_state('failed')
                return False
            
            # Mission completed
            self.update_state('completed')
            rospy.loginfo("Inspection mission completed successfully")
            return True
            
        except Exception as e:
            rospy.logerr(f"Inspection mission failed: {e}")
            self.update_state('failed')
            return False
    
    def generate_inspection_plan(self):
        """Generate systematic inspection plan for the structure."""
        rospy.loginfo("Generating inspection plan")
        
        # For a cylindrical structure (e.g., pipeline, pillar)
        if self.structure_model['type'] == 'cylinder':
            self.inspection_points = self.generate_cylinder_inspection_points()
        
        # For a planar structure (e.g., wall, panel)
        elif self.structure_model['type'] == 'plane':
            self.inspection_points = self.generate_plane_inspection_points()
        
        # For a complex structure
        elif self.structure_model['type'] == 'complex':
            self.inspection_points = self.generate_complex_inspection_points()
        
        else:
            rospy.logerr(f"Unknown structure type: {self.structure_model['type']}")
            return False
        
        rospy.loginfo(f"Generated {len(self.inspection_points)} inspection points")
        return True
    
    def generate_cylinder_inspection_points(self):
        """Generate inspection points for cylindrical structure."""
        points = []
        
        center = self.structure_model['center']
        radius = self.structure_model['radius']
        height = self.structure_model['height']
        
        # Calculate inspection parameters
        inspection_radius = radius + self.inspection_distance
        
        # Vertical spacing based on camera field of view
        vertical_spacing = 1.0  # meters
        num_levels = int(height / vertical_spacing) + 1
        
        # Angular spacing for complete coverage
        angular_spacing = np.pi / 6  # 30 degrees
        num_angles = int(2 * np.pi / angular_spacing)
        
        for level in range(num_levels):
            z = center[2] - height/2 + level * vertical_spacing
            
            for angle_idx in range(num_angles):
                angle = angle_idx * angular_spacing
                
                x = center[0] + inspection_radius * np.cos(angle)
                y = center[1] + inspection_radius * np.sin(angle)
                
                # Calculate orientation to face the structure
                yaw = angle + np.pi  # Face inward
                
                point = {
                    'position': Point(x, y, z),
                    'orientation': self.yaw_to_quaternion(yaw),
                    'level': level,
                    'angle': angle
                }
                
                points.append(point)
        
        return points
    
    def generate_plane_inspection_points(self):
        """Generate inspection points for planar structure."""
        points = []
        
        # Structure parameters
        corner1 = self.structure_model['corner1']
        corner2 = self.structure_model['corner2']
        normal = self.structure_model['normal']
        
        # Calculate inspection grid
        width = np.linalg.norm(np.array(corner2) - np.array(corner1))
        height = self.structure_model['height']
        
        grid_spacing = 1.5  # meters
        num_x = int(width / grid_spacing) + 1
        num_z = int(height / grid_spacing) + 1
        
        for i in range(num_x):
            for j in range(num_z):
                # Calculate position on the structure surface
                u = i / (num_x - 1) if num_x > 1 else 0
                v = j / (num_z - 1) if num_z > 1 else 0
                
                surface_point = (
                    np.array(corner1) * (1 - u) + np.array(corner2) * u +
                    np.array([0, 0, v * height])
                )
                
                # Offset by inspection distance along normal
                inspection_point = surface_point + np.array(normal) * self.inspection_distance
                
                # Calculate orientation to face the structure
                orientation = self.normal_to_quaternion(normal)
                
                point = {
                    'position': Point(inspection_point[0], inspection_point[1], inspection_point[2]),
                    'orientation': orientation,
                    'grid_x': i,
                    'grid_z': j
                }
                
                points.append(point)
        
        return points
    
    def execute_inspection(self):
        """Execute the inspection by visiting all planned points."""
        rospy.loginfo("Starting inspection execution")
        
        for i, point in enumerate(self.inspection_points):
            if rospy.is_shutdown() or self.is_aborted():
                return False
            
            rospy.loginfo(f"Moving to inspection point {i+1}/{len(self.inspection_points)}")
            
            # Create pose from point
            pose = Pose()
            pose.position = point['position']
            pose.orientation = point['orientation']
            
            # Move to inspection point
            success = self.motion_client.goto_pose(pose, timeout=30.0)
            
            if not success:
                rospy.logwarn(f"Failed to reach inspection point {i+1}")
                continue
            
            # Perform inspection at this point
            self.perform_point_inspection(point)
            
            # Update progress
            progress = (i + 1) / len(self.inspection_points) * 100
            self.update_progress(progress)
        
        rospy.loginfo("Inspection execution completed")
        return True
    
    def perform_point_inspection(self, point):
        """Perform detailed inspection at a specific point."""
        # Stabilize at position
        rospy.sleep(1.0)
        
        # Collect data for specified duration
        inspection_duration = 3.0  # seconds
        start_time = rospy.Time.now()
        
        while (rospy.Time.now() - start_time).to_sec() < inspection_duration:
            rospy.sleep(0.1)
        
        rospy.loginfo(f"Completed inspection at point {point['position']}")
    
    def analyze_inspection_data(self):
        """Analyze collected inspection data and generate report."""
        rospy.loginfo("Analyzing inspection data")
        
        # Categorize findings
        defects = []
        anomalies = []
        measurements = []
        
        for data in self.inspection_data:
            analysis = data['analysis']
            
            if 'defects' in analysis:
                defects.extend(analysis['defects'])
            
            if 'anomalies' in analysis:
                anomalies.extend(analysis['anomalies'])
            
            if 'measurements' in analysis:
                measurements.extend(analysis['measurements'])
        
        # Generate summary report
        report = {
            'total_points_inspected': len(self.inspection_points),
            'total_data_collected': len(self.inspection_data),
            'defects_found': len(defects),
            'anomalies_detected': len(anomalies),
            'measurements_taken': len(measurements),
            'inspection_coverage': self.calculate_coverage(),
            'recommendations': self.generate_recommendations(defects, anomalies)
        }
        
        # Save report
        self.save_inspection_report(report)
        
        rospy.loginfo(f"Analysis completed. Found {len(defects)} defects and {len(anomalies)} anomalies")
        return True
    
    def calculate_coverage(self):
        """Calculate inspection coverage percentage."""
        # Simplified coverage calculation
        # In practice, this would be more sophisticated
        planned_points = len(self.inspection_points)
        completed_points = len([d for d in self.inspection_data if d['data_type'] == 'visual'])
        
        return (completed_points / planned_points) * 100 if planned_points > 0 else 0
    
    def generate_recommendations(self, defects, anomalies):
        """Generate maintenance recommendations based on findings."""
        recommendations = []
        
        if len(defects) > 5:
            recommendations.append("High number of defects detected. Immediate maintenance recommended.")
        
        if len(anomalies) > 0:
            recommendations.append("Structural anomalies detected. Further investigation required.")
        
        if len(defects) == 0 and len(anomalies) == 0:
            recommendations.append("Structure appears to be in good condition. Continue regular monitoring.")
        
        return recommendations
    
    def save_inspection_report(self, report):
        """Save inspection report to file."""
        import json
        import os
        
        timestamp = rospy.Time.now().to_sec()
        filename = f"inspection_report_{timestamp}.json"
        filepath = os.path.join("/tmp", filename)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        rospy.loginfo(f"Inspection report saved to {filepath}")
    
    def yaw_to_quaternion(self, yaw):
        """Convert yaw angle to quaternion."""
        import math
        return Quaternion(
            x=0.0,
            y=0.0,
            z=math.sin(yaw / 2.0),
            w=math.cos(yaw / 2.0)
        )
    
    def normal_to_quaternion(self, normal):
        """Convert normal vector to quaternion orientation."""
        import math
        
        # Calculate yaw and pitch from normal vector
        yaw = math.atan2(normal[1], normal[0])
        pitch = math.asin(-normal[2])
        
        # Convert to quaternion
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        
        return Quaternion(
            x=sp * cy,
            y=0.0,
            z=sy * cp,
            w=cp * cy
        )

if __name__ == '__main__':
    try:
        rospy.init_node('inspection_mission')
        
        # Define structure to inspect
        structure_model = {
            'type': 'cylinder',
            'center': [10, 10, -5],
            'radius': 2.0,
            'height': 10.0
        }
        
        mission = InspectionMission(structure_model)
        success = mission.execute()
        
        if success:
            rospy.loginfo("Inspection mission completed successfully")
        else:
            rospy.logwarn("Inspection mission failed")
            
    except rospy.ROSInterruptException:
        pass
```

## Sensor Integration

### Example 7: Multi-Sensor Fusion

```python
#!/usr/bin/env python
"""
Multi-Sensor Fusion Example

Demonstrates integration and fusion of multiple sensor modalities
for robust state estimation and environmental perception.
"""

import rospy
import numpy as np
from sensor_msgs.msg import Imu, Image, PointCloud2
from geometry_msgs.msg import PoseWithCovariance, TwistWithCovariance
from modcube_msgs.msg import DVLData, FluidDepth, NavState
from modcube_common.filters import ExtendedKalmanFilter
from modcube_common.vision import VisualOdometry

class MultiSensorFusion:
    def __init__(self):
        rospy.init_node('multi_sensor_fusion')
        
        # Initialize filters and processors
        self.ekf = ExtendedKalmanFilter(state_dim=15)  # [pos, vel, acc, orient, ang_vel]
        self.visual_odometry = VisualOdometry()
        
        # Sensor data buffers
        self.imu_data = None
        self.dvl_data = None
        self.depth_data = None
        self.visual_data = None
        
        # Sensor subscribers
        self.imu_sub = rospy.Subscriber('/modcube/imu/data', Imu, self.imu_callback)
        self.dvl_sub = rospy.Subscriber('/modcube/dvl/data', DVLData, self.dvl_callback)
        self.depth_sub = rospy.Subscriber('/modcube/depth', FluidDepth, self.depth_callback)
        self.image_sub = rospy.Subscriber('/modcube/camera/image_raw', Image, self.image_callback)
        self.cloud_sub = rospy.Subscriber('/modcube/pointcloud', PointCloud2, self.cloud_callback)
        
        # State publisher
        self.nav_pub = rospy.Publisher('/modcube/nav_state', NavState, queue_size=10)
        
        # Fusion timer
        self.fusion_timer = rospy.Timer(rospy.Duration(0.02), self.fusion_callback)  # 50 Hz
        
        # Sensor health monitoring
        self.sensor_health = {
            'imu': {'last_update': None, 'status': 'unknown'},
            'dvl': {'last_update': None, 'status': 'unknown'},
            'depth': {'last_update': None, 'status': 'unknown'},
            'visual': {'last_update': None, 'status': 'unknown'}
        }
        
        rospy.loginfo("Multi-sensor fusion initialized")
    
    def imu_callback(self, msg):
        """Process IMU data."""
        self.imu_data = {
            'timestamp': msg.header.stamp,
            'orientation': msg.orientation,
            'angular_velocity': msg.angular_velocity,
            'linear_acceleration': msg.linear_acceleration,
            'orientation_covariance': np.array(msg.orientation_covariance).reshape(3, 3),
            'angular_velocity_covariance': np.array(msg.angular_velocity_covariance).reshape(3, 3),
            'linear_acceleration_covariance': np.array(msg.linear_acceleration_covariance).reshape(3, 3)
        }
        
        self.update_sensor_health('imu', msg.header.stamp)
    
    def dvl_callback(self, msg):
        """Process DVL data."""
        self.dvl_data = {
            'timestamp': msg.header.stamp,
            'velocity': msg.velocity,
            'altitude': msg.altitude,
            'beam_ranges': msg.beam_ranges,
            'beam_velocities': msg.beam_velocities,
            'status': msg.status
        }
        
        self.update_sensor_health('dvl', msg.header.stamp)
    
    def depth_callback(self, msg):
        """Process depth sensor data."""
        self.depth_data = {
            'timestamp': msg.header.stamp,
            'depth': msg.depth,
            'pressure': msg.pressure,
            'temperature': msg.temperature
        }
        
        self.update_sensor_health('depth', msg.header.stamp)
    
    def image_callback(self, msg):
        """Process camera images for visual odometry."""
        # Process visual odometry
        vo_result = self.visual_odometry.process_image(msg)
        
        if vo_result:
            self.visual_data = {
                'timestamp': msg.header.stamp,
                'pose_delta': vo_result['pose_delta'],
                'confidence': vo_result['confidence'],
                'features': vo_result['features']
            }
            
            self.update_sensor_health('visual', msg.header.stamp)
    
    def cloud_callback(self, msg):
        """Process point cloud data for obstacle detection and mapping."""
        # Process point cloud for environmental awareness
        # This could include obstacle detection, SLAM, etc.
        pass
    
    def update_sensor_health(self, sensor_name, timestamp):
        """Update sensor health status."""
        self.sensor_health[sensor_name]['last_update'] = timestamp
        
        # Check if sensor is healthy (received data recently)
        current_time = rospy.Time.now()
        time_diff = (current_time - timestamp).to_sec()
        
        if time_diff < 1.0:  # Less than 1 second old
            self.sensor_health[sensor_name]['status'] = 'healthy'
        elif time_diff < 5.0:  # Less than 5 seconds old
            self.sensor_health[sensor_name]['status'] = 'degraded'
        else:
            self.sensor_health[sensor_name]['status'] = 'failed'
    
    def fusion_callback(self, event):
        """Main sensor fusion loop."""
        current_time = rospy.Time.now()
        
        # Prediction step
        self.ekf.predict(dt=0.02)
        
        # Update with available sensor data
        if self.imu_data and self.is_data_fresh(self.imu_data, current_time, 0.1):
            self.update_with_imu()
        
        if self.dvl_data and self.is_data_fresh(self.dvl_data, current_time, 0.5):
            self.update_with_dvl()
        
        if self.depth_data and self.is_data_fresh(self.depth_data, current_time, 1.0):
            self.update_with_depth()
        
        if self.visual_data and self.is_data_fresh(self.visual_data, current_time, 0.2):
            self.update_with_visual()
        
        # Publish fused navigation state
        self.publish_nav_state()
    
    def is_data_fresh(self, data, current_time, max_age):
        """Check if sensor data is fresh enough to use."""
        age = (current_time - data['timestamp']).to_sec()
        return age < max_age
    
    def update_with_imu(self):
        """Update EKF with IMU measurements."""
        # Extract measurements
        orientation = self.imu_data['orientation']
        angular_velocity = self.imu_data['angular_velocity']
        linear_acceleration = self.imu_data['linear_acceleration']
        
        # Convert quaternion to euler angles
        roll, pitch, yaw = self.quaternion_to_euler(orientation)
        
        # Measurement vector: [roll, pitch, yaw, wx, wy, wz, ax, ay, az]
        z_imu = np.array([
            roll, pitch, yaw,
            angular_velocity.x, angular_velocity.y, angular_velocity.z,
            linear_acceleration.x, linear_acceleration.y, linear_acceleration.z
        ])
        
        # Measurement covariance
        R_imu = np.block([
            [self.imu_data['orientation_covariance'], np.zeros((3, 6))],
            [np.zeros((3, 3)), self.imu_data['angular_velocity_covariance'], np.zeros((3, 3))],
            [np.zeros((3, 6)), self.imu_data['linear_acceleration_covariance']]
        ])
        
        # Update EKF
        self.ekf.update(z_imu, R_imu, measurement_model='imu')
    
    def update_with_dvl(self):
        """Update EKF with DVL measurements."""
        if self.dvl_data['status'] != 0:  # Check DVL status
            return
        
        # Extract velocity measurements
        velocity = self.dvl_data['velocity']
        
        # Measurement vector: [vx, vy, vz]
        z_dvl = np.array([velocity.x, velocity.y, velocity.z])
        
        # Measurement covariance (simplified)
        R_dvl = np.diag([0.01, 0.01, 0.01])  # 1 cm/s standard deviation
        
        # Update EKF
        self.ekf.update(z_dvl, R_dvl, measurement_model='dvl')
    
    def update_with_depth(self):
        """Update EKF with depth measurements."""
        depth = self.depth_data['depth']
        
        # Measurement vector: [z]
        z_depth = np.array([depth])
        
        # Measurement covariance
        R_depth = np.array([[0.01]])  # 1 cm standard deviation
        
        # Update EKF
        self.ekf.update(z_depth, R_depth, measurement_model='depth')
    
    def update_with_visual(self):
        """Update EKF with visual odometry measurements."""
        if self.visual_data['confidence'] < 0.5:  # Low confidence
            return
        
        pose_delta = self.visual_data['pose_delta']
        
        # Extract position and orientation changes
        dx = pose_delta['position']['x']
        dy = pose_delta['position']['y']
        dz = pose_delta['position']['z']
        dyaw = pose_delta['orientation']['yaw']
        
        # Measurement vector: [dx, dy, dz, dyaw]
        z_visual = np.array([dx, dy, dz, dyaw])
        
        # Measurement covariance (based on confidence)
        confidence = self.visual_data['confidence']
        base_variance = 0.1 / confidence  # Higher variance for lower confidence
        R_visual = np.diag([base_variance, base_variance, base_variance, base_variance * 0.1])
        
        # Update EKF
        self.ekf.update(z_visual, R_visual, measurement_model='visual')
    
    def publish_nav_state(self):
        """Publish the fused navigation state."""
        state = self.ekf.get_state()
        covariance = self.ekf.get_covariance()
        
        nav_msg = NavState()
        nav_msg.header.stamp = rospy.Time.now()
        nav_msg.header.frame_id = 'odom'
        
        # Position
        nav_msg.pose.pose.position.x = state[0]
        nav_msg.pose.pose.position.y = state[1]
        nav_msg.pose.pose.position.z = state[2]
        
        # Orientation (convert from euler to quaternion)
        nav_msg.pose.pose.orientation = self.euler_to_quaternion(state[9], state[10], state[11])
        
        # Velocity
        nav_msg.twist.twist.linear.x = state[3]
        nav_msg.twist.twist.linear.y = state[4]
        nav_msg.twist.twist.linear.z = state[5]
        
        # Angular velocity
        nav_msg.twist.twist.angular.x = state[12]
        nav_msg.twist.twist.angular.y = state[13]
        nav_msg.twist.twist.angular.z = state[14]
        
        # Covariances (simplified)
        nav_msg.pose.covariance = covariance[:6, :6].flatten().tolist() + [0] * 30
        nav_msg.twist.covariance = covariance[3:9, 3:9].flatten().tolist() + [0] * 30
        
        self.nav_pub.publish(nav_msg)
    
    def quaternion_to_euler(self, q):
        """Convert quaternion to euler angles."""
        import math
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (q.w * q.y - q.z * q.x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    def euler_to_quaternion(self, roll, pitch, yaw):
        """Convert euler angles to quaternion."""
        import math
        from geometry_msgs.msg import Quaternion
        
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        return Quaternion(
            x=sr * cp * cy - cr * sp * sy,
            y=cr * sp * cy + sr * cp * sy,
            z=cr * cp * sy - sr * sp * cy,
            w=cr * cp * cy + sr * sp * sy
        )

if __name__ == '__main__':
    try:
        fusion = MultiSensorFusion()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
```

## Multi-Vehicle Coordination

### Example 8: Formation Control

```python
#!/usr/bin/env python
"""
Formation Control Example

Demonstrates coordinated control of multiple ModCube vehicles
in various formation patterns.
"""

import rospy
import numpy as np
from geometry_msgs.msg import Point, Pose, Twist
from modcube_msgs.msg import FormationCommand, VehicleState
from modcube_common.formation import FormationController

class FormationControl:
    def __init__(self, vehicle_id, num_vehicles):
        rospy.init_node(f'formation_control_{vehicle_id}')
        
        self.vehicle_id = vehicle_id
        self.num_vehicles = num_vehicles
        
        # Formation controller
        self.formation_controller = FormationController(vehicle_id, num_vehicles)
        
        # Vehicle states
        self.vehicle_states = {}
        self.formation_command = None
        
        # Publishers and subscribers
        self.cmd_pub = rospy.Publisher(f'/modcube_{vehicle_id}/cmd_vel', Twist, queue_size=10)
        self.state_pub = rospy.Publisher('/formation/vehicle_states', VehicleState, queue_size=10)
        
        # Subscribe to other vehicles' states
        for i in range(num_vehicles):
            if i != vehicle_id:
                rospy.Subscriber(f'/modcube_{i}/nav_state', NavState, 
                               lambda msg, vid=i: self.vehicle_state_callback(msg, vid))
        
        # Formation command subscriber
        rospy.Subscriber('/formation/command', FormationCommand, self.formation_command_callback)
        
        # Control timer
        self.control_timer = rospy.Timer(rospy.Duration(0.1), self.control_callback)
        
        rospy.loginfo(f"Formation control initialized for vehicle {vehicle_id}")
    
    def vehicle_state_callback(self, msg, vehicle_id):
        """Update state of other vehicles."""
        self.vehicle_states[vehicle_id] = {
            'position': msg.pose.pose.position,
            'velocity': msg.twist.twist.linear,
            'timestamp': msg.header.stamp
        }
    
    def formation_command_callback(self, msg):
        """Receive formation command."""
        self.formation_command = msg
        rospy.loginfo(f"Received formation command: {msg.formation_type}")
    
    def control_callback(self, event):
        """Main formation control loop."""
        if not self.formation_command:
            return
        
        # Get current vehicle state
        current_state = self.get_current_state()
        if not current_state:
            return
        
        # Calculate formation control
        control_cmd = self.formation_controller.compute_control(
            current_state=current_state,
            neighbor_states=self.vehicle_states,
            formation_command=self.formation_command
        )
        
        # Publish control command
        if control_cmd:
            self.cmd_pub.publish(control_cmd)
    
    def get_current_state(self):
        """Get current vehicle state."""
        # In practice, this would come from the navigation system
        # For this example, we'll simulate it
        return {
            'position': Point(0, 0, -2),
            'velocity': Point(0, 0, 0),
            'timestamp': rospy.Time.now()
        }

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        rospy.logerr("Usage: formation_control.py <vehicle_id> <num_vehicles>")
        sys.exit(1)
    
    try:
        vehicle_id = int(sys.argv[1])
        num_vehicles = int(sys.argv[2])
        
        formation = FormationControl(vehicle_id, num_vehicles)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
```

## Advanced Applications

### Example 9: Adaptive Control

```python
#!/usr/bin/env python
"""
Adaptive Control Example

Demonstrates adaptive control algorithms that adjust
to changing vehicle dynamics and environmental conditions.
"""

import rospy
import numpy as np
from modcube_msgs.msg import ControllerCommand, NavState
from modcube_common.adaptive import AdaptiveController

class AdaptiveControlSystem:
    def __init__(self):
        rospy.init_node('adaptive_control')
        
        # Adaptive controller
        self.adaptive_controller = AdaptiveController()
        
        # System identification
        self.system_id = SystemIdentification()
        
        # Data buffers
        self.control_history = []
        self.state_history = []
        
        # Publishers and subscribers
        self.cmd_pub = rospy.Publisher('/modcube/controller_command', 
                                     ControllerCommand, queue_size=10)
        self.nav_sub = rospy.Subscriber('/modcube/nav_state', NavState, 
                                      self.nav_callback)
        
        # Adaptation timer
        self.adapt_timer = rospy.Timer(rospy.Duration(1.0), self.adaptation_callback)
        
        rospy.loginfo("Adaptive control system initialized")
    
    def nav_callback(self, msg):
        """Process navigation state updates."""
        # Store state history for system identification
        state = {
            'timestamp': msg.header.stamp,
            'position': msg.pose.pose.position,
            'velocity': msg.twist.twist.linear,
            'orientation': msg.pose.pose.orientation
        }
        
        self.state_history.append(state)
        
        # Limit history size
        if len(self.state_history) > 1000:
            self.state_history.pop(0)
    
    def adaptation_callback(self, event):
        """Perform parameter adaptation."""
        if len(self.state_history) < 10 or len(self.control_history) < 10:
            return
        
        # Identify current system parameters
        identified_params = self.system_id.identify(
            states=self.state_history[-100:],
            controls=self.control_history[-100:]
        )
        
        # Update controller parameters
        if identified_params:
            self.adaptive_controller.update_parameters(identified_params)
            rospy.loginfo("Controller parameters adapted")

class SystemIdentification:
    """System identification for parameter estimation."""
    
    def __init__(self):
        self.model_order = 2
        self.forgetting_factor = 0.95
    
    def identify(self, states, controls):
        """Identify system parameters from data."""
        # Simplified system identification
        # In practice, use more sophisticated methods
        
        if len(states) < self.model_order + 1:
            return None
        
        # Extract position and velocity data
        positions = np.array([[s['position'].x, s['position'].y, s['position'].z] 
                            for s in states])
        velocities = np.array([[s['velocity'].x, s['velocity'].y, s['velocity'].z] 
                             for s in states])
        
        # Simple parameter estimation
        try:
            # Estimate damping and mass parameters
            damping = self.estimate_damping(velocities)
            mass = self.estimate_mass(velocities, controls)
            
            return {
                'damping': damping,
                'mass': mass,
                'confidence': 0.8
            }
        except:
            return None
    
    def estimate_damping(self, velocities):
        """Estimate damping coefficients."""
        # Simplified damping estimation
        return np.array([1.0, 1.0, 1.0])
    
    def estimate_mass(self, velocities, controls):
        """Estimate vehicle mass."""
        # Simplified mass estimation
        return 50.0  # kg

if __name__ == '__main__':
    try:
        adaptive_system = AdaptiveControlSystem()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
```

## Custom Development

### Example 10: Custom Mission Plugin

```python
#!/usr/bin/env python
"""
Custom Mission Plugin Example

Demonstrates how to create custom mission types
by extending the base mission framework.
"""

import rospy
from modcube_mission import BaseMission
from modcube_common.motion import MotionClient
from geometry_msgs.msg import Point

class CustomSurveyMission(BaseMission):
    """Custom survey mission with specific requirements."""
    
    def __init__(self, survey_area, resolution=1.0):
        super().__init__()
        
        self.survey_area = survey_area
        self.resolution = resolution
        self.motion_client = MotionClient()
        
        # Mission-specific parameters
        self.altitude = -2.0
        self.speed = 0.5
        self.overlap = 0.3
        
        rospy.loginfo("Custom survey mission initialized")
    
    def execute(self):
        """Execute the custom survey mission."""
        rospy.loginfo("Starting custom survey mission")
        
        try:
            # Generate survey pattern
            survey_points = self.generate_survey_pattern()
            
            # Execute survey
            for i, point in enumerate(survey_points):
                if self.is_aborted():
                    return False
                
                success = self.motion_client.goto_position(
                    point.x, point.y, point.z
                )
                
                if not success:
                    rospy.logwarn(f"Failed to reach survey point {i}")
                    continue
                
                # Perform survey at this point
                self.perform_survey_at_point(point)
                
                # Update progress
                progress = (i + 1) / len(survey_points) * 100
                self.update_progress(progress)
            
            self.update_state('completed')
            return True
            
        except Exception as e:
            rospy.logerr(f"Survey mission failed: {e}")
            self.update_state('failed')
            return False
    
    def generate_survey_pattern(self):
        """Generate survey waypoints."""
        points = []
        
        x_min, x_max = self.survey_area['x_range']
        y_min, y_max = self.survey_area['y_range']
        
        x_steps = int((x_max - x_min) / self.resolution) + 1
        y_steps = int((y_max - y_min) / self.resolution) + 1
        
        for i in range(x_steps):
            x = x_min + i * self.resolution
            
            if i % 2 == 0:  # Even rows: bottom to top
                y_range = range(y_steps)
            else:  # Odd rows: top to bottom
                y_range = range(y_steps - 1, -1, -1)
            
            for j in y_range:
                y = y_min + j * self.resolution
                points.append(Point(x, y, self.altitude))
        
        return points
    
    def perform_survey_at_point(self, point):
        """Perform survey operations at a specific point."""
        # Custom survey operations
        rospy.sleep(2.0)  # Simulate data collection time

if __name__ == '__main__':
    try:
        rospy.init_node('custom_survey_mission')
        
        survey_area = {
            'x_range': (0, 20),
            'y_range': (0, 15)
        }
        
        mission = CustomSurveyMission(survey_area, resolution=2.0)
        success = mission.execute()
        
        if success:
            rospy.loginfo("Custom survey mission completed")
        else:
            rospy.logwarn("Custom survey mission failed")
            
    except rospy.ROSInterruptException:
        pass
```

## Running the Examples

### Prerequisites

1. **ROS Environment**: Ensure ROS is properly installed and sourced
2. **ModCube Workspace**: Build and source the ModCube workspace
3. **Simulation Environment**: Launch Gazebo simulation if testing in simulation

### Basic Setup

```bash
# Source ROS and workspace
source /opt/ros/melodic/setup.bash
source ~/modcube_ws/devel/setup.bash

# Launch simulation (optional)
roslaunch modcube_sim_worlds umd.launch

# Launch ModCube system
roslaunch modcube_config system.launch
```

### Running Individual Examples

```bash
# Example 1: Basic Position Control
rosrun modcube_examples basic_position_control.py

# Example 2: Velocity Control
rosrun modcube_examples velocity_control.py

# Example 3: Waypoint Navigation
rosrun modcube_examples waypoint_navigation.py

# Example 5: Search and Rescue Mission
rosrun modcube_examples search_rescue_mission.py

# Example 8: Formation Control (multiple terminals)
rosrun modcube_examples formation_control.py 0 3  # Vehicle 0 of 3
rosrun modcube_examples formation_control.py 1 3  # Vehicle 1 of 3
rosrun modcube_examples formation_control.py 2 3  # Vehicle 2 of 3
```

### Monitoring and Debugging

```bash
# Monitor system status
rostopic echo /modcube/nav_state
rostopic echo /modcube/controller_command

# Visualize in RViz
rosrun rviz rviz -d $(rospack find modcube_config)/rviz/modcube.rviz

# Check system diagnostics
rosrun rqt_robot_monitor rqt_robot_monitor
```

## Troubleshooting

### Common Issues

1. **No response to commands**
   - Check if the controller is running
   - Verify topic names and message types
   - Ensure proper coordinate frames

2. **Simulation not starting**
   - Check Gazebo installation
   - Verify world files and model paths
   - Check for conflicting processes

3. **Sensor data not available**
   - Verify sensor plugins are loaded
   - Check topic remapping
   - Ensure proper sensor configuration

### Debug Commands

```bash
# List active topics
rostopic list

# Check message types
rostopic type /modcube/nav_state

# Monitor message frequency
rostopic hz /modcube/imu/data

# View node graph
rosrun rqt_graph rqt_graph

# Check parameter server
rosparam list
rosparam get /modcube/controller/pid_gains
```

## Next Steps

1. **Modify Examples**: Adapt the examples to your specific requirements
2. **Create Custom Missions**: Use the mission framework to develop custom applications
3. **Integrate Hardware**: Connect real sensors and actuators
4. **Performance Tuning**: Optimize parameters for your specific use case
5. **Advanced Features**: Explore machine learning integration and advanced control algorithms

For more detailed information, refer to the [API Documentation](api.md) and [Tutorials](tutorials.md).