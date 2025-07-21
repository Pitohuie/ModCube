---
layout: default
title: API Documentation
nav_order: 4
has_children: false
permalink: /api/
---

# API Documentation

This document provides comprehensive API documentation for the RS-ModCubes system, including ROS topics, services, messages, and Python APIs. The system implements advanced underwater robotics capabilities with modular architecture for autonomous underwater vehicles (AUVs).

## Table of Contents

1. [ROS Topics](#ros-topics)
2. [ROS Services](#ros-services)
3. [Message Definitions](#message-definitions)
4. [Python APIs](#python-apis)
5. [C++ APIs](#cpp-apis)
6. [State Estimation APIs](#state-estimation-apis)
7. [Trajectory Generation APIs](#trajectory-generation-apis)
8. [Configuration Parameters](#configuration-parameters)
9. [Plugin APIs](#plugin-apis)
10. [Examples](#examples)

## ROS Topics

### Control Topics

#### `/modcube/controller_command`
**Type**: `modcube_msgs/ControllerCommand`  
**Description**: High-level control commands for the vehicle  
**Publisher**: Mission planner, teleop nodes  
**Subscriber**: PID controller  

```yaml
# Message structure
header:
  stamp: time
  frame_id: string
mode: int32          # Control mode (1=position, 2=velocity, 3=acceleration)
setpoint:
  position:          # Target position (mode 1)
    x: float64
    y: float64
    z: float64
  orientation:       # Target orientation (quaternion)
    x: float64
    y: float64
    z: float64
    w: float64
  linear:           # Target linear velocity (mode 2)
    x: float64
    y: float64
    z: float64
  angular:          # Target angular velocity (mode 2)
    x: float64
    y: float64
    z: float64
```

#### `/modcube/thrust_command`
**Type**: `modcube_msgs/ThrusterCommand`  
**Description**: Direct thruster control commands  
**Publisher**: Thruster manager, controllers  
**Subscriber**: Hardware drivers  

```yaml
# Message structure
header:
  stamp: time
  frame_id: string
thrusters: float64[14]  # Thrust values for 14 thrusters (-1.0 to 1.0)
```

### State Topics

#### `/modcube/nav_state`
**Type**: `modcube_msgs/NavState`  
**Description**: Current navigation state of the vehicle  
**Publisher**: State estimator  
**Subscriber**: Controllers, mission planner  

```yaml
# Message structure
header:
  stamp: time
  frame_id: string
pose:
  pose:
    position:
      x: float64
      y: float64
      z: float64
    orientation:
      x: float64
      y: float64
      z: float64
      w: float64
  covariance: float64[36]
twist:
  twist:
    linear:
      x: float64
      y: float64
      z: float64
    angular:
      x: float64
      y: float64
      z: float64
  covariance: float64[36]
```

### Sensor Topics

#### `/modcube/imu/data`
**Type**: `sensor_msgs/Imu`  
**Description**: IMU sensor data  
**Publisher**: IMU driver  
**Subscriber**: State estimator  

#### `/modcube/dvl/data`
**Type**: `modcube_msgs/DVLData`  
**Description**: DVL sensor data  
**Publisher**: DVL driver  
**Subscriber**: State estimator  

```yaml
# DVLData message structure
header:
  stamp: time
  frame_id: string
velocity:
  x: float64
  y: float64
  z: float64
altitude: float64
beam_ranges: float64[4]
beam_velocities: float64[4]
status: int32
```

#### `/modcube/depth`
**Type**: `modcube_msgs/FluidDepth`  
**Description**: Depth sensor data  
**Publisher**: Depth sensor driver  
**Subscriber**: State estimator  

### Mission Topics

#### `/modcube/mission_state`
**Type**: `modcube_msgs/MissionState`  
**Description**: Current mission status  
**Publisher**: Mission manager  
**Subscriber**: Mission monitor, GUI  

#### `/modcube/alarms`
**Type**: `modcube_msgs/Alarm`  
**Description**: System alarms and warnings  
**Publisher**: Various nodes  
**Subscriber**: Mission manager, safety monitor  

### Trajectory and Path Planning Topics

#### `/modcube/trajectory/path`
**Type**: `nav_msgs/Path`  
**Description**: Planned trajectory path for visualization  
**Publisher**: PID planner, trajectory generator  
**Subscriber**: RViz, mission monitor  

#### `/modcube/trajectory/target`
**Type**: `geometry_msgs/PoseStamped`  
**Description**: Current trajectory target pose  
**Publisher**: Trajectory generator  
**Subscriber**: PID planner, controllers  

#### `/modcube/trajectory/request`
**Type**: `modcube_msgs/GetTrajectoryRequest`  
**Description**: Request for trajectory generation  
**Publisher**: Mission planner  
**Subscriber**: Trajectory generator  

### Debug and Monitoring Topics

#### `/modcube/debug/pid_output`
**Type**: `modcube_msgs/PIDDebug`  
**Description**: PID controller debug information  
**Publisher**: PID planner  
**Subscriber**: Debug monitor, tuning tools  

#### `/modcube/debug/ekf_state`
**Type**: `modcube_msgs/EKFDebug`  
**Description**: Extended Kalman Filter internal state  
**Publisher**: State estimator  
**Subscriber**: Debug monitor, analysis tools  

#### `/modcube/debug/thruster_allocation`
**Type**: `modcube_msgs/ThrusterAllocation`  
**Description**: Thruster allocation matrix output  
**Publisher**: Thruster manager  
**Subscriber**: Debug monitor, performance analysis  

## ROS Services

### Control Services

#### `/modcube/set_pid_params`
**Type**: `modcube_msgs/SetPIDParams`  
**Description**: Set PID controller parameters  

```yaml
# Request
position_gains:
  p: float64[3]  # [x, y, z]
  i: float64[3]
  d: float64[3]
orientation_gains:
  p: float64[3]  # [roll, pitch, yaw]
  i: float64[3]
  d: float64[3]

# Response
success: bool
message: string
```

#### `/modcube/get_pid_params`
**Type**: `modcube_msgs/GetPIDParams`  
**Description**: Get current PID controller parameters  

#### `/modcube/set_thruster_config`
**Type**: `modcube_msgs/SetThrusterManagerConfig`  
**Description**: Configure thruster manager  

```yaml
# Request
thruster_ids: int32[]
max_thrust: float64[]
thruster_topic_prefix: string
thruster_topic_suffix: string
thruster_frame_base: string
max_thrust_pc: float64

# Response
success: bool
message: string
```

### Mission Services

#### `/modcube/mission_control`
**Type**: `modcube_msgs/MissionControl`  
**Description**: Control mission execution  

```yaml
# Request
command: string        # 'start', 'stop', 'pause', 'resume'
mission_type: string   # Mission type identifier
parameters: string[]   # Mission-specific parameters

# Response
success: bool
message: string
mission_id: string
```

### State Estimation Services

#### `/modcube/state_estimator/reset`
**Type**: `modcube_msgs/ResetStateEstimator`  
**Description**: Reset state estimator with optional initial state  

```yaml
# Request
initial_pose:
  position: {x: float64, y: float64, z: float64}
  orientation: {x: float64, y: float64, z: float64, w: float64}
initial_velocity:
  linear: {x: float64, y: float64, z: float64}
  angular: {x: float64, y: float64, z: float64}
reset_covariance: bool

# Response
success: bool
message: string
```

#### `/modcube/state_estimator/set_sensor_config`
**Type**: `modcube_msgs/SetSensorConfig`  
**Description**: Configure sensor parameters for state estimation  

### Trajectory Generation Services

#### `/modcube/trajectory/generate`
**Type**: `modcube_msgs/GenerateTrajectory`  
**Description**: Generate optimal trajectory between waypoints  

```yaml
# Request
waypoints: geometry_msgs/PoseStamped[]
start_velocity: geometry_msgs/Twist
end_velocity: geometry_msgs/Twist
aggressiveness: float64  # 0.0 to 1.0
max_velocity: float64
max_acceleration: float64

# Response
success: bool
trajectory: nav_msgs/Path
total_time: float64
total_distance: float64
average_speed: float64
```

#### `/modcube/trajectory/get_target`
**Type**: `modcube_msgs/GetTrajectoryTarget`  
**Description**: Get current trajectory target at specified time  

### Dynamic Parameter Services

#### `/modcube/tune_pid_planner`
**Type**: `modcube_msgs/TunePIDPlanner`  
**Description**: Dynamically tune PID planner parameters  

```yaml
# Request
axis: string  # 'x', 'y', 'z', 'roll', 'pitch', 'yaw'
kp: float64
ki: float64
kd: float64
tau: float64
limit: float64

# Response
success: bool
message: string
```

#### `/modcube/tune_dynamics`
**Type**: `modcube_msgs/TuneDynamics`  
**Description**: Tune vehicle dynamics parameters  

```yaml
# Request
mass: float64
volume: float64
water_density: float64
center_of_gravity: geometry_msgs/Vector3
center_of_buoyancy: geometry_msgs/Vector3
moments_of_inertia: float64[6]  # [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
linear_damping: float64[6]
quadratic_damping: float64[6]
added_mass: float64[36]  # 6x6 matrix

# Response
success: bool
message: string
```

### Diagnostic Services

#### `/modcube/get_system_status`
**Type**: `modcube_msgs/GetSystemStatus`  
**Description**: Get comprehensive system status  

#### `/modcube/test_thrusters`
**Type**: `modcube_msgs/TestThrusters`  
**Description**: Test individual thrusters  

#### `/modcube/calibrate_sensors`
**Type**: `modcube_msgs/CalibrateSensors`  
**Description**: Perform sensor calibration procedures  

## Message Definitions

### Core Messages

#### `modcube_msgs/ControllerCommand`
```yaml
std_msgs/Header header
int32 mode                    # 1=position, 2=velocity, 3=acceleration
geometry_msgs/Pose setpoint_pose
geometry_msgs/Twist setpoint_twist
geometry_msgs/Accel setpoint_accel
bool enable_position_hold
bool enable_depth_hold
bool enable_heading_hold
float64 timeout              # Command timeout in seconds
```

#### `modcube_msgs/NavigationState`
```yaml
std_msgs/Header header
geometry_msgs/PoseWithCovariance pose
geometry_msgs/TwistWithCovariance twist
geometry_msgs/AccelWithCovariance accel
geometry_msgs/Vector3 euler_angles     # [roll, pitch, yaw]
geometry_msgs/Vector3 euler_rates      # [roll_rate, pitch_rate, yaw_rate]
geometry_msgs/Vector3 euler_accels     # [roll_accel, pitch_accel, yaw_accel]
float64 depth
float64 altitude
bool is_valid
float64 timestamp
```

#### `modcube_msgs/ThrusterCommand`
```yaml
std_msgs/Header header
float64[14] thrusters        # Normalized thrust values [-1.0, 1.0]
float64[14] thrust_forces    # Actual thrust forces in Newtons
bool emergency_stop
float64 max_thrust_percentage # Global thrust scaling [0.0, 1.0]
```

### Sensor Messages

#### `modcube_msgs/DVLData`
```yaml
std_msgs/Header header
geometry_msgs/Vector3 velocity
float64 altitude
float64[4] beam_ranges
float64[4] beam_velocities
int32 status
```

#### `modcube_msgs/FluidDepth`
```yaml
std_msgs/Header header
float64 depth
float64 pressure
float64 temperature
```

#### `modcube_msgs/SonarPulse`
```yaml
std_msgs/Header header
float64 range
float64 intensity
float64 angle
```

### Detection Messages

#### `modcube_msgs/AprilTagDetection`
```yaml
std_msgs/Header header
int32 id
geometry_msgs/PoseWithCovariance pose
float64 size
float64 confidence
```

#### `modcube_msgs/GateDetection`
```yaml
std_msgs/Header header
geometry_msgs/Point[] corners
float64 width
float64 height
float64 confidence
```

### System Messages

#### `modcube_msgs/Alarm`
```yaml
std_msgs/Header header
string alarm_name
int32 severity      # 0=info, 1=warning, 2=error, 3=critical
string description
bool active
```

#### `modcube_msgs/BatteryStatus`
```yaml
std_msgs/Header header
float64 voltage
float64 current
float64 charge_percentage
float64 temperature
int32 status        # 0=unknown, 1=charging, 2=discharging, 3=full
```

### Debug and Analysis Messages

#### `modcube_msgs/PIDDebug`
```yaml
std_msgs/Header header
string axis                  # 'x', 'y', 'z', 'roll', 'pitch', 'yaw'
float64 setpoint
float64 process_variable
float64 error
float64 error_integral
float64 error_derivative
float64 proportional_term
float64 integral_term
float64 derivative_term
float64 output
float64 output_limited
float64 kp
float64 ki
float64 kd
float64 tau
float64 limit
```

#### `modcube_msgs/EKFDebug`
```yaml
std_msgs/Header header
float64[15] state_vector     # [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, 
                             #  acc_x, acc_y, acc_z, roll, pitch, yaw, 
                             #  ang_vel_x, ang_vel_y, ang_vel_z]
float64[225] covariance      # 15x15 covariance matrix (row-major)
float64[15] innovation       # Innovation vector
float64[225] innovation_cov  # Innovation covariance
string last_update_sensor    # Last sensor that updated the filter
float64 mahalanobis_distance # Mahalanobis distance for outlier detection
bool is_converged
```

#### `modcube_msgs/ThrusterAllocation`
```yaml
std_msgs/Header header
geometry_msgs/Wrench desired_wrench
float64[14] allocated_thrusts
float64[6] achieved_wrench   # [Fx, Fy, Fz, Tx, Ty, Tz]
float64 allocation_error     # Norm of (desired - achieved)
float64[14] saturation_flags # 1.0 if thruster is saturated, 0.0 otherwise
```

### Trajectory Messages

#### `modcube_msgs/TrajectoryPoint`
```yaml
float64 time_from_start
geometry_msgs/Pose pose
geometry_msgs/Twist velocity
geometry_msgs/Accel acceleration
```

#### `modcube_msgs/OptimalTrajectory`
```yaml
std_msgs/Header header
modcube_msgs/TrajectoryPoint[] points
float64 total_time
float64 total_distance
float64 average_speed
float64 max_velocity
float64 max_acceleration
float64 aggressiveness
string optimization_method   # 'minimum_snap', 'minimum_jerk', etc.
```

## Python APIs

### Controller API

#### `modcube_common.controllers.PIDController`

```python
class PIDController:
    def __init__(self, namespace='/modcube'):
        """Initialize PID controller.
        
        Args:
            namespace (str): ROS namespace for topics and services
        """
    
    def set_gains(self, pos_gains, orient_gains):
        """Set PID gains.
        
        Args:
            pos_gains (dict): Position gains {'p': [x,y,z], 'i': [x,y,z], 'd': [x,y,z]}
            orient_gains (dict): Orientation gains {'p': [r,p,y], 'i': [r,p,y], 'd': [r,p,y]}
        
        Returns:
            bool: Success status
        """
    
    def compute_wrench(self, current_state, desired_state):
        """Compute control wrench.
        
        Args:
            current_state (NavState): Current vehicle state
            desired_state (ControllerCommand): Desired state
        
        Returns:
            geometry_msgs/Wrench: Control wrench
        """
    
    def enable(self):
        """Enable controller."""
    
    def disable(self):
        """Disable controller."""
```

#### Example Usage

```python
import rospy
from modcube_common.controllers import PIDController
from modcube_msgs.msg import ControllerCommand
from geometry_msgs.msg import Point, Quaternion

# Initialize controller
controller = PIDController(namespace='/modcube')

# Set gains
pos_gains = {'p': [10.0, 10.0, 10.0], 'i': [0.1, 0.1, 0.1], 'd': [5.0, 5.0, 5.0]}
orient_gains = {'p': [20.0, 20.0, 20.0], 'i': [0.2, 0.2, 0.2], 'd': [8.0, 8.0, 8.0]}
controller.set_gains(pos_gains, orient_gains)

# Enable controller
controller.enable()

# Send command
cmd = ControllerCommand()
cmd.mode = 1  # Position control
cmd.setpoint.position = Point(1.0, 0.0, -1.0)
cmd.setpoint.orientation = Quaternion(0, 0, 0, 1)

controller.send_command(cmd)
```

### Motion Client API

#### `modcube_common.motion.MotionClient`

```python
class MotionClient:
    def __init__(self, namespace='/modcube'):
        """Initialize motion client."""
    
    def goto_position(self, x, y, z, timeout=30.0):
        """Move to specified position.
        
        Args:
            x, y, z (float): Target position
            timeout (float): Maximum time to reach target
        
        Returns:
            bool: Success status
        """
    
    def goto_pose(self, pose, timeout=30.0):
        """Move to specified pose.
        
        Args:
            pose (geometry_msgs/Pose): Target pose
            timeout (float): Maximum time to reach target
        
        Returns:
            bool: Success status
        """
    
    def follow_trajectory(self, trajectory):
        """Follow a trajectory.
        
        Args:
            trajectory (nav_msgs/Path): Trajectory to follow
        
        Returns:
            bool: Success status
        """
    
    def stop(self):
        """Stop all motion."""
    
    def get_current_pose(self):
        """Get current vehicle pose.
        
        Returns:
            geometry_msgs/Pose: Current pose
        """
```

### Mission Manager API

#### `modcube_mission.MissionManager`

```python
class MissionManager:
    def __init__(self):
        """Initialize mission manager."""
    
    def start_mission(self, mission_type, parameters=None):
        """Start a mission.
        
        Args:
            mission_type (str): Type of mission
            parameters (dict): Mission parameters
        
        Returns:
            str: Mission ID
        """
    
    def stop_mission(self, mission_id):
        """Stop a running mission.
        
        Args:
            mission_id (str): Mission ID to stop
        
        Returns:
            bool: Success status
        """
    
    def get_mission_status(self, mission_id):
        """Get mission status.
        
        Args:
            mission_id (str): Mission ID
        
        Returns:
            dict: Mission status information
        """
    
    def register_mission_type(self, mission_type, mission_class):
        """Register a new mission type.
        
        Args:
            mission_type (str): Mission type identifier
            mission_class (class): Mission implementation class
        """
```

### State Estimation API

#### `modcube_common.state_estimation.StateEstimator`

```python
class StateEstimator:
    def __init__(self, sensors=None):
        """Initialize state estimator.
        
        Args:
            sensors (list): List of sensor configurations
        """
    
    def add_sensor(self, sensor_type, topic, frame_id):
        """Add a sensor to the estimator.
        
        Args:
            sensor_type (str): Type of sensor ('imu', 'dvl', 'depth', etc.)
            topic (str): ROS topic for sensor data
            frame_id (str): Sensor frame ID
        """
    
    def get_state(self):
        """Get current estimated state.
        
        Returns:
            NavState: Current navigation state
        """
    
    def reset_state(self, initial_state=None):
        """Reset estimator state.
        
        Args:
            initial_state (NavState): Initial state (optional)
        """

## C++ APIs

### Extended Kalman Filter API

#### `modcube_common::Ekf`

```cpp
#include <modcube_common/ekf.h>

namespace modcube_common {

class Ekf {
public:
    /**
     * @brief Constructor
     */
    Ekf();
    
    /**
     * @brief Destructor
     */
    ~Ekf();
    
    /**
     * @brief Set DVL offset from vehicle center
     * @param offset DVL offset vector [x, y, z]
     */
    void setDvlOffset(const Eigen::Vector3d& offset);
    
    /**
     * @brief Set process noise covariance
     * @param cov Process noise covariance matrix (15x15)
     */
    void setProcessCovariance(const Eigen::MatrixXd& cov);
    
    /**
     * @brief Get current state vector
     * @return State vector [pos, vel, acc, euler, ang_vel]
     */
    Eigen::VectorXd getState() const;
    
    /**
     * @brief Set state vector
     * @param state State vector (15x1)
     */
    void setState(const Eigen::VectorXd& state);
    
    /**
     * @brief Get state covariance matrix
     * @return Covariance matrix (15x15)
     */
    Eigen::MatrixXd getCovariance() const;
    
    /**
     * @brief Set state covariance matrix
     * @param cov Covariance matrix (15x15)
     */
    void setCovariance(const Eigen::MatrixXd& cov);
    
    /**
     * @brief Process IMU measurement
     * @param imu_data IMU sensor data
     * @param dt Time step
     */
    void handleImuMeasurement(const sensor_msgs::Imu& imu_data, double dt);
    
    /**
     * @brief Process DVL measurement
     * @param dvl_data DVL sensor data
     * @param dt Time step
     */
    void handleDvlMeasurement(const modcube_msgs::DVLData& dvl_data, double dt);
    
    /**
     * @brief Process depth measurement
     * @param depth_data Depth sensor data
     * @param dt Time step
     */
    void handleDepthMeasurement(const modcube_msgs::FluidDepth& depth_data, double dt);
    
private:
    /**
     * @brief Prediction step of EKF
     * @param dt Time step
     */
    void predict(double dt);
    
    /**
     * @brief Update step of EKF
     * @param measurement Measurement vector
     * @param measurement_cov Measurement covariance
     * @param H Measurement Jacobian
     */
    void update(const Eigen::VectorXd& measurement,
                const Eigen::MatrixXd& measurement_cov,
                const Eigen::MatrixXd& H);
    
    /**
     * @brief Extrapolate state forward in time
     * @param state Current state
     * @param dt Time step
     * @return Extrapolated state
     */
    Eigen::VectorXd extrapolateState(const Eigen::VectorXd& state, double dt);
    
    /**
     * @brief Extrapolate covariance forward in time
     * @param cov Current covariance
     * @param F State transition Jacobian
     * @param Q Process noise covariance
     * @return Extrapolated covariance
     */
    Eigen::MatrixXd extrapolateCovariance(const Eigen::MatrixXd& cov,
                                          const Eigen::MatrixXd& F,
                                          const Eigen::MatrixXd& Q);
    
    /**
     * @brief Wrap angles to [-pi, pi]
     * @param angle Input angle
     * @return Wrapped angle
     */
    double wrapAngle(double angle);
    
    /**
     * @brief Calculate state transition Jacobian
     * @param state Current state
     * @param dt Time step
     * @return Jacobian matrix
     */
    Eigen::MatrixXd calculateStateJacobian(const Eigen::VectorXd& state, double dt);
    
    // State vector: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z,
    //                acc_x, acc_y, acc_z, roll, pitch, yaw,
    //                ang_vel_x, ang_vel_y, ang_vel_z]
    Eigen::VectorXd state_;           // 15x1 state vector
    Eigen::MatrixXd covariance_;      // 15x15 covariance matrix
    Eigen::MatrixXd process_cov_;     // 15x15 process noise covariance
    Eigen::Vector3d dvl_offset_;      // DVL offset from vehicle center
    
    bool is_initialized_;
    double last_update_time_;
};

} // namespace modcube_common
```

### State Estimator API

#### `modcube_common::StateEstimator`

```cpp
#include <modcube_common/state_estimator.h>

namespace modcube_common {

class StateEstimator {
public:
    /**
     * @brief Constructor
     * @param nh ROS node handle
     */
    explicit StateEstimator(ros::NodeHandle& nh);
    
    /**
     * @brief Destructor
     */
    ~StateEstimator();
    
    /**
     * @brief Initialize state estimator
     * @return Success status
     */
    bool initialize();
    
    /**
     * @brief Start state estimation
     */
    void start();
    
    /**
     * @brief Stop state estimation
     */
    void stop();
    
    /**
     * @brief Get current navigation state
     * @return Current navigation state
     */
    modcube_msgs::NavigationState getCurrentState() const;
    
private:
    /**
     * @brief IMU callback
     * @param msg IMU message
     */
    void imuCallback(const sensor_msgs::Imu::ConstPtr& msg);
    
    /**
     * @brief DVL callback
     * @param msg DVL message
     */
    void dvlCallback(const modcube_msgs::DVLData::ConstPtr& msg);
    
    /**
     * @brief Depth callback
     * @param msg Depth message
     */
    void depthCallback(const modcube_msgs::FluidDepth::ConstPtr& msg);
    
    /**
     * @brief Publish navigation state
     */
    void publishState();
    
    /**
     * @brief Load configuration parameters
     * @return Success status
     */
    bool loadConfig();
    
    ros::NodeHandle nh_;
    
    // Subscribers
    ros::Subscriber imu_sub_;
    ros::Subscriber dvl_sub_;
    ros::Subscriber depth_sub_;
    
    // Publishers
    ros::Publisher nav_state_pub_;
    ros::Publisher debug_pub_;
    
    // Services
    ros::ServiceServer reset_service_;
    ros::ServiceServer config_service_;
    
    // EKF instance
    std::unique_ptr<Ekf> ekf_;
    
    // Configuration
    double publish_rate_;
    std::string base_frame_;
    std::string world_frame_;
    
    // Sensor delays and covariances
    double imu_delay_;
    double dvl_delay_;
    double depth_delay_;
    
    Eigen::MatrixXd imu_cov_;
    Eigen::MatrixXd dvl_cov_;
    Eigen::MatrixXd depth_cov_;
    
    // State
    bool is_initialized_;
    bool is_running_;
    
    mutable std::mutex state_mutex_;
};

} // namespace modcube_common
```

## State Estimation APIs

### Multi-Sensor Fusion

#### `modcube_common.state_estimation.ExtendedKalmanFilter`

```python
class ExtendedKalmanFilter:
    """Extended Kalman Filter for underwater vehicle state estimation."""
    
    def __init__(self, initial_state=None, initial_covariance=None):
        """Initialize EKF with optional initial conditions.
        
        Args:
            initial_state (np.ndarray): Initial 15-element state vector
            initial_covariance (np.ndarray): Initial 15x15 covariance matrix
        """
    
    def predict(self, dt, control_input=None):
        """Prediction step of EKF.
        
        Args:
            dt (float): Time step in seconds
            control_input (np.ndarray): Optional control input vector
        """
    
    def update_imu(self, acceleration, angular_velocity, orientation=None):
        """Update with IMU measurement.
        
        Args:
            acceleration (np.ndarray): Linear acceleration [ax, ay, az]
            angular_velocity (np.ndarray): Angular velocity [wx, wy, wz]
            orientation (np.ndarray): Optional orientation quaternion [x, y, z, w]
        """
    
    def update_dvl(self, velocity, altitude=None):
        """Update with DVL measurement.
        
        Args:
            velocity (np.ndarray): Velocity measurement [vx, vy, vz]
            altitude (float): Optional altitude measurement
        """
    
    def update_depth(self, depth):
        """Update with depth measurement.
        
        Args:
            depth (float): Depth measurement in meters
        """
    
    def update_position(self, position, covariance=None):
        """Update with absolute position measurement.
        
        Args:
            position (np.ndarray): Position measurement [x, y, z]
            covariance (np.ndarray): Optional measurement covariance
        """
    
    def get_state(self):
        """Get current state estimate.
        
        Returns:
            dict: State dictionary with position, velocity, acceleration, orientation
        """
    
    def get_covariance(self):
        """Get current state covariance.
        
        Returns:
            np.ndarray: 15x15 covariance matrix
        """
    
    def reset(self, state=None, covariance=None):
        """Reset filter state.
        
        Args:
            state (np.ndarray): Optional new state vector
            covariance (np.ndarray): Optional new covariance matrix
        """
```

#### `modcube_common.state_estimation.VisualOdometry`

```python
class VisualOdometry:
    """Visual odometry for underwater environments."""
    
    def __init__(self, camera_params, feature_detector='ORB'):
        """Initialize visual odometry.
        
        Args:
            camera_params (dict): Camera calibration parameters
            feature_detector (str): Feature detector type ('ORB', 'SIFT', 'SURF')
        """
    
    def process_frame(self, image, timestamp):
        """Process new camera frame.
        
        Args:
            image (np.ndarray): Input image
            timestamp (float): Image timestamp
        
        Returns:
            dict: Pose estimate and confidence
        """
    
    def get_pose_estimate(self):
        """Get current pose estimate.
        
        Returns:
            geometry_msgs/PoseWithCovariance: Current pose estimate
        """
    
    def reset_tracking(self):
        """Reset visual tracking."""
```

## Trajectory Generation APIs

### Optimal Trajectory Planning

#### `modcube_common.trajectory.DroneTrajectory`

```python
class DroneTrajectory:
    """Optimal spline trajectory generator for underwater vehicles."""
    
    def __init__(self):
        """Initialize trajectory generator."""
    
    def solve(self, waypoints, aggressiveness=0.5, start_vel=None, end_vel=None):
        """Generate optimal trajectory through waypoints.
        
        Args:
            waypoints (list): List of TrajectoryWaypoint objects
            aggressiveness (float): Trajectory aggressiveness [0.0, 1.0]
            start_vel (np.ndarray): Initial velocity [vx, vy, vz]
            end_vel (np.ndarray): Final velocity [vx, vy, vz]
        
        Returns:
            bool: Success status
        """
    
    def evaluate(self, t):
        """Evaluate trajectory at given time.
        
        Args:
            t (float): Time from trajectory start
        
        Returns:
            dict: Position, velocity, acceleration at time t
        """
    
    def get_position(self, t):
        """Get position at time t.
        
        Args:
            t (float): Time from trajectory start
        
        Returns:
            np.ndarray: Position [x, y, z]
        """
    
    def get_velocity(self, t):
        """Get velocity at time t.
        
        Args:
            t (float): Time from trajectory start
        
        Returns:
            np.ndarray: Velocity [vx, vy, vz]
        """
    
    def get_acceleration(self, t):
        """Get acceleration at time t.
        
        Args:
            t (float): Time from trajectory start
        
        Returns:
            np.ndarray: Acceleration [ax, ay, az]
        """
    
    def to_path_msg(self, dt=0.1):
        """Convert trajectory to ROS Path message.
        
        Args:
            dt (float): Time step for path discretization
        
        Returns:
            nav_msgs/Path: ROS path message
        """
    
    def get_total_time(self):
        """Get total trajectory time.
        
        Returns:
            float: Total time in seconds
        """
    
    def get_total_distance(self):
        """Get total trajectory distance.
        
        Returns:
            float: Total distance in meters
        """
    
    def get_average_speed(self):
        """Get average trajectory speed.
        
        Returns:
            float: Average speed in m/s
        """
```

#### `modcube_common.trajectory.TrajectoryWaypoint`

```python
class TrajectoryWaypoint:
    """Waypoint for trajectory generation."""
    
    def __init__(self, position, orientation=None, constraints=None):
        """Initialize waypoint.
        
        Args:
            position (np.ndarray): Waypoint position [x, y, z]
            orientation (np.ndarray): Optional orientation quaternion [x, y, z, w]
            constraints (dict): Optional velocity/acceleration constraints
        """
    
    def set_velocity_constraint(self, velocity):
        """Set velocity constraint at waypoint.
        
        Args:
            velocity (np.ndarray): Velocity constraint [vx, vy, vz]
        """
    
    def set_acceleration_constraint(self, acceleration):
        """Set acceleration constraint at waypoint.
        
        Args:
            acceleration (np.ndarray): Acceleration constraint [ax, ay, az]
        """
```

### Minimum Snap Trajectory

#### `modcube_common.trajectory.OptimalTrajectory`

```python
class OptimalTrajectory:
    """Minimum snap trajectory optimization."""
    
    def __init__(self, dimension=3):
        """Initialize trajectory optimizer.
        
        Args:
            dimension (int): Spatial dimension (3 for 3D trajectories)
        """
    
    def generate_trajectory(self, waypoints, time_allocation, 
                          derivative_order=4, continuity_order=3):
        """Generate minimum snap trajectory.
        
        Args:
            waypoints (list): List of waypoint positions
            time_allocation (list): Time allocation between waypoints
            derivative_order (int): Order of derivative to minimize (4 for snap)
            continuity_order (int): Continuity order at waypoints
        
        Returns:
            dict: Trajectory coefficients and metadata
        """
    
    def evaluate_polynomial(self, coefficients, t, derivative=0):
        """Evaluate polynomial trajectory.
        
        Args:
            coefficients (np.ndarray): Polynomial coefficients
            t (float): Time parameter
            derivative (int): Derivative order (0=position, 1=velocity, etc.)
        
        Returns:
            np.ndarray: Evaluated trajectory point
        """
```

### Path Planning and Navigation

#### `modcube_common.navigation.PathPlanner`

```python
class PathPlanner:
    """A* and RRT* path planning for underwater environments."""
    
    def __init__(self, occupancy_map=None, planning_algorithm='A*'):
        """Initialize path planner.
        
        Args:
            occupancy_map (OccupancyGrid): Optional occupancy grid map
            planning_algorithm (str): Planning algorithm ('A*', 'RRT*', 'PRM')
        """
    
    def plan_path(self, start, goal, constraints=None):
        """Plan path from start to goal.
        
        Args:
            start (np.ndarray): Start position [x, y, z]
            goal (np.ndarray): Goal position [x, y, z]
            constraints (dict): Optional planning constraints
        
        Returns:
            list: List of waypoints forming the path
        """
    
    def update_map(self, occupancy_grid):
        """Update occupancy map.
        
        Args:
            occupancy_grid (nav_msgs/OccupancyGrid): New occupancy grid
        """
    
    def set_vehicle_constraints(self, max_velocity, max_acceleration, turning_radius):
        """Set vehicle dynamic constraints.
        
        Args:
            max_velocity (float): Maximum velocity in m/s
            max_acceleration (float): Maximum acceleration in m/s²
            turning_radius (float): Minimum turning radius in meters
        """
```

#### `modcube_common.navigation.ObstacleAvoidance`

```python
class ObstacleAvoidance:
    """Real-time obstacle avoidance using potential fields."""
    
    def __init__(self, safety_distance=2.0, max_avoidance_force=5.0):
        """Initialize obstacle avoidance.
        
        Args:
            safety_distance (float): Minimum distance to obstacles in meters
            max_avoidance_force (float): Maximum avoidance force magnitude
        """
    
    def compute_avoidance_force(self, current_position, obstacles, target_velocity):
        """Compute obstacle avoidance force.
        
        Args:
            current_position (np.ndarray): Current vehicle position [x, y, z]
            obstacles (list): List of obstacle positions and sizes
            target_velocity (np.ndarray): Desired velocity [vx, vy, vz]
        
        Returns:
            np.ndarray: Avoidance force vector [fx, fy, fz]
        """
    
    def update_obstacles(self, sensor_data):
        """Update obstacle map from sensor data.
        
        Args:
            sensor_data (dict): Sensor data including sonar, lidar, camera
        """
```

## Advanced Configuration Parameters

### Controller Parameters

```yaml
# PID Controller
controller:
  position_gains:
    p: [10.0, 10.0, 10.0]  # Proportional gains [x, y, z]
    i: [0.1, 0.1, 0.1]     # Integral gains [x, y, z]
    d: [5.0, 5.0, 5.0]     # Derivative gains [x, y, z]
  orientation_gains:
    p: [20.0, 20.0, 20.0]  # Proportional gains [roll, pitch, yaw]
    i: [0.2, 0.2, 0.2]     # Integral gains [roll, pitch, yaw]
    d: [8.0, 8.0, 8.0]     # Derivative gains [roll, pitch, yaw]
  max_thrust: 100.0        # Maximum thrust (N)
  control_frequency: 100   # Control loop frequency (Hz)
  
  # Advanced controller settings
  adaptive_gains:
    enabled: true
    adaptation_rate: 0.01
    min_gain_factor: 0.1
    max_gain_factor: 5.0
  
  anti_windup:
    enabled: true
    max_integral: [10.0, 10.0, 10.0, 5.0, 5.0, 5.0]
  
  feedforward:
    enabled: true
    velocity_ff: [0.8, 0.8, 0.8]
    acceleration_ff: [0.2, 0.2, 0.2]
  
  saturation_limits:
    position_error_max: [5.0, 5.0, 5.0]  # Maximum position error (m)
    velocity_max: [2.0, 2.0, 2.0]        # Maximum velocity (m/s)
    angular_velocity_max: [1.0, 1.0, 1.0] # Maximum angular velocity (rad/s)
```

### Thruster Manager Parameters

```yaml
# Thruster Manager
thruster_manager:
  thrusters:
    - id: 0
      frame: "thruster_0"
      max_thrust: 50.0
      topic: "/modcube/thrusters/0/input"
    - id: 1
      frame: "thruster_1"
      max_thrust: 50.0
      topic: "/modcube/thrusters/1/input"
    # ... additional thrusters
  
  allocation_matrix: "tam_14_thrusters.yaml"
  update_rate: 50  # Hz
```

### Sensor Parameters

```yaml
# IMU Configuration
imu:
  frame_id: "imu_link"
  topic: "/modcube/imu/data"
  frequency: 100
  orientation_covariance: [0.01, 0.01, 0.01]
  angular_velocity_covariance: [0.001, 0.001, 0.001]
  linear_acceleration_covariance: [0.01, 0.01, 0.01]
  bias_estimation:
    enabled: true
    gyro_bias_std: 0.001
    accel_bias_std: 0.01
  calibration:
    auto_calibrate: true
    calibration_time: 30.0  # seconds

# DVL Configuration
dvl:
  frame_id: "dvl_link"
  topic: "/modcube/dvl/data"
  frequency: 10
  velocity_covariance: [0.01, 0.01, 0.01]
  altitude_covariance: 0.1
  beam_configuration:
    num_beams: 4
    beam_angle: 30.0  # degrees
    max_range: 100.0  # meters
  outlier_rejection:
    enabled: true
    velocity_threshold: 5.0  # m/s
    altitude_threshold: 200.0  # m

# Depth Sensor Configuration
depth_sensor:
  frame_id: "depth_link"
  topic: "/modcube/depth"
  frequency: 20
  depth_covariance: 0.01
  pressure_to_depth:
    water_density: 1025.0  # kg/m³
    gravity: 9.81  # m/s²
    atmospheric_pressure: 101325.0  # Pa
  filtering:
    enabled: true
    filter_type: "low_pass"
    cutoff_frequency: 5.0  # Hz

# Camera Configuration
camera:
  frame_id: "camera_link"
  image_topic: "/modcube/camera/image_raw"
  info_topic: "/modcube/camera/camera_info"
  frequency: 30
  resolution: [1920, 1080]
  field_of_view: [90.0, 60.0]  # [horizontal, vertical] degrees
  exposure:
    auto_exposure: true
    exposure_time: 0.033  # seconds
  white_balance:
    auto_white_balance: true
    color_temperature: 5000  # K

# Sonar Configuration
sonar:
  frame_id: "sonar_link"
  topic: "/modcube/sonar/data"
  frequency: 10
  range_min: 0.5  # meters
  range_max: 50.0  # meters
  field_of_view: 120.0  # degrees
  resolution: 1.0  # degrees
```

## Plugin APIs

### Gazebo Plugin API

#### Custom Thruster Plugin

```cpp
// custom_thruster_plugin.h
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <ros/ros.h>
#include <modcube_msgs/ThrusterCommand.h>

namespace gazebo {
    class CustomThrusterPlugin : public ModelPlugin {
    public:
        CustomThrusterPlugin();
        virtual ~CustomThrusterPlugin();
        
        virtual void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf);
        virtual void Update();
        
    private:
        void ThrusterCallback(const modcube_msgs::ThrusterCommand::ConstPtr& msg);
        
        physics::ModelPtr model_;
        physics::LinkPtr link_;
        event::ConnectionPtr update_connection_;
        
        ros::NodeHandle nh_;
        ros::Subscriber thruster_sub_;
        
        double thrust_force_;
        math::Vector3 thrust_direction_;
    };
}
```

### ROS Plugin API

#### Custom Sensor Plugin

```python
# custom_sensor_plugin.py
import rospy
from sensor_msgs.msg import PointCloud2
from modcube_msgs.msg import CustomSensorData

class CustomSensorPlugin:
    def __init__(self):
        self.pub = rospy.Publisher('/modcube/custom_sensor', CustomSensorData, queue_size=10)
        self.sub = rospy.Subscriber('/camera/depth/points', PointCloud2, self.pointcloud_callback)
    
    def pointcloud_callback(self, msg):
        # Process point cloud data
        processed_data = self.process_pointcloud(msg)
        
        # Publish custom sensor data
        custom_msg = CustomSensorData()
        custom_msg.header = msg.header
        custom_msg.data = processed_data
        
        self.pub.publish(custom_msg)
    
    def process_pointcloud(self, pointcloud):
        # Custom processing logic
        return processed_data
```

## Examples

### Basic Control Example

```python
#!/usr/bin/env python
import rospy
from modcube_common.motion import MotionClient
from geometry_msgs.msg import Pose, Point, Quaternion

def main():
    rospy.init_node('basic_control_example')
    
    # Initialize motion client
    motion = MotionClient()
    
    # Wait for system to be ready
    rospy.sleep(2.0)
    
    # Move to different positions
    positions = [
        (0, 0, -1),
        (5, 0, -1),
        (5, 5, -1),
        (0, 5, -1),
        (0, 0, -1)
    ]
    
    for pos in positions:
        rospy.loginfo(f"Moving to position: {pos}")
        success = motion.goto_position(pos[0], pos[1], pos[2], timeout=30.0)
        
        if success:
            rospy.loginfo("Position reached successfully")
        else:
            rospy.logwarn("Failed to reach position")
        
        rospy.sleep(2.0)
    
    rospy.loginfo("Mission completed")

if __name__ == '__main__':
    main()
```

### Advanced Trajectory Following Example

```python
#!/usr/bin/env python
import rospy
import numpy as np
from modcube_common.trajectory import DroneTrajectory, TrajectoryWaypoint
from modcube_common.motion import MotionClient
from geometry_msgs.msg import Point

def main():
    rospy.init_node('trajectory_following_example')
    
    # Initialize trajectory generator and motion client
    trajectory = DroneTrajectory()
    motion = MotionClient()
    
    # Define waypoints for a complex 3D trajectory
    waypoints = [
        TrajectoryWaypoint([0, 0, -1]),
        TrajectoryWaypoint([10, 0, -1]),
        TrajectoryWaypoint([10, 10, -3]),
        TrajectoryWaypoint([0, 10, -5]),
        TrajectoryWaypoint([0, 0, -3]),
        TrajectoryWaypoint([0, 0, -1])
    ]
    
    # Set velocity constraints for smooth motion
    waypoints[0].set_velocity_constraint([0, 0, 0])  # Start from rest
    waypoints[-1].set_velocity_constraint([0, 0, 0])  # End at rest
    
    # Generate optimal trajectory
    rospy.loginfo("Generating optimal trajectory...")
    success = trajectory.solve(
        waypoints=waypoints,
        aggressiveness=0.7,
        start_vel=np.array([0, 0, 0]),
        end_vel=np.array([0, 0, 0])
    )
    
    if not success:
        rospy.logerr("Failed to generate trajectory")
        return
    
    # Convert to ROS path message
    path_msg = trajectory.to_path_msg(dt=0.1)
    
    # Follow the trajectory
    rospy.loginfo(f"Following trajectory (duration: {trajectory.get_total_time():.2f}s)")
    success = motion.follow_trajectory(path_msg)
    
    if success:
        rospy.loginfo("Trajectory completed successfully")
    else:
        rospy.logwarn("Trajectory following failed")

if __name__ == '__main__':
    main()
```

### Multi-Sensor State Estimation Example

```python
#!/usr/bin/env python
import rospy
import numpy as np
from modcube_common.state_estimation import ExtendedKalmanFilter
from sensor_msgs.msg import Imu
from modcube_msgs.msg import DVLData, FluidDepth, NavState
from geometry_msgs.msg import PoseWithCovariance

class StateEstimationNode:
    def __init__(self):
        rospy.init_node('state_estimation_example')
        
        # Initialize EKF
        initial_state = np.zeros(15)  # [pos, vel, acc, euler, ang_vel]
        initial_cov = np.eye(15) * 0.1
        self.ekf = ExtendedKalmanFilter(initial_state, initial_cov)
        
        # Subscribers
        self.imu_sub = rospy.Subscriber('/modcube/imu/data', Imu, self.imu_callback)
        self.dvl_sub = rospy.Subscriber('/modcube/dvl/data', DVLData, self.dvl_callback)
        self.depth_sub = rospy.Subscriber('/modcube/depth', FluidDepth, self.depth_callback)
        
        # Publishers
        self.nav_pub = rospy.Publisher('/modcube/nav_state', NavState, queue_size=10)
        
        # Timing
        self.last_prediction_time = rospy.Time.now()
        
        # Timer for regular state publishing
        self.timer = rospy.Timer(rospy.Duration(0.02), self.publish_state)  # 50 Hz
        
        rospy.loginfo("State estimation node initialized")
    
    def imu_callback(self, msg):
        # Extract IMU data
        accel = np.array([msg.linear_acceleration.x,
                         msg.linear_acceleration.y,
                         msg.linear_acceleration.z])
        
        ang_vel = np.array([msg.angular_velocity.x,
                           msg.angular_velocity.y,
                           msg.angular_velocity.z])
        
        # Optional orientation from IMU
        orientation = None
        if msg.orientation_covariance[0] > 0:
            orientation = np.array([msg.orientation.x,
                                   msg.orientation.y,
                                   msg.orientation.z,
                                   msg.orientation.w])
        
        # Prediction step
        current_time = rospy.Time.now()
        dt = (current_time - self.last_prediction_time).to_sec()
        if dt > 0:
            self.ekf.predict(dt)
            self.last_prediction_time = current_time
        
        # Update with IMU measurement
        self.ekf.update_imu(accel, ang_vel, orientation)
    
    def dvl_callback(self, msg):
        # Extract DVL velocity
        velocity = np.array([msg.velocity.x,
                            msg.velocity.y,
                            msg.velocity.z])
        
        # Update with DVL measurement
        self.ekf.update_dvl(velocity, msg.altitude)
    
    def depth_callback(self, msg):
        # Update with depth measurement
        self.ekf.update_depth(msg.depth)
    
    def publish_state(self, event):
        # Get current state estimate
        state = self.ekf.get_state()
        covariance = self.ekf.get_covariance()
        
        # Create NavState message
        nav_msg = NavState()
        nav_msg.header.stamp = rospy.Time.now()
        nav_msg.header.frame_id = "world"
        
        # Position and orientation
        nav_msg.pose.pose.position.x = state['position'][0]
        nav_msg.pose.pose.position.y = state['position'][1]
        nav_msg.pose.pose.position.z = state['position'][2]
        
        # Convert Euler angles to quaternion
        from tf.transformations import quaternion_from_euler
        q = quaternion_from_euler(state['orientation'][0],
                                 state['orientation'][1],
                                 state['orientation'][2])
        nav_msg.pose.pose.orientation.x = q[0]
        nav_msg.pose.pose.orientation.y = q[1]
        nav_msg.pose.pose.orientation.z = q[2]
        nav_msg.pose.pose.orientation.w = q[3]
        
        # Velocity
        nav_msg.twist.twist.linear.x = state['velocity'][0]
        nav_msg.twist.twist.linear.y = state['velocity'][1]
        nav_msg.twist.twist.linear.z = state['velocity'][2]
        
        # Angular velocity
        nav_msg.twist.twist.angular.x = state['angular_velocity'][0]
        nav_msg.twist.twist.angular.y = state['angular_velocity'][1]
        nav_msg.twist.twist.angular.z = state['angular_velocity'][2]
        
        # Covariance (simplified - position only)
        nav_msg.pose.covariance[0] = covariance[0, 0]  # x-x
        nav_msg.pose.covariance[7] = covariance[1, 1]  # y-y
        nav_msg.pose.covariance[14] = covariance[2, 2]  # z-z
        
        # Publish
        self.nav_pub.publish(nav_msg)
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = StateEstimationNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
```

### Autonomous Survey Mission Example

```python
#!/usr/bin/env python
import rospy
import numpy as np
from modcube_mission import MissionManager
from modcube_common.navigation import PathPlanner, ObstacleAvoidance
from modcube_common.motion import MotionClient
from modcube_msgs.msg import NavState
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid

class AutonomousSurveyMission:
    def __init__(self):
        rospy.init_node('autonomous_survey_mission')
        
        # Initialize components
        self.motion = MotionClient()
        self.path_planner = PathPlanner(planning_algorithm='A*')
        self.obstacle_avoidance = ObstacleAvoidance(safety_distance=3.0)
        
        # Mission parameters
        self.survey_area = {
            'min_x': 0, 'max_x': 50,
            'min_y': 0, 'max_y': 30,
            'depth': -5.0
        }
        self.line_spacing = 5.0  # meters between survey lines
        self.survey_speed = 1.0  # m/s
        
        # State tracking
        self.current_pose = None
        self.obstacles = []
        
        # Subscribers
        self.nav_sub = rospy.Subscriber('/modcube/nav_state', NavState, self.nav_callback)
        self.sonar_sub = rospy.Subscriber('/modcube/sonar/pointcloud', PointCloud2, self.sonar_callback)
        self.map_sub = rospy.Subscriber('/modcube/occupancy_grid', OccupancyGrid, self.map_callback)
        
        rospy.loginfo("Autonomous survey mission initialized")
    
    def nav_callback(self, msg):
        self.current_pose = msg.pose.pose
    
    def sonar_callback(self, msg):
        # Process sonar data for obstacle detection
        self.obstacle_avoidance.update_obstacles({'sonar': msg})
    
    def map_callback(self, msg):
        # Update path planner with new map
        self.path_planner.update_map(msg)
    
    def generate_survey_pattern(self):
        """Generate lawnmower survey pattern."""
        waypoints = []
        
        y = self.survey_area['min_y']
        direction = 1  # 1 for positive x, -1 for negative x
        
        while y <= self.survey_area['max_y']:
            if direction == 1:
                # Left to right
                start_x = self.survey_area['min_x']
                end_x = self.survey_area['max_x']
            else:
                # Right to left
                start_x = self.survey_area['max_x']
                end_x = self.survey_area['min_x']
            
            waypoints.append([start_x, y, self.survey_area['depth']])
            waypoints.append([end_x, y, self.survey_area['depth']])
            
            y += self.line_spacing
            direction *= -1
        
        return waypoints
    
    def execute_survey(self):
        """Execute the survey mission."""
        rospy.loginfo("Starting autonomous survey mission")
        
        # Wait for initial position
        while self.current_pose is None and not rospy.is_shutdown():
            rospy.loginfo("Waiting for navigation state...")
            rospy.sleep(1.0)
        
        # Generate survey pattern
        waypoints = self.generate_survey_pattern()
        rospy.loginfo(f"Generated {len(waypoints)} waypoints")
        
        # Execute waypoints with obstacle avoidance
        for i, waypoint in enumerate(waypoints):
            rospy.loginfo(f"Navigating to waypoint {i+1}/{len(waypoints)}: {waypoint}")
            
            # Plan path to waypoint
            current_pos = np.array([
                self.current_pose.position.x,
                self.current_pose.position.y,
                self.current_pose.position.z
            ])
            
            path = self.path_planner.plan_path(
                start=current_pos,
                goal=np.array(waypoint),
                constraints={'max_velocity': self.survey_speed}
            )
            
            if path is None:
                rospy.logwarn(f"Failed to plan path to waypoint {i+1}")
                continue
            
            # Execute path with real-time obstacle avoidance
            success = self.execute_path_with_avoidance(path)
            
            if not success:
                rospy.logwarn(f"Failed to reach waypoint {i+1}")
                # Decide whether to continue or abort mission
                continue
            
            rospy.loginfo(f"Reached waypoint {i+1}")
        
        rospy.loginfo("Survey mission completed")
    
    def execute_path_with_avoidance(self, path):
        """Execute path with real-time obstacle avoidance."""
        rate = rospy.Rate(10)  # 10 Hz control loop
        
        for waypoint in path:
            # Check for obstacles and compute avoidance
            current_pos = np.array([
                self.current_pose.position.x,
                self.current_pose.position.y,
                self.current_pose.position.z
            ])
            
            target_velocity = (np.array(waypoint) - current_pos)
            target_velocity = target_velocity / np.linalg.norm(target_velocity) * self.survey_speed
            
            # Compute obstacle avoidance force
            avoidance_force = self.obstacle_avoidance.compute_avoidance_force(
                current_pos, self.obstacles, target_velocity
            )
            
            # Combine target velocity with avoidance
            modified_velocity = target_velocity + avoidance_force
            
            # Send velocity command
            success = self.motion.goto_position(
                waypoint[0], waypoint[1], waypoint[2],
                timeout=10.0
            )
            
            if not success:
                return False
            
            rate.sleep()
        
        return True
    
    def run(self):
        """Run the mission."""
        try:
            self.execute_survey()
        except rospy.ROSInterruptException:
            rospy.loginfo("Mission interrupted")
        except Exception as e:
            rospy.logerr(f"Mission failed: {e}")

if __name__ == '__main__':
    mission = AutonomousSurveyMission()
    mission.run()
```

### Mission Planning Example

```python
#!/usr/bin/env python
import rospy
from modcube_mission import MissionManager
from modcube_mission.missions import WaypointMission, SearchMission

def main():
    rospy.init_node('mission_example')
    
    # Initialize mission manager
    mission_mgr = MissionManager()
    
    # Register custom mission types
    mission_mgr.register_mission_type('waypoint', WaypointMission)
    mission_mgr.register_mission_type('search', SearchMission)
    
    # Start waypoint mission
    waypoint_params = {
        'waypoints': [
            {'x': 0, 'y': 0, 'z': -1},
            {'x': 10, 'y': 0, 'z': -1},
            {'x': 10, 'y': 10, 'z': -1},
            {'x': 0, 'y': 10, 'z': -1}
        ],
        'speed': 0.5
    }
    
    mission_id = mission_mgr.start_mission('waypoint', waypoint_params)
    rospy.loginfo(f"Started mission: {mission_id}")
    
    # Monitor mission progress
    rate = rospy.Rate(1)  # 1 Hz
    while not rospy.is_shutdown():
        status = mission_mgr.get_mission_status(mission_id)
        rospy.loginfo(f"Mission status: {status['state']}")
        
        if status['state'] == 'completed':
            break
        
        rate.sleep()
    
    rospy.loginfo("Mission completed successfully")

if __name__ == '__main__':
    main()
```

### Sensor Integration Example

```python
#!/usr/bin/env python
import rospy
from modcube_common.state_estimation import StateEstimator
from sensor_msgs.msg import Imu
from modcube_msgs.msg import DVLData, NavState

def main():
    rospy.init_node('sensor_integration_example')
    
    # Initialize state estimator
    estimator = StateEstimator()
    
    # Add sensors
    estimator.add_sensor('imu', '/modcube/imu/data', 'imu_link')
    estimator.add_sensor('dvl', '/modcube/dvl/data', 'dvl_link')
    estimator.add_sensor('depth', '/modcube/depth', 'depth_link')
    
    # Publisher for estimated state
    nav_pub = rospy.Publisher('/modcube/nav_state', NavState, queue_size=10)
    
    rate = rospy.Rate(50)  # 50 Hz
    
    while not rospy.is_shutdown():
        # Get current estimated state
        state = estimator.get_state()
        
        # Publish state
        nav_pub.publish(state)
        
        rate.sleep()

if __name__ == '__main__':
    main()
```

This API documentation provides a comprehensive reference for developing with the RS-ModCubes system. For more examples and detailed usage, refer to the source code and example implementations in the repository.