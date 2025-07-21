---
layout: default
title: FAQ
nav_order: 7
has_children: false
permalink: /faq/
---

# Frequently Asked Questions

This page addresses common questions about the RS-ModCubes system, covering installation, configuration, troubleshooting, and usage.

## Table of Contents

1. [General Questions](#general-questions)
2. [Installation and Setup](#installation-and-setup)
3. [Configuration](#configuration)
4. [Hardware Integration](#hardware-integration)
5. [Simulation](#simulation)
6. [Control and Navigation](#control-and-navigation)
7. [Mission Planning](#mission-planning)
8. [Troubleshooting](#troubleshooting)
9. [Performance and Optimization](#performance-and-optimization)
10. [Development and Customization](#development-and-customization)

## General Questions

### Q: What is RS-ModCubes?

**A:** RS-ModCubes is a comprehensive robotic system designed for autonomous underwater vehicles (AUVs). It provides a complete software stack including control systems, navigation, mission planning, and simulation capabilities. The system is built on ROS (Robot Operating System) and supports both simulation and real hardware deployment.

### Q: What are the main features of RS-ModCubes?

**A:** Key features include:
- **Advanced Control Systems**: PID controllers, thruster management, and adaptive control
- **State Estimation**: Multi-sensor fusion with IMU, DVL, depth sensors, and cameras
- **Mission Planning**: Waypoint navigation, search patterns, and custom mission types
- **Simulation Environment**: Gazebo-based underwater simulation with realistic physics
- **Hardware Integration**: Support for various sensors and actuators
- **Modular Architecture**: Extensible design for custom applications

### Q: What platforms does RS-ModCubes support?

**A:** RS-ModCubes is primarily designed for Linux systems and supports:
- Ubuntu 18.04 LTS (with ROS Melodic)
- Ubuntu 20.04 LTS (with ROS Noetic)
- Other Linux distributions with compatible ROS versions

### Q: Is RS-ModCubes open source?

**A:** Yes, RS-ModCubes is released under an open-source license. You can find the source code, contribute to development, and modify it according to your needs.

## Installation and Setup

### Q: What are the system requirements for RS-ModCubes?

**A:** Minimum requirements:
- **OS**: Ubuntu 18.04/20.04 LTS
- **RAM**: 8GB (16GB recommended)
- **Storage**: 20GB free space
- **CPU**: Intel i5 or equivalent (i7 recommended)
- **GPU**: NVIDIA GPU recommended for simulation

### Q: How do I install ROS for RS-ModCubes?

**A:** Follow these steps:

```bash
# For Ubuntu 18.04 (ROS Melodic)
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
sudo apt update
sudo apt install ros-melodic-desktop-full

# For Ubuntu 20.04 (ROS Noetic)
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
sudo apt update
sudo apt install ros-noetic-desktop-full
```

### Q: How do I build the RS-ModCubes workspace?

**A:** After cloning the repository:

```bash
# Create workspace
mkdir -p ~/modcube_ws/src
cd ~/modcube_ws/src

# Clone repository
git clone https://github.com/your-org/RS-ModCubes.git

# Install dependencies
cd ~/modcube_ws
rosdep install --from-paths src --ignore-src -r -y

# Build workspace
catkin_make

# Source workspace
source devel/setup.bash
```

### Q: I'm getting build errors. What should I do?

**A:** Common solutions:

1. **Check dependencies**:
   ```bash
   rosdep check --from-paths src --ignore-src
   ```

2. **Clean and rebuild**:
   ```bash
   catkin_make clean
   catkin_make
   ```

3. **Check ROS version compatibility**:
   ```bash
   rosversion -d
   ```

4. **Update package lists**:
   ```bash
   sudo apt update
   rosdep update
   ```

## Configuration

### Q: How do I configure the vehicle parameters?

**A:** Vehicle parameters are configured in YAML files located in `modcube_config/modcube_description/yaml/`. Key files include:

- `vehicle.yaml`: Basic vehicle parameters (mass, dimensions)
- `thruster_manager.yaml`: Thruster configuration
- `controller.yaml`: Control system parameters
- `sensors.yaml`: Sensor configurations

Example vehicle configuration:
```yaml
vehicle:
  mass: 50.0  # kg
  length: 1.5  # m
  width: 0.8   # m
  height: 0.6  # m
  center_of_mass: [0.0, 0.0, 0.0]
  center_of_buoyancy: [0.0, 0.0, 0.1]
```

### Q: How do I tune the PID controller?

**A:** PID parameters can be tuned in several ways:

1. **Static configuration** in `controller.yaml`:
   ```yaml
   pid_gains:
     position:
       x: {p: 2.0, i: 0.1, d: 0.5}
       y: {p: 2.0, i: 0.1, d: 0.5}
       z: {p: 3.0, i: 0.2, d: 0.8}
   ```

2. **Dynamic reconfiguration**:
   ```bash
   rosrun rqt_reconfigure rqt_reconfigure
   ```

3. **Service calls**:
   ```bash
   rosservice call /modcube/controller/set_pid_gains "gains: {...}"
   ```

### Q: How do I add a new sensor?

**A:** To add a new sensor:

1. **Add sensor configuration** to `sensors.yaml`
2. **Create sensor plugin** (if needed)
3. **Update launch files** to include sensor
4. **Modify sensor fusion** if required

Example sensor configuration:
```yaml
sensors:
  new_sensor:
    type: "custom_sensor"
    topic: "/modcube/new_sensor/data"
    frame_id: "new_sensor_link"
    rate: 10.0
    parameters:
      param1: value1
      param2: value2
```

## Hardware Integration

### Q: What hardware is supported?

**A:** RS-ModCubes supports various hardware components:

**Sensors:**
- IMU: Xsens MTi series, VectorNav VN-100/200
- DVL: Teledyne RDI, Nortek
- Depth sensors: Pressure-based depth sensors
- Cameras: USB cameras, GigE cameras
- Sonar: Forward-looking sonar, imaging sonar

**Actuators:**
- Thrusters: BlueRobotics T100/T200, VideoRay thrusters
- Servos: Standard PWM servos
- Lights: LED arrays, strobes

### Q: How do I connect an IMU?

**A:** IMU integration steps:

1. **Physical connection**: Connect IMU via USB/Serial/Ethernet
2. **Install driver**: Install manufacturer's ROS driver
3. **Configure parameters**:
   ```yaml
   imu:
     device: "/dev/ttyUSB0"
     frame_id: "imu_link"
     rate: 100
   ```
4. **Update launch file**:
   ```xml
   <node name="imu_driver" pkg="imu_driver" type="imu_node">
     <rosparam file="$(find modcube_config)/yaml/imu.yaml"/>
   </node>
   ```

### Q: How do I configure thrusters?

**A:** Thruster configuration involves:

1. **Define thruster layout** in `thruster_manager.yaml`:
   ```yaml
   thrusters:
     - id: 0
       position: [0.3, 0.2, 0.0]
       orientation: [0.0, 0.0, 0.0]
       max_thrust: 50.0  # N
     - id: 1
       position: [0.3, -0.2, 0.0]
       orientation: [0.0, 0.0, 0.0]
       max_thrust: 50.0  # N
   ```

2. **Configure thruster allocation matrix**
3. **Set up hardware interface**

### Q: My sensors are not publishing data. What should I check?

**A:** Troubleshooting steps:

1. **Check physical connections**
2. **Verify device permissions**:
   ```bash
   sudo chmod 666 /dev/ttyUSB0
   ```
3. **Check driver installation**:
   ```bash
   rospack find sensor_driver_package
   ```
4. **Monitor topics**:
   ```bash
   rostopic list | grep sensor
   rostopic echo /sensor/topic
   ```
5. **Check launch file configuration**
6. **Review log files**:
   ```bash
   roscd && cd ../log
   ```

## Simulation

### Q: How do I start the simulation?

**A:** To launch the simulation:

```bash
# Source workspace
source ~/modcube_ws/devel/setup.bash

# Launch simulation world
roslaunch modcube_sim_worlds umd.launch

# In another terminal, launch vehicle
roslaunch modcube_config system.launch
```

### Q: The simulation is running slowly. How can I improve performance?

**A:** Performance optimization tips:

1. **Reduce physics update rate**:
   ```xml
   <physics type="ode">
     <max_step_size>0.01</max_step_size>
     <real_time_factor>1.0</real_time_factor>
   </physics>
   ```

2. **Disable unnecessary sensors**
3. **Use lower resolution meshes**
4. **Reduce GUI rendering**:
   ```bash
   roslaunch modcube_sim_worlds umd.launch gui:=false
   ```

5. **Enable GPU acceleration**

### Q: How do I create a custom simulation world?

**A:** To create a custom world:

1. **Create world file** in `modcube_sim_worlds/worlds/`:
   ```xml
   <?xml version="1.0"?>
   <sdf version="1.6">
     <world name="custom_world">
       <!-- Add models, lighting, physics -->
     </world>
   </sdf>
   ```

2. **Create launch file**:
   ```xml
   <launch>
     <include file="$(find gazebo_ros)/launch/empty_world.launch">
       <arg name="world_name" value="$(find modcube_sim_worlds)/worlds/custom_world.world"/>
     </include>
   </launch>
   ```

3. **Add custom models** to `modcube_sim_worlds/models/`

### Q: How do I add obstacles to the simulation?

**A:** Add obstacles by:

1. **Creating SDF models**
2. **Adding to world file**:
   ```xml
   <include>
     <uri>model://obstacle_model</uri>
     <pose>10 5 -2 0 0 0</pose>
   </include>
   ```
3. **Using Gazebo GUI** to place models interactively

## Control and Navigation

### Q: The vehicle is not responding to commands. What should I check?

**A:** Troubleshooting steps:

1. **Check controller status**:
   ```bash
   rostopic echo /modcube/controller/status
   ```

2. **Verify command topics**:
   ```bash
   rostopic list | grep cmd
   rostopic echo /modcube/controller_command
   ```

3. **Check thruster allocation**:
   ```bash
   rostopic echo /modcube/thrusters/commands
   ```

4. **Verify coordinate frames**:
   ```bash
   rosrun tf tf_echo odom base_link
   ```

5. **Check parameter values**:
   ```bash
   rosparam get /modcube/controller/
   ```

### Q: How do I switch between control modes?

**A:** Control modes can be switched using:

1. **Service calls**:
   ```bash
   rosservice call /modcube/controller/set_mode "mode: 1"  # Position control
   rosservice call /modcube/controller/set_mode "mode: 2"  # Velocity control
   ```

2. **Command messages**:
   ```python
   cmd = ControllerCommand()
   cmd.mode = 1  # Position control
   cmd_pub.publish(cmd)
   ```

3. **Dynamic reconfigure**:
   ```bash
   rosrun rqt_reconfigure rqt_reconfigure
   ```

### Q: How do I implement custom control algorithms?

**A:** To implement custom controllers:

1. **Create controller class** inheriting from base controller
2. **Implement control logic**:
   ```python
   class CustomController(BaseController):
       def compute_control(self, state, setpoint):
           # Custom control algorithm
           return control_output
   ```
3. **Register controller** in the controller manager
4. **Configure parameters** in YAML files

## Mission Planning

### Q: How do I create a simple waypoint mission?

**A:** Create a waypoint mission:

```python
from modcube_mission import WaypointMission
from geometry_msgs.msg import Point

# Define waypoints
waypoints = [
    Point(0, 0, -2),
    Point(10, 0, -2),
    Point(10, 10, -2),
    Point(0, 10, -2)
]

# Create and execute mission
mission = WaypointMission(waypoints)
success = mission.execute()
```

### Q: How do I implement custom mission types?

**A:** Extend the base mission class:

```python
from modcube_mission import BaseMission

class CustomMission(BaseMission):
    def __init__(self, custom_params):
        super().__init__()
        self.custom_params = custom_params
    
    def execute(self):
        # Implement custom mission logic
        try:
            # Mission execution code
            self.update_state('completed')
            return True
        except Exception as e:
            self.update_state('failed')
            return False
```

### Q: How do I monitor mission progress?

**A:** Monitor missions using:

1. **Mission status topic**:
   ```bash
   rostopic echo /modcube/mission/status
   ```

2. **RViz visualization**
3. **Custom monitoring scripts**:
   ```python
   def mission_status_callback(msg):
       print(f"Mission: {msg.state}, Progress: {msg.progress}%")
   
   rospy.Subscriber('/modcube/mission/status', MissionStatus, mission_status_callback)
   ```

## Troubleshooting

### Q: ROS nodes are not communicating. What should I check?

**A:** Communication troubleshooting:

1. **Check ROS master**:
   ```bash
   echo $ROS_MASTER_URI
   rosnode list
   ```

2. **Verify network configuration**:
   ```bash
   echo $ROS_IP
   echo $ROS_HOSTNAME
   ```

3. **Check topic connections**:
   ```bash
   rostopic info /topic_name
   rosnode info /node_name
   ```

4. **Monitor network traffic**:
   ```bash
   rostopic bw /topic_name
   rostopic hz /topic_name
   ```

### Q: The system crashes frequently. How do I debug?

**A:** Debugging crashes:

1. **Check log files**:
   ```bash
   roscd && cd ../log
   tail -f latest/rosout.log
   ```

2. **Use debugging tools**:
   ```bash
   gdb --args rosrun package_name node_name
   valgrind rosrun package_name node_name
   ```

3. **Monitor system resources**:
   ```bash
   htop
   iotop
   ```

4. **Check for memory leaks**
5. **Review error messages** in terminal output

### Q: Sensor data is noisy or unreliable. What can I do?

**A:** Improve sensor data quality:

1. **Check physical connections**
2. **Implement filtering**:
   ```python
   from scipy.signal import butter, filtfilt
   
   def low_pass_filter(data, cutoff, fs):
       nyquist = 0.5 * fs
       normal_cutoff = cutoff / nyquist
       b, a = butter(6, normal_cutoff, btype='low', analog=False)
       return filtfilt(b, a, data)
   ```

3. **Calibrate sensors**
4. **Adjust sensor fusion parameters**
5. **Check environmental conditions**

## Performance and Optimization

### Q: How can I improve system performance?

**A:** Performance optimization strategies:

1. **Optimize control loop rates**:
   ```yaml
   control_frequency: 50  # Hz
   estimation_frequency: 100  # Hz
   ```

2. **Use efficient data structures**
3. **Minimize memory allocations**
4. **Profile code performance**:
   ```bash
   perf record rosrun package_name node_name
   perf report
   ```

5. **Optimize ROS communication**:
   - Use appropriate queue sizes
   - Minimize message size
   - Use efficient serialization

### Q: How do I monitor system performance?

**A:** Performance monitoring tools:

1. **ROS diagnostics**:
   ```bash
   rosrun rqt_robot_monitor rqt_robot_monitor
   ```

2. **System monitoring**:
   ```bash
   rosrun rqt_top rqt_top
   ```

3. **Custom performance metrics**:
   ```python
   import time
   
   start_time = time.time()
   # ... code to measure ...
   execution_time = time.time() - start_time
   ```

4. **Network monitoring**:
   ```bash
   rostopic bw /topic_name
   rostopic delay /topic_name
   ```

## Development and Customization

### Q: How do I contribute to RS-ModCubes development?

**A:** Contributing guidelines:

1. **Fork the repository** on GitHub
2. **Create feature branch**:
   ```bash
   git checkout -b feature/new-feature
   ```
3. **Follow coding standards**:
   - Use consistent naming conventions
   - Add documentation and comments
   - Write unit tests
4. **Submit pull request** with detailed description
5. **Respond to review feedback**

### Q: How do I create custom plugins?

**A:** Plugin development:

1. **Create plugin class**:
   ```cpp
   #include <pluginlib/class_list_macros.h>
   #include <modcube_common/base_plugin.h>
   
   class CustomPlugin : public modcube_common::BasePlugin {
   public:
       void initialize() override {
           // Plugin initialization
       }
   };
   
   PLUGINLIB_EXPORT_CLASS(CustomPlugin, modcube_common::BasePlugin)
   ```

2. **Create plugin description**:
   ```xml
   <library path="lib/libcustom_plugin">
     <class name="custom_plugin/CustomPlugin" type="CustomPlugin" base_class_type="modcube_common::BasePlugin">
       <description>Custom plugin description</description>
     </class>
   </library>
   ```

3. **Register plugin** in package.xml

### Q: How do I add support for new hardware?

**A:** Hardware integration steps:

1. **Create hardware interface**:
   ```cpp
   class NewHardwareInterface : public hardware_interface::RobotHW {
       // Implement hardware interface
   };
   ```

2. **Develop ROS driver**
3. **Create configuration files**
4. **Add launch file integration**
5. **Write documentation and examples**
6. **Test thoroughly**

### Q: Where can I find more help?

**A:** Additional resources:

1. **Documentation**: Check the [official documentation](https://your-org.github.io/RS-ModCubes/)
2. **GitHub Issues**: Report bugs and request features
3. **Community Forum**: Join discussions with other users
4. **Email Support**: Contact the development team
5. **Video Tutorials**: Watch demonstration videos
6. **Academic Papers**: Read research publications

---

## Still Have Questions?

If you can't find the answer to your question here, please:

1. **Search the documentation** for related topics
2. **Check GitHub issues** for similar problems
3. **Ask on the community forum**
4. **Contact the development team**

We're here to help you succeed with RS-ModCubes!