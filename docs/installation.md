---
layout: default
title: Installation Guide
nav_order: 6
has_children: false
permalink: /installation/
---

# Installation Guide

This guide provides step-by-step instructions for installing and configuring the RS-ModCubes system on your development environment.

## Prerequisites

### System Requirements

**Operating System**:
- Ubuntu 18.04 LTS (Bionic Beaver) - Recommended
- Ubuntu 20.04 LTS (Focal Fossa) - Supported
- Other Linux distributions may work but are not officially supported

**Hardware Requirements**:
- **CPU**: Intel i5 or AMD equivalent (minimum), Intel i7 or better (recommended)
- **RAM**: 8 GB (minimum), 16 GB or more (recommended)
- **Storage**: 20 GB free space (minimum), SSD recommended
- **Graphics**: Dedicated GPU recommended for Gazebo simulation

### Software Dependencies

**ROS Distribution**:
- ROS Melodic (Ubuntu 18.04)
- ROS Noetic (Ubuntu 20.04)

**Required Packages**:
- Python 2.7 or 3.x
- Git
- CMake 3.10+
- GCC/G++ compiler
- Gazebo 9.0+

## Installation Steps

### Step 1: Install ROS

#### For Ubuntu 18.04 (ROS Melodic)

```bash
# Setup sources.list
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

# Setup keys
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# Update package index
sudo apt update

# Install ROS Melodic Desktop Full
sudo apt install ros-melodic-desktop-full

# Initialize rosdep
sudo rosdep init
rosdep update

# Setup environment
echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
source ~/.bashrc

# Install additional tools
sudo apt install python-rosinstall python-rosinstall-generator python-wstool build-essential
```

#### For Ubuntu 20.04 (ROS Noetic)

```bash
# Setup sources.list
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

# Setup keys
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# Update package index
sudo apt update

# Install ROS Noetic Desktop Full
sudo apt install ros-noetic-desktop-full

# Initialize rosdep
sudo rosdep init
rosdep update

# Setup environment
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc

# Install additional tools
sudo apt install python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
```

### Step 2: Install Additional Dependencies

```bash
# Install Gazebo (if not already installed with ROS)
sudo apt install gazebo9 libgazebo9-dev

# Install additional ROS packages
sudo apt install ros-$ROS_DISTRO-gazebo-ros-pkgs ros-$ROS_DISTRO-gazebo-ros-control
sudo apt install ros-$ROS_DISTRO-joint-state-publisher ros-$ROS_DISTRO-robot-state-publisher
sudo apt install ros-$ROS_DISTRO-xacro ros-$ROS_DISTRO-tf2-tools
sudo apt install ros-$ROS_DISTRO-rqt ros-$ROS_DISTRO-rqt-common-plugins

# Install Python dependencies
sudo apt install python3-pip
pip3 install numpy scipy matplotlib

# Install Git LFS (for large files)
sudo apt install git-lfs
git lfs install
```

### Step 3: Create Workspace

```bash
# Create catkin workspace
mkdir -p ~/modcube_ws/src
cd ~/modcube_ws/
catkin_make

# Setup workspace environment
echo "source ~/modcube_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Step 4: Clone Repository

```bash
# Navigate to workspace source directory
cd ~/modcube_ws/src

# Clone the ModCube repository
git clone https://github.com/jiaxi-zheng/ModCube.git

# Navigate to the cloned repository
cd ModCube

# Initialize and update submodules
git submodule init
git submodule update
```

### Step 5: Install Package Dependencies

```bash
# Navigate to workspace root
cd ~/modcube_ws

# Install dependencies using rosdep
rosdep install --from-paths src --ignore-src -r -y

# Install additional Python dependencies
cd src/ModCube
pip3 install -r requirements.txt  # If requirements.txt exists
```

### Step 6: Build the Workspace

```bash
# Navigate to workspace root
cd ~/modcube_ws

# Build all packages
catkin_make

# Alternative: Build with specific number of jobs
catkin_make -j4

# Source the workspace
source devel/setup.bash
```

## Configuration

### Environment Setup

Add the following to your `~/.bashrc` file:

```bash
# ROS Environment
source /opt/ros/$ROS_DISTRO/setup.bash
source ~/modcube_ws/devel/setup.bash

# Gazebo Environment
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/modcube_ws/src/ModCube/packages/modcube_sim_worlds/models
export GAZEBO_RESOURCE_PATH=$GAZEBO_RESOURCE_PATH:~/modcube_ws/src/ModCube/packages/modcube_sim_worlds

# ModCube specific environment variables
export MODCUBE_CONFIG_PATH=~/modcube_ws/src/ModCube/packages/modcube_config
export MODCUBE_WORKSPACE=~/modcube_ws
```

### Network Configuration

For multi-machine setups:

```bash
# Set ROS Master URI (replace with actual master IP)
export ROS_MASTER_URI=http://192.168.1.100:11311

# Set ROS IP (replace with your machine's IP)
export ROS_IP=192.168.1.101
```

## Verification

### Test Basic Installation

```bash
# Test ROS installation
roscore &
rostopic list
killall roscore

# Test Gazebo installation
gazebo --version

# Test ModCube packages
rospack find modcube_common
rospack find modcube_mission
rospack find modcube_vehicle
```

### Run Simple Simulation

```bash
# Launch basic simulation
roslaunch modcube_mission system.launch model_name:=modcube simulated:=true

# In another terminal, check running nodes
rosnode list

# Check topics
rostopic list
```

## Hardware Setup (Optional)

### IMU Configuration

For Xsens IMU systems:

```bash
# Install Xsens drivers (follow manufacturer instructions)
# Configure USB permissions
sudo usermod -a -G dialout $USER

# Create udev rules for IMU
sudo nano /etc/udev/rules.d/99-xsens.rules
# Add: SUBSYSTEM=="usb", ATTRS{idVendor}=="2639", MODE="0666"

# Reload udev rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### DVL Configuration

For Teledyne DVL systems:

```bash
# Configure serial port permissions
sudo usermod -a -G dialout $USER

# Create udev rules for DVL
sudo nano /etc/udev/rules.d/99-teledyne.rules
# Add appropriate rules based on your DVL model

# Test serial communication
sudo apt install minicom
minicom -D /dev/ttyUSB0 -b 115200
```

### Thruster Configuration

For Pololu Maestro PWM controllers:

```bash
# Install Pololu software
wget https://www.pololu.com/file/0J315/pololu-usb-sdk-120404.tar.gz
tar -xzf pololu-usb-sdk-120404.tar.gz
cd pololu-usb-sdk-120404
make
sudo make install

# Configure USB permissions
sudo nano /etc/udev/rules.d/99-pololu.rules
# Add: SUBSYSTEM=="usb", ATTRS{idVendor}=="1ffb", MODE="0666"

# Reload udev rules
sudo udevadm control --reload-rules
```

## Troubleshooting

### Common Issues

**Issue**: `catkin_make` fails with dependency errors
```bash
# Solution: Install missing dependencies
rosdep install --from-paths src --ignore-src -r -y
```

**Issue**: Gazebo crashes or runs slowly
```bash
# Solution: Check graphics drivers and reduce simulation complexity
export LIBGL_ALWAYS_SOFTWARE=1  # For software rendering
```

**Issue**: Cannot find ModCube packages
```bash
# Solution: Source the workspace
source ~/modcube_ws/devel/setup.bash
```

**Issue**: Permission denied for hardware devices
```bash
# Solution: Add user to appropriate groups
sudo usermod -a -G dialout $USER
sudo usermod -a -G plugdev $USER
# Log out and log back in
```

### Debug Commands

```bash
# Check ROS environment
env | grep ROS

# Check package dependencies
rospack depends modcube_common

# Check for missing dependencies
rosdep check --from-paths src --ignore-src

# Verbose catkin build
catkin_make --verbose

# Check Gazebo plugins
gazebo --verbose
```

### Log Files

Important log locations:
- ROS logs: `~/.ros/log/`
- Gazebo logs: `~/.gazebo/`
- System logs: `/var/log/syslog`

## Performance Optimization

### Build Optimization

```bash
# Use multiple cores for building
catkin_make -j$(nproc)

# Release build for better performance
catkin_make -DCMAKE_BUILD_TYPE=Release
```

### Runtime Optimization

```bash
# Increase ROS message buffer sizes
export ROSCONSOLE_CONFIG_FILE=~/modcube_ws/src/ModCube/config/rosconsole.conf

# Optimize Gazebo performance
export GAZEBO_MASTER_URI=http://localhost:11345
export GAZEBO_MODEL_DATABASE_URI=http://gazebosim.org/models
```

## Next Steps

After successful installation:

1. **Read the [Tutorials](tutorials.md)** to learn basic system usage
2. **Explore [Examples](examples.md)** for practical applications
3. **Review [API Documentation](api.md)** for development
4. **Join the community** for support and contributions

## Support

If you encounter issues during installation:

1. Check the [Troubleshooting](#troubleshooting) section
2. Search existing [GitHub Issues](https://github.com/jiaxi-zheng/ModCube/issues)
3. Create a new issue with detailed error information
4. Contact the development team

## Contributing

Interested in contributing? See our [Contributing Guidelines](https://github.com/jiaxi-zheng/ModCube/blob/main/CONTRIBUTING.md) for development setup and contribution process.