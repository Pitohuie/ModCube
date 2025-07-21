---
layout: default
title: Contributing
nav_order: 8
has_children: false
permalink: /contributing/
---

# Contributing to RS-ModCubes

We welcome contributions to the RS-ModCubes project! This guide will help you understand how to contribute effectively to our open-source underwater robotics platform.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Environment](#development-environment)
3. [Contribution Types](#contribution-types)
4. [Development Workflow](#development-workflow)
5. [Coding Standards](#coding-standards)
6. [Testing Guidelines](#testing-guidelines)
7. [Documentation](#documentation)
8. [Pull Request Process](#pull-request-process)
9. [Community Guidelines](#community-guidelines)
10. [Recognition](#recognition)

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- **Git** installed and configured
- **GitHub account** with SSH keys set up
- **ROS development environment** (Melodic or Noetic)
- **Basic understanding** of underwater robotics concepts
- **Familiarity** with C++ and Python programming

### First Steps

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone git@github.com:YOUR_USERNAME/RS-ModCubes.git
   cd RS-ModCubes
   ```

3. **Add upstream remote**:
   ```bash
   git remote add upstream git@github.com:ORIGINAL_OWNER/RS-ModCubes.git
   ```

4. **Set up development environment** following the [Installation Guide](installation.md)

## Development Environment

### Required Tools

```bash
# Install development tools
sudo apt install -y \
    build-essential \
    cmake \
    git \
    python3-pip \
    clang-format \
    cppcheck \
    valgrind

# Install Python tools
pip3 install --user \
    pre-commit \
    black \
    flake8 \
    pytest \
    sphinx
```

### IDE Setup

Recommended IDEs with ROS support:

- **VS Code** with ROS extension
- **CLion** with ROS plugin
- **Qt Creator** with ROS plugin
- **Vim/Neovim** with appropriate plugins

### Pre-commit Hooks

Set up pre-commit hooks to ensure code quality:

```bash
# Install pre-commit
pip3 install pre-commit

# Install hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## Contribution Types

We welcome various types of contributions:

### Code Contributions

- **Bug fixes**: Fix existing issues
- **New features**: Add functionality
- **Performance improvements**: Optimize existing code
- **Hardware support**: Add new sensor/actuator drivers
- **Algorithm implementations**: Add new control or estimation algorithms

### Documentation

- **API documentation**: Document functions and classes
- **Tutorials**: Create step-by-step guides
- **Examples**: Provide usage examples
- **Wiki articles**: Write explanatory content
- **Translation**: Translate documentation to other languages

### Testing

- **Unit tests**: Test individual components
- **Integration tests**: Test component interactions
- **Hardware tests**: Test with real hardware
- **Performance benchmarks**: Measure system performance

### Community

- **Issue reporting**: Report bugs and suggest features
- **Issue triaging**: Help categorize and prioritize issues
- **Code review**: Review pull requests
- **Community support**: Help other users

## Development Workflow

### Branch Strategy

We use a Git flow-based branching strategy:

- **main**: Stable release branch
- **develop**: Integration branch for new features
- **feature/**: Feature development branches
- **bugfix/**: Bug fix branches
- **hotfix/**: Critical fixes for main branch

### Creating a Feature Branch

```bash
# Update your fork
git checkout develop
git pull upstream develop

# Create feature branch
git checkout -b feature/your-feature-name

# Make your changes
# ...

# Commit changes
git add .
git commit -m "Add your feature description"

# Push to your fork
git push origin feature/your-feature-name
```

### Keeping Your Branch Updated

```bash
# Fetch latest changes
git fetch upstream

# Rebase your branch
git rebase upstream/develop

# Force push if needed (be careful!)
git push --force-with-lease origin feature/your-feature-name
```

## Coding Standards

### C++ Guidelines

We follow the [ROS C++ Style Guide](http://wiki.ros.org/CppStyleGuide) with some modifications:

#### Naming Conventions

```cpp
// Classes: PascalCase
class VehicleController {
public:
    // Methods: camelCase
    void updateControl();
    
    // Public members: camelCase
    double maxThrust;
    
private:
    // Private members: camelCase with trailing underscore
    double current_thrust_;
    
    // Constants: UPPER_CASE
    static const double MAX_VELOCITY;
};

// Functions: camelCase
void calculateTrajectory();

// Variables: snake_case
double target_depth = -5.0;

// Namespaces: lowercase
namespace modcube_control {
    // ...
}
```

#### Code Formatting

Use `clang-format` with our configuration:

```bash
# Format single file
clang-format -i src/controller.cpp

# Format all C++ files
find . -name '*.cpp' -o -name '*.h' | xargs clang-format -i
```

#### Header Structure

```cpp
/**
 * @file vehicle_controller.h
 * @brief Vehicle control system implementation
 * @author Your Name
 * @date 2024-01-01
 */

#ifndef MODCUBE_CONTROL_VEHICLE_CONTROLLER_H
#define MODCUBE_CONTROL_VEHICLE_CONTROLLER_H

#include <ros/ros.h>
#include <geometry_msgs/Twist.h>

namespace modcube_control {

/**
 * @brief Vehicle controller class
 * 
 * Detailed description of the class functionality.
 */
class VehicleController {
public:
    /**
     * @brief Constructor
     * @param nh ROS node handle
     */
    explicit VehicleController(ros::NodeHandle& nh);
    
    /**
     * @brief Update control output
     * @param target_twist Desired velocity
     * @return Control success status
     */
    bool updateControl(const geometry_msgs::Twist& target_twist);
    
private:
    ros::NodeHandle nh_;
    ros::Publisher cmd_pub_;
};

}  // namespace modcube_control

#endif  // MODCUBE_CONTROL_VEHICLE_CONTROLLER_H
```

### Python Guidelines

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some ROS-specific conventions:

#### Code Style

```python
#!/usr/bin/env python3
"""
Vehicle Controller Module

This module implements the main vehicle control system.

Author: Your Name
Date: 2024-01-01
"""

import rospy
import numpy as np
from geometry_msgs.msg import Twist
from modcube_msgs.msg import ControllerCommand


class VehicleController:
    """Vehicle controller class.
    
    This class implements PID control for underwater vehicles.
    
    Attributes:
        node_name (str): ROS node name
        control_rate (float): Control loop frequency in Hz
    """
    
    def __init__(self, node_name='vehicle_controller'):
        """Initialize the vehicle controller.
        
        Args:
            node_name (str): Name of the ROS node
        """
        self.node_name = node_name
        self.control_rate = 50.0  # Hz
        
        # Initialize ROS
        rospy.init_node(self.node_name)
        
        # Publishers and subscribers
        self.cmd_pub = rospy.Publisher(
            '/modcube/cmd_vel', Twist, queue_size=10
        )
        
    def update_control(self, target_twist):
        """Update control output.
        
        Args:
            target_twist (Twist): Desired velocity
            
        Returns:
            bool: True if control update successful
        """
        try:
            # Control logic here
            self.cmd_pub.publish(target_twist)
            return True
        except Exception as e:
            rospy.logerr(f"Control update failed: {e}")
            return False


def main():
    """Main function."""
    try:
        controller = VehicleController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
```

#### Code Formatting

Use `black` for Python code formatting:

```bash
# Format single file
black src/controller.py

# Format all Python files
black .

# Check formatting
black --check .
```

### ROS-Specific Guidelines

#### Package Structure

```
modcube_package/
├── CMakeLists.txt
├── package.xml
├── README.md
├── include/
│   └── modcube_package/
│       └── header.h
├── src/
│   ├── cpp_source.cpp
│   └── python_script.py
├── scripts/
│   └── executable_script.py
├── launch/
│   └── package.launch
├── config/
│   └── parameters.yaml
├── msg/
│   └── CustomMessage.msg
├── srv/
│   └── CustomService.srv
└── test/
    ├── test_cpp.cpp
    └── test_python.py
```

#### Launch Files

```xml
<?xml version="1.0"?>
<launch>
    <!-- Arguments -->
    <arg name="vehicle_name" default="modcube"/>
    <arg name="debug" default="false"/>
    
    <!-- Parameters -->
    <rosparam file="$(find modcube_config)/config/controller.yaml" command="load"/>
    
    <!-- Nodes -->
    <node name="vehicle_controller" 
          pkg="modcube_control" 
          type="vehicle_controller_node" 
          output="screen"
          if="$(arg debug)">
        <param name="vehicle_name" value="$(arg vehicle_name)"/>
    </node>
    
    <!-- Include other launch files -->
    <include file="$(find modcube_sensors)/launch/sensors.launch">
        <arg name="vehicle_name" value="$(arg vehicle_name)"/>
    </include>
</launch>
```

## Testing Guidelines

### Unit Testing

#### C++ Tests (Google Test)

```cpp
#include <gtest/gtest.h>
#include <modcube_control/vehicle_controller.h>

class VehicleControllerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Test setup
    }
    
    void TearDown() override {
        // Test cleanup
    }
    
    modcube_control::VehicleController controller_;
};

TEST_F(VehicleControllerTest, InitializationTest) {
    // Test controller initialization
    EXPECT_TRUE(controller_.isInitialized());
}

TEST_F(VehicleControllerTest, ControlUpdateTest) {
    geometry_msgs::Twist target;
    target.linear.x = 1.0;
    
    EXPECT_TRUE(controller_.updateControl(target));
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

#### Python Tests (pytest)

```python
import pytest
import rospy
from modcube_control.vehicle_controller import VehicleController


class TestVehicleController:
    """Test cases for VehicleController class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.controller = VehicleController()
    
    def test_initialization(self):
        """Test controller initialization."""
        assert self.controller.node_name == 'vehicle_controller'
        assert self.controller.control_rate == 50.0
    
    def test_control_update(self):
        """Test control update functionality."""
        from geometry_msgs.msg import Twist
        
        target = Twist()
        target.linear.x = 1.0
        
        result = self.controller.update_control(target)
        assert result is True
    
    @pytest.mark.parametrize("velocity", [0.0, 1.0, -1.0, 2.5])
    def test_velocity_range(self, velocity):
        """Test different velocity values."""
        from geometry_msgs.msg import Twist
        
        target = Twist()
        target.linear.x = velocity
        
        result = self.controller.update_control(target)
        assert result is True
```

### Integration Testing

```python
#!/usr/bin/env python3
"""
Integration test for vehicle control system.
"""

import unittest
import rospy
import rostest
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


class VehicleControlIntegrationTest(unittest.TestCase):
    """Integration test for vehicle control."""
    
    def setUp(self):
        """Set up test environment."""
        rospy.init_node('vehicle_control_test')
        
        self.cmd_pub = rospy.Publisher('/modcube/cmd_vel', Twist, queue_size=1)
        self.odom_sub = rospy.Subscriber('/modcube/odom', Odometry, self.odom_callback)
        
        self.latest_odom = None
        
        # Wait for connections
        rospy.sleep(1.0)
    
    def odom_callback(self, msg):
        """Handle odometry messages."""
        self.latest_odom = msg
    
    def test_forward_motion(self):
        """Test forward motion command."""
        # Send forward command
        cmd = Twist()
        cmd.linear.x = 1.0
        self.cmd_pub.publish(cmd)
        
        # Wait for response
        timeout = rospy.Time.now() + rospy.Duration(5.0)
        while self.latest_odom is None and rospy.Time.now() < timeout:
            rospy.sleep(0.1)
        
        # Check that vehicle is moving forward
        self.assertIsNotNone(self.latest_odom)
        self.assertGreater(self.latest_odom.twist.twist.linear.x, 0.1)


if __name__ == '__main__':
    rostest.rosrun('modcube_control', 'vehicle_control_integration_test', 
                   VehicleControlIntegrationTest)
```

### Running Tests

```bash
# Run C++ tests
catkin_make run_tests_modcube_control

# Run Python tests
python -m pytest src/modcube_control/test/

# Run ROS tests
rostest modcube_control vehicle_control_test.test

# Run all tests
catkin_make run_tests
```

## Documentation

### Code Documentation

#### Doxygen for C++

```cpp
/**
 * @brief Calculate PID control output
 * 
 * This function computes the PID control output based on the error
 * between the setpoint and current value.
 * 
 * @param setpoint Desired value
 * @param current_value Current measured value
 * @param dt Time step in seconds
 * @return Control output value
 * 
 * @note This function assumes that the PID gains have been properly set
 * @warning Large time steps may cause instability
 * 
 * @see setPIDGains()
 * @since Version 1.0
 */
double calculatePID(double setpoint, double current_value, double dt);
```

#### Docstrings for Python

```python
def calculate_pid(self, setpoint, current_value, dt):
    """Calculate PID control output.
    
    This function computes the PID control output based on the error
    between the setpoint and current value.
    
    Args:
        setpoint (float): Desired value
        current_value (float): Current measured value
        dt (float): Time step in seconds
        
    Returns:
        float: Control output value
        
    Raises:
        ValueError: If dt is negative or zero
        
    Note:
        This function assumes that the PID gains have been properly set.
        
    Warning:
        Large time steps may cause instability.
        
    Example:
        >>> controller = PIDController()
        >>> output = controller.calculate_pid(1.0, 0.8, 0.02)
        >>> print(f"Control output: {output}")
    """
    if dt <= 0:
        raise ValueError("Time step must be positive")
    
    # PID calculation logic
    pass
```

### README Files

Each package should have a comprehensive README:

```markdown
# ModCube Control

This package provides control algorithms for the ModCube underwater vehicle.

## Overview

The control system implements various control strategies including:
- PID control for position and velocity
- Adaptive control for changing conditions
- Formation control for multi-vehicle operations

## Installation

```bash
# Clone repository
git clone https://github.com/your-org/RS-ModCubes.git

# Build package
catkin_make
```

## Usage

### Basic Control

```bash
# Launch control system
roslaunch modcube_control vehicle_control.launch

# Send velocity command
rostopic pub /modcube/cmd_vel geometry_msgs/Twist "linear: {x: 1.0}"
```

### Configuration

Edit `config/controller.yaml` to adjust control parameters:

```yaml
controller:
  pid_gains:
    position:
      x: {p: 2.0, i: 0.1, d: 0.5}
```

## API Reference

### Classes

- `VehicleController`: Main control class
- `PIDController`: PID control implementation

### Topics

- `/modcube/cmd_vel` (geometry_msgs/Twist): Velocity commands
- `/modcube/odom` (nav_msgs/Odometry): Vehicle odometry

### Services

- `/modcube/set_pid_gains`: Update PID parameters

## Testing

```bash
# Run unit tests
catkin_make run_tests_modcube_control

# Run integration tests
rostest modcube_control integration_test.test
```

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](../LICENSE) for details.
```

## Pull Request Process

### Before Submitting

1. **Test your changes** thoroughly
2. **Update documentation** as needed
3. **Follow coding standards**
4. **Write descriptive commit messages**
5. **Rebase on latest develop branch**

### Pull Request Template

Use this template for your pull requests:

```markdown
## Description

Brief description of the changes made.

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Hardware testing completed (if applicable)

## Checklist

- [ ] My code follows the style guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes

## Screenshots (if applicable)

Add screenshots to help explain your changes.

## Additional Notes

Any additional information that reviewers should know.
```

### Review Process

1. **Automated checks** must pass (CI/CD)
2. **Code review** by at least one maintainer
3. **Testing** in simulation and hardware (if applicable)
4. **Documentation review**
5. **Final approval** and merge

## Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:

- **Be respectful** and considerate
- **Be collaborative** and helpful
- **Be patient** with newcomers
- **Focus on constructive feedback**
- **Respect different viewpoints**

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Slack/Discord**: Real-time communication (if available)
- **Mailing List**: Announcements and important updates

### Getting Help

If you need help:

1. **Check documentation** first
2. **Search existing issues** on GitHub
3. **Ask in discussions** or community channels
4. **Create a new issue** if needed

## Recognition

We value all contributions and recognize contributors through:

### Contributors List

All contributors are listed in:
- `CONTRIBUTORS.md` file
- GitHub contributors page
- Release notes

### Contribution Types

We recognize various contribution types:
- 💻 Code
- 📖 Documentation
- 🐛 Bug reports
- 💡 Ideas
- 🤔 Answering questions
- ⚠️ Tests
- 🔧 Tools
- 🌍 Translation

### Special Recognition

- **Maintainer status** for consistent, high-quality contributions
- **Featured contributions** in release announcements
- **Conference presentations** opportunities
- **Academic collaboration** for research contributions

## Development Roadmap

### Current Priorities

1. **Performance optimization**
2. **Hardware driver expansion**
3. **Advanced control algorithms**
4. **Multi-vehicle coordination**
5. **Machine learning integration**

### How to Get Involved

- **Check the roadmap** for current priorities
- **Look for "good first issue" labels**
- **Join planning discussions**
- **Propose new features**

## Resources

### Learning Materials

- [ROS Tutorials](http://wiki.ros.org/ROS/Tutorials)
- [Underwater Robotics Handbook](https://example.com)
- [Control Systems Theory](https://example.com)
- [Git Workflow Guide](https://example.com)

### Tools and References

- [ROS Style Guide](http://wiki.ros.org/StyleGuide)
- [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- [PEP 8 Python Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [Conventional Commits](https://www.conventionalcommits.org/)

---

Thank you for contributing to RS-ModCubes! Your efforts help advance underwater robotics research and applications worldwide.