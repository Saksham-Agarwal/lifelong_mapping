# AMCL Confidence Calculation

## Overview

This document describes the AMCL confidence estimation node used in the lifelong mapping project.

The purpose of this node is to convert the localization uncertainty reported by AMCL into a single confidence score that can be monitored and used by other modules.

The node subscribes to the `amcl_pose` topic, extracts relevant covariance values from the AMCL pose estimate, computes a combined uncertainty measure, and publishes a confidence value between 0 and 1.

---

# Input Topic

## `/amcl_pose`

**Message Type**

```text
geometry_msgs/PoseWithCovarianceStamped
```

AMCL publishes the robot pose estimate together with a 6×6 covariance matrix representing uncertainty in position and orientation.

---

# Covariance Matrix Definition


For a planar mobile robot, only the following values are used:

| Covariance Index | Meaning                        |
| ---------------- | ------------------------------ |
| 0                | Uncertainty in x position      |
| 7                | Uncertainty in y position      |
| 35               | Uncertainty in yaw orientation |

---

# Parameter Used

## `angle_relevance`

**Default Value**

```yaml
angle_relevance: 0.5
```

### Purpose

This parameter determines how much orientation uncertainty contributes to the overall localization uncertainty.


---

# Confidence Calculation


```text
pose_uncertainty =
covariance[0] + covariance[7]
```

This combines the uncertainty in the x and y directions.

---


```text
angle_uncertainty =
covariance[35]
```

This represents the uncertainty in the robot's yaw angle.

---

## Step 3: Total Uncertainty

```text
total_uncertainty =
pose_uncertainty +
(angle_relevance × angle_uncertainty)
```



---

## Step 4: Confidence Score

The confidence value is calculated using an exponential decay function:

```text
confidence =
exp(-total_uncertainty)
```

### Properties

* Confidence ranges between 0 and 1.
* Low uncertainty results in confidence values close to 1.
* High uncertainty results in confidence values close to 0.
* The decay is smooth and continuous.



---

# Output Topic

## `/amcl_confidence`

**Message Type**

```text
std_msgs/Float32
```

Published value:

```text
0.0 ≤ confidence ≤ 1.0
```


---
