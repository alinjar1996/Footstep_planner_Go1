#!/usr/bin/env python3
#load go1 robot in gazebo before running this module
import rclpy
from geometry_msgs.msg import Point
from geometry_msgs.msg import Polygon
from visualization_msgs.msg import Marker
import numpy as np
import math
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import TransformStamped
import xml.etree.ElementTree as ET
import math
import tf2_ros
#import data2 as dt

class LegData:
    def __init__(self):
        self.name = ""  # name of the leg
        self.id = 0  # unique id for the leg
        self.leg_theta = 0.0  # Leg trajectory cycle parameter
        self.true_is_stance = False  # If leg is in stance or swing

        self.leg_frame = np.zeros(3)  # leg frame location wrt base frame
        self.foot_pos = np.zeros(3)  # foot position in leg frame
        self.prev_foot_pos = np.zeros(3)  # previous foot position in leg frame
        self.shifts = np.zeros(3)  # Translational shifts in leg frame
        self.motor_angles = np.zeros(3)  # Joint angles
        self.prev_motor_angles = np.zeros(3)  # Previous joint angles


class RobotData:
    def __init__(self):
        #self.link_lengths = np.array([0.0782, 0.2968, 0.0965])  # Leg link lengths of robot
        #self.robot_body_dimensions = np.array([0.3649, 0.076])  # Torso length and width
        
        self.link_lengths = np.array([0.08, 0.23, 0.24])  # Leg link lengths of robot
        self.robot_body_dimensions = np.array([0.4, 0.15])  # Torso length and width taken from (/home/alinjar/go1_ros2_ws/install/go1_description/share/go1_description/config/robot_params.yaml)


        self.front_left = LegData()
        self.front_right = LegData()
        self.back_left = LegData()
        self.back_right = LegData()

        self.legs = np.array([self.front_left, self.front_right, self.back_left, self.back_right])

        self.front_left.name = "fl"
        self.front_right.name = "fr"
        self.back_left.name = "bl"
        self.back_right.name = "br"

        self.front_left.id = 0
        self.front_right.id = 1
        self.back_left.id = 2
        self.back_right.id = 3

        self.front_left.foot_pos.fill(0.0)
        self.front_right.foot_pos.fill(0.0)
        self.back_left.foot_pos.fill(0.0)
        self.back_right.foot_pos.fill(0.0)

        self.front_left.prev_foot_pos.fill(0.0)
        self.front_right.prev_foot_pos.fill(0.0)
        self.back_left.prev_foot_pos.fill(0.0)
        self.back_right.prev_foot_pos.fill(0.0)

        self.front_left.shifts.fill(0.0)
        self.front_right.shifts.fill(0.0)
        self.back_left.shifts.fill(0.0)
        self.back_right.shifts.fill(0.0)

        self.front_left.motor_angles.fill(0.0)
        self.front_right.motor_angles.fill(0.0)
        self.back_left.motor_angles.fill(0.0)
        self.back_right.motor_angles.fill(0.0)

        self.front_left.prev_motor_angles.fill(0.0)
        self.front_right.prev_motor_angles.fill(0.0)
        self.back_left.prev_motor_angles.fill(0.0)
        self.back_right.prev_motor_angles.fill(0.0)

        self.front_left.leg_frame = np.array([self.robot_body_dimensions[0] / 2.0, self.robot_body_dimensions[1] / 2.0, 0.0])
        self.front_right.leg_frame = np.array([self.robot_body_dimensions[0] / 2.0, -self.robot_body_dimensions[1] / 2.0, 0.0])
        self.back_left.leg_frame = np.array([-self.robot_body_dimensions[0] / 2.0, self.robot_body_dimensions[1] / 2.0, 0.0])
        self.back_right.leg_frame = np.array([-self.robot_body_dimensions[0] / 2.0, -self.robot_body_dimensions[1] / 2.0, 0.0])

    #def initialize(self):
      

        #print("self.back_right.leg_frame",self.back_right.leg_frame)

    def set_robot_dimensions(self, link_lengths, robot_body_dimensions):
        assert not np.isnan(link_lengths).any(), "Invalid! set_robot_dimensions input Vector, link_lengths has NaN element"
        assert not np.isnan(robot_body_dimensions).any(), "Invalid! set_robot_dimensions input Vector, robot_body_dimensions has NaN element"

        self.robot_body_dimensions = robot_body_dimensions
        self.link_lengths = link_lengths

        self.front_left.leg_frame = np.array([robot_body_dimensions[0] / 2.0, robot_body_dimensions[1] / 2.0, 0.0])
        self.front_right.leg_frame = np.array([robot_body_dimensions[0] / 2.0, -robot_body_dimensions[1] / 2.0, 0.0])
        self.back_left.leg_frame = np.array([-robot_body_dimensions[0] / 2.0, robot_body_dimensions[1] / 2.0, 0.0])
        self.back_right.leg_frame = np.array([-robot_body_dimensions[0] / 2.0, -robot_body_dimensions[1] / 2.0, 0.0])

    def set_robot_link_lengths(self, link_lengths):
        assert not np.isnan(link_lengths).any(), "Invalid! set_robot_link_lengths input Vector, link_lengths has NaN element"
        self.link_lengths = link_lengths

    def get_robot_dimensions(self):
        return self.link_lengths, self.robot_body_dimensions

    def initialize_shift(self, shifts):
        assert not np.isnan(shifts).any(), "Invalid! initialize_shift input Matrix, shifts has NaN element"

        self.front_left.shifts = shifts[:, 0]
        self.front_right.shifts = shifts[:, 1]
        self.back_left.shifts = shifts[:, 2]
        self.back_right.shifts = shifts[:, 3]

        return self.front_left.shifts,self.front_right.shifts,self.back_left.shifts,self.back_right.shifts 

    def initialize_prev_foot_pos(self, prev_leg_pos):
        assert not np.isnan(prev_leg_pos).any(), "Invalid! initialize_prev_foot_pos input Matrix, prev_leg_pos has NaN element"

        self.front_left.prev_foot_pos = prev_leg_pos[:, 0]
        self.front_right.prev_foot_pos = prev_leg_pos[:, 1]
        self.back_left.prev_foot_pos = prev_leg_pos[:, 2]
        self.back_right.prev_foot_pos = prev_leg_pos[:, 3]

    def initialize_foot_pos(self, leg_pos):
        assert not np.isnan(leg_pos).any(), "Invalid! initialize_foot_pos input Matrix, leg_pos has NaN element"

        self.front_left.foot_pos = leg_pos[:, 0]
        self.front_right.foot_pos = leg_pos[:, 1]
        self.back_left.foot_pos = leg_pos[:, 2]
        self.back_right.foot_pos = leg_pos[:, 3]

    def initialize_leg_state(self, shifts, prev_leg_pos):
        self.front_left.leg_frame = np.array([self.robot_body_dimensions[0] / 2.0, self.robot_body_dimensions[1] / 2.0, 0.0])
        self.front_right.leg_frame = np.array([self.robot_body_dimensions[0] / 2.0, -self.robot_body_dimensions[1] / 2.0, 0.0])
        self.back_left.leg_frame = np.array([-self.robot_body_dimensions[0] / 2.0, self.robot_body_dimensions[1] / 2.0, 0.0])
        self.back_right.leg_frame = np.array([-self.robot_body_dimensions[0] / 2.0, -self.robot_body_dimensions[1] / 2.0, 0.0])

        self.initialize_shift(shifts)
        self.initialize_prev_foot_pos(prev_leg_pos)

class Trajectory_gen:

    def __init__(self):
        self.traj_theta_ = 0.0
        self.no_of_points_ = 0.0
        self.gait_ = None
        self.robot_ = RobotData()

    def setGaitConfig(self, new_gait):
        self.gait_ = new_gait

    def getGaitConfig(self, current_gait):
        current_gait = self.gait_

    def setRobotData(self, rdata):
        self.robot_ = rdata

    def getRobotData(self, rdata):
        rdata = self.robot_

    def constrainTheta(self, theta):
        theta_c = theta % (2 * math.pi)
        if theta < 0:
            theta_c = theta + 2 * math.pi
        return theta_c

    def updateTrajTheta(self, dt):
        self.traj_theta_ = self.constrainTheta(self.traj_theta_ + self.gait_.omega_ * dt)

    def resetTheta(self):
        self.traj_theta_ = 0

    def getTheta(self):
        return self.traj_theta_
    
    def getLegPhase(self, leg):
        leg_phase = self.traj_theta_ - self.gait_.stance_start_[leg.id_]
        if leg_phase < 0:
            leg_phase += 2 * math.pi

        return leg_phase

    def cspline_coeff(self, z_initial, z_final, d_initial, d_final, t, T):
        coeff = np.array([0.0,0.0,0.0,0.0])

        coeff[0] = z_initial

        coeff[1] = d_initial
    
        #d_final/T^2 + d_initial/T^2 - (2*z_final)/T^3 + (2*z_initial)/T^3
    
        coeff[2] = (3 * z_final) / (T ** 2) - (2 * d_initial) / T - (d_final / T) - (3 * z_initial) / (T ** 2)
    
        coeff[3] = (d_final / (T ** 2)) + (d_initial / (T ** 2)) - (2 * z_final / (T ** 3)) + (2 * z_initial / (T ** 3))

        return coeff

def publish_marker(node, publisher, position, ns, marker_id, color, size):
    marker = Marker()
    marker.header.frame_id = "base"
    marker.header.stamp = node.get_clock().now().to_msg()
    marker.ns = ns
    marker.id = marker_id
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = position
    marker.pose.orientation.w = 1.0
    # marker.scale.x = 0.05
    # marker.scale.y = 0.05
    # marker.scale.z = 0.05
    marker.scale.x, marker.scale.y, marker.scale.z = size
    marker.color.a = 1.0
    marker.color.r, marker.color.g, marker.color.b = color
    publisher.publish(marker)

def publish_marker_cube(node, publisher, position, orientation, ns, marker_id, color, size):
    marker = Marker()
    marker.header.frame_id = "base"
    marker.header.stamp = node.get_clock().now().to_msg()
    marker.ns = ns
    marker.id = marker_id
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = position
    marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z, marker.pose.orientation.w = orientation
    # marker.scale.x = 0.05
    # marker.scale.y = 0.05
    # marker.scale.z = 0.05
    marker.scale.x, marker.scale.y, marker.scale.z = size
    marker.color.a = 1.0
    marker.color.r, marker.color.g, marker.color.b = color
    publisher.publish(marker)

def publish_polygon(node, publisher, points, ns, marker_id, color, size):
    marker = Marker()
    marker.header.frame_id = "base"
    marker.header.stamp = node.get_clock().now().to_msg()
    marker.ns = ns
    marker.id = marker_id
    marker.type = Marker.LINE_STRIP #or Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.points = points
    marker.pose.orientation.w = 1.0
    marker.scale.x, marker.scale.y, marker.scale.z = size
    marker.color.a = 1.0
    marker.color.r, marker.color.g, marker.color.b = color
    publisher.publish(marker)

def point_inside_polygon(point, polygon_points):
    x, y, z = point
    odd_nodes = False
    j = len(polygon_points) - 1

    for i in range(len(polygon_points)):

        xi, yi, zi = polygon_points[i].x, polygon_points[i].y, polygon_points[i].z
        xj, yj, zj = polygon_points[j].x, polygon_points[j].y, polygon_points[j].z 

        if (
            (yi < y and yj >= y or yj < y and yi >= y)
            and (xi <= x or xj <= x)
        ):
            odd_nodes ^= (
                xi + (y - yi) / (yj - yi) * (xj - xi) < x
            )

        j = i

    return odd_nodes    

def projected_point_inside_polygon(point, polygon_points):
    x, y = point[0:2]
    odd_nodes = False
    j = len(polygon_points) - 1

    for i in range(len(polygon_points)):
        xi, yi = polygon_points[i].x, polygon_points[i].y
        xj, yj = polygon_points[j].x, polygon_points[j].y

        if (
            (yi < y and yj >= y or yj < y and yi >= y)
            and (xi <= x or xj <= x)
        ):
            odd_nodes ^= (
                xi + (y - yi) / (yj - yi) * (xj - xi) < x
            )

        j = i

    return odd_nodes




def foot_step_planner(leg, linear_cmd_vel, desired_cmd_vel,raibert_gain_):
    assert not np.any(np.isnan(linear_cmd_vel)), "Invalid! footStepPlanner input Vector, linear_cmd_vel has NaN element"

    pos = np.array([0, 0, 0])
    #s = np.array([0, 0, 0])

    #stance_time = (Gait().stance_duration[leg.id] / (2 * np.pi)) * (1.0 / Gait().frequency)

    stance_time = 0.2

    s = linear_cmd_vel * (stance_time / 2.0) + raibert_gain_ * (linear_cmd_vel - desired_cmd_vel)

    #print("linear_cmd_vel * (stance_time / 2.0)", linear_cmd_vel * (stance_time / 2.0))

    #pos = s + leg.shifts + leg.leg_frame 

    pos = s #New one, removing shifts as that does not make any sense as of now, leg.legframe will be added for first calculation only. Done at the references of this present function.
    
    #print("leg.shifts", leg.shifts)

    assert not np.any(np.isnan(pos)), "Invalid! footStepPlanner output Vector, pos has NaN element"

    return pos

def calculate_orientation(points):
    # Assume points are given in counter-clockwise order
    p0, p1, p2, p3 = points

    # Calculate vectors representing two adjacent sides of the rectangle
    side1 = -p1 + p0
    side2 = -p2 + p1

    # Convert to floats if needed
    side1 = side1.astype(float)
    side2 = side2.astype(float)

    # Normalize the vectors
    side1 /= np.linalg.norm(side1)
    side2 /= np.linalg.norm(side2)

    # Calculate the cross product of the vectors to find the normal vector
    normal = np.cross(side1, side2)
    normal /= np.linalg.norm(normal)

    # Calculate the angle between the first side and the reference direction (X-axis)
    angle = np.arccos(np.dot(side1, np.array([1, 0, 0])))

    # Create a quaternion representing the rotation around the normal vector by the calculated angle
    quaternion = Rotation.from_rotvec(angle * normal).as_quat()

    return tuple(quaternion)

def joint_angles_to_quaternion(joint_angles):
    # Define rotation axes (assuming rotation about z-axis for each joint)
    rotation_axes = [(0, 0, 1) for _ in range(len(joint_angles))]
    
    # Initialize identity rotation
    combined_rotation = Rotation.identity()
    
    # Apply rotation for each joint angle
    for angle, axis in zip(joint_angles, rotation_axes):
        # Create rotation matrix for the current joint
        rotation = Rotation.from_euler('z', angle, degrees=True)
        # Multiply with previous rotations
        combined_rotation = rotation * combined_rotation
    
    # Convert combined rotation to quaternion
    quaternion = combined_rotation.as_quat()
    
    return quaternion

def generate_link_transform_new(node, link_name, parent_link, pos, ori):
    """
    Generate a TransformStamped message representing the transformation between a foot and the world frame.
    
    Parameters:
        foot_name (str): Name of the foot (e.g., 'FL_foot', 'FR_foot', 'RL_foot', 'RR_foot').
        nextstep (numpy.array): Position of the foot in the world frame.
    
    Returns:
        geometry_msgs.msg.TransformStamped: TransformStamped message.
    """
    transform_msg = TransformStamped()
    transform_msg.header.stamp = node.get_clock().now().to_msg()
    transform_msg.header.frame_id = parent_link  # Change 'map' to your world frame
    transform_msg.child_frame_id = link_name
    transform_msg.transform.translation.x = pos[0]
    transform_msg.transform.translation.y = pos[1]
    transform_msg.transform.translation.z = pos[2]
    transform_msg.transform.rotation.x = ori[0]  # Replace with your desired quaternion values
    transform_msg.transform.rotation.y = ori[1]
    transform_msg.transform.rotation.z = ori[2]
    transform_msg.transform.rotation.w = ori[3]
    
    return transform_msg

def generate_link_transform(node, link_name, pos, ori):
    """
    Generate a TransformStamped message representing the transformation between a foot and the world frame.
    
    Parameters:
        foot_name (str): Name of the foot (e.g., 'FL_foot', 'FR_foot', 'RL_foot', 'RR_foot').
        nextstep (numpy.array): Position of the foot in the world frame.
    
    Returns:
        geometry_msgs.msg.TransformStamped: TransformStamped message.
    """
    transform_msg = TransformStamped()
    transform_msg.header.stamp = node.get_clock().now().to_msg()
    transform_msg.header.frame_id = 'base'  # Change 'map' to your world frame
    transform_msg.child_frame_id = link_name
    transform_msg.transform.translation.x = pos[0]
    transform_msg.transform.translation.y = pos[1]
    transform_msg.transform.translation.z = pos[2]
    transform_msg.transform.rotation.x = ori[0]  # Replace with your desired quaternion values
    transform_msg.transform.rotation.y = ori[1]
    transform_msg.transform.rotation.z = ori[2]
    transform_msg.transform.rotation.w = ori[3]
    
    return transform_msg


def quaternion_from_axis_angle(axis, angle):
    """
    Convert rotation represented by axis and angle to quaternion.

    :param axis: Rotation axis, a 3D vector.
    :param angle: Rotation angle in radians.
    :return: Quaternion representing the rotation.
    """
    axis = np.array(axis)
    axis /= np.linalg.norm(axis)  # Normalize axis vector
    half_angle = angle / 2
    w = np.cos(half_angle)
    x = axis[0] * np.sin(half_angle)
    y = axis[1] * np.sin(half_angle)
    z = axis[2] * np.sin(half_angle)
    return np.array([x, y, z, w])

def extract_rotation_axes_from_urdf(urdf_file, child_link_name):
    tree = ET.parse(urdf_file)
    root = tree.getroot()

    rotation_axes = []

    # Find all joints
    for joint in root.findall(".//joint"):
        # Find the child link associated with the joint
        child_link = joint.find("child")
        if child_link is not None and child_link.attrib.get("link") == child_link_name:
            # Get the axis of the joint if it's a revolute joint
            joint_type = joint.attrib.get("type")
            if joint_type == "revolute":
                axis_element = joint.find("axis")
                if axis_element is not None:
                    axis_str = axis_element.attrib.get("xyz", "0 0 0")  # Default axis value
                    axis = tuple(float(comp) for comp in axis_str.split())
                    #rotation_axes.append(axis)
                    rotation_axes = axis

    return rotation_axes

def extract_link_names(urdf_file):
    link_names = []
    tree = ET.parse(urdf_file)
    root = tree.getroot()

    # Find all link elements in the URDF
    for link in root.findall('.//link'):
        # Get the name attribute of each link
        name = link.attrib.get('name')
        if name:
            link_names.append(name)

    return link_names

def extract_joint_names(urdf_file):
    joint_names = []
    tree = ET.parse(urdf_file)
    root = tree.getroot()

    # Find all link elements in the URDF
    for joint in root.findall('.//joint'):
        # Get the name attribute of each link
        name = joint.attrib.get('name')

        if name:
            joint_names.append(name)

    return joint_names


def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Convert Euler angles to rotation matrix (XYZ convention).
    
    Parameters:
        roll (float): Rotation around X-axis in radians.
        pitch (float): Rotation around Y-axis in radians.
        yaw (float): Rotation around Z-axis in radians.
    
    Returns:
        rotation_matrix (numpy.ndarray): 3x3 rotation matrix.
    """
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    
    rotation_matrix = np.dot(R_z, np.dot(R_y, R_x))
    
    return rotation_matrix

def extract_transform(joint):
    origin = joint.find('origin')
    #print('origin', origin)
    if origin is not None:
        translation = [float(x) for x in origin.attrib['xyz'].split()]
        rotation = [float(x) for x in origin.attrib.get('rpy').split()]
        roll, pitch, yaw = rotation
        rotation_matrix = euler_to_rotation_matrix(roll, pitch, yaw)
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = translation

        padded_rotation_matrix = np.eye(4)
        padded_rotation_matrix[:3, :3] = rotation_matrix
        
        #print('rotation_matrix', padded_rotation_matrix)
        #print('translation_matrix', translation_matrix)

        transformation_matrix = np.dot(translation_matrix, padded_rotation_matrix)

        return transformation_matrix
    else:
        return np.eye(4)

def calculate_transformation(parent_link_name, child_link_name, root):
    parent = root.find(".//link[@name='{}']".format(parent_link_name))
    child = root.find(".//link[@name='{}']".format(child_link_name))
    
    if parent is None or child is None:
        raise ValueError("One or both of the provided link names not found in the URDF.")
    
    ancestor = find_common_ancestor(parent, child)
    transformation_matrix = np.eye(4)
    
    while parent != ancestor:
        parent_parent_link = parent.find('parent')
        if parent_parent_link is None:
            break
        joint = find_joint(parent, parent_parent_link.attrib['link'], root)
        transformation_matrix = np.dot(extract_transform(joint), transformation_matrix)
        parent = parent_parent_link
    
    while child != ancestor:
        child_child_link = child.find('child')
        if child_child_link is None:
            break
        joint = find_joint(child_child_link.attrib['link'], child, root)
        transformation_matrix = np.dot(transformation_matrix, np.linalg.inv(extract_transform(joint)))
        child = child_child_link
    
    return transformation_matrix

def find_common_ancestor(link1, link2):
    ancestors1 = set()
    while link1 is not None:
        ancestors1.add(link1)
        link1 = link1.find('parent')
    while link2 is not None:
        if link2 in ancestors1:
            return link2
        link2 = link2.find('parent')
    return None

def find_joint(parent_link_name, child_link_name, root):
    for joint in root.findall(".//joint"):
        parent = joint.find("./parent")
        child = joint.find("./child")
        if parent is not None and child is not None:
            if parent.attrib["link"] == parent_link_name and child.attrib["link"] == child_link_name:
                return joint
    return None

def calculate_transformation_between_frames(parent_frame_name, child_frame_name, root):
    parent = root.find(".//link[@name='{}']".format(parent_frame_name))
    child = root.find(".//link[@name='{}']".format(child_frame_name))

    if parent is None or child is None:
        raise ValueError("One or both of the provided frame names not found in the URDF.")

    transformation_matrix = np.eye(4)
    frames = [child]

    while frames:
        frame = frames.pop(0)

        #print("frame:", frame.attrib['name'])

        # Find the parent joint of the current frame
        parent_joint = None
        for joint in root.findall(".//joint"):
            child_elem = joint.find("./child")
            if child_elem is not None and child_elem.attrib.get('link') == frame.attrib['name']:
                parent_joint = joint
                break

        if parent_joint is None:
            raise ValueError("Parent joint not found for frame '{}'.".format(frame.attrib['name']))

        parent_link_name = parent_joint.find("./parent").attrib['link']

        # print('parent_link_name:', parent_link_name)

        if parent_link_name != parent_frame_name:
            frames.append(root.find(".//link[@name='{}']".format(parent_link_name)))
            continue

        # Get the joint connecting the parent and child frames
        joint = find_joint(parent_link_name, frame.attrib['name'], root)
        transformation_matrix = np.dot(transformation_matrix, extract_transform(joint))

    return transformation_matrix

    


def inverseKinematics2R(ee_pos, leg, link_lengths, safety=False):
    assert len(ee_pos) == 2, "Invalid! In serial 2r Inverse Kinematics, input ee_pos size != 2"
    assert not np.isnan(ee_pos).any(), "Invalid! In serial 2r Inverse Kinematics, input ee_pos has NaN element"

    joint_angles = [0,0]
    l1 = link_lengths[1]
    l2 = link_lengths[2]
    theta1, theta2 = 0.0, 0.0
    t1, t2 = 0.0, 0.0
    r_theta = 0.0
    valid_ik = 0
    pos = ee_pos

    # # Zero position is invalid
    # if np.linalg.norm(ee_pos) < 0.0001:
    #     return -1

    # # If not in workspace, then find the point in workspace
    # # closest to the desired end-effector position. Return
    # # false if such a point is not found.
    # if not inWorkspace(ee_pos):
    #     if safety:
    #         pos = searchSafePosition(ee_pos)
    #         valid_ik = 1
    #     else:
    #         return -1
    # else:
    #     pos = ee_pos

    # l1 = link_lengths[0]
    # l2 = link_lengths[1]

    r_theta = math.atan2(-pos[0], -pos[1])  # angle made by radial line w.r.t the reference

    # Ensure the output lies in the range [-PI , PI]
    if r_theta > math.pi:
        r_theta = r_theta - 2 * math.pi
    elif r_theta < -math.pi:
        r_theta = r_theta + 2 * math.pi

    t1 = cosineRule(l2, np.linalg.norm(pos), l1)  # internal angle opposite to l2
    t2 = cosineRule(np.linalg.norm(pos), l1, l2)  # internal angle opposite to radial line

    theta2 = -(math.pi - t2)

    if leg in ('FR', 'BR'):

      #  t1 = -t1
      #  theta2 = -theta2
      t1 = t1
      theta2 = theta2  

    theta1 = r_theta + t1

    

    joint_angles[0] = theta1
    joint_angles[1] = theta2

    assert not np.isnan(joint_angles).any(), "Invalid! In serial 2r Inverse Kinematics, output joint_angles has NaN element"

    return joint_angles


def inverseKinematics3R(ee_pos, leg, link_lengths, joint_angles_2r, safety=False):
    assert len(ee_pos) == 3, "Invalid! In serial 3r Inverse Kinematics, input ee_pos size != 3"
    assert not np.isnan(ee_pos).any(), "Invalid! In serial 3r Inverse Kinematics, input ee_pos has NaN element"

    

    #right_leg = leg in ["FR", "BR"]

    t1, t2 = 0.0, 0.0
    hip_angle, thigh_angle, calf_angle = 0.0, 0.0, 0.0
    h, r = 0.0, 0.0
    x, y, z = 0.0, 0.0, 0.0
    valid_ik = 0
    l1 = link_lengths[0]

   
    # pos = np.array([0.0, 0.0, 0.0])

    # # Zero position is invalid
    # if np.linalg.norm(ee_pos) < 0.0001:
    #     return -1

    # If not in workspace, then find the point in workspace
    # closest to the desired end-effector position. Return
    # false if such a point is not found.
    # if not inWorkspace(ee_pos, link_lengths):
    #     if safety:
    #         pos = searchSafePosition(ee_pos)
    #         valid_ik = 1
    #     else:
    #         return -1
    # else:
    #     pos = ee_pos

    pos = ee_pos

    joint_angles = [0,0,0]

    x, y, z = pos

    l1 = link_lengths[0]

    r = np.linalg.norm(pos[1:])

    h = math.sqrt(r * r - l1 * l1)

    t1 = math.atan2(h, l1)
    t2 = math.atan2(y, -z)

    if leg in ('FR', 'BR'):
        hip_angle = math.pi / 2 - t1 + t2
    else:
        hip_angle = t1 + t2 - math.pi / 2
        #hip_angle = 0
     
    
    thigh_angle = joint_angles_2r[0]
    calf_angle = joint_angles_2r[1]
    
  
    joint_angles[0] = hip_angle
    joint_angles[1] = thigh_angle
    joint_angles[2] = calf_angle

    assert not np.isnan(joint_angles).any(), "Invalid! In serial 3r Inverse Kinematics, output joint_angles has NaN element"

    return joint_angles

def cosineRule(a, b, c):
    assert a >= 0, "Invalid! For cosineRule length a of the triangle must be >= 0"
    assert b >= 0, "Invalid! For cosineRule length b of the triangle must be >= 0"
    assert c >= 0, "Invalid! For cosineRule length c of the triangle must be >= 0"

    assert a + b >= c, "Invalid! In cosineRule triangle inequality, a + b >= c not satisfied"
    assert c + a >= b, "Invalid! In cosineRule triangle inequality, c + a >= b not satisfied"
    assert b + c >= a, "Invalid! In cosineRule triangle inequality, b + c >= a not satisfied"

    return math.acos((c * c + b * b - a * a) / (2 * b * c))

def inWorkspace(ee_pos, link_lengths):
    if np.linalg.norm(ee_pos) ** 2 > np.sum(link_lengths) ** 2:
        return False

    if np.linalg.norm(ee_pos) ** 2 < (link_lengths[0] - link_lengths[1]) ** 2:
        return False

    return True

def searchSafePosition(desired_pos, link_lengths):
    RADIAL_DISTANCE = 0.2  # radial distance for a point in workspace
    DELTA_MAX = 0.001
    MAX_ITERATIONS = 20

    p_in = np.array([0.0, 0.0])
    p_out = np.array([0.0, 0.0])
    p = np.array([0.0, 0.0])
    unit_vector = np.array([0.0, 0.0])
    n = 0

    # If the input is valid, then there is no need to search
    if inWorkspace(desired_pos, link_lengths):
        return desired_pos

    # p_out is always an invalid point (lies outside the workspace)
    p_out = desired_pos

    unit_vector = p_out / np.linalg.norm(p_out)

    # p_in is always a valid point (lies inside the workspace)
    p_in = RADIAL_DISTANCE * unit_vector

    while np.linalg.norm(p_in - p_out) > DELTA_MAX and n < MAX_ITERATIONS:
        p = (p_in + p_out) / 2
        if inWorkspace(p, link_lengths):
            p_in = p
        else:
            p_out = p
        n += 1

    print("WARNING: Serial 2r Inverse Kinematics safety is being applied")

    return p_in

def quaternion_to_homogeneous_matrix(quaternion, translation):
    # Convert quaternion to rotation matrix
    rotation_matrix = Rotation.from_quat(quaternion).as_matrix()

    # Create a 4x4 homogeneous transformation matrix
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = rotation_matrix
    homogeneous_matrix[:3, 3] = translation

    return homogeneous_matrix

def determine_phase(leg_name, current_time, leg_period):
    """
    Determine whether a leg is in swing or stance phase based on its name and current time.

    Args:
    leg_name (str): The name of the leg (e.g., "FL", "BR", "FR", "BL").
    current_time (float): The current time in seconds.
    leg_period (float): The duration of the leg gait cycle in seconds.

    Returns:
    str: "swing" if the leg is in swing phase, "stance" otherwise.
    """

    phase_index = current_time % leg_period

    phase_id = int(2*current_time/leg_period)

    start_time = phase_id * leg_period/2

    x = start_time

    #print("phase_index", phase_index)

    
    if leg_name == "fl" or leg_name == "br":
        # FL and BR are in swing phase between 0 to leg_period
        
        if phase_index < leg_period / 2:
           
            r =  "swing"

        # elif phase_index < leg_period / 2:

        #     r = "stance"
                   
        elif phase_index >= leg_period / 2:
            
            r =  "stance"
              

    elif leg_name == "fr" or leg_name == "bl":


        if phase_index >= leg_period / 2:

            #print("start_time", start_time)

            r =  "swing"

        elif phase_index < leg_period / 2:

            r =  "stance"

        # elif phase_index < leg_period / 2:

        #     r = "swing"    

    return r, x  


def nextstep_update_legs(leg,linear_cmd_vel, desired_cmd_vel, raibert_gain_, polygon_points, front_back_dist, left_right_dist, currentstep):
    
    #robot_data = RobotData()

    #leg = robot_data.front_left
    x = foot_step_planner(leg, linear_cmd_vel, desired_cmd_vel, raibert_gain_)

    nextstep = currentstep + x

    point_to_check = nextstep

    raibert_gain_n = raibert_gain_

    polygon_points_1 = polygon_points[0]

    polygon_points_2 = polygon_points[1]

    polygon_points_3 = polygon_points[2]
    

    
    
    is_inside_1 = point_inside_polygon(point_to_check, polygon_points_1)

    is_inside_2 = point_inside_polygon(point_to_check, polygon_points_2)

    is_inside_3 = point_inside_polygon(point_to_check, polygon_points_3)

    is_inside_1_proj = projected_point_inside_polygon(point_to_check, polygon_points_1)
    is_inside_2_proj = projected_point_inside_polygon(point_to_check, polygon_points_2)
    is_inside_3_proj = projected_point_inside_polygon(point_to_check, polygon_points_3)



    if is_inside_1 or is_inside_2 or is_inside_3:

        # print("fl(red) is inside the polygon.")

        visual = True

        #print('visual_fl', visual_fl)

        if is_inside_1_proj:
        
                #nextstep[2] = z1_sol

            nextstep[2] = z1_sol
        

        if is_inside_2_proj:
        
                #nextstep[2] = z2_sol

            nextstep[2] = z2_sol
        

        if is_inside_3_proj:
        
            nextstep[2] = z3_sol

    else:

        # print("fl(red) is outside the polygon.")

        
        while not (is_inside_1 or is_inside_2 or is_inside_3):

            visual = False

            # Perform actions or calculations as needed within the while loop
            # For example, you can update variables or take corrective actions

            # Update Raibert's gain for front left leg
            raibert_gain_n = raibert_gain_n + 0.05  # Update with your new value


            # Update other variables or take corrective actions as needed

            # Update is_inside conditions for the next iteration

            x = foot_step_planner(leg, linear_cmd_vel, desired_cmd_vel, raibert_gain_n)

            nextstep = nextstep + x

            

            if is_inside_1_proj:
        
                #nextstep[2] = z1_sol

                nextstep[2] = z1_sol
        

            if is_inside_2_proj:
        
                #nextstep[2] = z2_sol

                nextstep[2] = z2_sol
        

            if is_inside_3_proj:
        
                nextstep[2] = z3_sol

            point_to_check = nextstep

            is_inside_1 = point_inside_polygon(point_to_check, polygon_points_1)

            is_inside_2 = point_inside_polygon(point_to_check, polygon_points_2)

            is_inside_3 = point_inside_polygon(point_to_check, polygon_points_3)
        
    return nextstep, visual

def quaternion_to_rotation_matrix(quaternion):
    """
    Convert a quaternion [x, y, z, w] to a 3x3 rotation matrix.
    
    Parameters:
        quaternion (list or array-like): Quaternion in the format [x, y, z, w].
        
    Returns:
        numpy.ndarray: 3x3 rotation matrix.
    """
    x, y, z, w = quaternion
    rotation_matrix = np.array([
        [1 - 2*y**2 - 2*z**2,   2*x*y - 2*w*z,       2*x*z + 2*w*y],
        [2*x*y + 2*w*z,         1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,         2*y*z + 2*w*x,       1 - 2*x**2 - 2*y**2]
    ])
    return rotation_matrix



def main():



    rclpy.init()

    node = rclpy.create_node('footstep_planner_node')
    publisher = node.create_publisher(Marker, '/visualization_marker', 10)
    rate = node.create_rate(10)  # 10 Hz


    urdf_file_path = 'go1.urdf'
    

    robot_data = RobotData()  # Create an instance of RobotData

    nextstep_fl = robot_data.front_left.leg_frame
    nextstep_fr = robot_data.front_right.leg_frame
    nextstep_bl = robot_data.back_left.leg_frame
    nextstep_br = robot_data.back_right.leg_frame

    print("front_back_dist", np.linalg.norm(nextstep_fl - nextstep_bl))
    print("left_right_dist", np.linalg.norm(nextstep_bl - nextstep_br))

    min_fb_dist = 0.22; max_fb_dist = 0.43
    min_rl_dist = 0.05; max_rl_dist = 0.25


    # Define parameters
    linear_cmd_vel_x = 'linear_cmd_vel_x'
    linear_cmd_vel_y = 'linear_cmd_vel_y'
    linear_cmd_vel_z = 'linear_cmd_vel_z'
    desired_cmd_vel_x = 'desired_cmd_vel_x'
    desired_cmd_vel_y = 'desired_cmd_vel_y'
    desired_cmd_vel_z = 'desired_cmd_vel_z'
    raibert_gain = 'raibert_gain'


    # Declare each parameter individually with its default value
    node.declare_parameter(linear_cmd_vel_x, 0.1)
    node.declare_parameter(linear_cmd_vel_y, 0.0)
    node.declare_parameter(linear_cmd_vel_z, 0.05) #0.05 is the threshold value between a math error and proper stair crossing without collision
    node.declare_parameter(desired_cmd_vel_x, 0.05)
    node.declare_parameter(desired_cmd_vel_y, 0.0)
    node.declare_parameter(desired_cmd_vel_z, 0.0)
    node.declare_parameter(raibert_gain, 1.0)


    # Get parameter values
    linear_cmd_vel_x_value = node.get_parameter('linear_cmd_vel_x').get_parameter_value().double_value
    linear_cmd_vel_y_value = node.get_parameter('linear_cmd_vel_y').get_parameter_value().double_value
    linear_cmd_vel_z_value = node.get_parameter('linear_cmd_vel_z').get_parameter_value().double_value
    desired_cmd_vel_x_value = node.get_parameter('desired_cmd_vel_x').get_parameter_value().double_value
    desired_cmd_vel_y_value = node.get_parameter('desired_cmd_vel_y').get_parameter_value().double_value
    desired_cmd_vel_z_value = node.get_parameter('desired_cmd_vel_z').get_parameter_value().double_value
    raibert_gain_value = node.get_parameter('raibert_gain').get_parameter_value().double_value

    history_positions_fl = []  # Store historical positions for front left leg
    history_positions_fr = []  # Store historical positions for front right leg
    history_positions_bl = []  # Store historical positions for back left leg
    history_positions_br = []  # Store historical positions for back right leg

    r_small = 0.03 #Small marker size
    r_big = 0.05 #Big marker size
    
    #Color codings
    r = [1.0, 0.0, 0.0]
    g = [0.0, 1.0, 0.0]
    b = [0.0, 0.0, 1.0]
    y = [0.5, 0.5, 0.0]

    t = 0.0


    leg_period = 0.4

    swing_time = leg_period/2  #duration of a leg to be in swinging mode

    swing_phase = np.array([0.0, swing_time, swing_time, 0.0])

    step_time = 0.05*1

    #Initial foot locations
         
    leg = robot_data.front_left
    legframe_fl = leg.leg_frame
    currentstep_fl = legframe_fl.copy() #initialization

    leg = robot_data.back_right
    legframe_br = leg.leg_frame
    currentstep_br = legframe_br.copy() #initialization

    leg = robot_data.front_right
    legframe_fr = leg.leg_frame
    currentstep_fr = legframe_fr.copy() #initialization


    leg = robot_data.back_left
    legframe_bl = leg.leg_frame
    currentstep_bl = legframe_bl.copy() #initialization

    foot_pos_fl =  currentstep_fl
    foot_pos_br =  currentstep_br 
    foot_pos_fr =  currentstep_fr
    foot_pos_bl =  currentstep_bl
  
    while rclpy.ok():

        trajectory_generator = Trajectory_gen()

        print("t", t)
        
        visual_fl = False
        visual_br = False
        visual_fr = False
        visual_bl = False


        linear_cmd_vel = np.array([linear_cmd_vel_x_value, linear_cmd_vel_y_value, linear_cmd_vel_z_value])
        desired_cmd_vel = np.array([desired_cmd_vel_x_value, desired_cmd_vel_y_value, desired_cmd_vel_z_value])
        raibert_gain_ = raibert_gain_value

        publish_marker_cube(node, publisher, legframe_fl,(0.0,0.0,0.0,1.0), "fl_leg_frame/step", 1, r , [r_small, r_small, r_small])
   

        publish_marker_cube(node, publisher, legframe_br, (0.0,0.0,0.0,1.0), "br_leg_frame/step", 3, g, [r_small, r_small, r_small])



        publish_marker_cube(node, publisher, legframe_fr, (0.0,0.0,0.0,1.0), "fr_leg_frame/step", 5, b, [r_small, r_small, r_small])


        publish_marker_cube(node, publisher, legframe_bl, (0.0,0.0,0.0,1.0), "bl_leg_frame/step", 7, y, [r_small, r_small, r_small])

        rclpy.spin_once(node, timeout_sec=0.1)
        rate.sleep()

    



    #create Polygon
    
        #z1 = 0.1

        # x1 = -0.2 
        # y1 = -0.2

        x1a = -0.3
        x1b = 0.3
        y1a = -0.3
        y1b = 0.3
        z1 = 0


        polygon_points_1 = [Point(x=x1a, y=y1a, z=0.0),
                          Point(x=x1a, y=y1b, z=0.0),
                          Point(x=x1b, y=y1b, z=0.0),
                          Point(x=x1b, y=y1a, z=0.0),
                          Point(x=x1a, y=y1a, z=0.0)] #First and last points should be the same to create a closed polygon
        
        LL = 0.7 #Distance between two steps + length of squared step
        HH = 0.1 #Height between two steps

        x2a = x1a + LL
        x2b = x1b + LL
        y2a = y1a
        y2b = y1b 
        z2 = HH

        
        polygon_points_2 = [Point(x=x2a, y=y2a, z=z2),
                          Point(x=x2a, y=y2b, z=z2),
                          Point(x=x2b, y=y2b, z=z2),
                          Point(x=x2b, y=y2a, z=z2),
                          Point(x=x2a, y=y2a, z=z2)] #First and last points should be the same to create a closed polygon
        
        x3a = x2a + LL
        x3b = x2b + LL
        y3a = y2a
        y3b = y2b 
        z3 = 2 * HH

        polygon_points_3 = [Point(x=x3a, y=y3a, z=z3),
                          Point(x=x3a, y=y3b, z=z3),
                          Point(x=x3b, y=y3b, z=z3),
                          Point(x=x3b, y=y3a, z=z3),
                          Point(x=x3a, y=y3a, z=z3)] #First and last points should be the same to create a closed polygon


        publish_polygon(node, publisher, polygon_points_1, "surface/polygon", 0, [1.0, 1.0, 1.0], [0.01, 0.01, 0.01])

        publish_polygon(node, publisher, polygon_points_2, "surface/polygon", 1, [1.0, 1.0, 1.0], [0.01, 0.01, 0.01])

        publish_polygon(node, publisher, polygon_points_3, "surface/polygon", 2, [1.0, 1.0, 1.0], [0.01, 0.01, 0.01])

        torso = (currentstep_fl+currentstep_fr+currentstep_bl+currentstep_br)/4 #For initialization

        if np.linalg.norm(nextstep_fl-nextstep_bl) > 0.3 or np.linalg.norm(nextstep_fl-nextstep_bl) > 0.3:

            torso = torso + 2*linear_cmd_vel * step_time #higher velocity when polygon is to be crossed

        else:

            torso = torso + linear_cmd_vel * step_time

        torso_height = 0.3

        #torso[2] = torso_height 

        torso[2] = torso_height + 1*(currentstep_fl[2]+currentstep_fr[2]+currentstep_bl[2]+currentstep_br[2])/4
        #publish_marker_cube(node, publisher, position, orientation, ns, marker_id, color, size)

        #history_positions_torso.append(torso.copy())  # Store the historical position

    #Calculating z's for points on stairs 
        
        global W1, W2, W3, z1_sol, z2_sol, z3_sol
        global z_arr_1, z_arr_2, z_arr_3, xm_1, ym_1, xm_2, ym_2, xm_3, ym_3, coeff_1, coeff_2, coeff_3 
        
        W1= np.array([[1, x1a, y1a],
             [1, x1a, y1b],
             [1, x1b, y1b],
             [1, x1b, y1a]])

        z_arr_1 = np.array([[z1],[z1],[z1],[z1]])

        coeff_1 = np.linalg.pinv(np.transpose(W1)@W1)@np.transpose(W1)@z_arr_1

        

        xm_1 = (x1a + x1b)/2
        ym_1 = (y1a + y1b)/2          

        z1_sol = coeff_1[0] + coeff_1[1]*xm_1 + coeff_1[2]*ym_1

        W2= np.array([[1, x2a, y2a],
             [1, x2a, y2b],
             [1, x2b, y2b],
             [1, x2b, y2a]])

        z_arr_2 = np.array([[z2],[z2],[z2],[z2]])

        coeff_2 = np.linalg.pinv(np.transpose(W2)@W2)@np.transpose(W2)@z_arr_2

         

        xm_2 = (x2a + x2b)/2
        ym_2 = (y2a + y2b)/2          

        z2_sol = coeff_2[0] + coeff_2[1]*xm_2 + coeff_2[2]*ym_2


        W3= np.array([[1, x3a, y3a],
             [1, x3a, y3b],
             [1, x3b, y3b],
             [1, x3b, y3a]])

        z_arr_3 = np.array([[z3],[z3],[z3],[z3]])

        coeff_3 = np.linalg.pinv(np.transpose(W3)@W3)@np.transpose(W3)@z_arr_3

        # print("coeff", coeff)    

        xm_3 = (x3a + x3b)/2
        ym_3 = (y3a + y3b)/2          

        z3_sol = coeff_3[0] + coeff_3[1]*xm_3 + coeff_3[2]*ym_3

        polygon_points = [polygon_points_1, polygon_points_2, polygon_points_3]

        front_back_dist = np.linalg.norm(nextstep_fl - nextstep_bl)
        left_right_dist = np.linalg.norm(nextstep_bl - nextstep_br)

        
    #start
        
        z_vel_stepping = 0.25
        
        leg_name = robot_data.front_left.name

        fl_xx = determine_phase(leg_name = leg_name, current_time = t, leg_period = leg_period) 

        fl_phase = fl_xx[0]

        fl_start = fl_xx[1]

        fl_end = fl_start + 0.5* leg_period

        #print("t- fl_start", t- fl_start)

        if fl_phase == "stance":  #leg is in stance phase

            print('fl in stance')
            
            leg = robot_data.front_left

            visual_fl = True

            
            if abs(t- fl_start)<1e-3 or abs(t-fl_end)< 1e-3:

                visual_fl = False

                currentstep_fl =  nextstep_fl


                front_back_dist = np.linalg.norm(nextstep_fl - nextstep_bl)

                left_right_dist = np.linalg.norm(nextstep_fl - nextstep_fr)

                polygon_points = [polygon_points_1, polygon_points_2, polygon_points_3]


                xxx = nextstep_update_legs(leg,linear_cmd_vel, desired_cmd_vel, raibert_gain_, polygon_points, front_back_dist, left_right_dist, currentstep_fl)
                               
                nextstep_fl = xxx[0]

                visual_fl =  xxx[1]





                #print("visual_fl", visual_fl)

                # nextstep_fl[2] = 0       
        
        elif fl_phase == "swing":  #leg is in swing phase

            #nextstep_fl[2] = 0

            visual_fl = True

               
            print('fl in swing')
            
            # print("remainder_fl", t % (swing_phase[0]+ swing_time))

            # div = int(t / (swing_phase[0]+ swing_time))

            # print("div", div)

            #tdash = t - div * (swing_phase[0]+ swing_time)

            if (nextstep_fl[2] - currentstep_fl[2])>1e-2:

                linear_cmd_vel_z_value_new = z_vel_stepping

            else:

                linear_cmd_vel_z_value_new = linear_cmd_vel_z_value


            tdash = t - fl_start

            coeffs_fl_x = trajectory_generator.cspline_coeff(currentstep_fl[0], nextstep_fl[0], 1*linear_cmd_vel_x_value,  -1*linear_cmd_vel_x_value, tdash, swing_time)

            coeffs_fl_y = trajectory_generator.cspline_coeff(currentstep_fl[1], nextstep_fl[1], 1*linear_cmd_vel_y_value,  -1*linear_cmd_vel_y_value, tdash, swing_time)

            coeffs_fl_z = trajectory_generator.cspline_coeff(currentstep_fl[2], nextstep_fl[2], 3*linear_cmd_vel_z_value_new,  -1*linear_cmd_vel_z_value_new, tdash, swing_time)

            t_vec = np.array([tdash ** 0, tdash ** 1, tdash ** 2, tdash ** 3])

            foot_pos_fl[0] = np.dot(coeffs_fl_x, t_vec)

            foot_pos_fl[1] = np.dot(coeffs_fl_y, t_vec)

            foot_pos_fl[2] = np.dot(coeffs_fl_z, t_vec)

            # print("currentstep_fl", currentstep_fl)

            # print("foot_pos_fl", foot_pos_fl)

            # print("nextstep_fl", nextstep_fl) 

        # else:

        #     foot_pos_fl = nextstep_fl

        
        #print("foot_pos_fl", foot_pos_fl)

        #print("nextstep_fl", nextstep_fl) 


        
        front_back_dist = np.linalg.norm(nextstep_fr - nextstep_br)
        left_right_dist = np.linalg.norm(nextstep_bl - nextstep_br)

        leg_name = robot_data.back_right.name

        br_xx = determine_phase(leg_name = leg_name, current_time = t, leg_period = leg_period) 

        br_phase = br_xx [0]

        br_start = br_xx [1]

        br_end = br_start + 0.5 * leg_period

        #raibert_gain_br = raibert_gain_

        #print("t- br_start", t- br_start)
        
        if br_phase == "stance":  #leg is in stance phase

            print("br in stance")
            
            leg = robot_data.back_right

            visual_br = True

            
            if abs(t-br_start)< 1e-3 or abs(t-br_end) < 1e-3:

                visual_br = False

                currentstep_br =  nextstep_br

                #currentstep_br[2] = 0

                front_back_dist = np.linalg.norm(nextstep_fr - nextstep_br)

                left_right_dist = np.linalg.norm(nextstep_bl - nextstep_br)

                xxx = nextstep_update_legs(leg,linear_cmd_vel, desired_cmd_vel, raibert_gain_, polygon_points, front_back_dist, left_right_dist, currentstep_br)                

                nextstep_br = xxx[0]

                visual_br = xxx[1]

                #print("visual_br", visual_br)

        elif br_phase == "swing":  #leg is in swing phase

            # div = int(t / (swing_phase[3]+ swing_time))

            #print("div", div)

            #tdash = t - div * (swing_phase[3]+ swing_time)

            visual_br = True

            print("br in swing")

            tdash = t - br_start

            if (nextstep_br[2] - currentstep_br[2])>1e-2:

                linear_cmd_vel_z_value_new = z_vel_stepping

            else:

                linear_cmd_vel_z_value_new = linear_cmd_vel_z_value


            coeffs_br_x = trajectory_generator.cspline_coeff(currentstep_br[0], nextstep_br[0], 1*linear_cmd_vel_x_value,  -1*linear_cmd_vel_x_value, tdash, swing_time)

            coeffs_br_y = trajectory_generator.cspline_coeff(currentstep_br[1], nextstep_br[1], 1*linear_cmd_vel_y_value,  -1*linear_cmd_vel_y_value, tdash, swing_time)

            coeffs_br_z = trajectory_generator.cspline_coeff(currentstep_br[2], nextstep_br[2], 3*linear_cmd_vel_z_value_new,  -1*linear_cmd_vel_z_value_new, tdash, swing_time)

            t_vec = np.array([tdash ** 0, tdash ** 1, tdash ** 2, tdash ** 3])

            foot_pos_br[0] = np.dot(coeffs_br_x, t_vec)

            foot_pos_br[1] = np.dot(coeffs_br_y, t_vec)

            foot_pos_br[2] = np.dot(coeffs_br_z, t_vec)

            # print("currentstep_br", currentstep_br)

            # print("foot_pos_br", foot_pos_br)

            # print("nextstep_br", nextstep_br) 

        
        #else:   
              
        
        leg_name = robot_data.front_right.name

        fr_xx = determine_phase(leg_name = leg_name, current_time = t, leg_period = leg_period) 

        fr_phase = fr_xx[0]

        fr_start = fr_xx[1]

        fr_end = fr_start + 0.5* leg_period

        #print("t- fr_start", t- fr_start)

        if fr_phase == "stance":  #leg is in stance phase

            print('fr in stance')

            visual_fr = True
            
            if abs(t- fr_start)<1e-3 or abs(t-fr_end)<1e-3:

                visual_fr = False

                currentstep_fr =  nextstep_fr

                #currentstep_fr[2] = 0

                front_back_dist = np.linalg.norm(nextstep_fr - nextstep_br)

                left_right_dist = np.linalg.norm(nextstep_fr - nextstep_fl)

                polygon_points = [polygon_points_1, polygon_points_2, polygon_points_3]

                xxx = nextstep_update_legs(leg,linear_cmd_vel, desired_cmd_vel, raibert_gain_, polygon_points, front_back_dist, left_right_dist, currentstep_fr)
                
                nextstep_fr = xxx[0]

                visual_fr = xxx[1]

                # if visual_fr == False:
                     
                #     nextstep_fr = currentstep_fr


                

                #nextstep_fr[2] = 0

            

        elif fr_phase == "swing": # swing_phase[3] for fr
            
            #print("t % (swing_phase[1]+ swing_time)", t % (swing_phase[1]+ swing_time))

            # div = int(t / (swing_phase[1]+ swing_time))

            #print("div", div)

            #tdash = t - div * (swing_phase[1]+ swing_time)

            print("fr in swing")

            visual_fr = True

            tdash = t - fr_start
            
            if (nextstep_fr[2] - currentstep_fr[2])>1e-2:

                linear_cmd_vel_z_value_new = z_vel_stepping

            else:

                linear_cmd_vel_z_value_new = linear_cmd_vel_z_value

            
            coeffs_fr_x = trajectory_generator.cspline_coeff(currentstep_fr[0], nextstep_fr[0], 1*linear_cmd_vel_x_value,  -1*linear_cmd_vel_x_value, t, swing_time)

            coeffs_fr_y = trajectory_generator.cspline_coeff(currentstep_fr[1], nextstep_fr[1], 1*linear_cmd_vel_y_value,  -1*linear_cmd_vel_y_value, t, swing_time)

            coeffs_fr_z = trajectory_generator.cspline_coeff(currentstep_fr[2], nextstep_fr[2], 3*linear_cmd_vel_z_value_new,  -1*linear_cmd_vel_z_value_new, t, swing_time)

            t_vec = np.array([tdash ** 0, tdash ** 1, tdash ** 2, tdash ** 3])

            foot_pos_fr[0] = np.dot(coeffs_fr_x, t_vec)

            foot_pos_fr[1] = np.dot(coeffs_fr_y, t_vec)

            foot_pos_fr[2] = np.dot(coeffs_fr_z, t_vec)

            # print("currentstep_fr", currentstep_fr)

            #print("foot_pos_fr", foot_pos_fr)

            # print("nextstep_fr", nextstep_fr) 

        
        front_back_dist = np.linalg.norm(nextstep_fl - nextstep_bl)
        left_right_dist = np.linalg.norm(nextstep_bl - nextstep_br)

        leg_name = robot_data.back_left.name

        bl_xx = determine_phase(leg_name = leg_name, current_time = t, leg_period = leg_period) 

        bl_phase = bl_xx [0]

        bl_start = bl_xx [1]

        bl_end = bl_start + 0.5* leg_period

        #raibert_gain_br = raibert_gain_

        #print("t- bl_start", t- bl_start)
        
        if bl_phase == "stance":  #leg is in stance phase

            print("bl in stance")
            
            leg = robot_data.back_left

            visual_bl = True
            
            if abs(t-bl_start)< 1e-3 or abs(t-bl_end)< 1e-3:

                visual_bl = False

                currentstep_bl =  nextstep_bl

                #currentstep_bl[2] = 0

                front_back_dist = np.linalg.norm(nextstep_fl - nextstep_bl)

                left_right_dist = np.linalg.norm(nextstep_bl - nextstep_br)

                xxx = nextstep_update_legs(leg,linear_cmd_vel, desired_cmd_vel, raibert_gain_, polygon_points, front_back_dist, left_right_dist, currentstep_bl)  
        
                nextstep_bl = xxx[0]

                visual_bl = xxx[1]

                #print("visual_bl", visual_bl)
                #nextstep_bl[2] = 0
                

        elif bl_phase == "swing": # swing_phase[3] for bl
        
            print("bl in swing")

            visual_bl = True

            tdash = t - bl_start

            if (nextstep_bl[2] - currentstep_bl[2])>1e-2:

                linear_cmd_vel_z_value_new = z_vel_stepping

            else:

                linear_cmd_vel_z_value_new = linear_cmd_vel_z_value

            coeffs_bl_x = trajectory_generator.cspline_coeff(currentstep_bl[0], nextstep_bl[0], 1*linear_cmd_vel_x_value,  -1*linear_cmd_vel_x_value, tdash, swing_time)

            coeffs_bl_y = trajectory_generator.cspline_coeff(currentstep_bl[1], nextstep_bl[1], 1*linear_cmd_vel_y_value,  -1*linear_cmd_vel_y_value, tdash, swing_time)

            coeffs_bl_z = trajectory_generator.cspline_coeff(currentstep_bl[2], nextstep_bl[2], 3*linear_cmd_vel_z_value_new,  -1*linear_cmd_vel_z_value_new, tdash, swing_time)

            t_vec = np.array([tdash ** 0, tdash ** 1, tdash ** 2, tdash ** 3])

            foot_pos_bl[0] = np.dot(coeffs_bl_x, t_vec)

            foot_pos_bl[1] = np.dot(coeffs_bl_y, t_vec)

            foot_pos_bl[2] = np.dot(coeffs_bl_z, t_vec)


            # print("currentstep_bl", currentstep_bl)

            # print("foot_pos_bl", foot_pos_bl)

            # print("nextstep_bl", nextstep_bl)
    
        

        publish_marker(node, publisher, nextstep_fl, "fl_foot/step", 6, r, [r_small, r_small, r_small])
                

        publish_marker(node, publisher, nextstep_br, "br_foot/step", 7, g, [r_small, r_small, r_small])


        publish_marker(node, publisher, nextstep_fr, "fr_foot/step", 8, b, [r_small, r_small, r_small])


        publish_marker(node, publisher, nextstep_bl, "bl_foot/step", 9, y, [r_small, r_small, r_small])

        rclpy.spin_once(node, timeout_sec=0.1)
        rate.sleep()



        

        tree = ET.parse(urdf_file_path)
        root = tree.getroot()


        transformations = {
        'base_to_trunk': calculate_transformation_between_frames("base", "trunk", root),
        # FL leg
        'trunk_to_fl_hip': calculate_transformation_between_frames("trunk", "FL_hip", root),
        'trunk_to_fl_hip_rotor': calculate_transformation_between_frames('trunk', 'FL_hip_rotor', root),
        'fl_hip_to_fl_thigh': calculate_transformation_between_frames('FL_hip', 'FL_thigh', root),
        'fl_hip_to_fl_thigh_rotor': calculate_transformation_between_frames('FL_hip', 'FL_thigh_rotor', root),
        'fl_thigh_to_fl_calf': calculate_transformation_between_frames('FL_thigh', 'FL_calf', root),
        'fl_thigh_to_fl_calf_rotor': calculate_transformation_between_frames('FL_thigh', 'FL_calf_rotor', root),
        'fl_calf_to_fl_foot': calculate_transformation_between_frames('FL_calf', 'FL_foot', root),
        
        # FR leg
        'trunk_to_fr_hip': calculate_transformation_between_frames('trunk', 'FR_hip', root),
        'trunk_to_fr_hip_rotor': calculate_transformation_between_frames('trunk', 'FR_hip_rotor', root),
        'fr_hip_to_fr_thigh': calculate_transformation_between_frames('FR_hip', 'FR_thigh', root),
        'fr_hip_to_fr_thigh_rotor': calculate_transformation_between_frames('FR_hip', 'FR_thigh_rotor', root),
        'fr_thigh_to_fr_calf': calculate_transformation_between_frames('FR_thigh', 'FR_calf', root),
        'fr_thigh_to_fr_calf_rotor': calculate_transformation_between_frames('FR_thigh', 'FR_calf_rotor', root),
        'fr_calf_to_fr_foot': calculate_transformation_between_frames('FR_calf', 'FR_foot', root),

        # BL leg
        'trunk_to_bl_hip': calculate_transformation_between_frames('trunk', 'RL_hip', root),
        'trunk_to_bl_hip_rotor': calculate_transformation_between_frames('trunk', 'RL_hip_rotor', root),
        'bl_hip_to_bl_thigh': calculate_transformation_between_frames('RL_hip', 'RL_thigh', root),
        'bl_hip_to_bl_thigh_rotor': calculate_transformation_between_frames('RL_hip', 'RL_thigh_rotor', root),
        'bl_thigh_to_bl_calf': calculate_transformation_between_frames('RL_thigh', 'RL_calf', root),
        'bl_thigh_to_bl_calf_rotor': calculate_transformation_between_frames('RL_thigh', 'RL_calf_rotor', root),
        'bl_calf_to_bl_foot': calculate_transformation_between_frames('RL_calf', 'RL_foot', root),
        # BR leg
        'trunk_to_br_hip': calculate_transformation_between_frames('trunk', 'RR_hip', root),
        'trunk_to_br_hip_rotor': calculate_transformation_between_frames('trunk', 'RR_hip_rotor', root),
        'br_hip_to_br_thigh': calculate_transformation_between_frames('RR_hip', 'RR_thigh', root),
        'br_hip_to_br_thigh_rotor': calculate_transformation_between_frames('RR_hip', 'RR_thigh_rotor', root),
        'br_thigh_to_br_calf': calculate_transformation_between_frames('RR_thigh', 'RR_calf', root),
        'br_thigh_to_br_calf_rotor': calculate_transformation_between_frames('RR_thigh', 'RR_calf_rotor', root),
        'br_calf_to_br_foot': calculate_transformation_between_frames('RR_calf', 'RR_foot', root),
        }

        




        #print('bl_jj', bl_jj)

        print("visual_fl", visual_fl)
        print("visual_br", visual_br)
        print("visual_fr", visual_fr)
        print("visual_bl", visual_bl)

        #zx = 0 #dummy variable4
        
        #if zx == 0:

        if visual_fl and visual_br and visual_bl and visual_fr: 
            
            new_trunk_position = torso

            #new_trunk_position_homogeneous = np.append(new_trunk_position, 1)

            

            #torso_orientation = quaternion_from_axis_angle(np.array([0.0,1.0,0.0]), -1*aa) #aa is the slope of incline

            slope_est_1 = math.atan2((currentstep_fl[2]-currentstep_bl[2]),(currentstep_fl[0]-currentstep_bl[0]))

            slope_est_2 = math.atan2((currentstep_fr[2]-currentstep_br[2]),(currentstep_fr[0]-currentstep_br[0]))

            slope_est = (slope_est_1 + slope_est_2)/2

            torso_orientation = quaternion_from_axis_angle(np.array([0.0,1.0,0.0]), -slope_est) 

            #torso_orientation = calculate_orientation([currentstep_fl, currentstep_bl, currentstep_fr, currentstep_br])

            torso_ori = torso_orientation

            

            #torso_ori_matrix = Rotation.from_quat(torso_ori).as_matrix()

            torso_ori_matrix = quaternion_to_rotation_matrix(torso_ori)

            base_to_trunk_rot = torso_ori_matrix

            #print("torso_ori_matrix", torso_ori_matrix)

            base_to_trunk_hom = np.eye(4)

            base_to_trunk_hom[:3,:3] = base_to_trunk_rot #base_to_trunk_hom will be multiplied to every matrices to get correct position

            base_to_trunk_hom[:3,3] = new_trunk_position #base_to_trunk_hom will be multiplied to every matrices to get correct position

            new_trunk_position_homogeneous = transformations['base_to_trunk'][:4,3] #In local frame

            #new_trunk_position_homogeneous = np.append(new_trunk_position,1)

            #print("transformations['base_to_trunk']", transformations['base_to_trunk'])

            #torso_ori = torso_orientation

            broadcaster = tf2_ros.TransformBroadcaster(node)

            transform_msg_torso = generate_link_transform_new(node, 'trunk', 'base', torso, torso_ori)

            broadcaster.sendTransform(transform_msg_torso)

            
        # leg Fl
            new_fl_foot_position =  foot_pos_fl

            new_fl_foot_position_homogeneous = np.append(new_fl_foot_position, 1)

            #rotational_matrix(angle, axis)
            
            #print('transform', transformations['trunk_to_fl_hip'])
            
            #print("new_trunk_position_homogeneous", new_trunk_position_homogeneous)

            #print("base_to_trunk_hom @ new_trunk_position_homogeneous", base_to_trunk_hom @ new_trunk_position_homogeneous)

            

            trunk_pos_hom = base_to_trunk_hom @ new_trunk_position_homogeneous

            trunk_pos = trunk_pos_hom[:3]

            fl_hip_rotor_pos_hom = base_to_trunk_hom @ transformations['trunk_to_fl_hip_rotor']

            fl_hip_pos_hom = base_to_trunk_hom @ transformations['trunk_to_fl_hip']

            fl_hip_rotor_pos = fl_hip_rotor_pos_hom[:3,3]
            
            fl_hip_pos = fl_hip_pos_hom[:3,3]


            
            
            
            ee_pos = base_to_trunk_rot.T @(foot_pos_fl - fl_hip_pos)
            ee_pos_hom = np.append(ee_pos,1)
            ee_pos = ee_pos_hom[:3]


            ee_pos_fl = np.array([ee_pos[0], ee_pos[2]])
            leg = 'FL'
            #print('ee_pos', ee_pos)
            fl_jj_2R = inverseKinematics2R(ee_pos_fl, leg, robot_data.link_lengths, safety=False)
            

            ee_pos_fl_3R = ee_pos

            fl_jj = inverseKinematics3R(ee_pos_fl_3R, leg,robot_data.link_lengths, joint_angles_2r = fl_jj_2R, safety=False)

        #axes and rotation
            axes_fl_hip =  extract_rotation_axes_from_urdf(urdf_file_path, 'FL_hip') # hip angle 

            #print('axes_fl_hip', axes_fl_hip)

            quat_fl_hip = quaternion_from_axis_angle(axes_fl_hip, fl_jj[0])

            #Rot_fl_hip = rotational_matrix(fl_jj[0], axes_fl_hip)

            torso_fl_hip_rel_pos = transformations['trunk_to_fl_hip'][:3,3]

            print("torso_fl_hip_rel_pos",torso_fl_hip_rel_pos)
            
            transform_msg_fl_hip = generate_link_transform_new(node, 'fl_hip', 'trunk', torso_fl_hip_rel_pos, quat_fl_hip)

            

            axes_fl_thigh =  extract_rotation_axes_from_urdf(urdf_file_path, 'FL_thigh') # thigh angle 

            quat_fl_thigh = quaternion_from_axis_angle(axes_fl_thigh, fl_jj[1])

            fl_hip_thigh_rel_pos = transformations['fl_hip_to_fl_thigh'][:3,3]

            transform_msg_fl_thigh = generate_link_transform_new(node, 'fl_thigh', 'fl_hip', fl_hip_thigh_rel_pos, quat_fl_thigh)

            

            axes_fl_calf = extract_rotation_axes_from_urdf(urdf_file_path, 'FL_calf') # calf angle

            quat_fl_calf = quaternion_from_axis_angle(axes_fl_calf, fl_jj[2])

            fl_thigh_calf_rel_pos = transformations['fl_thigh_to_fl_calf'][:3,3]

            transform_msg_fl_calf = generate_link_transform_new(node, 'fl_calf', 'fl_thigh', fl_thigh_calf_rel_pos, quat_fl_calf)

            
            #fl_hip_ori = torso_ori

            zero_rel_ori = np.array([0.0,0.0,0.0,1.0])

            quat_fl_foot = quat_fl_calf #Since foot and calf are joined by fixed joint 

            fl_calf_foot_rel_pos = transformations['fl_calf_to_fl_foot'][:3,3]

            transform_msg_fl_foot = generate_link_transform_new(node, 'fl_foot', 'fl_calf', fl_calf_foot_rel_pos, zero_rel_ori)

            
            
            torso_fl_hip_rotor_rel_pos = transformations['trunk_to_fl_hip_rotor'][:3,3]

            transform_msg_fl_hip_rotor = generate_link_transform_new(node, 'fl_hip_rotor', 'trunk', torso_fl_hip_rotor_rel_pos, zero_rel_ori)

            

            fl_hip_thigh_rotor_rel_pos = transformations['fl_hip_to_fl_thigh_rotor'][:3,3]
            
            transform_msg_fl_thigh_rotor = generate_link_transform_new(node, 'fl_thigh_rotor', 'fl_hip', fl_hip_thigh_rotor_rel_pos, zero_rel_ori)

            

            fl_thigh_calf_rotor_rel_pos = transformations['fl_thigh_to_fl_calf_rotor'][:3,3]

            transform_msg_fl_calf_rotor = generate_link_transform_new(node, 'fl_calf_rotor', 'fl_thigh', fl_thigh_calf_rotor_rel_pos, zero_rel_ori)

            

 

            
        # leg BR
            new_br_foot_position =  foot_pos_br

            new_br_foot_position_homogeneous = np.append(new_br_foot_position, 1)

            br_hip_rotor_pos_hom = base_to_trunk_hom @ transformations['trunk_to_br_hip_rotor']

            br_hip_pos_hom = base_to_trunk_hom @ transformations['trunk_to_br_hip']

            br_hip_rotor_pos = br_hip_rotor_pos_hom[:3,3]
            
            br_hip_pos = br_hip_pos_hom[:3,3]


            
            ee_pos = base_to_trunk_rot.T @(foot_pos_br - br_hip_pos)
            ee_pos_hom = np.append(ee_pos,1)
            ee_pos = ee_pos_hom[:3]


            ee_pos_br = np.array([ee_pos[0], ee_pos[2]])
            leg = 'BR'
            #print('ee_pos', ee_pos)
            br_jj_2R = inverseKinematics2R(ee_pos_br, leg, robot_data.link_lengths, safety=False)
            

            ee_pos_br_3R = ee_pos
            br_jj = inverseKinematics3R(ee_pos_br_3R, leg,robot_data.link_lengths, joint_angles_2r = br_jj_2R, safety=False)

        #axes and rotation
            axes_br_hip =  extract_rotation_axes_from_urdf(urdf_file_path, 'RR_hip') # hip angle 

            #print('axes_fl_hip', axes_fl_hip)

            quat_br_hip = quaternion_from_axis_angle(axes_br_hip, br_jj[0])

            #Rot_fl_hip = rotational_matrix(fl_jj[0], axes_fl_hip)

            torso_br_hip_rel_pos = transformations['trunk_to_br_hip'][:3,3]

            #print("torso_br_hip_rel_pos",torso_br_hip_rel_pos)
            
            transform_msg_br_hip = generate_link_transform_new(node, 'br_hip', 'trunk', torso_br_hip_rel_pos, quat_br_hip)

            

            

            axes_br_thigh =  extract_rotation_axes_from_urdf(urdf_file_path, 'RR_thigh') # thigh angle 

            quat_br_thigh = quaternion_from_axis_angle(axes_br_thigh, br_jj[1])

            br_hip_thigh_rel_pos = transformations['br_hip_to_br_thigh'][:3,3]

            transform_msg_br_thigh = generate_link_transform_new(node, 'br_thigh', 'br_hip', br_hip_thigh_rel_pos, quat_br_thigh)

            

            axes_br_calf = extract_rotation_axes_from_urdf(urdf_file_path, 'RR_calf') # calf angle

            quat_br_calf = quaternion_from_axis_angle(axes_br_calf, br_jj[2])

            br_thigh_calf_rel_pos = transformations['br_thigh_to_br_calf'][:3,3]

            transform_msg_br_calf = generate_link_transform_new(node, 'br_calf', 'br_thigh', br_thigh_calf_rel_pos, quat_br_calf)

            


            zero_rel_ori = np.array([0.0,0.0,0.0,1.0])

            
            br_calf_foot_rel_pos = transformations['br_calf_to_br_foot'][:3,3]

            transform_msg_br_foot = generate_link_transform_new(node, 'br_foot', 'br_calf', br_calf_foot_rel_pos, zero_rel_ori)

            
            
            torso_br_hip_rotor_rel_pos = transformations['trunk_to_br_hip_rotor'][:3,3]

            transform_msg_br_hip_rotor = generate_link_transform_new(node, 'br_hip_rotor', 'trunk', torso_br_hip_rotor_rel_pos, zero_rel_ori)

            

            br_hip_thigh_rotor_rel_pos = transformations['br_hip_to_br_thigh_rotor'][:3,3]
            
            transform_msg_br_thigh_rotor = generate_link_transform_new(node, 'br_thigh_rotor', 'br_hip', br_hip_thigh_rotor_rel_pos, zero_rel_ori)

            

            br_thigh_calf_rotor_rel_pos = transformations['br_thigh_to_br_calf_rotor'][:3,3]

            transform_msg_br_calf_rotor = generate_link_transform_new(node, 'br_calf_rotor', 'br_thigh', br_thigh_calf_rotor_rel_pos, zero_rel_ori)

            

  

        #leg FR
        
            new_fr_foot_position =  foot_pos_fr

            new_fr_foot_position_homogeneous = np.append(new_fr_foot_position, 1)

            fr_hip_rotor_pos_hom = base_to_trunk_hom @ transformations['trunk_to_fr_hip_rotor']

            fr_hip_pos_hom = base_to_trunk_hom @ transformations['trunk_to_fr_hip']

            fr_hip_rotor_pos = fr_hip_rotor_pos_hom[:3,3]
            
            fr_hip_pos = fr_hip_pos_hom[:3,3]


            
            
            
            ee_pos = base_to_trunk_rot.T @(foot_pos_fr - fr_hip_pos)
            ee_pos_hom = np.append(ee_pos,1)
            ee_pos = ee_pos_hom[:3]


            ee_pos_fr = np.array([ee_pos[0], ee_pos[2]])
            leg = 'FR'
            #print('ee_pos', ee_pos)
            fr_jj_2R = inverseKinematics2R(ee_pos_fr, leg, robot_data.link_lengths, safety=False)
            

            ee_pos_fr_3R = ee_pos
            fr_jj = inverseKinematics3R(ee_pos_fr_3R, leg,robot_data.link_lengths, joint_angles_2r = fr_jj_2R, safety=False)

        #axes and rotation
            axes_fr_hip =  extract_rotation_axes_from_urdf(urdf_file_path, 'FR_hip') # hip angle 

            #print('axes_fl_hip', axes_fl_hip)

            quat_fr_hip = quaternion_from_axis_angle(axes_fr_hip, fr_jj[0])

            torso_fr_hip_rel_pos = transformations['trunk_to_fr_hip'][:3,3]

            transform_msg_fr_hip = generate_link_transform_new(node, 'fr_hip', 'trunk', torso_fr_hip_rel_pos, quat_fr_hip)

            

            

            axes_fr_thigh =  extract_rotation_axes_from_urdf(urdf_file_path, 'FR_thigh') # thigh angle 

            quat_fr_thigh = quaternion_from_axis_angle(axes_fr_thigh, fr_jj[1])

            fr_hip_thigh_rel_pos = transformations['fr_hip_to_fr_thigh'][:3,3]

            transform_msg_fr_thigh = generate_link_transform_new(node, 'fr_thigh', 'fr_hip', fr_hip_thigh_rel_pos, quat_fr_thigh)

            

            axes_fr_calf = extract_rotation_axes_from_urdf(urdf_file_path, 'FR_calf') # calf angle

            quat_fr_calf = quaternion_from_axis_angle(axes_fr_calf, fr_jj[2])

            fr_thigh_calf_rel_pos = transformations['fr_thigh_to_fr_calf'][:3,3]

            transform_msg_fr_calf = generate_link_transform_new(node, 'fr_calf', 'fr_thigh', fr_thigh_calf_rel_pos, quat_fr_calf)

            

            zero_rel_ori = np.array([0.0,0.0,0.0,1.0])

            fr_calf_foot_rel_pos = transformations['fr_calf_to_fr_foot'][:3,3]

            transform_msg_fr_foot = generate_link_transform_new(node, 'fr_foot', 'fr_calf', fr_calf_foot_rel_pos, zero_rel_ori)

            
            
            torso_fr_hip_rotor_rel_pos = transformations['trunk_to_fr_hip_rotor'][:3,3]

            transform_msg_fr_hip_rotor = generate_link_transform_new(node, 'fr_hip_rotor', 'trunk', torso_fr_hip_rotor_rel_pos, zero_rel_ori)

            

            fr_hip_thigh_rotor_rel_pos = transformations['fr_hip_to_fr_thigh_rotor'][:3,3]
            
            transform_msg_fr_thigh_rotor = generate_link_transform_new(node, 'fr_thigh_rotor', 'fr_hip', fr_hip_thigh_rotor_rel_pos, zero_rel_ori)

            

            fr_thigh_calf_rotor_rel_pos = transformations['fr_thigh_to_fr_calf_rotor'][:3,3]

            transform_msg_fr_calf_rotor = generate_link_transform_new(node, 'fr_calf_rotor', 'fr_thigh', fr_thigh_calf_rotor_rel_pos, zero_rel_ori)



        #leg BL
            
            new_bl_foot_position =  foot_pos_bl

            new_bl_foot_position_homogeneous = np.append(new_bl_foot_position, 1)

            bl_hip_rotor_pos_hom = base_to_trunk_hom @ transformations['trunk_to_bl_hip_rotor']

            bl_hip_pos_hom = base_to_trunk_hom @ transformations['trunk_to_bl_hip']

            bl_hip_rotor_pos = bl_hip_rotor_pos_hom[:3,3]
            
            bl_hip_pos = bl_hip_pos_hom[:3,3]


            
            ee_pos = base_to_trunk_rot.T @(foot_pos_bl - bl_hip_pos)
            ee_pos_hom = np.append(ee_pos,1)
            ee_pos = ee_pos_hom[:3]


            ee_pos_bl = np.array([ee_pos[0], ee_pos[2]])
            leg = 'BL'
            #print('ee_pos', ee_pos)
            bl_jj_2R = inverseKinematics2R(ee_pos_bl, leg, robot_data.link_lengths, safety=False)
            

            ee_pos_bl_3R = ee_pos
            bl_jj = inverseKinematics3R(ee_pos_bl_3R, leg,robot_data.link_lengths, joint_angles_2r = bl_jj_2R, safety=False)

        #axes and rotation
            axes_bl_hip =  extract_rotation_axes_from_urdf(urdf_file_path, 'FR_hip') # hip angle 

            #print('axes_fl_hip', axes_fl_hip)

            quat_bl_hip = quaternion_from_axis_angle(axes_bl_hip, bl_jj[0])

            torso_bl_hip_rel_pos = transformations['trunk_to_bl_hip'][:3,3]

            transform_msg_bl_hip = generate_link_transform_new(node, 'bl_hip', 'trunk', torso_bl_hip_rel_pos, quat_bl_hip)

            

            

            axes_bl_thigh =  extract_rotation_axes_from_urdf(urdf_file_path, 'FR_thigh') # thigh angle 

            quat_bl_thigh = quaternion_from_axis_angle(axes_bl_thigh, bl_jj[1])

            bl_hip_thigh_rel_pos = transformations['bl_hip_to_bl_thigh'][:3,3]

            transform_msg_bl_thigh = generate_link_transform_new(node, 'bl_thigh', 'bl_hip', bl_hip_thigh_rel_pos, quat_bl_thigh)

            

            axes_bl_calf = extract_rotation_axes_from_urdf(urdf_file_path, 'FR_calf') # calf angle

            quat_bl_calf = quaternion_from_axis_angle(axes_bl_calf, bl_jj[2])

            bl_thigh_calf_rel_pos = transformations['bl_thigh_to_bl_calf'][:3,3]

            transform_msg_bl_calf = generate_link_transform_new(node, 'bl_calf', 'bl_thigh', bl_thigh_calf_rel_pos, quat_bl_calf)

            

            zero_rel_ori = np.array([0.0,0.0,0.0,1.0])

            bl_calf_foot_rel_pos = transformations['bl_calf_to_bl_foot'][:3,3]

            transform_msg_bl_foot = generate_link_transform_new(node, 'bl_foot', 'bl_calf', bl_calf_foot_rel_pos, zero_rel_ori)

            
            
            torso_bl_hip_rotor_rel_pos = transformations['trunk_to_bl_hip_rotor'][:3,3]

            transform_msg_bl_hip_rotor = generate_link_transform_new(node, 'bl_hip_rotor', 'trunk', torso_bl_hip_rotor_rel_pos, zero_rel_ori)

            

            bl_hip_thigh_rotor_rel_pos = transformations['bl_hip_to_bl_thigh_rotor'][:3,3]
            
            transform_msg_bl_thigh_rotor = generate_link_transform_new(node, 'bl_thigh_rotor', 'bl_hip', bl_hip_thigh_rotor_rel_pos, zero_rel_ori)

            

            bl_thigh_calf_rotor_rel_pos = transformations['bl_thigh_to_bl_calf_rotor'][:3,3]

            transform_msg_bl_calf_rotor = generate_link_transform_new(node, 'bl_calf_rotor', 'bl_thigh', bl_thigh_calf_rotor_rel_pos, zero_rel_ori)

            


            broadcaster.sendTransform(transform_msg_torso)

            broadcaster.sendTransform(transform_msg_fl_hip)

            broadcaster.sendTransform(transform_msg_fl_thigh)

            broadcaster.sendTransform(transform_msg_fl_calf)

            broadcaster.sendTransform(transform_msg_fl_foot)

            broadcaster.sendTransform(transform_msg_fl_hip_rotor)

            broadcaster.sendTransform(transform_msg_fl_thigh_rotor)

            broadcaster.sendTransform(transform_msg_fl_calf_rotor)


            


            broadcaster.sendTransform(transform_msg_br_hip)

            broadcaster.sendTransform(transform_msg_br_thigh)

            broadcaster.sendTransform(transform_msg_br_calf)

            broadcaster.sendTransform(transform_msg_br_foot)

            broadcaster.sendTransform(transform_msg_br_hip_rotor)

            broadcaster.sendTransform(transform_msg_br_thigh_rotor)

            broadcaster.sendTransform(transform_msg_br_calf_rotor)



            broadcaster.sendTransform(transform_msg_fr_hip)

            broadcaster.sendTransform(transform_msg_fr_thigh)

            broadcaster.sendTransform(transform_msg_fr_calf)

            broadcaster.sendTransform(transform_msg_fr_foot)

            broadcaster.sendTransform(transform_msg_fr_hip_rotor)

            broadcaster.sendTransform(transform_msg_fr_thigh_rotor)

            broadcaster.sendTransform(transform_msg_fr_calf_rotor)


            broadcaster.sendTransform(transform_msg_bl_hip)

            broadcaster.sendTransform(transform_msg_bl_thigh)

            broadcaster.sendTransform(transform_msg_bl_calf)

            broadcaster.sendTransform(transform_msg_bl_foot)

            broadcaster.sendTransform(transform_msg_bl_hip_rotor)

            broadcaster.sendTransform(transform_msg_bl_thigh_rotor)

            broadcaster.sendTransform(transform_msg_bl_calf_rotor) 

            rclpy.spin_once (node, timeout_sec= 0.1)

            rate.sleep()   

            t_prev = t

            t += step_time

            if (t - t_prev) > step_time:

                t = t - 1e-7



    

    #node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()