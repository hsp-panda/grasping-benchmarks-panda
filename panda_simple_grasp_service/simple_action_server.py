#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import copy
import numpy as np
import rospy
import geometry_msgs
import moveit_commander
import moveit_msgs.msg
from tf.transformations import quaternion_from_matrix, quaternion_matrix

from panda_grasp_srv.srv import PandaGrasp, PandaGraspRequest, PandaGraspResponse, UserCmd


def all_close(goal, actual, tolerance):

    all_equal = True
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False
    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        return all_close(pose_to_list(goal), pose_to_list(actual), tolerance)

    return True


class PandaGraspServer(object):

    def __init__(self, service_name, publish_rviz):
        # --- configure moveit --- #
        moveit_commander.roscpp_initialize(sys.argv)
        self._robot = moveit_commander.RobotCommander()
        self._group_name = "panda_arm"
        self._move_group = moveit_commander.MoveGroupCommander(
            self._group_name)
        self._move_group.set_end_effector_link("panda_hand")
        self._eef_link = self._move_group.get_end_effector_link()

        self._move_group_hand = moveit_commander.MoveGroupCommander(
            "hand")

        self._move_group.set_max_velocity_scaling_factor(0.1)

        self._move_group.set_planner_id("RRTkConfigDefault")

        # display trajectories in Rviz
        self._publish_rviz = publish_rviz
        self._display_trajectory_publisher = None
        if self._publish_rviz:
            self._display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                                 moveit_msgs.msg.DisplayTrajectory,
                                                                 queue_size=10)

        self._scene = moveit_commander.PlanningSceneInterface()
        self._scene.remove_world_object()

        # --- User input service --- #
        self._cmd_srv = rospy.Service("PandaGraspServer/cmd", UserCmd, self.user_cmd)

        # --- configure ROS service server --- #
        self._grasp_service = rospy.Service(service_name, PandaGrasp,
                                            self.do_grasp)

        self._home_pose = geometry_msgs.msg.Pose()
        # top view
        self._home_pose.orientation.x = 1.
        self._home_pose.orientation.y = 0.0
        self._home_pose.orientation.z = 0.
        self._home_pose.orientation.w = 0.

        self._home_pose.position.x = 0.5
        self._home_pose.position.y = 0.0
        self._home_pose.position.z = 0.6

    def user_cmd(self, req):
        print("Received new command from user...")

        cmd = req.cmd.data

        if cmd=="help":
            print("available commands are:")
            print("go_home\nset_home\njoints_state\npose_ee\nmove_gripper")
            return True

        elif cmd == "go_home":
            self.go_home()
            return True

        elif cmd == "set_home":
            pos, quat = self._get_pose_from_user()
            if len(pos)==3 and len(quat)==4:
                self.set_home(pos, quat)
                return True
            else:
                return False

        elif cmd == "joints_state":
            joint_states = self.get_joints_state()
            print("joint poses: ", joint_states)
            gripper_poses = self.get_gripper_state()
            print("gripper poses: ", gripper_poses)
            return True

        elif cmd == "pose_ee":
            pos, quat = self.get_current_pose_EE()
            print("current gripper pose: ")
            print(pos)
            print(quat)
            return True
        elif cmd == "move_gripper":
            user_cmd = raw_input("Set desired gripper width:")
            width = float(user_cmd)
            print("required width ", width)

            self.command_gripper(width)
            return True

        else:
            print("unvalid command ", cmd)
            return False

    def set_home(self, pos, quat):
        self._home_pose.orientation.x = quat[0]
        self._home_pose.orientation.y = quat[1]
        self._home_pose.orientation.z = quat[2]
        self._home_pose.orientation.w = quat[3]

        self._home_pose.position.x = pos[0]
        self._home_pose.position.y = pos[1]
        self._home_pose.position.z = pos[2]

    def do_grasp(self, req):
        rospy.loginfo('%s: Executing grasp' %
                      (self._grasp_service.resolved_name))

        # move fingers in pre grasp pose
        # self.command_gripper(req.width.data)
        self.open_gripper()

        # --- define a pre-grasp point along the approach axis --- #
        p1 = quaternion_matrix([0., 0., 0., 1.])
        p1[:3, 3] = np.array([0., 0., -0.1])

        # transform grasp in matrix notation
        q_gp = req.grasp.pose.orientation
        p_gp = req.grasp.pose.position
        gp = quaternion_matrix([q_gp.x, q_gp.y, q_gp.z, q_gp.w])
        gp[:3, 3] = np.array([p_gp.x, p_gp.y, p_gp.z])

        # create pregrasp pose
        pregrasp = np.matmul(gp, p1)
        q_pregrasp = quaternion_from_matrix(pregrasp)

        pregrasp_pose = geometry_msgs.msg.Pose()

        pregrasp_pose.orientation.x = q_pregrasp[0]
        pregrasp_pose.orientation.y = q_pregrasp[1]
        pregrasp_pose.orientation.z = q_pregrasp[2]
        pregrasp_pose.orientation.w = q_pregrasp[3]

        pregrasp_pose.position.x = pregrasp[0, 3]
        pregrasp_pose.position.y = pregrasp[1, 3]
        pregrasp_pose.position.z = pregrasp[2, 3]

        print("pregrasp")

        self._move_group.set_pose_target(pregrasp_pose)
        self._move_group.go(wait=True)
        self._move_group.stop()
        self._move_group.clear_pose_targets()

        print("grasp pose ")
        print(req.grasp.pose)

        self._move_group.set_pose_target(req.grasp.pose)
        self._move_group.go(wait=True)
        self._move_group.stop()
        self._move_group.clear_pose_targets()


        # --- Execute planned trajectory to grasp pose --- #
        # self._move_group.execute(plan, wait=True)

        # --- Close fingers to try the grasp --- #
        ok = self.close_gripper()

        print("lift")
        lift_pose = req.grasp.pose
        lift_pose.position.z += 0.30

        self._move_group.set_pose_target(lift_pose)
        self._move_group.go(wait=True)
        self._move_group.stop()
        self._move_group.clear_pose_targets()

        # --- Check if grasp was successful --- #
        gripper_state = self.get_gripper_state()
        success = False if sum(gripper_state) <= 0.01 else True
        print("gripper_state ", gripper_state)
        print("Grasp success? :", success)

        # --- drop object out of workspace --- #
        print("gripper home")
        next_pose = lift_pose
        next_pose.orientation.x = 1
        next_pose.orientation.y = 0
        next_pose.orientation.z = 0
        next_pose.orientation.w = 0

        self._move_group.set_pose_target(next_pose)
        self._move_group.go(wait=True)
        self._move_group.stop()
        self._move_group.clear_pose_targets()

        print("back/right")
        next_pose.position.x = 0.4
        next_pose.position.y = -0.4

        self._move_group.set_pose_target(next_pose)
        self._move_group.go(wait=True)
        self._move_group.stop()
        self._move_group.clear_pose_targets()

        print("down")
        next_pose.position.z = 0.45

        self._move_group.set_pose_target(next_pose)
        self._move_group.go(wait=True)
        self._move_group.stop()
        self._move_group.clear_pose_targets()

        # --- Open fingers to drop the object --- #
        print("release object")
        ok = self.open_gripper()

        print("up")
        next_pose.position.z = 0.60

        self._move_group.set_pose_target(next_pose)
        self._move_group.go(wait=True)
        self._move_group.stop()
        self._move_group.clear_pose_targets()

        # --- go in home pose --- #
        print("home")
        self.go_home()

        return success

    def go_home(self):
        self._move_group.set_pose_target(self._home_pose)
        self._move_group.go(wait=True)
        self._move_group.stop()
        self._move_group.clear_pose_targets()

    def close_gripper(self):
        joint_goal = self._move_group_hand.get_current_joint_values()
        if joint_goal[0] > 0.03 and joint_goal[1] > 0.03:
            self.command_gripper(0.0)
        else:
            print("gripper already closed")

    def open_gripper(self):
        joint_goal = self._move_group_hand.get_current_joint_values()
        if joint_goal[0] <= 0.03 and joint_goal[1] <= 0.03:
            self.command_gripper(0.08)
        else:
            print("gripper already open")

    def command_gripper(self, gripper_width):

        joint_goal = self._move_group_hand.get_current_joint_values()
        joint_goal[0] = gripper_width/2.
        joint_goal[1] = gripper_width/2.

        self._move_group_hand.go(joint_goal, wait=True)
        self._move_group_hand.stop()
        current_joints = self._move_group_hand.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.001)

    def get_gripper_state(self):
        joint_poses = self._move_group_hand.get_current_joint_values()
        return joint_poses

    def get_joints_state(self):
        joint_poses = self._move_group.get_current_joint_values()
        return joint_poses

    def get_current_pose_EE(self):

        quaternion = [self._move_group.get_current_pose().pose.orientation.x,
                      self._move_group.get_current_pose().pose.orientation.y,
                      self._move_group.get_current_pose().pose.orientation.z,
                      self._move_group.get_current_pose().pose.orientation.w]

        position = [self._move_group.get_current_pose().pose.position.x,
                    self._move_group.get_current_pose().pose.position.y,
                    self._move_group.get_current_pose().pose.position.z]

        # print("current gripper pose is:")
        # print(position)
        # print(quaternion)
        return [position, quaternion]

    def _get_pose_from_user(self):
        position = [0]*3
        quaternion = [0, 1, 0, 0]

        user_cmd = raw_input("Set desired EE home position as 'x y z':")
        user_cmd = user_cmd.split()
        if not len(user_cmd) == 3:
            user_cmd = input("Wrong input. Try again: ")
            user_cmd = user_cmd.split()
            if not len(user_cmd) == 3:
                return [], []

        for i, cmd in enumerate(user_cmd):
            position[i] = float(cmd)

        user_cmd = raw_input("Set desired EE home orientation as quaternion 'x y z w': ")
        user_cmd = user_cmd.split()
        if not len(user_cmd) == 4:
            user_cmd = input("Wrong input. Try again: ")
            user_cmd = user_cmd.split()
            if not len(user_cmd) == 4:
                return [], []

        for i, cmd in enumerate(user_cmd):
            quaternion[i] = float(cmd)

        return position, quaternion


if __name__ == "__main__":
    # Initialize the ROS node.
    rospy.init_node("Panda_grasp_service")

    # Get configs.
    # rospy.get_param("~grasp_planner_service_name")
    grasp_service_name = "panda_grasp"
    publish_rviz = True  # rospy.get_param("~publish_rviz")

    # Instantiate the grasp planner.
    grasp_planner = PandaGraspServer(grasp_service_name, publish_rviz)

    # Spin forever.
    rospy.spin()
