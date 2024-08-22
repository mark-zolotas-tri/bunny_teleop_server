import time
from copy import deepcopy
from functools import partial
from threading import Lock
from typing import Optional, Tuple

import numpy as np
from dex_retargeting.seq_retarget import SeqRetargeting
from bunny_teleop.single_teleop_server import TeleopServer
from bunny_teleop.init_config import SingleInitializationConfig
from hand_msgs.msg import BimanualHandDetection
from pytransform3d import rotations
from pytransform3d import transformations as pt
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

from bunny_teleop_server.communication.visualizer_base import TeleopVisualizerBase
from bunny_teleop_server.control.base import BaseMotionControl
from bunny_teleop_server.nodes.bimanual_hand_monitor_node import BimanualMonitorNode
from bunny_teleop_server.utils.robot_utils import LPFilter, LPRotationFilter


class SingleRobotTeleopNode(BimanualMonitorNode):
    def __init__(
        self,
        detection_topic_name: str,
        hand_name: str,
        teleop_port: int,
        motion_control: BaseMotionControl,
        need_init=True,
        retargeting_optimizer: Optional[SeqRetargeting] = None,
        robot_viz: Optional[TeleopVisualizerBase] = None,
        low_pass_smoothing_wrist=0.1,
        disable_orientation_control=False,
        motion_scaling_factor=1.0,
        teleop_host="localhost",
        verbose=False,
    ):
        super().__init__(
            detection_topic_name,
            need_init,
            verbose,
        )

        # Control class of two hands
        use_gpu = motion_control.is_use_gpu()
        self.motion_control_group = (
            MutuallyExclusiveCallbackGroup() if use_gpu else ReentrantCallbackGroup()
        )
        self.hand_arm = SingleArmHandNode(
            hand_name,
            self,
            retargeting_optimizer=retargeting_optimizer,
            low_pass_smoothing_wrist=low_pass_smoothing_wrist,
            motion_control=motion_control,
            disable_orientation_control=disable_orientation_control,
            motion_scaling_factor=motion_scaling_factor,
        )

        # Server publisher
        self._teleop_publish_group = MutuallyExclusiveCallbackGroup()
        self.publish_dt = 1 / 60
        self.publish_timer = self.create_timer(
            self.publish_dt,
            self.publish_periodically,
            callback_group=self._teleop_publish_group,
        )
        self.publish_timer.cancel()
        self.hand_index = 0 if "left" in hand_name else 1

        # Web visualizer for teleoperation and environment
        # A port negative port number will disable the web visualizer
        if robot_viz is not None:
            self.viz_dt = 1 / 30
            self.use_web_viz = True
            self.robot_viz = robot_viz
            self._viz_group = MutuallyExclusiveCallbackGroup()
            self.viz_timer = self.create_timer(
                self.viz_dt,
                self.viz_periodically,
                callback_group=self._viz_group,
            )
            self.viz_timer.cancel()
        else:
            self.use_web_viz = False

        # Teleoperation server, which needs to be initialized based the init_config from teleoperation client
        self.teleop_server = TeleopServer(teleop_port, teleop_host)
        self.robot_base_pose = np.zeros(7)
        self.client_qpos = np.array([])
        self.client_ee_pose = np.zeros(7) # EE pose at robot arm's base
        self.client_lock = Lock()

        # Wait for the first initialization config to come
        init_config = self.teleop_server.wait_for_init_config()
        self.apply_teleop_init_config(init_config)

    def apply_teleop_init_config(self, init_config: SingleInitializationConfig):
        self.client_qpos = init_config.init_qpos
        self.robot_base_pose = init_config.robot_base_pose
        print(f"Single Arm Initialization!")

        # Build index mapping for retargeting optimizer and check joint completeness
        self.hand_arm.set_optimizer_index(
            init_config.get_joint_index_mapping(
                self.hand_arm.retargeting_joint_names
            )
        )

        # Build index mapping for motion control and check joint completeness
        joint_names = self.hand_arm.motion_control.get_joint_names()
        self.hand_arm.set_motion_control_index(
            init_config.get_joint_index_mapping(joint_names)
        )

        if set(self.hand_arm.retargeting_joint_names).union(set(joint_names)) != set(
            init_config.joint_names
        ):
            raise ValueError(
                f"Server and client side joint name mismatch for hand_arm.\n"
                f"Optimizer joints: {self.hand_arm.retargeting_joint_names}\n"
                f"Motion control joints: {joint_names}\n"
                f"Teleoperation client joints: {init_config.joint_names}"
            )

        motion_control_qpos = self.client_qpos[
            self.hand_arm.index_client2control
        ]
        self.hand_arm.set_control_qpos(motion_control_qpos)
        motion_control_ee_pose = self.hand_arm.motion_control.compute_ee_pose(
            motion_control_qpos
        )

        self.client_ee_pose[:] = motion_control_ee_pose
        init_ee_pose = motion_control_ee_pose

        # Setup transformation matrix between robot and world
        base_mat = pt.transform_from_pq(self.robot_base_pose)
        global_ee_pose = base_mat @ pt.transform_from_pq(self.client_ee_pose)

        # Compute the initialization frame pose in the robot world
        init_frame_rot = global_ee_pose[:3, :3]
        init_frame_pos = global_ee_pose[:3, 3]

        # Set up the global pose for each robot controller
        global_init_pose = pt.transform_from(init_frame_rot, init_frame_pos)
        self.hand_arm.set_init2base(
            pt.pq_from_transform(np.linalg.inv(base_mat) @ global_init_pose)
        )

        if self.use_web_viz:
            # Wait for the viz timer to be completed
            while not self.viz_timer.is_canceled():
                time.sleep(1e-3)

            # Init robot base pose in visualizer
            self.robot_viz.init_robot_base_pose(self.robot_base_pose)

            # Create initialization frame visualization
            self.robot_viz.create_init_frame(global_init_pose)

            # Set joint mapping for the viz
            self.robot_viz.set_joint_index_mapping(
                init_config.get_joint_index_mapping(
                    self.robot_viz.get_robot_joint_names()
                )
            )
            self.robot_viz.update_robot(init_config.init_qpos)

    def update_last_retargeted_qpos(self):
        joints = self.get_last_hand_joints()
        qpos = self.hand_arm.update_last_retargeted_qpos(joints[self.hand_index])

        # Update last retargeted qpos
        with self.client_lock:
            self.client_qpos[self.hand_arm.index_client2optimizer] = qpos[
                self.hand_arm.retargeting_joint_indices
            ]

    def publish_periodically(self):
        if self.initialized:
            with self.hand_arm.motion_control_lock:
                self.client_qpos[self.hand_arm.index_client2control[:7]] = (
                    self.hand_arm.last_control_qpos[:7]
                )
                ee_target_pose = self.hand_arm.last_target_ee_pose

            with self.client_lock:
                self.client_ee_pose[:] = ee_target_pose
                qpos = deepcopy(self.client_qpos)
                ee_pose = deepcopy(self.client_ee_pose)

            self.teleop_server.send_teleop_cmd(qpos, ee_pose)

    def viz_periodically(self):
        if self.initialized:
            with self.client_lock:
                qpos = deepcopy(self.client_qpos)
                ee_pose = deepcopy(self.client_ee_pose)

            # Update ee_pose target visualization in web visualizer
            self.robot_viz.update_ee_target(ee_pose)

            # Update robot joint position visualization in web visualizer
            self.robot_viz.update_robot(qpos)

    def _after_init(self):
        print("Teleop Server: Initialization finished.")
        self.teleop_server.set_initialized()
        self.ready_to_reinit = True
        if self.use_web_viz:
            with self.client_lock:
                ee_poses = deepcopy(self.client_ee_pose)
            self.robot_viz.create_ee_target(ee_poses, self.robot_base_pose)

        self.hand_arm.start()
        self.publish_timer.reset()
        if self.use_web_viz:
            self.viz_timer.reset()

    def on_hand_detection(self, data: BimanualHandDetection):
        super().on_hand_detection(data)

        # If the monitor is already initialized but server is not, it means that we need to reinitialize
        if (
            not self.teleop_server.initialized
            and self.initialized
            and self.ready_to_reinit
        ):
            self.ready_to_reinit = False
            self.prepare_reinit()
            return

        # Teleoperation server will start to compute action only when the initialization has finished
        is_success = data.detected
        if self.initialized and is_success:
            if not self.teleop_server.initialized:
                self._after_init()

            # Update retargeting results
            self.update_last_retargeted_qpos()

        elif not self.initialized:
            if self.use_web_viz:
                self.robot_viz.update_init_viz(1 - self.init_process)
            else:
                print(self.init_process)

    def prepare_reinit(self):
        # First clear the initialization cache for perception part
        super().prepare_reinit()

        # Then clear the cache in retargeting, control, and motion filter
        if self.use_web_viz:
            self.viz_timer.cancel()
        self.publish_timer.cancel()
        self.hand_arm.clean_up()
        self.hand_arm.retargeting.reset()
        self.apply_teleop_init_config(self.teleop_server.last_init_config)


class SingleArmHandNode:
    def __init__(
        self,
        hand_type,
        node: SingleRobotTeleopNode,
        retargeting_optimizer: SeqRetargeting,
        low_pass_smoothing_wrist: float,
        motion_control: Optional[BaseMotionControl],
        disable_orientation_control: bool,
        motion_scaling_factor: float,
    ):
        self.hand_index = 0 if "left" in hand_type else 1
        self.node = node
        self.init2base = None

        self.retargeting = retargeting_optimizer
        robot = retargeting_optimizer.optimizer.robot
        indices = []
        names = []
        for index, name in enumerate(robot.dof_joint_names):
            indices.append(index)
            names.append(name)
        self.retargeting_joint_indices = np.array(indices, dtype=int)
        self.retargeting_joint_names = names

        # Filter
        self.wrist_pos_filter = LPFilter(alpha=low_pass_smoothing_wrist)
        self.wrist_rot_filter = LPRotationFilter(alpha=low_pass_smoothing_wrist)

        # Motion control
        # If motion_control is None, then we only use retargeted result for teleoperation
        # For example, in-hand manipulation with only a box, no online collision detection is necessary
        self.disable_orientation_control = disable_orientation_control
        self.motion_scaling_factor = motion_scaling_factor
        self.action_update_dt = 1 / 60
        self.motion_control = motion_control
        control_repeat = int(
            max(1, round(self.action_update_dt / self.motion_control.get_timestep()))
        )
        # self._action_sub_group = MutuallyExclusiveCallbackGroup()
        self._action_sub_group = self.node.motion_control_group
        self.action_timer = self.node.create_timer(
            self.action_update_dt,
            partial(self.update_action_periodically, repeat_times=control_repeat),
            callback_group=self._action_sub_group,
        )
        # Stop the timer since we do not want to start it before initialization
        self.action_timer.cancel()
        self.motion_control_lock = Lock()
        self.last_control_qpos = None
        self.last_target_ee_pose = None

        # Initialization config
        self.index_client2optimizer = None
        self.index_client2control = None

    def update_last_retargeted_qpos(self, joint: np.ndarray):
        # Compute the input vector
        retargeting_type = self.retargeting.optimizer.retargeting_type
        indices = self.retargeting.optimizer.target_link_human_indices
        if retargeting_type == "POSITION":
            indices = indices
            ref_value = joint[indices, :]
        else:
            origin_indices = indices[0, :]
            task_indices = indices[1, :]
            ref_value = joint[task_indices, :] - joint[origin_indices, :]

        retargeted_qpos = self.retargeting.retarget(ref_value)

        return retargeted_qpos

    def update_action_periodically(self, repeat_times):
        if self.node.initialized:
            ee_poses = self.node.get_last_wrist_poses()
            ee_pos = ee_poses[self.hand_index][:3] * self.motion_scaling_factor
            if self.disable_orientation_control:
                ee_quat = np.array([1, 0, 0, 0])
            else:
                ee_quat = ee_poses[self.hand_index][3:7]

            ee_pos = (
                rotations.q_prod_vector(self.init2base[3:7], ee_pos)
                + self.init2base[0:3]
            )
            ee_quat = rotations.concatenate_quaternions(self.init2base[3:7], ee_quat)

            filter_ee_pos = self.wrist_pos_filter.next(ee_pos)
            filter_ee_quat = self.wrist_rot_filter.next(ee_quat)

            # Combine with simulated environment
            target_ee_pose = np.concatenate([filter_ee_pos, filter_ee_quat])
            self.motion_control.step(filter_ee_pos, filter_ee_quat, repeat_times)

            control_qpos = self.motion_control.get_current_qpos()
            with self.motion_control_lock:
                self.last_control_qpos = control_qpos
                self.last_target_ee_pose = target_ee_pose

    def set_init2base(self, init2base: np.ndarray):
        self.init2base = init2base

    def set_control_qpos(self, qpos: np.ndarray):
        self.motion_control.set_current_qpos(qpos)
        with self.motion_control_lock:
            self.last_control_qpos = qpos

    def set_motion_control_index(self, indices: np.ndarray):
        self.index_client2control = indices

    def set_optimizer_index(self, indices: np.ndarray):
        self.index_client2optimizer = indices

    def start(self):
        self.action_timer.reset()

    def stop(self):
        self.action_timer.cancel()

    def clean_up(self):
        self.stop()
        self.wrist_pos_filter.reset()
        self.wrist_rot_filter.reset()
        self.index_client2control = None
        self.index_client2optimizer = None
        self.init2base = None
