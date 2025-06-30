# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SO100 Real Robot
from re import T
import time
from contextlib import contextmanager

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
from lerobot.common.robot_devices.robots.configs import So100RobotConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError

# NOTE:
# Sometimes we would like to abstract different env, or run this on a separate machine
# User can just move this single python class method gr00t/eval/service.py
# to their code or do the following line below
# sys.path.append(os.path.expanduser("~/Isaac-GR00T/gr00t/eval/"))
from service import ExternalRobotInferenceClient

# Import tqdm for progress bar
from tqdm import tqdm


class SO100Robot:
    def __init__(self, calibrate=False, enable_camera=False, camera_index=9):
        self.config = So100RobotConfig()
        self.calibrate = calibrate
        self.enable_camera = enable_camera
        self.camera_index = camera_index
        if not enable_camera:
            self.config.cameras = {}
        else:
            self.config.cameras = {"webcam": OpenCVCameraConfig(camera_index, 30, 640, 480, "bgr")}
        self.config.leader_arms = {}

        # remove the .cache/calibration/so100 folder
        if self.calibrate:
            import os
            import shutil

            calibration_folder = os.path.join(os.getcwd(), ".cache", "calibration", "so100")
            print("========> Deleting calibration_folder:", calibration_folder)
            if os.path.exists(calibration_folder):
                shutil.rmtree(calibration_folder)

        # Create the robot
        self.robot = make_robot_from_config(self.config)
        self.motor_bus = self.robot.follower_arms["main"]

    @contextmanager
    def activate(self):
        try:
            self.connect()
            # self.move_to_initial_pose()
            yield
        finally:
            self.disconnect()

    def connect(self):
        if self.robot.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "ManipulatorRobot is already connected. Do not run `robot.connect()` twice."
            )

        # Connect the arms
        self.motor_bus.connect()

        # We assume that at connection time, arms are in a rest position, and torque can
        # be safely disabled to run calibration and/or set robot preset configurations.
        self.motor_bus.write("Torque_Enable", TorqueMode.DISABLED.value)

        # Calibrate the robot
        self.robot.activate_calibration()

        self.set_so100_robot_preset()

        # Enable torque on all motors of the follower arms
        self.motor_bus.write("Torque_Enable", TorqueMode.ENABLED.value)
        print("robot present position:", self.motor_bus.read("Present_Position"))
        self.robot.is_connected = True

        self.camera = self.robot.cameras["webcam"] if self.enable_camera else None
        if self.camera is not None:
            self.camera.connect()
        print("================> SO100 Robot is fully connected =================")

    def set_so100_robot_preset(self):
        # 设置电机控制模式
        # Mode=0 表示位置控制模式（Position Control）
        # 这种模式下，电机将精确控制到指定位置
        self.motor_bus.write("Mode", 0)
        
        # 设置PID控制器参数
        # P_Coefficient (比例系数):
        # - 调低P值（从默认32改为10）以减少抖动
        # - 较低的P值使运动更平滑，但响应速度会稍慢
        # self.motor_bus.write("P_Coefficient", 16)
        self.motor_bus.write("P_Coefficient", 10)
        
        # I_Coefficient (积分系数) 和 D_Coefficient (微分系数)
        # - I_Coefficient=0: 不使用积分项，避免累积误差
        # - D_Coefficient=32: 使用微分项来抑制抖动
        self.motor_bus.write("I_Coefficient", 0)
        self.motor_bus.write("D_Coefficient", 32)
        
        # 锁定配置以确保加速参数生效
        # 这一步是必要的，因为加速参数需要写入EPROM地址
        # 才能在重启后保持生效
        self.motor_bus.write("Lock", 0)
        
        # 设置加速参数
        # Maximum_Acceleration=254:
        # - 设置最大加速度为254（最大值）
        # - 这将使电机加速和减速更快
        # - 注意：这个配置不在官方STS3215内存表中
        self.motor_bus.write("Maximum_Acceleration", 254)
        
        # 设置加速度
        # Acceleration=254:
        # - 设置加速度为254（最大值）
        # - 这将使电机运动更快速
        self.motor_bus.write("Acceleration", 254)

    def move_to_initial_pose(self):
        current_state = self.robot.capture_observation()["observation.state"]
        print("current_state", current_state)
        # print all keys of the observation
        print("observation keys:", self.robot.capture_observation().keys())

        current_state[0] = 90
        current_state[2] = 90
        current_state[3] = 90
        self.robot.send_action(current_state)
        time.sleep(2)

        current_state[4] = -70
        current_state[5] = 30
        current_state[1] = 90
        self.robot.send_action(current_state)
        time.sleep(2)

        print("----------------> SO100 Robot moved to initial pose")

    def go_home(self):
        # [ 88.0664, 156.7090, 135.6152,  83.7598, -89.1211,  16.5107]
        print("----------------> SO100 Robot moved to home pose")
        home_state = torch.tensor([88.0664, 156.7090, 135.6152, 83.7598, -89.1211, 16.5107])
        self.set_target_state(home_state)
        time.sleep(2)

    def get_observation(self):
        return self.robot.capture_observation()

    def get_current_state(self):
        return self.get_observation()["observation.state"].data.numpy()

    def get_current_img(self):
        img = self.get_observation()["observation.images.webcam"].data.numpy()
        # convert bgr to rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def set_target_state(self, target_state: torch.Tensor):
        self.robot.send_action(target_state)

    def enable(self):
        self.motor_bus.write("Torque_Enable", TorqueMode.ENABLED.value)

    def disable(self):
        self.motor_bus.write("Torque_Enable", TorqueMode.DISABLED.value)

    def disconnect(self):
        self.disable()
        self.robot.disconnect()
        self.robot.is_connected = False
        print("================> SO100 Robot disconnected")



if __name__ == "__main__":
       
        robot = SO100Robot(calibrate=False, enable_camera=False)
        with robot.activate():
            home_state = torch.tensor([0, -34, -89, 44, -0, 0])
            # robot.set_target_state(home_state)
            print("current state:", robot.get_current_state())
            

