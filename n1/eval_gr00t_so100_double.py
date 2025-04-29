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
from lerobot.common.robot_devices.robots.configs import So100DoubleRobotConfig
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

#################################################################################


class SO100Robot:
    def __init__(self, calibrate=False, enable_camera=False, camera_index=9):
        self.config = So100DoubleRobotConfig()
        self.calibrate = calibrate
        self.enable_camera = enable_camera
        self.camera_index = camera_index
        if not enable_camera:
            self.config.cameras = {}
        else:
            self.config.cameras = {"top": OpenCVCameraConfig(camera_index, 30, 640, 480, "bgr")}

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

        for name in self.robot.follower_arms:
            print(f"Connecting {name} follower arm.")
            self.robot.follower_arms[name].connect()

        for name in self.robot.follower_arms:
            self.robot.follower_arms[name].write("Torque_Enable", TorqueMode.DISABLED.value)

        # Calibrate the robot
        self.robot.activate_calibration()

        self.set_so100_robot_preset()

        # Enable torque on all motors of the follower arms
        self.enable()
        self.robot.is_connected = True

        self.camera = self.robot.cameras["top"] if self.enable_camera else None
        if self.camera is not None:
            self.camera.connect()
        print("================> SO100 Robot is fully connected =================")

    def set_so100_robot_preset(self):
        for name in self.robot.follower_arms:
            # Mode=0 for Position Control
            self.robot.follower_arms[name].write("Mode", 0)
            # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
            self.robot.follower_arms[name].write("P_Coefficient", 16)
            # Set I_Coefficient and D_Coefficient to default value 0 and 32
            self.robot.follower_arms[name].write("I_Coefficient", 0)
            self.robot.follower_arms[name].write("D_Coefficient", 32)
            # Close the write lock so that Maximum_Acceleration gets written to EPROM address,
            # which is mandatory for Maximum_Acceleration to take effect after rebooting.
            self.robot.follower_arms[name].write("Lock", 0)
            # Set Maximum_Acceleration to 254 to speedup acceleration and deceleration of
            # the motors. Note: this configuration is not in the official STS3215 Memory Table
            self.robot.follower_arms[name].write("Maximum_Acceleration", 254)
            self.robot.follower_arms[name].write("Acceleration", 254)

    def get_observation(self):
        return self.robot.capture_observation()

    def get_current_state(self):
        return self.get_observation()["observation.state"].data.numpy()

    def get_current_img(self):
        img = self.get_observation()["observation.images.top"].data.numpy()
        # convert bgr to rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def set_target_state(self, target_state: torch.Tensor):
        self.robot.send_action(target_state)

    def enable(self):
        for name in self.robot.follower_arms:
            self.robot.follower_arms[name].write("Torque_Enable", TorqueMode.ENABLED.value)

    def disable(self):
        for name in self.robot.follower_arms:
            self.robot.follower_arms[name].write("Torque_Enable", TorqueMode.DISABLED.value)

    def disconnect(self):
        self.disable()
        self.robot.disconnect()
        self.robot.is_connected = False
        print("================> SO100 Robot disconnected")

    def __del__(self):
        self.disconnect()


#################################################################################


class Gr00tRobotInferenceClient:
    def __init__(
        self,
        host="localhost",
        port=5555,
        language_instruction="Pick up the fruits and place them on the plate.",
    ):
        self.language_instruction = language_instruction
        # 480, 640
        self.img_size = (480, 640)
        self.policy = ExternalRobotInferenceClient(host=host, port=port)

    def get_action(self, img, state):
        obs_dict = {
            "video.left": img[np.newaxis, :, :, :],
            "state.right_arm": state[:5][np.newaxis, :].astype(np.float64),
            "state.gripper": state[5:6][np.newaxis, :].astype(np.float64),
            "state.left_arm": state[6:11][np.newaxis, :].astype(np.float64),
            "annotation.human.action.task_description": [self.language_instruction],
        }
        start_time = time.time()
        res = self.policy.get_action(obs_dict)
        print("Inference query time taken", time.time() - start_time)
        return res


#################################################################################


def view_img(img, img2=None):
    """
    This is a matplotlib viewer since cv2.imshow can be flaky in lerobot env
    also able to overlay the image to ensure camera view is alligned to training settings
    """
    plt.imshow(img)
    if img2 is not None:
        plt.imshow(img2, alpha=0.5)
    plt.axis("off")
    plt.pause(0.001)  # Non-blocking show
    plt.clf()  # Clear the figure for the next frame


#################################################################################

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--action_horizon", type=int, default=16)
    parser.add_argument("--actions_to_execute", type=int, default=350000)
    parser.add_argument("--camera_index", type=int, default=0)
    args = parser.parse_args()
    args.use_policy = True

    ACTIONS_TO_EXECUTE = args.actions_to_execute
    ACTION_HORIZON = (
        args.action_horizon
    )  # we will execute only some actions from the action_chunk of 16
    MODALITY_KEYS = ["right_arm", "gripper", "left_arm"]

    client = Gr00tRobotInferenceClient(
        host=args.host,
        port=args.port,
        language_instruction="pick_put_double",
    )

    robot = SO100Robot(calibrate=False, enable_camera=True, camera_index=args.camera_index)
    with robot.activate():
        for i in tqdm(range(ACTIONS_TO_EXECUTE), desc="Executing actions"):
            img = robot.get_current_img()
            view_img(img)
            state = robot.get_current_state()
            print(f'state {state}')
            action = client.get_action(img, state)
            start_time = time.time()
            for i in range(ACTION_HORIZON):
                concat_action = np.concatenate(
                    [np.atleast_1d(action[f"action.{key}"][i]) for key in MODALITY_KEYS],
                    axis=0,
                )
                assert concat_action.shape == (11,), concat_action.shape
                robot.set_target_state(torch.from_numpy(concat_action))
                print(f'send action {i} to robot {concat_action}')
                # get the realtime image
                img = robot.get_current_img()
                view_img(img)

                # 0.05*16 = 0.8 seconds
                print("executing action", i, "time taken", time.time() - start_time)
            print("Action chunk execution time taken", time.time() - start_time)
