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
            self.move_to_initial_pose()
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

    def __del__(self):
        self.disconnect()




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
            "video.webcam": img[np.newaxis, :, :, :],
            "state.single_arm": state[:5][np.newaxis, :].astype(np.float64),
            "state.gripper": state[5:6][np.newaxis, :].astype(np.float64),
            "annotation.human.action.task_description": [self.language_instruction],
        }
        start_time = time.time()
        res = self.policy.get_action(obs_dict)
        print("Inference query time taken", time.time() - start_time)
        return res

    def sample_action(self):
        obs_dict = {
            "video.webcam": np.zeros((1, self.img_size[0], self.img_size[1], 3), dtype=np.uint8),
            "state.single_arm": np.zeros((1, 5)),
            "state.gripper": np.zeros((1, 1)),
            "annotation.human.action.task_description": [self.language_instruction],
        }
        return self.policy.get_action(obs_dict)




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



if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--action_horizon", type=int, default=16)
    parser.add_argument("--actions_to_execute", type=int, default=350)
    parser.add_argument("--camera_index", type=int, default=0)
    args = parser.parse_args()

    ACTIONS_TO_EXECUTE = args.actions_to_execute
    ACTION_HORIZON = args.action_horizon
    MODALITY_KEYS = ["single_arm", "gripper"]

    client = Gr00tRobotInferenceClient(
        host=args.host,
        port=args.port,
        language_instruction="Pick up the blocks and place them on the plate.",
    )

    robot = SO100Robot(calibrate=False, enable_camera=True, camera_index=args.camera_index)
    with robot.activate():
        for i in tqdm(range(ACTIONS_TO_EXECUTE), desc="Executing actions"):
            img = robot.get_current_img()
            view_img(img)
            state = robot.get_current_state()
            action = client.get_action(img, state)
            start_time = time.time()
            for i in range(ACTION_HORIZON):
                concat_action = np.concatenate(
                    [np.atleast_1d(action[f"action.{key}"][i]) for key in MODALITY_KEYS],
                    axis=0,
                )
                assert concat_action.shape == (6,), concat_action.shape
                robot.set_target_state(torch.from_numpy(concat_action))
                time.sleep(0.01)

                # get the realtime image
                img = robot.get_current_img()
                view_img(img)

                # 0.05*16 = 0.8 seconds
                print("executing action", i, "time taken", time.time() - start_time)
            print("Action chunk execution time taken", time.time() - start_time)
