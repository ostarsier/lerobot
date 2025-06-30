import time
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus

def main():

    config = FeetechMotorsBusConfig(
                port="/dev/ttyACM0",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            )

    motors_bus = FeetechMotorsBus(config)
    motors_bus.connect()

    position = motors_bus.read("Present_Position")
    print("position:", position)

    # move from a few motor steps as an example
    # few_steps = 500
    # motors_bus.write("Goal_Position", position + few_steps)

    # when done, consider disconnecting
    motors_bus.disconnect()

if __name__ == "__main__":
    main()
