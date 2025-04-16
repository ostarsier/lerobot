# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import abc
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, TypeVar

import draccus
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import CONFIG_NAME
from huggingface_hub.errors import HfHubHTTPError

from lerobot.common.optim.optimizers import OptimizerConfig
from lerobot.common.optim.schedulers import LRSchedulerConfig
from lerobot.common.utils.hub import HubMixin
from lerobot.common.utils.utils import auto_select_torch_device, is_amp_available, is_torch_device_available
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature

# Generic variable that is either PreTrainedConfig or a subclass thereof
T = TypeVar("T", bound="PreTrainedConfig")


@dataclass
class PreTrainedConfig(draccus.ChoiceRegistry, HubMixin, abc.ABC):
    """
    策略模型的基础配置类。
    
    主要功能：
    1. 定义策略模型的输入输出特征
    2. 管理特征的归一化和反归一化
    3. 提供模型配置的保存和加载功能
    4. 支持从本地或HuggingFace Hub加载预训练模型
    
    Args:
        n_obs_steps: 观测窗口大小，定义了策略需要考虑的过去时间步数
        normalization_mapping: 特征归一化映射，用于定义不同特征的归一化方式
        input_features: 输入特征字典，定义了模型的输入数据结构
        output_features: 输出特征字典，定义了模型的输出数据结构
        device: 模型运行的设备（cuda/cpu/mp）
        use_amp: 是否使用混合精度训练（Automatic Mixed Precision）
    """

    n_obs_steps: int = 1
    normalization_mapping: dict[str, NormalizationMode] = field(default_factory=dict)

    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)

    device: str | None = None  # cuda | cpu | mp
    use_amp: bool = False

    def __post_init__(self):
        """
        初始化后处理：
        1. 自动选择可用的设备
        2. 根据设备类型自动调整AMP设置
        """
        self.pretrained_path = None
        if not self.device or not is_torch_device_available(self.device):
            auto_device = auto_select_torch_device()
            logging.warning(f"Device '{self.device}' is not available. Switching to '{auto_device}'.")
            self.device = auto_device.type

        # 自动禁用不支持AMP的设备
        if self.use_amp and not is_amp_available(self.device):
            logging.warning(
                f"Automatic Mixed Precision (amp) is not available on device '{self.device}'. Deactivating AMP."
            )
            self.use_amp = False

    @property
    def type(self) -> str:
        """
        获取策略类型名称
        """
        return self.get_choice_name(self.__class__)

    @abc.abstractproperty
    def observation_delta_indices(self) -> list | None:
        """
        抽象属性：定义观测值的差分索引
        用于处理观测值的时间差分
        """
        raise NotImplementedError

    @abc.abstractproperty
    def action_delta_indices(self) -> list | None:
        """
        抽象属性：定义动作的差分索引
        用于处理动作值的时间差分
        """
        raise NotImplementedError

    @abc.abstractproperty
    def reward_delta_indices(self) -> list | None:
        """
        抽象属性：定义奖励的差分索引
        用于处理奖励值的时间差分
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_optimizer_preset(self) -> OptimizerConfig:
        """
        获取优化器预设配置
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        """
        获取学习率调度器预设配置
        """
        raise NotImplementedError

    @abc.abstractmethod
    def validate_features(self) -> None:
        """
        验证特征配置的有效性
        """
        raise NotImplementedError

    @property
    def robot_state_feature(self) -> PolicyFeature | None:
        """
        获取机器人状态特征
        """
        for _, ft in self.input_features.items():
            if ft.type is FeatureType.STATE:
                return ft
        return None

    @property
    def env_state_feature(self) -> PolicyFeature | None:
        """
        获取环境状态特征
        """
        for _, ft in self.input_features.items():
            if ft.type is FeatureType.ENV:
                return ft
        return None

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        """
        获取所有视觉特征
        """
        return {key: ft for key, ft in self.input_features.items() if ft.type is FeatureType.VISUAL}

    @property
    def action_feature(self) -> PolicyFeature | None:
        """
        获取动作特征
        """
        for _, ft in self.output_features.items():
            if ft.type is FeatureType.ACTION:
                return ft
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        """
        保存模型配置到指定目录
        """
        with open(save_directory / CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(self, f, indent=4)

    @classmethod
    def from_pretrained(
        cls: Type[T],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **policy_kwargs,
    ) -> T:
        """
        从本地路径或HuggingFace Hub加载预训练模型配置
        
        Args:
            pretrained_name_or_path: 模型名称或路径
            force_download: 是否强制下载
            resume_download: 是否续传
            proxies: 代理设置
            token: HuggingFace访问令牌
            cache_dir: 缓存目录
            local_files_only: 是否只使用本地文件
            revision: 版本号
            policy_kwargs: 其他策略参数
        
        Returns:
            配置对象实例
        """
        model_id = str(pretrained_name_or_path)
        config_file: str | None = None
        if Path(model_id).is_dir():
            if CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, CONFIG_NAME)
            else:
                print(f"{CONFIG_NAME} not found in {Path(model_id).resolve()}")
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                ) from e

        # HACK: 临时解决方案，理想情况下应该通过draccus原生支持
        cli_overrides = policy_kwargs.pop("cli_overrides", [])
        return draccus.parse(cls, config_file, args=cli_overrides)
