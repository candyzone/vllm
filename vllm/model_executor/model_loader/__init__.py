from typing import Optional

from torch import nn

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ParallelConfig, SchedulerConfig,
                         VisionLanguageConfig)
from vllm.model_executor.model_loader.loader import (BaseModelLoader,
                                                     get_model_loader)
from vllm.model_executor.model_loader.utils import (
    get_architecture_class_name, get_model_architecture)

loader=None
def read_weight(model_config: ModelConfig, load_config: LoadConfig) -> None:
    global loader
    loader = get_model_loader(load_config)
    loader.weight = loader._get_weights_iterator(model_config.model,
                                                 model_config.revision,
                                                 True)

def get_model(*, model_config: ModelConfig, load_config: LoadConfig,
              device_config: DeviceConfig, parallel_config: ParallelConfig,
              scheduler_config: SchedulerConfig,
              lora_config: Optional[LoRAConfig],
              vision_language_config: Optional[VisionLanguageConfig],
              cache_config: CacheConfig) -> nn.Module:
    return loader.init_model(model_config=model_config,
                             device_config=device_config,
                             lora_config=lora_config,
                             vision_language_config=vision_language_config,
                             parallel_config=parallel_config,
                             scheduler_config=scheduler_config,
                             cache_config=cache_config)

def load_weight(*, model_config: ModelConfig, load_config: LoadConfig,
              device_config: DeviceConfig, parallel_config: ParallelConfig,
              scheduler_config: SchedulerConfig,
              lora_config: Optional[LoRAConfig],
              vision_language_config: Optional[VisionLanguageConfig],
              cache_config: CacheConfig) -> nn.Module:
    return loader.load_weight(model_config=model_config,
                             device_config=device_config,
                             lora_config=lora_config,
                             vision_language_config=vision_language_config,
                             parallel_config=parallel_config,
                             scheduler_config=scheduler_config,
                             cache_config=cache_config)

__all__ = [
    "get_model", "get_model_loader", "BaseModelLoader", "load_weight",
    "get_architecture_class_name", "get_model_architecture"
]
