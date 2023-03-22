# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Lint as: python3
"""Config file for synthetic dataset."""

import ml_collections
import torch
import math
import numpy as np
from datetime import timedelta

from configs.ksphere.sample_complexity.base import get_config as get_base_config

def get_config():
  config = get_base_config()
  config.data.data_samples = 1
  #logging
  logging = config.logging
  logging.log_path = f'logs/ksphere/sample_complexity/'
  logging.log_name = f'n={config.data.data_samples}'

  return config
