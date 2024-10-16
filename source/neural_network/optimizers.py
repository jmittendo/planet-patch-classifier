# This file is part of planet-patch-classifier, a Python tool for generating and
# classifying planet patches from satellite imagery via unsupervised machine learning
# Copyright (C) 2024  Jan Mittendorf (jmittendo)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from collections.abc import Iterable

import torch
from torch import no_grad
from torch.optim import Optimizer


class LARS(Optimizer):
    # See: https://arxiv.org/pdf/1708.03888.pdf

    def __init__(
        self,
        parameters: Iterable,
        base_learning_rate: float,
        lars_coefficient: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 0.0001,
    ) -> None:
        defaults = {
            "base_learning_rate": base_learning_rate,
            "lars_coefficient": lars_coefficient,
            "momentum": momentum,
            "weight_decay": weight_decay,
        }

        super().__init__(parameters, defaults)

        for param_group in self.param_groups:
            param_group["velocities"] = [
                torch.zeros(param.size()) for param in param_group["params"]
            ]

    @no_grad
    def step(self):
        for param_group in self.param_groups:
            params = param_group["params"]
            param_velocities = param_group["velocities"]
            lars_coefficient = param_group["lars_coefficient"]
            weight_decay = param_group["weight_decay"]
            momentum = param_group["momentum"]
            base_learning_rate = param_group["base_learning_rate"]

            for param, param_velocity in zip(params, param_velocities):
                if param.grad is None:
                    continue

                param_velocity = param_velocity.to(param.device)

                param_gradient = param.grad.data
                param_norm = param.norm()

                local_learning_rate = lars_coefficient * (
                    param_norm / (param_gradient.norm() + weight_decay * param_norm)
                )

                param_velocity.mul_(momentum).add_(
                    base_learning_rate
                    * local_learning_rate
                    * (param_gradient + weight_decay * param)
                )

                param.sub_(param_velocity)
