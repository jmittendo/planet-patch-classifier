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

import torch
from torch import Tensor
from torch.nn import Module


class NTXentLoss(Module):
    # See: https://arxiv.org/abs/2002.05709

    def __init__(self, temperature: float) -> None:
        super().__init__()

        self._temperature = temperature

    def forward(self, batch_tensor: Tensor) -> Tensor:
        batch_size = batch_tensor.size(0)

        if batch_size % 2 != 0:
            raise ValueError(
                "NT-Xent loss expects an even batch size.\n"
                f"Size of the input batch: {batch_size}"
            )

        vector_norms = batch_tensor.norm(dim=1)

        similarity_matrix = torch.matmul(batch_tensor, batch_tensor.t()) / torch.outer(
            vector_norms, vector_norms
        )

        # est = exp, similarity, temperature
        est_matrix = torch.exp(similarity_matrix / self._temperature)

        est_matrix = est_matrix.clone().fill_diagonal_(0)

        loss_denominators = est_matrix.sum(dim=1)

        loss_sum_indexes_1 = torch.arange(0, batch_size, 2)
        loss_sum_indexes_2 = torch.arange(1, batch_size, 2)

        loss_sum_1 = torch.sum(
            -torch.log(
                est_matrix[loss_sum_indexes_1, loss_sum_indexes_2]
                / loss_denominators[loss_sum_indexes_1]
            )
        )
        loss_sum_2 = torch.sum(
            -torch.log(
                est_matrix[loss_sum_indexes_2, loss_sum_indexes_1]
                / loss_denominators[loss_sum_indexes_2]
            )
        )

        loss = (loss_sum_1 + loss_sum_2) / batch_size

        return loss
