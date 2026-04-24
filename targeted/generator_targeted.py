import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from attention import SpatialTransformer


class Generator(nn.Module):
    def __init__(
        self,
        input_dim,
        num_filters,
        output_dim,
        batch_size,
        first_kernel_size,
        num_heads=1,
        context_dim=256
    ):
        super(Generator, self).__init__()

        self.length = len(num_filters)
        self.hidden_layer = nn.ModuleList()

        # =========================
        # BUILD NETWORK
        # =========================
        for i in range(len(num_filters)):
            layers = []

            for j in range(len(num_filters[i])):

                if (i == 0) and (j == 0):
                    layers.append(
                        nn.ConvTranspose2d(
                            input_dim,
                            num_filters[i][j],
                            kernel_size=first_kernel_size,
                            stride=1,
                            padding=0
                        )
                    )
                elif j == 0:
                    layers.append(
                        nn.ConvTranspose2d(
                            num_filters[i-1][-1],
                            num_filters[i][j],
                            kernel_size=4,
                            stride=2,
                            padding=1
                        )
                    )
                else:
                    layers.append(
                        nn.ConvTranspose2d(
                            num_filters[i][j-1],
                            num_filters[i][j],
                            kernel_size=4,
                            stride=2,
                            padding=1
                        )
                    )

                layers.append(nn.BatchNorm2d(num_filters[i][j]))
                layers.append(nn.ReLU(True))

            self.hidden_layer.append(nn.Sequential(*layers))

            # Cross attention
            if i < len(num_filters) - 1:
                self.hidden_layer.append(
                    SpatialTransformer(
                        num_filters[i][-1],
                        num_heads,
                        num_filters[i][-1] // num_heads,
                        depth=1,
                        context_dim=context_dim
                    )
                )

        # =========================
        # OUTPUT LAYER
        # =========================
        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(
                num_filters[-1][-1],
                output_dim,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.Tanh()   # keep but we scale later
        )

    def forward(self, x, cond):
        """
        x: (B, z_dim, 1, 1)
        cond: (B, D)
        """

        text_cond = cond.unsqueeze(1)  # (B,1,D)

        for i in range(self.length - 1):
            x = self.hidden_layer[2*i](x)
            x = self.hidden_layer[2*i + 1](x, text_cond)

        x = self.hidden_layer[2 * (self.length - 1)](x)

        out = self.output_layer(x)

        return out