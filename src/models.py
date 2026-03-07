from flax import linen as nn
from jax import numpy as jnp
import jax
import numpyro
from numpyro import handlers
from numpyro import distributions as dist
from src.subspace_sampling import init_curve_frame_cp, ortho_at_one_t
from src.subspace_curve import vec_to_pytree, bezier_curve, pytree_to_vec, vec_to_single_pytree
from functools import partial
from typing import Callable


class MLPModel(nn.Module):
    depth: int = 3
    width: int = 10
    activation: str = "relu"

    def setup(self) -> None:
        self.activation_fn = getattr(nn, self.activation)
        return super().setup()
    
    @nn.compact
    def __call__(self, x, train=True):
        x = jnp.concat([x, x**2, x**3], axis=-1)
        for _ in range(self.depth):
            x = nn.Dense(self.width)(x)
            x = self.activation_fn(x)
        x = nn.Dense(1)(x)
        return x
    

class MLPModelUCI(nn.Module):
    depth: int = 3
    width: int = 16
    activation: str = "relu"

    def setup(self) -> None:
        self.activation_fn = getattr(nn, self.activation)
        return super().setup()

    @nn.compact
    def __call__(self, x, train=True):
        for _ in range(self.depth):
            x = nn.Dense(self.width)(x)
            x = self.activation_fn(x)
        x = nn.Dense(1)(x)
        return x
    

# class LeNetti(nn.Module):
#     """
#     A super simple LeNet version.

#     Args:
#         config (LeNettiConfig): The configuration for the model.
#     """

#     config: LeNettiConfig

#     @nn.compact
#     def __call__(self, x: jnp.ndarray):
#         """
#         Forward pass.

#         Args:
#             x (jnp.ndarray): The input data of
#             shape (batch_size, channels, height, width).
#         """
#         activation = self.config.activation.flax_activation
#         # x = x.transpose((0, 2, 3, 1))
#         x = nn.Conv(
#             features=1, kernel_size=(3, 3), strides=(1, 1), padding=2, name='conv1'
#         )(x)
#         x = activation(x)
#         x = x.reshape((x.shape[0], -1))
#         x = nn.Dense(features=8, use_bias=self.config.use_bias, name='fc1')(x)
#         x = activation(x)
#         x = nn.Dense(features=8, use_bias=self.config.use_bias, name='fc2')(x)
#         x = activation(x)
#         x = nn.Dense(features=8, use_bias=self.config.use_bias, name='fc3')(x)
#         x = activation(x)
#         x = nn.Dense(
#             features=self.config.out_dim, use_bias=self.config.use_bias, name='fc4'
#         )(x)
#         return x
    

# class LeNet(nn.Module):
#     """
#     Implementation of LeNet.

#     Args:
#         config (LeNetConfig): The configuration for the model.
#     """

#     config: LeNetConfig

#     @nn.compact
#     def __call__(self, x: jnp.ndarray):
#         """
#         Forward pass.

#         Args:
#             x (jnp.ndarray): The input data of
#             shape (batch_size, channels, height, width).
#         """
#         activation = self.config.activation.flax_activation
#         # x = x.transpose((0, 2, 3, 1))
#         x = nn.Conv(
#             features=6, kernel_size=(5, 5), strides=(1, 1), padding=2, name='conv1'
#         )(x)
#         x = activation(x)
#         x = nn.avg_pool(x, window_shape=(2, 2),
#                         strides=(2, 2), padding='VALID')
#         x = nn.Conv(
#             features=16, kernel_size=(5, 5), strides=(1, 1), padding=0, name='conv2'
#         )(x)
#         x = activation(x)
#         x = nn.avg_pool(x, window_shape=(2, 2),
#                         strides=(2, 2), padding='VALID')
#         x = x.reshape((x.shape[0], -1))
#         x = nn.Dense(
#             features=120, use_bias=self.config.use_bias, name='fc1')(x)
#         x = activation(x)
#         x = nn.Dense(features=84, use_bias=self.config.use_bias, name='fc2')(x)
#         x = activation(x)
#         x = nn.Dense(
#             features=self.config.out_dim, use_bias=self.config.use_bias, name='fc3'
#         )(x)
#         return x


resnet_kernel_init = nn.initializers.variance_scaling(
    2.0, mode='fan_out', distribution='normal')


class ResNetBlock(nn.Module):
    act_fn: Callable  # Activation function
    c_out: int   # Output feature size
    subsample: bool = False  # If True, we apply a stride inside F

    @nn.compact
    def __call__(self, x, train=True):
        # Network representing F
        z = nn.Conv(self.c_out, kernel_size=(3, 3),
                    strides=(1, 1) if not self.subsample else (2, 2),
                    kernel_init=resnet_kernel_init,
                    use_bias=False)(x)
        z = nn.BatchNorm()(z, use_running_average=not train)
        z = self.act_fn(z)
        z = nn.Conv(self.c_out, kernel_size=(3, 3),
                    kernel_init=resnet_kernel_init,
                    use_bias=False)(z)
        z = nn.BatchNorm()(z, use_running_average=not train)

        if self.subsample:
            x = nn.Conv(self.c_out, kernel_size=(1, 1), strides=(
                2, 2), kernel_init=resnet_kernel_init)(x)

        x_out = self.act_fn(z + x)
        return x_out


class PreActResNetBlock(ResNetBlock):

    @nn.compact
    def __call__(self, x, train=True):
        # Network representing F
        z = nn.BatchNorm()(x, use_running_average=not train)
        z = self.act_fn(z)
        z = nn.Conv(self.c_out, kernel_size=(3, 3),
                    strides=(1, 1) if not self.subsample else (2, 2),
                    kernel_init=resnet_kernel_init,
                    use_bias=False)(z)
        z = nn.BatchNorm()(z, use_running_average=not train)
        z = self.act_fn(z)
        z = nn.Conv(self.c_out, kernel_size=(3, 3),
                    kernel_init=resnet_kernel_init,
                    use_bias=False)(z)

        if self.subsample:
            x = nn.BatchNorm()(x, use_running_average=not train)
            x = self.act_fn(x)
            x = nn.Conv(self.c_out,
                        kernel_size=(1, 1),
                        strides=(2, 2),
                        kernel_init=resnet_kernel_init,
                        use_bias=False)(x)

        x_out = z + x
        return x_out


class ResNet(nn.Module):
    num_classes: int
    act_fn: Callable
    block_class: nn.Module
    num_blocks: tuple = (3, 3, 3)
    c_hidden: tuple = (16, 32, 64)

    @nn.compact
    def __call__(self, x, train=True):
        # A first convolution on the original image to scale up the channel size
        x = nn.Conv(self.c_hidden[0], kernel_size=(
            3, 3), kernel_init=resnet_kernel_init, use_bias=False)(x)
        if self.block_class == ResNetBlock:  # If pre-activation block, we do not apply non-linearities yet
            x = nn.BatchNorm()(x, use_running_average=not train)
            x = self.act_fn(x)

        # Creating the ResNet blocks
        for block_idx, block_count in enumerate(self.num_blocks):
            for bc in range(block_count):
                # Subsample the first block of each group, except the very first one.
                subsample = (bc == 0 and block_idx > 0)
                # ResNet block
                x = self.block_class(c_out=self.c_hidden[block_idx],
                                     act_fn=self.act_fn,
                                     subsample=subsample)(x, train=train)

        # Mapping to classification output
        x = x.mean(axis=(-2, -3))
        x = nn.Dense(self.num_classes)(x)
        return x
