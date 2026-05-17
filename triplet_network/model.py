"""
MLP in pytorch

Created on Thu Aug 10 16:03:59 2023
"""

import torch as T
from torch import Tensor
import torch.nn as nn
from typing import Callable, List, Optional


class DenseBlock(nn.Module):
    """
    Dense blocks for feedforward network
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: Optional[int] = None,
        dropout: float = 0.0,
        dropout_layer: Callable[[float], nn.Module] = nn.Dropout,
        activation: Callable[[], nn.Module] = nn.ReLU,
        bias: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        in_dim : int
            Input dimensions to dense layer.
        out_dim : Optional[int], optional
            Number of neurons in dense layer. The default is None (uses input size).
        dropout: float, optional
            Amount of dropout after activation. The default is 0.
        dropout_layer: Callable[[float], nn.Module], optional
            Layer to be used for dropout.
        activation : Callable[[], nn.Module], optional
            Constructor function for activation layer. The default is nn.ReLU.
        bias : bool, optional
            Use bias term in linear layer. The default is True.
        """

        super().__init__()
        self.in_dim = in_dim
        self.out_dim = (
            out_dim or in_dim
        )  # use input dimenstions if output size not provided
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=bias), activation(), dropout_layer(dropout)
        )

    def forward(self, x: T.Tensor) -> T.Tensor:
        """
        Forward pass through the dense block.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Output after linear, activation, and dropout.
        """
        return self.block(x)


def make_mlp(
    d_in: int,
    neurons: List[int],
    activation: Callable[[], nn.Module] = nn.ReLU,
    dropout: float = 0.0,
    dropout_layer: Callable[[float], nn.Module] = nn.Dropout,
    d_out: Optional[int] = None,
    final_layer_activation: Optional[Callable[[], nn.Module]] = None,
    bias: bool = True,
) -> nn.Sequential:
    """
    Constructor function to make mlp

    Parameters
    ----------
    d_in : int
        Number of input features.
    neurons : list
        Number of neurons in hidden layer, len(neurons) is depth of network.
    activation : Callable[[], nn.Module], optional
        Activation function. The default is nn.ReLU.
    dropout : float, optional
        Probability of dropping out connections. The default is 0.
    dropout_layer : Callable[[float], nn.Module], optional
        Dropout layer. The default is nn.Dropout.
    d_out : Optional[int], optional
        Number of output features. The default is None (uses size of last hidden layer).
    final_layer_activation : Optional[Callable[[], nn.Module]], optional
        Activation function for final layer. The default is None (no activation).
    bias : bool, optional
        Whether to use bias term in linear layers. The default is True.

    Returns
    -------
    nn.Sequential
        MLP model.

    Examples
    --------
    import torch
    model = make_mlp(10, [64, 32], d_out=2)
    output = model(torch.randn(5, 10))
    """

    if not isinstance(neurons, list):
        neurons = [neurons]

    neurons = [d_in] + neurons  # prepend input size to list of neurons
    if d_out is not None:
        neurons = neurons + [d_out]
    layers = []  # init layers

    # iterate over layers
    for i in range(len(neurons) - 1):
        # add layer with actiation function and dropout
        if i != len(neurons) - 2:
            # add dense block
            layers += [
                DenseBlock(
                    neurons[i],
                    neurons[i + 1],
                    activation=activation,
                    dropout=dropout,
                    dropout_layer=dropout_layer,
                    bias=bias,
                )
            ]

        else:  # final layer has no activation or dropout
            layers += [
                DenseBlock(
                    neurons[i],
                    neurons[i + 1],
                    activation=nn.Identity
                    if final_layer_activation is None
                    else final_layer_activation,
                    dropout=0.0,
                    bias=bias,
                )
            ]

    return nn.Sequential(*layers)  # create sequential network from layers


class ContrastiveMLP(nn.Module):
    """
    Contrastive MLP model for feature extraction, projection, and classification.

    This model supports contrastive learning by providing separate forward passes
    for features, projections, and classification heads.
    """

    def __init__(
        self,
        d_in: int,
        neurons: List[int],
        activation: Callable[[], nn.Module] = nn.ReLU,
        dropout: float = 0.0,
        dropout_layer: Callable[[int], nn.Module] = nn.Dropout,
        n_classes: Optional[int] = None,
        d_out: Optional[int] = None,
        final_layer_activation: Optional[Callable[[], nn.Module]] = None,
        bias: bool = True,
    ) -> None:
        """
        Initialize the ContrastiveMLP model.

        Parameters
        ----------
        d_in : int
            Number of input features.
        neurons : List[int]
            Number of neurons in hidden layers.
        activation : Callable[[], nn.Module], optional
            Activation function. Default is nn.ReLU.
        dropout : float, optional
            Dropout probability. Default is 0.0.
        dropout_layer : Callable[[float], nn.Module], optional
            Dropout layer constructor. Default is nn.Dropout.
        n_classes : Optional[int], optional
            Number of classes for classification head. Default is None.
        d_out : Optional[int], optional
            Output dimension for projection head. Default is None.
        final_layer_activation : Optional[Callable[[], nn.Module]], optional
            Activation for final layer. Default is None.
        bias : bool, optional
            Whether to use bias in linear layers. Default is True.
        """
        super().__init__()
        neurons = neurons if isinstance(neurons, list) else [neurons]

        self.mlp = make_mlp(
            d_in=d_in,
            neurons=neurons,
            activation=activation,
            dropout=dropout,
            dropout_layer=dropout_layer,
            final_layer_activation=final_layer_activation,
            bias=bias,
        )

        self.proj = nn.Identity() if d_out is None else nn.Linear(neurons[-1], d_out)
        self.probe = (
            nn.Identity() if n_classes is None else nn.Linear(neurons[-1], n_classes)
        )

    def forward_features(self, x: Tensor) -> Tensor:
        """
        Forward pass through the MLP to extract features.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Feature representations.
        """
        return self.mlp(x)

    def forward_cls(self, x: Tensor) -> Tensor:
        """
        Forward pass through the classification head.

        Parameters
        ----------
        x : Tensor
            Input features.

        Returns
        -------
        Tensor
            Classification logits.
        """
        return self.probe(x)

    def forward_finetune(self, x: Tensor) -> Tensor:
        """
        Forward pass for fine-tuning: features through classification.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Classification outputs.
        """
        x = self.forward_features(x)
        x = self.forward_cls(x)
        return x

    def forward_probe(self, x: Tensor) -> Tensor:
        """
        Forward pass through the projection head.

        Parameters
        ----------
        x : Tensor
            Input features.

        Returns
        -------
        Tensor
            Projected representations.
        """
        x = self.proj(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """
        Default forward pass: features through projection.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Projected features.
        """
        x = self.forward_features(x)
        return self.proj(x)

    def reset_probe(self):
        """Reinitialize all Linear layers inside the probe"""
        for m in self.probe.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()


class ContrastiveFNN(ContrastiveMLP):
    """
    Contrastive FNN model that uses fine-tuning forward pass by default.
    """

    def forward(self, x: Tensor):
        """
        Forward pass using fine-tuning mode.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Classification outputs.
        """
        return self.forward_finetune(x)
