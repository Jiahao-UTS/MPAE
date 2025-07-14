import torch


class EnforcedPresenceLoss(torch.nn.Module):
    """
    This class defines the Enforced Presence loss.
    """

    def __init__(self, loss_type: str = "enforced_presence", eps: float = 1e-10):
        super(EnforcedPresenceLoss, self).__init__()
        self.loss_type = loss_type
        self.eps = eps
        self.grid_x = None
        self.grid_y = None
        self.mask = None

    def forward(self, maps):
        """
        Forward function for the Enforced Presence loss.
        :param maps: Attention map with shape (batch_size, channels, height, width) where channels is the landmark probability
        :return: The Enforced Presence loss
        """
        if self.loss_type == "enforced_presence":
            avg_pooled_maps = torch.nn.functional.avg_pool2d(
                maps, 3, stride=1)
            if self.grid_x is None or self.grid_y is None:
                grid_x, grid_y = torch.meshgrid(torch.arange(avg_pooled_maps.shape[2]),
                                                torch.arange(avg_pooled_maps.shape[3]), indexing='ij')
                grid_x = grid_x.unsqueeze(0).unsqueeze(0).contiguous().to(avg_pooled_maps.device,
                                                                          non_blocking=True)
                grid_y = grid_y.unsqueeze(0).unsqueeze(0).contiguous().to(avg_pooled_maps.device,
                                                                          non_blocking=True)
                grid_x = (grid_x / grid_x.max()) * 2 - 1
                grid_y = (grid_y / grid_y.max()) * 2 - 1

                mask = grid_x ** 2 + grid_y ** 2
                mask = mask / mask.max()
                self.grid_x = grid_x
                self.grid_y = grid_y
                self.mask = mask

            masked_part_activation = avg_pooled_maps * self.mask
            masked_bg_part_activation = masked_part_activation[:, -1, :, :]

            max_pooled_maps = torch.nn.functional.adaptive_max_pool2d(masked_bg_part_activation, 1).flatten(start_dim=0)
            loss_area = torch.nn.functional.binary_cross_entropy(max_pooled_maps, torch.ones_like(max_pooled_maps))
        else:
            part_activation_sums = torch.nn.functional.adaptive_avg_pool2d(maps, 1).flatten(start_dim=1)
            background_part_activation = part_activation_sums[:, -1]
            if self.loss_type == "log":
                loss_area = torch.nn.functional.binary_cross_entropy(background_part_activation,
                                                                     torch.ones_like(background_part_activation))

            elif self.loss_type == "linear":
                loss_area = (1 - background_part_activation).mean()

            elif self.loss_type == "mse":
                loss_area = torch.nn.functional.mse_loss(background_part_activation,
                                                         torch.ones_like(background_part_activation))
            else:
                raise ValueError(f"Invalid loss type: {self.loss_type}")

        return loss_area


def presence_loss_soft_constraint(maps: torch.Tensor, beta: float = 0.1):
    """
    Calculate presence loss for a feature map
    :param maps: Attention map with shape (batch_size, channels, height, width) where channels is the landmark probability
    :param beta: Weight of soft constraint
    :return: value of the presence loss
    """
    loss_max = torch.nn.functional.adaptive_max_pool2d(torch.nn.functional.avg_pool2d(
        maps, 3, stride=1), 1).flatten(start_dim=1).max(dim=0)[0]
    loss_max_detach = loss_max.detach().clone()
    loss_max_p1 = 1 - loss_max
    loss_max_p2 = ((1 - beta) * loss_max_detach) + beta
    loss_max_final = (loss_max_p1 * loss_max_p2).mean()
    return loss_max_final


def presence_loss_tanh(maps: torch.Tensor):
    """
    Calculate presence loss for a feature map with tanh formulation from the paper PIP-NET
    Ref: https://github.com/M-Nauta/PIPNet/blob/68054822ee405b5f292369ca846a9c6233f2df69/pipnet/train.py#L111
    :param maps: Attention map with shape (batch_size, channels, height, width) where channels is the landmark probability
    :return:
    """
    pooled_maps = torch.tanh(torch.sum(torch.nn.functional.adaptive_max_pool2d(torch.nn.functional.avg_pool2d(
        maps, 3, stride=1), 1).flatten(start_dim=1), dim=0))

    loss_max = torch.nn.functional.binary_cross_entropy(pooled_maps, target=torch.ones_like(pooled_maps))

    return loss_max


def presence_loss_soft_tanh(maps: torch.Tensor):
    """
    Calculate presence loss for a feature map with tanh formulation (non-log/softer version)
    :param maps: Attention map with shape (batch_size, channels, height, width) where channels is the landmark probability
    :return:
    """
    pooled_maps = torch.tanh(torch.sum(torch.nn.functional.adaptive_max_pool2d(torch.nn.functional.avg_pool2d(
        maps, 3, stride=1), 1).flatten(start_dim=1), dim=0))

    loss_max = 1 - pooled_maps

    return loss_max.mean()


def presence_loss_original(maps: torch.Tensor, num_group: int):
    """
    Calculate presence loss for a feature map
    Modified from: https://github.com/robertdvdk/part_detection/blob/eec53f2f40602113f74c6c1f60a2034823b0fcaf/train.py#L181
    :param maps: Attention map with shape (batch_size, channels, height, width) where channels is the landmark probability
    :return: value of the presence loss
    """
# .max(dim=0)[0].mean()
    loss_max = torch.nn.functional.adaptive_max_pool2d(torch.nn.functional.avg_pool2d(
        maps, 3, stride=1), 1).flatten(start_dim=1)
    loss_max_group = torch.split(loss_max, num_group, dim=0)
    total_channel_wise_loss = 0
    total_group_wise_loss = 0
    for temp_group in loss_max_group:
        total_channel_wise_loss += (1.0 - temp_group.max(dim=0)[0].mean())
        total_group_wise_loss += (1.0 - temp_group.max(dim=1)[0].mean())

    total_channel_wise_loss = total_channel_wise_loss / len(loss_max_group)
    total_group_wise_loss = total_group_wise_loss / len(loss_max_group)

    return total_channel_wise_loss + total_group_wise_loss

    # loss_max = torch.nn.functional.adaptive_max_pool2d(torch.nn.functional.avg_pool2d(
    #     maps, 3, stride=1), 1).flatten(start_dim=1).max(dim=0)[0].mean()
    #
    # return 1 - loss_max

# def presence_loss_original(maps: torch.Tensor, num_group: int):
#     """
#     Calculate presence loss for a feature map
#     Modified from: https://github.com/robertdvdk/part_detection/blob/eec53f2f40602113f74c6c1f60a2034823b0fcaf/train.py#L181
#     :param maps: Attention map with shape (batch_size, channels, height, width) where channels is the landmark probability
#     :return: value of the presence loss
#     """
# # torch.nn.functional.adaptive_max_pool2d(, 1).flatten(start_dim=1)
# # .max(dim=0)[0].mean()
#     loss_max = torch.nn.functional.avg_pool2d(maps, 3, stride=1)
#     loss_max_group = torch.split(loss_max, num_group, dim=0)
#     total_channel_wise_loss = 0
#     total_group_wise_loss = 0
#     total_area_wise_loss = 0
#     for temp_group in loss_max_group:
#         total_channel_wise_loss += (1.0 - torch.nn.functional.adaptive_max_pool2d(temp_group, 1).flatten(start_dim=1).max(dim=0)[0].mean())
#         total_group_wise_loss += (1.0 - torch.nn.functional.adaptive_max_pool2d(temp_group, 1).flatten(start_dim=1).max(dim=1)[0].mean())
#         total_area_wise_loss += torch.clip(10.0 - torch.sum(temp_group, dim=(2, 3)).max(dim=0)[0].mean(), 0) / 10.0
#
#     total_channel_wise_loss = total_channel_wise_loss / len(loss_max_group)
#     total_group_wise_loss = total_group_wise_loss / len(loss_max_group)
#     total_area_wise_loss = total_area_wise_loss / len(loss_max_group)
#
#     return total_channel_wise_loss + total_group_wise_loss + total_area_wise_loss

    # loss_max = torch.nn.functional.adaptive_max_pool2d(torch.nn.functional.avg_pool2d(
    #     maps, 3, stride=1), 1).flatten(start_dim=1).max(dim=0)[0].mean()
    #
    # return 1 - loss_max

class PresenceLoss(torch.nn.Module):
    """
    This class defines the presence loss.
    """

    def __init__(self, loss_type: str = "original", beta: float = 0.1):
        super(PresenceLoss, self).__init__()
        self.loss_type = loss_type
        self.beta = beta

    def forward(self, maps, group=None):
        """
        Forward function for the presence loss.
        :param maps: Attention map with shape (batch_size, channels, height, width) where channels is the landmark probability
        :return: The presence loss
        """
        if self.loss_type == "original":
            return presence_loss_original(maps, group)
        elif self.loss_type == "soft_constraint":
            return presence_loss_soft_constraint(maps, beta=self.beta)
        elif self.loss_type == "tanh":
            return presence_loss_tanh(maps)
        elif self.loss_type == "soft_tanh":
            return presence_loss_soft_tanh(maps)
        else:
            raise NotImplementedError(f"Presence loss {self.loss_type} not implemented")