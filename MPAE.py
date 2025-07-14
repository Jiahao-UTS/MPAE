import torch

import torch.nn as nn
import torch.nn.functional as F

from backbone import Get_vit, ArcMarginProduct, MaskedAutoencoderViT
from utils import EnforcedPresenceLoss, TotalVariationLoss, PresenceLoss

def pixel_wise_entropy_loss(maps):
    """
    Calculate pixel-wise entropy loss for a feature map
    :param maps: Attention map with shape (batch_size, channels, height, width) where channels is the landmark probability
    :return: value of the pixel-wise entropy loss
    """
    # Calculate entropy for each pixel with numerical stability
    entropy = torch.distributions.categorical.Categorical(probs=maps.permute(0, 2, 3, 1).contiguous()).entropy()
    # Take the mean of the entropy
    return entropy.mean()

class MPAE(nn.Module):
    def __init__(self, num_part, d_model, scale_factor, img_size, mini_group, mask_ratio,
                 cfg):
        super(MPAE, self).__init__()

        self.num_part = num_part
        self.d_model = d_model
        self.mask_ratio = mask_ratio
        self.background = cfg.MODEL.BACKGROUND
        self.scale_factor = scale_factor
        self.mini_group = mini_group

        self.TotalVariationLoss = TotalVariationLoss()
        self.EnforcedPresenceLoss = EnforcedPresenceLoss()
        self.PresenceLoss = PresenceLoss()

        self.ArcMarginProduct = ArcMarginProduct(d_model, self.num_part - self.background,
                                                 self.num_part - self.background)
        self.compress_layer = nn.Conv2d(768, d_model, (1, 1), (1, 1), bias=False)
        self.apply(self._init_weights)

        self.backbone1 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        self.Transformer = Get_vit(num_part, img_size, 14, 256, 6, 8, d_model)
        self.Decoder = MaskedAutoencoderViT(img_size=img_size, patch_size=14, embed_dim=256, depth=2, num_heads=4,
                                            decoder_embed_dim=d_model, decoder_depth=2, decoder_num_heads=d_model // 64,
                                            mlp_ratio=4, rec_patch=8)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def high_level_loss(self, probmap, img_l, img, vgg, return_img=False):
        Bs = img.size(0)

        R_img = self.Decoder(img_l, probmap.flatten(2).permute(0, 2, 1).contiguous(), mask_ratio=self.mask_ratio)
        vgg_features = vgg(torch.cat((R_img, img), dim=0))

        loss_weight = [5.0 / 32.0, 5.0 / 16.0, 5.0 / 8.0, 5.0 / 4.0, 5.0]
        loss = 0.1 * torch.mean((R_img - img).abs())

        for i in range(len(vgg_features)):
            feature_diff = (vgg_features[i][0:Bs] - vgg_features[i][Bs:])
            value = torch.abs(feature_diff).mean()
            loss += value * loss_weight[i]

        if return_img is False:
            return loss
        else:
            return loss, R_img

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, input, input_small=None, VGG=None, trainable=False):
        Bs, _, H, W = input.size()
        backbone_out = self.backbone1(input, is_training=True)
        Local_Features = backbone_out['x_norm_patchtokens'].permute(0, 2, 1).view(Bs, 768, H // 14, W // 14)
        Local_Features = self.compress_layer(Local_Features)
        Global_Feature = self.Transformer(input)

        if trainable:
            output_distribution = torch.einsum("bqc,bchw->bqhw", Global_Feature, Local_Features)
            output_distribution = F.softmax(output_distribution, dim=1)

            local_part_rep = (output_distribution.unsqueeze(1) * Local_Features.unsqueeze(2)).contiguous()
            local_part_rep = local_part_rep.mean(-1).mean(-1).contiguous()
            local_part_rep = local_part_rep.permute(0, 2, 1).contiguous()

            background_loss = self.EnforcedPresenceLoss(output_distribution)
            variation_loss = self.TotalVariationLoss(output_distribution)
            entropy_loss = pixel_wise_entropy_loss(output_distribution)

            semantic_loss = self.ArcMarginProduct(local_part_rep[:, :self.num_part - 1, :],
                                                  Global_Feature[:, :self.num_part - 1, :],
                                                  output_distribution[:, :self.num_part - 1, :, :].detach())

            foreground_loss = self.PresenceLoss(output_distribution[:, :self.num_part - 1, :, :], self.mini_group)

            Syn_feat = torch.einsum("bln,bnc->blc", Global_Feature.permute(0, 2, 1), output_distribution.flatten(2))
            Syn_feat = Syn_feat.view(Bs, self.d_model, int(H // 14), int(W // 14))

            reconstruction_loss = self.high_level_loss(Syn_feat, input, input_small, VGG)

            return background_loss + foreground_loss, reconstruction_loss, variation_loss, entropy_loss, semantic_loss

        else:
            Local_Features = F.interpolate(Local_Features, scale_factor=self.scale_factor, mode='bilinear')
            output_distribution = torch.einsum("bqc,bchw->bqhw", Global_Feature, Local_Features)
            output_distribution = F.softmax(output_distribution, dim=1)
            return output_distribution
