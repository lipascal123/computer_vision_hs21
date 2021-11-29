import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureNet(nn.Module):
    def __init__(self, kernel_sizes=[3, 3, 5, 3, 3, 5, 3, 3, 3], 
                 channels=[8, 8, 16, 16, 16, 32, 32, 32, 32],
                 conv_strides=[1, 1, 2, 1, 1, 2, 1, 1, 1],
                 pad = [1,1,2,1,1,2,1,1,1]):
        super(FeatureNet, self).__init__()
        # TODO
        # check for same length
        assert(len(kernel_sizes) == len(channels) == len(conv_strides))
        
        # build CNN network
        input_size = 3 # initial number of input channels
        self.num_layers = len(kernel_sizes)
        layers_conv = []
        layers_bn = []
        # first conv layer (0)
        layers_conv.append(nn.Conv2d(input_size, channels[0], kernel_sizes[0], conv_strides[0], pad[0]))
        layers_bn.append(nn.BatchNorm2d(channels[0]))
        # intermediate conv and bn layers (1-7)
        for i in range(1,self.num_layers-1):
            layers_conv.append(nn.Conv2d(channels[i-1], channels[i], kernel_sizes[i], conv_strides[i], pad[i]))
            layers_bn.append(nn.BatchNorm2d(channels[i]))
        # last conv layer (8)
        layers_conv.append(nn.Conv2d(
            channels[self.num_layers-2], channels[self.num_layers-1], kernel_sizes[self.num_layers-1], conv_strides[self.num_layers-1], pad[i]))

        # covnert python list to nn.ModuleList
        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)

        # activation
        self.relu = nn.ReLU(True)


    def forward(self, x):
        # x: [B,3,H,W]
        # TODO
        # forward for layers (1-8)
        for i in range(0,self.num_layers-2):
            x = self.layers_conv[i](x)
            print("x shape1", x.shape, " ", i)
            x = self.layers_bn[i](x)
            print("x shape2", x.shape, " ", i)
            x = self.relu(x)
            print("x shape3", x.shape, " ", i)
        # last conv layer (9)
        x = self.layers_conv[self.num_layers-1](x)
        print("x last", x.shape)

        return x

class SimlarityRegNet(nn.Module):
    def __init__(self, G):
        super(SimlarityRegNet, self).__init__()
        # TODO

    def forward(self, x):
        # x: [B,G,D,H,W]
        # out: [B,D,H,W]
        # TODO
        print("hello")


def warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W], B,32,160,128
    # src_proj: [B, 4, 4], Projection matrix 4x4
    # ref_proj: [B, 4, 4], Projection matrix 4x4
    # depth_values: [B, D]
    # out: [B, C, D, H, W]
    B,C,H,W = src_fea.size()
    D = depth_values.size(1)
    # compute the warped positions with depth values
    with torch.no_grad():
        # relative transformation from reference to source view
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]
        y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, W, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous() # memory reshape
        y, x = y.view(H * W), x.view(H * W) # reshape tensor
        # TODO
        # stack all reference image pixel coordinates
        uv1 = torch.stack((x, y, torch.ones(x.shape)), dim=0)  # [3,H*W]
        # add batch dimension
        uv1_b = torch.unsqueeze(uv1, 0).repeat((B, 1, 1))  # [B,3,H*W]
        
        # rotate
        uv1_b_rot = torch.matmul(rot,uv1_b)
        
        # add depth dimension
        uvd_b_rot = torch.unsqueeze(uv1_b_rot, 1).repeat((1, D, 1, 1))  # [B,D,3,H*W]
        # multiply with depth samples
        depth_values = torch.unsqueeze(depth_values,2)
        depth_values = torch.unsqueeze(depth_values, 3) # [B,D,1,1]
        uvd_b_rot = uvd_b_rot*depth_values # elementwise multiplication along dimension [-,-,3,H*W]; [B,D,3,H*W]

        # translate
        trans = torch.unsqueeze(trans,1) # [B,1,3,1]
        uvd_b_rot_trans = uvd_b_rot * trans # [B,D,3,H*W]

        # normalize homogenous image coordinates
        uvd_b_rot_trans[:, :, 0, :] = uvd_b_rot_trans[:, :, 0, :] / uvd_b_rot_trans[:, :, 2, :]
        uvd_b_rot_trans[:, :, 1, :] = uvd_b_rot_trans[:, :, 1, :] / uvd_b_rot_trans[:, :, 2, :]

        # normalize to [-1,1] for grid sample
        uvd_b_rot_trans_xn = (uvd_b_rot_trans[:, :, 0, :] / (W/2)) -1 # [B,D,H*W]
        uvd_b_rot_trans_yn = (uvd_b_rot_trans[:, :, 1, :] / (H/2)) -1
        
        uvd_b_rot_trans_n = torch.stack((uvd_b_rot_trans_xn, uvd_b_rot_trans_yn),dim=3) # [B,D,H*W,2]

    # get warped_src_fea with bilinear interpolation (use 'grid_sample' function from pytorch)
    # TODO
    warped_src_fea = F.grid_sample(src_fea,
        uvd_b_rot_trans_n.view(B, D * H, W, 2), mode="bilinear",
        padding_mode="zeros", align_corners=True,)

    return warped_src_fea.view(B, C, D, H, W)

def group_wise_correlation(ref_fea, warped_src_fea, G):
    # ref_fea: [B,C,H,W]
    # warped_src_fea: [B,C,D,H,W]
    # out: [B,G,D,H,W]
    # TODO
    B, C, D, H, W = warped_src_fea.size()
    warped_src_fea_re = warped_src_fea.view(B, G, C//G, D, H, W)
    ref_fea_re = ref_fea.view(B,G,C//G,1,H,W)
    print("warped_src_fea_re.shape", warped_src_fea_re.shape)
    print("ref_fea", ref_fea.shape)
    similarity = (warped_src_fea_re * ref_fea_re).mean(2)

    return similarity


def depth_regression(p, depth_values):
    # p: probability volume [B, D, H, W]
    # depth_values: discrete depth values [B, D]
    # TODO
    
    #! replace this, is copied from other code
    return torch.sum(p * depth_values.view(depth_values.shape[0], 1, 1), dim=1).unsqueeze(1)


def mvs_loss(depth_est, depth_gt, mask):
    # depth_est: [B,1,H,W]
    # depth_gt: [B,1,H,W]
    # mask: [B,1,H,W]
    # TODO
    print("hello")
