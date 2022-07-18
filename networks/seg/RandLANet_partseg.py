from typing import List, Optional, Tuple

import jittor as jt
import jittor.nn as nn
from jittor import init
from jittor.contrib import concat 

class ConfigS3DIS:
    feature_size = 6
    batch_size = 6  # batch_size during training
    num_layers = 5
    num_classes = 13
    d_out = [16, 64, 128, 256, 512]  # feature dimension

# TODO
class DilatedResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.build_dilated_res_block()
    
    def build_dilated_res_block(self):
        self.mlp1 = nn.Conv2d(self.in_dim, self.out_dim // 2, (1,1))
        # building_block
        # f_xyz = 10
        self.mlp2 = nn.Conv2d(self.out_dim, self.out_dim * 2, (1,1))
        self.mlp1_building_block = nn.Conv2d(10, self.out_dim // 2)
        self.mlp_shortcut = nn.Conv2d(self.in_dim, self.out_dim * 2, (1,1))
        self.leaky_relu = nn.LeakyReLU(0.2)
        # TODO

    def execute(self, feature, xyz, neigh_idx):
        # (?,?,1,in_dim)
        f_pc = self.mlp1(feature) # fc -> # (?,?,1,out_dim // 2) 
        f_pc = self.building_block(xyz, f_pc, neigh_idx) # building -> (?,?,1,out_dim)
        f_pc = self.mlp2(f_pc) # fc -> (?,?,1,2 * out_dim)
        shortcut = self.mlp_shortcut(feature)
        return self.leaky_relu(f_pc + shortcut)

    def building_block(self, xyz, feature, neigh_idx):
        # feature: (?,?,1, out_dim // 2)
        d_in = self.out_dim // 2
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx) # relativePosEncoding -> (?,?,1,10)
        f_xyz = self.mlp1_building_block(f_xyz) # fc -> (?,?,1,out_dim // 2)
        f_neighbours = self.gather_neighbour(jt.squeeze(feature, axis=2), neigh_idx)
        f_concat = jt.concat([f_neighbours, f_xyz], axis = -1)
        # TODO

    def relative_pos_encoding(self, xyz:jt.Var, neigh_idx:jt.Var):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)
        xyz_tile = jt.repeat(xyz.unsqueeze(2), [1, 1, neigh_idx.shape[-1], 1])
        relative_xyz = xyz_tile - neighbor_xyz
        relative_dis = jt.sqrt(jt.sum(relative_xyz * relative_xyz, axis=-1, keepdims=True))
        relative_feature = jt.concat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], axis=-1)
        return relative_feature
    
    @staticmethod
    def gather_neighbour(pc, neighbor_idx):
        # gather the coordinates or features of neighboring points
        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        index_input = jt.reshape(neighbor_idx, shape=[batch_size, -1])
        features = pc.gather(-1, index_input)
        features = jt.reshape(features, [batch_size, num_points, neighbor_idx.shape[-1], d])
        return features


# TODO
class RandomSampler(nn.Module):
    def __init__(self):
        pass
    
    def execute(self):
        pass

class RandLANetEncoder(nn.Module):
    def __init__(self, num_layers, out_dimensions):
        self.num_layers = num_layers
        self.out_dimensions = out_dimensions # [num_layers]
        self.build_encoder()

    def build_encoder(self):
        self.dilated_res_blocks = nn.ModuleList()
        self.random_sample_blocks = nn.ModuleList()
        in_dim = 8
        for i in range(self.num_layers):
            self.dilated_res_blocks.append(
                DilatedResBlock(in_dim, self.out_dimensions[i])
            )
            self.random_sample_blocks.append(
                RandomSampler()
            )
            in_dim = self.out_dimensions[i]

    def execute(self, feature, xyz, neigh_idx, sub_idx)
        # feature: [?,?,1,8]
        for i in range(self.num_layers):
            f_encoder_i = self.dilated_res_blocks[i](feature, xyz, neigh_idx)
            f_sampled_i = self.random_sample_blocks[i](f_encoder_i, sub_idx)
            feature = f_sampled_i
        return feature
        
# TODO
class RandLANetDecoder(nn.Module):
    def __init__(self, out_dimensions):
        pass

    def execute(self, feature, interp_idx, encoder_list)
        pass

class RandLANet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.feature_size, self.num_layers = cfg.feature_size, cfg.num_layers
        self.out_dimensions = cfg.d_out
        self.build_model()
    
    def build_model(self):
        self.fc1 = nn.Conv(self.feature_size, 8, kernel_size=1)
        self.bn = nn.BatchNorm1d(8, eps=1e-6, momentum=0.99)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.encoder = RandLANetEncoder(
            num_layers = self.num_layers,
            out_dimensions = self.out_dimensions,
        )
        self.fc2 = nn.Conv2d(self.out_dimensions[4] * 2, self.out_dimensions[4] * 2, (1,1))
        self.decoder = RandLANetDecoder(
            num_layers = self.num_layers,
            out_dimensions = self.out_dimensions,
        )
        self.fc3 = nn.Conv2d(32, 64, (1,1))
        self.fc4 = nn.Conv2d(64, 32, (1,1))
        self.dropout = nn.Dropout()
        self.fc5 = nn.Conv2d(32, 13, (1,1))

    def execute(self, xyz, feature, neigh_idx, sub_idx, interp_idx):
        # [?,?,6]
        feature = self.fc1(feature) # fc -> [?,?,8]
        feature = self.leaky_relu(self.bn(feature)) # batchNorm -> Relu -> [?,?,8]
        feature = feature.unsqueeze(2) # ExpandDims -> [?,?,1,8]
        feature = self.encoder(feature, xyz, neigh_idx, sub_idx) # Encoder -> [?,?,1,1024]
        feature = self.fc2(feature) # fc -> [?,?,1,1024]
        feature = self.decoder(feature, interp_idx, encoder_list)
        f_layer_fc1 = self.fc3(feature)
        f_layer_fc2 = self.fc4(f_layer_fc1)
        f_layer_drop = self.dropout(f_layer_fc2)
        f_layer_fc3 = self.fc5(f_layer_drop)
        f_out = f_layer_fc3.squeeze(2)
        return f_out

def main():
    model = RandLANet(ConfigS3DIS)
    input_point = init.gauss([2, 1024, 3], 'float', mean=0.0)
    input_feature = init.gauss([2, 1024, ConfigS3DIS.feature_size], 'float', mean=0.0)
    cls_label = init.gauss([2, 16], 'float', mean=0.0)

    print (input_point.shape)
    print (input_feature.shape)
    print (cls_label.shape)
    outputs = model(input_point, input_feature, cls_label)
    print (outputs.shape)

if __name__ == '__main__':
    main()