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
    d_interp = [32, 128, 256, 512, 1024]

cfg = ConfigS3DIS

class DilatedResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.build_dilated_res_block()
    
    def build_dilated_res_block(self):
        self.mlp1 = nn.Linear(self.in_dim, self.out_dim // 2)
        # building_block
        # f_xyz = 10
        self.mlp2 = nn.Linear(self.out_dim, self.out_dim * 2)
        self.mlp1_building_block = nn.Linear(10, self.out_dim // 2)
        self.mlp2_building_block = nn.Linear(self.out_dim // 2, self.out_dim // 2)
        self.mlp_shortcut = nn.Linear(self.in_dim, self.out_dim * 2)
        self.leaky_relu = nn.LeakyReLU(0.2)
        # att_pooling
        self.att_pooling1 = nn.ModuleList()
        self.att_pooling1.append(nn.Linear(self.out_dim, self.out_dim, bias=False))
        self.att_pooling1.append(nn.Softmax(dim=1))
        self.att_pooling1.append(nn.Linear(self.out_dim, self.out_dim // 2))
        self.att_pooling2 = nn.ModuleList()
        self.att_pooling2.append(nn.Linear(self.out_dim, self.out_dim, bias=False))
        self.att_pooling2.append(nn.Softmax(dim=1))
        self.att_pooling2.append(nn.Linear(self.out_dim, self.out_dim))

    def execute(self, feature, xyz, neigh_idx):
        # (6,40960,1,in_dim)
        # print("[DRB]in_dim:", self.in_dim)
        f_pc = self.mlp1(feature) # fc -> # (6,40960,1,out_dim // 2) 
        f_pc = self.building_block(xyz, f_pc, neigh_idx) # building -> (6,40960,1,out_dim)
        f_pc = self.mlp2(f_pc) # fc -> (?,?,1,2 * out_dim)
        # print("feature shape:", f_pc.shape)
        shortcut = self.mlp_shortcut(feature)
        return self.leaky_relu(f_pc + shortcut)

    def building_block(self, xyz, feature, neigh_idx):
        # feature: (6,40960,1, out_dim // 2)
        d_in = self.out_dim // 2
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx) # relativePosEncoding -> (6,40960,1,10)
        f_xyz = self.mlp1_building_block(f_xyz) # fc -> (?,?,1,out_dim // 2)
        f_neighbours = self.gather_neighbour(jt.squeeze(feature, dim=2), neigh_idx)
        f_concat = jt.concat([f_neighbours, f_xyz], dim = -1)
        f_pc_agg = self.att_pooling(f_concat, self.att_pooling1) # att_pooling -> (?,?,1,out_dim // 2)
        
        f_xyz = self.mlp2_building_block(f_xyz) # fc -> (?,?,1,out_dim // 2)
        f_neighbours = self.gather_neighbour(jt.squeeze(f_pc_agg, dim = 2), neigh_idx)
        f_concat = jt.concat([f_neighbours, f_xyz], dim=-1)
        f_pc_agg = self.att_pooling(f_concat, self.att_pooling2) # att_pooling -> (?,?,1,out_dim)

        return f_pc_agg

    def att_pooling(self, feature_set, module: nn.ModuleList):
        batch_size = feature_set.shape[0]
        num_points = feature_set.shape[1]
        num_neigh = feature_set.shape[2]
        d = feature_set.shape[3]
        f_reshaped = jt.reshape(feature_set, [-1, num_neigh, d])
        att_activation = module[0](f_reshaped)
        att_scores = module[1](att_activation)
        f_agg = f_reshaped * att_scores
        f_agg = jt.sum(f_agg, dim=1)
        f_agg = jt.reshape(f_agg, [batch_size, num_points, 1, d])
        f_agg = module[2](f_agg)
        return f_agg

    def relative_pos_encoding(self, xyz:jt.Var, neigh_idx:jt.Var):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)
        xyz_tile = jt.repeat(xyz.unsqueeze(dim=2), [1, 1, neigh_idx.shape[-1], 1])
        relative_xyz = xyz_tile - neighbor_xyz
        relative_dis = jt.sqrt(jt.sum(relative_xyz * relative_xyz, dim=-1, keepdims=True))
        relative_feature = jt.concat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], dim=-1)
        return relative_feature
    
    @staticmethod
    def gather_neighbour(pc, neighbor_idx):
        # gather the coordinates or features of neighboring points
        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        index_input = jt.reshape(neighbor_idx, [batch_size, -1])
        features = pc.gather(-1, index_input)
        features = jt.reshape(features, [batch_size, num_points, neighbor_idx.shape[-1], d])
        return features

class RandLANetEncoder(nn.Module):
    def __init__(self, num_layers, out_dimensions):
        self.num_layers = num_layers
        self.out_dimensions = out_dimensions # [num_layers]
        self.build_encoder()

    def build_encoder(self):
        self.dilated_res_blocks = nn.ModuleList()
        in_dim = 8
        for i in range(self.num_layers):
            self.dilated_res_blocks.append(
                DilatedResBlock(in_dim, self.out_dimensions[i])
            )
            in_dim = self.out_dimensions[i] * 2

    def execute(self, feature, xyz, neigh_idx, sub_idx):
        # feature: [6.40960,1,8]
        f_encoder_list = []
        for i in range(self.num_layers):
            f_encoder_i = self.dilated_res_blocks[i](feature, xyz[i], neigh_idx[i])
            # print("[layer {}]after dilated res block: ".format(i), f_encoder_i.shape)
            f_sampled_i = self.random_sample(f_encoder_i, sub_idx[i])
            # print("[layer {}]after random sampling: ".format(i), f_sampled_i.shape)
            feature = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)
        return feature, f_encoder_list
    
    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = jt.squeeze(feature, dim=2)
        num_neigh = pool_idx.shape[-1]
        d = feature.shape[-1]
        batch_size = pool_idx.shape[0]
        pool_idx = jt.reshape(pool_idx, [batch_size, -1])
        pool_features = feature.gather(-1, pool_idx)
        pool_features = jt.reshape(pool_features, [batch_size, -1, num_neigh, d])
        pool_features = jt.max(pool_features, dim=2, keepdims=True)
        return pool_features
        
class RandLANetDecoder(nn.Module):
    def __init__(self, num_layers, out_dimensions):
        self.num_layers = num_layers
        self.out_dimensions = out_dimensions
        self.build_decoder()
    
    def build_decoder(self):
        self.conv2d_transpose = nn.ModuleList()
        for i in range(len(self.out_dimensions)-1):
            self.conv2d_transpose.append(
                nn.ConvTranspose(
                    in_channels = self.out_dimensions[-i-2] * 2 + self.out_dimensions[-i-1] * 2, 
                    out_channels = self.out_dimensions[-i-2] * 2,
                    kernel_size=(1,1)
                    )
                )
        self.conv2d_transpose.append(
            nn.ConvTranspose(
                in_channels = self.out_dimensions[0] * 4,
                out_channels = self.out_dimensions[0] * 2,
                kernel_size=(1,1)
            )
        )

    def execute(self, feature, interp_idx, f_encoder_list):
        for j in range(self.num_layers):
            f_interp_i = self.nearest_interpolation(feature, interp_idx[-j - 1])
            f_concat = jt.concat([f_encoder_list[-j-2], f_interp_i],dim=3)
            f_concat = jt.transpose(f_concat, [0,3,1,2])
            # print("f_concat shape: ",f_concat.shape)
            # print("in_channels:", self.out_dimensions[-j-2] * 2 + self.out_dimensions[-j-1] * 2)
            f_decoder_i = self.conv2d_transpose[j](f_concat)
            f_decoder_i = jt.transpose(f_decoder_i, [0,2,3,1])
            feature = f_decoder_i
        return feature

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = jt.squeeze(feature, dim=2)
        batch_size = interp_idx.shape[0]
        up_num_points = interp_idx.shape[1]
        interp_idx = jt.reshape(interp_idx, [batch_size, up_num_points])
        interpolated_features = feature.gather(-1, interp_idx)
        interpolated_features = interpolated_features.unsqueeze(dim=2)
        return interpolated_features

class RandLANet(nn.Module):
    def __init__(self, part_num=50):
        super().__init__()
        self.feature_size, self.num_layers = cfg.feature_size, cfg.num_layers
        self.out_dimensions = cfg.d_out
        self.build_model()
    
    def build_model(self):
        self.fc1 = nn.Linear(self.feature_size, 8)
        self.bn = nn.BatchNorm1d(8, eps=1e-6, momentum=0.99) # operation on dim=1
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.encoder = RandLANetEncoder(
            num_layers = self.num_layers,
            out_dimensions = self.out_dimensions,
        )
        self.fc2 = nn.Linear(self.out_dimensions[-1] * 2, self.out_dimensions[-1] * 2)
        self.decoder = RandLANetDecoder(
            num_layers = self.num_layers,
            out_dimensions = self.out_dimensions,
        )
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, 32)
        self.dropout = nn.Dropout()
        self.fc5 = nn.Linear(32, 13)

    def execute(self, xyz, feature, neigh_idx, sub_idx, interp_idx):
        # [6,40960,6]
        feature = self.fc1(feature) # fc -> [6,40960,8]
        feature = jt.transpose(feature, (0,2,1)) # transpose -> [6,8,40960]
        feature = self.leaky_relu(self.bn(feature)) # batchNorm -> Relu -> [6,8,40960]
        feature = jt.transpose(feature,(0,2,1)).unsqueeze(dim=2) # ExpandDims -> [6,40960,1,8]
        feature, f_encoder_list = self.encoder(feature, xyz, neigh_idx, sub_idx) # Encoder -> [6,80,1,1024]
        feature = self.fc2(feature) # fc -> [6,?,1,1024]
        feature = self.decoder(feature, interp_idx, f_encoder_list)
        f_layer_fc1 = self.fc3(feature)
        f_layer_fc2 = self.fc4(f_layer_fc1)
        f_layer_drop = self.dropout(f_layer_fc2)
        f_layer_fc3 = self.fc5(f_layer_drop)
        f_out = f_layer_fc3.squeeze(2)
        return f_out

def main():
    model = RandLANet(part_num=50)

    sub_sampling_ratio = [4, 4, 4, 4, 2]  # sampling ratio of random sampling at each layer
    num_layers = cfg.num_layers

    for j in range(500):
        batch_xyz = init.gauss([6,40960,3], 'int32', mean=0)
        batch_features = init.gauss([6,40960,3], 'float32', mean=0.0)
        batch_labels = init.gauss([6,40960], 'int32', mean=0)
        batch_pc_idx = init.gauss([6,40960], 'int32', mean=0)
        batch_cloud_idx = init.gauss([6,1], 'int32', mean=0)

        batch_features = jt.concat([batch_xyz, batch_features], dim=-1)
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        for i in range(num_layers):
            neighbour_idx = (jt.knn(batch_xyz, batch_xyz, 16))[1]
            sub_points = batch_xyz[:, :batch_xyz.shape[1] // sub_sampling_ratio[i], :]
            pool_i = neighbour_idx[:, :batch_xyz.shape[1] // sub_sampling_ratio[i], :]
            up_i = jt.knn(batch_xyz, sub_points, 1)[1]
            input_points.append(batch_xyz)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)
            batch_xyz = sub_points

        flat_inputs = input_points + input_neighbors + input_pools + input_up_samples
        flat_inputs += [batch_features, batch_labels, batch_pc_idx, batch_cloud_idx]
        # for x in flat_inputs:
        #     print(x.shape)
        outputs = model(
            xyz=flat_inputs[:num_layers], 
            feature=flat_inputs[4 * num_layers],
            neigh_idx=flat_inputs[num_layers: 2 * num_layers],
            sub_idx=flat_inputs[2 * num_layers:3 * num_layers],
            interp_idx=flat_inputs[3 * num_layers:4 * num_layers])
        
        print("step{}".format(j), outputs.shape)

if __name__ == '__main__':
    main()