import jittor as jt
import jittor.nn as nn
from jittor import init, knn
from jittor.contrib import concat 
from jittor_utils import auto_diff

class SharedMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        transpose=False,
        padding_mode='zeros',
        bn=False,
        activation_fn=None
    ):
        conv_fn = nn.ConvTranspose if transpose else nn.Conv2d

        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            # padding_mode=padding_mode
        )
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
        self.activation_fn = activation_fn

    def execute(self, input):
        r"""
            execute pass of the network

            Parameters
            ----------
            input: jt.Tensor, shape (B, d_in, N, K)

            Returns
            -------
            jt.Tensor, shape (B, d_out, N, K)
        """
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class LocalSpatialEncoding(nn.Module):
    def __init__(self, d, num_neighbors):
        self.num_neighbors = num_neighbors
        self.mlp = SharedMLP(10, d, bn=True, activation_fn=nn.ReLU())

    def execute(self, coords, features, knn_output):
        r"""
            execute pass

            Parameters
            ----------
            coords: jt.Tensor, shape (B, N, 3)
                coordinates of the point cloud
            features: jt.Tensor, shape (B, d, N, 1)
                features of the point cloud
            neighbors: tuple

            Returns
            -------
            jt.Tensor, shape (B, 2*d, N, K)
        """
        # finding neighboring points
        idx, dist = knn_output
        B, N, K = idx.size()
        # idx(B, N, K), coords(B, N, 3)
        # neighbors[b, i, n, k] = coords[b, idx[b, n, k], i] = extended_coords[b, i, extended_idx[b, i, n, k], k]
        extended_idx = idx.unsqueeze(1).expand(B, 3, N, K)
        extended_coords = coords.transpose(-2,-1).unsqueeze(-1).expand(B, 3, N, K)
        neighbors = jt.gather(extended_coords, 2, extended_idx) # shape (B, 3, N, K)
        # if USE_CUDA:
        #     neighbors = neighbors.cuda()

        # relative point position encoding
        concat = jt.concat((
            extended_coords,
            neighbors,
            extended_coords - neighbors,
            dist.unsqueeze(-3)
        ), dim=-3)
        return jt.concat((
            self.mlp(concat),
            features.expand(B, -1, N, K)
        ), dim=-3)



class AttentivePooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.score_fn = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.Softmax(dim=-2)
        )
        self.mlp = SharedMLP(in_channels, out_channels, bn=True, activation_fn=nn.ReLU())

    def execute(self, x):
        r"""
            execute pass

            Parameters
            ----------
            x: jt.Tensor, shape (B, d_in, N, K)

            Returns
            -------
            jt.Tensor, shape (B, d_out, N, 1)
        """
        # computing attention scores
        scores = self.score_fn(x.permute(0,2,3,1)).permute(0,3,1,2)

        # sum over the neighbors
        features = jt.sum(scores * x, dim=-1, keepdims=True) # shape (B, d_in, N, 1)

        return self.mlp(features)



class LocalFeatureAggregation(nn.Module):
    def __init__(self, d_in, d_out, num_neighbors):

        self.num_neighbors = num_neighbors

        self.mlp1 = SharedMLP(d_in, d_out//2, activation_fn=nn.LeakyReLU(0.2))
        self.mlp2 = SharedMLP(d_out, 2*d_out)
        self.shortcut = SharedMLP(d_in, 2*d_out, bn=True)

        self.lse1 = LocalSpatialEncoding(d_out//2, num_neighbors)
        self.lse2 = LocalSpatialEncoding(d_out//2, num_neighbors)

        self.pool1 = AttentivePooling(d_out, d_out//2)
        self.pool2 = AttentivePooling(d_out, d_out)

        self.lrelu = nn.LeakyReLU()

    def execute(self, coords, features):
        r"""
            execute pass

            Parameters
            ----------
            coords: jt.Tensor, shape (B, N, 3)
                coordinates of the point cloud
            features: jt.Tensor, shape (B, d_in, N, 1)
                features of the point cloud

            Returns
            -------
            jt.Tensor, shape (B, 2*d_out, N, 1)
        """
        knn_output = jt.knn(coords, coords, self.num_neighbors)
        knn_output = (knn_output[1], knn_output[0])
        x = self.mlp1(features)

        x = self.lse1(coords, x, knn_output)
        x = self.pool1(x)

        x = self.lse2(coords, x, knn_output)
        x = self.pool2(x)

        return self.lrelu(self.mlp2(x) + self.shortcut(features))



class RandLANet(nn.Module):
    def __init__(self, d_in, num_classes, num_neighbors=16, decimation=4):
        self.num_neighbors = num_neighbors
        self.decimation = decimation

        self.fc_start = nn.Linear(d_in, 8, bias=True)
        self.bn_start = nn.Sequential(
            nn.BatchNorm2d(8, eps=1e-6, momentum=0.99),
            nn.LeakyReLU(0.2)
        )

        # encoding layers
        self.encoder = nn.ModuleList([
            LocalFeatureAggregation(8, 16, num_neighbors),
            LocalFeatureAggregation(32, 64, num_neighbors),
            LocalFeatureAggregation(128, 128, num_neighbors),
            LocalFeatureAggregation(256, 256, num_neighbors)
        ])

        self.mlp = SharedMLP(512, 512, activation_fn=nn.ReLU())

        # decoding layers
        decoder_kwargs = dict(
            transpose=True,
            bn=True,
            activation_fn=nn.ReLU()
        )
        decoder_kwargs_2 = dict(
            transpose=True,
            bn=True,
            activation_fn=None
        )
        self.decoder = nn.ModuleList([
            SharedMLP(1024, 256, **decoder_kwargs),
            SharedMLP(512, 128, **decoder_kwargs),
            SharedMLP(256, 32, **decoder_kwargs),
            SharedMLP(64, 8, **decoder_kwargs)
        ])

        # final semantic prediction
        self.fc_end = nn.Sequential(
            SharedMLP(8, 64, bn=True, activation_fn=nn.ReLU()),
            SharedMLP(64, 32, bn=True, activation_fn=nn.ReLU()),
            nn.Dropout(),
            SharedMLP(32, num_classes)
        )

    def execute(self, input):
        r"""
            execute pass

            Parameters
            ----------
            input: jt.Tensor, shape (B, N, d_in)
                input points

            Returns
            -------
            jt.Tensor, shape (B, num_classes, N)
                segmentation scores for each point
        """
        N = input.size(1)
        d = self.decimation

        coords = input[...,:3].clone()
        x = self.fc_start(input).transpose(-2,-1).unsqueeze(-1)
        x = self.bn_start(x) # shape (B, d, N, 1)
        decimation_ratio = 1
        # <<<<<<<<<< ENCODER
        x_stack = []

        permutation = jt.randperm(N)
        coords = coords[:,permutation]
        x = x[:,:,permutation]

        for i, lfa in enumerate(self.encoder):
            print("at iteration {}, x.shape = ".format(i), x.shape) # (B, N//(d**i), d_in)
            x = lfa(coords[:,:N//decimation_ratio], x)
            x_stack.append(x.clone())
            decimation_ratio *= d
            x = x[:,:,:N//decimation_ratio]


        # # >>>>>>>>>> ENCODER

        x = self.mlp(x)

        # <<<<<<<<<< DECODER
        for i, mlp in enumerate(self.decoder):
            print(coords[:,:N//decimation_ratio].shape, coords[:,:d*N//decimation_ratio].shape)
            _, neighbors = jt.knn(
                coords[:,:d*N//decimation_ratio], # upsampled set
                coords[:,:N//decimation_ratio], # original set
                1
            ) # shape (B, N, 1)
            print(_.shape)
            print("at iteration {}, neighbors.shape = ".format(i), neighbors.shape)
            extended_neighbors = neighbors.unsqueeze(1).expand(-1, x.size(1), -1, 1)

            x_neighbors = jt.gather(x, -2, extended_neighbors)
            print("at iteration {}, x_neighbors.shape = ".format(i), x_neighbors.shape)
            x = jt.concat((x_neighbors, x_stack.pop()), dim=1)

            x = mlp(x)

            decimation_ratio //= d

        # >>>>>>>>>> DECODER
        # inverse permutation
        print(x.shape)
        print(permutation)
        x = x[:,:,jt.argsort(permutation)[0]]

        scores = self.fc_end(x)

        return scores.squeeze(-1)


if __name__ == '__main__':
    import time
    jt.set_global_seed(1)
    d_in = 7
    cloud = 1000* jt.randn(1, 2**16, d_in)
    model = RandLANet(d_in, 6, 16, 4)


    # model.load_state_dict(jt.load('checkpoints/checkpoint_100.pth'))
    model.eval()
    hook = auto_diff.Hook("RandLA")
    hook.hook_module(model)
    print("cloud:", cloud)
    # t0 = time.time()
    pred = model(cloud)
    # t1 = time.time()
    # print("pred:", pred)
    # print("time:", t1-t0)
