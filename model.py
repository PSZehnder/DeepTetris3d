import torch.nn as nn
import torch
from itertools import  product

class DenseUnit(nn.Module):

    def __init__(self, in_channels, conv_channels=256, kernel_size=3, pool=5):

        super(DenseUnit, self).__init__()
        self.in_channels = in_channels
        self.out_channels = conv_channels

        self.conv1 = nn.Conv3d(in_channels, conv_channels, kernel_size, padding=(kernel_size // 2))
        self.relu = nn.Sequential(nn.Dropout(0.5), nn.ReLU())
        self.conv2 = nn.Conv3d(conv_channels, conv_channels, kernel_size, padding=(kernel_size // 2))
        self.relupool = RegularizePool(pool)

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relupool(x)
        return x

class ConcatConv(nn.Module):

    def __init__(self, in_channels, out_channels, ncopies=1, kernel_size=3):
        super(ConcatConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ncopies = ncopies
        self.kernel_size = kernel_size
        self.conv = nn.Sequential(nn.Conv3d(ncopies * in_channels, out_channels,
                                            kernel_size, padding=(kernel_size // 2)),
                                   nn.BatchNorm3d(out_channels),
                                   nn.Dropout(0.5),
                                   nn.ReLU())

    def forward(self, *inputs):
        x = torch.cat(inputs, 1)
        return self.conv(x)

class DenseBlock(nn.Module):

    def __init__(self, in_channels, out_channels, concat_channels, kernel_size, pool=5):

        super(DenseBlock, self).__init__()

        self.dunit1 = DenseUnit(in_channels, out_channels, kernel_size, pool)
        self.ccat1 = ConcatConv(out_channels, concat_channels, ncopies=1, kernel_size=kernel_size)
        self.dunit2 = DenseUnit(concat_channels, out_channels, kernel_size, pool)
        self.ccat2 = ConcatConv(out_channels, concat_channels, ncopies=2, kernel_size=kernel_size )
        self.dunit3 = DenseUnit(concat_channels, out_channels, kernel_size + 2, pool)
        self.ccat3 = ConcatConv(out_channels, concat_channels, ncopies=3, kernel_size=kernel_size + 2)

        self.residual = None

        self.pool1 = RegularizePool(pool)
        self.pool2 = RegularizePool(pool)
        self.pool3 = RegularizePool(pool)

    def forward(self, input):
        self.residual = None
        x = self.dunit1(input)
        x = self.pool1(x)
        x = self.ccat1(x)
        self.gather_residual(x)
        x = self.dunit2(x)
        x = self.pool2(x)
        x = self.ccat2(x, self.residual)
        self.gather_residual(x)
        x = self.dunit3(x)
        x = self.pool3(x)
        x = self.ccat3(x, self.residual)
        return x

    def gather_residual(self, residual):
        if self.residual is None:
            self.residual = residual
        else:
            self.residual = torch.cat((self.residual, residual), 1)

class FullyConnected(nn.Module):

    def __init__(self, in_dim, out_dim, neurons):

        super(FullyConnected, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc1 = nn.Sequential(nn.Linear(in_dim, neurons), nn.ReLU(), nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(neurons, neurons), nn.ReLU(), nn.Dropout(0.5))
        self.fc3 = nn.Sequential(nn.Linear(neurons, out_dim))

    def forward(self, input):
        x = self.fc1(input)
        x = self.fc2(x)
        return self.fc3(x)

class RegularizePool(nn.Module):

    def __init__(self, pool):

        super(RegularizePool, self).__init__()
        self.pool = nn.Sequential(nn.ReLU(),
                                  nn.Dropout(),
                                  nn.AdaptiveMaxPool3d(pool))
    def forward(self, input):
        return self.pool(input)

class DenseNet(nn.Module):

    def __init__(self, out_channels, in_channels=512, neurons=512, fc_channels=64, dense_channels=512,
                 concat_channels=512, pool=4, piece_net_dim=128):

        self.init_args = {
            'out_channels': out_channels,
            'in_channels': in_channels,
            'neurons': neurons,
            'fc_channels': fc_channels,
            'dense_channels': dense_channels,
            'concat_channels': concat_channels,
            'pool': pool,
            'piece_net_dim': piece_net_dim
        }

        super(DenseNet, self).__init__()
        self.init_conv = nn.Conv3d(in_channels=1, out_channels=in_channels, kernel_size=1)
        self.dblock1 = DenseBlock(in_channels, dense_channels, in_channels, pool=pool, kernel_size=3)
        self.dblock2 = DenseBlock(in_channels, dense_channels, dense_channels, pool=pool, kernel_size=5)
        self.dblock3 = DenseBlock(dense_channels, dense_channels, dense_channels, pool=pool, kernel_size=5)
        #self.dblock4 = DenseBlock(dense_channels, dense_channels, concat_channels, pool=pool, kernel_size=5)

        self.pool1 = RegularizePool(pool)
        self.pool2 = RegularizePool(pool)

        self.piecenet = PieceNet(piece_net_dim)

        self.final_conv = nn.Sequential(nn.Conv3d(dense_channels, fc_channels, kernel_size=3, padding=1),
                                        nn.BatchNorm3d(fc_channels),
                                        nn.Dropout(0.5),
                                        nn.ReLU())

        self.fc_indim = fc_channels * pool ** 3 + piece_net_dim
        self.piece_net_dim = piece_net_dim
        self.fc = FullyConnected(self.fc_indim, out_channels, neurons)

    def forward(self, board, piece, location):
        piece = self.piecenet(piece, location)

        x = self.init_conv(board)
        x = self.pool1(x)
        x = self.dblock1(x)
        x = self.dblock2(x)
        # x = self.dblock3(x) # disabled for 2-20 tests
        # x = self.dblock4(x)
        x = self.pool2(x)
        x = self.final_conv(x)
        x = x.view(-1, self.fc_indim - self.piece_net_dim)

        x = torch.cat((x, piece), 1)

        x = self.fc(x)
        return x

    def copy(self):
        new_model = self.__class__(**self.init_args)
        new_model.load_state_dict(self.state_dict())
        new_model.eval()
        return new_model

class PieceNet(nn.Module):

    # used to process the piece matrix and the location of the piece
    def __init__(self, out_dim, embedding_size=(4, 4, 4)):

        super(PieceNet, self).__init__()
        self.embedding_size = embedding_size
        flattened_size = embedding_size[0] * embedding_size[1] * embedding_size[2] + 3 #location tuple
        self.fc1 = FullyConnected(flattened_size, flattened_size, 256)
        self.relu = nn.Sequential(nn.ReLU(), nn.Dropout(0.5))
        self.fc2 = FullyConnected(flattened_size, out_dim, 256)

    def embed(self, piece):
        device = piece.device
        out = torch.zeros((piece.shape[0], piece.shape[1], *self.embedding_size))
        for coord in product(*[range(x) for x in piece.shape]):
            try:
                out[coord] = piece[coord]
            except:
                print(out.shape, piece.shape)
        return out.to(device)

    def forward(self, piece, location):
        piece = self.embed(piece)
        x = piece.view(piece.shape[0], -1).float()
        location = location.view(location.shape[0], -1).float()
        x = torch.cat((x, location), 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x