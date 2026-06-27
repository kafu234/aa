"""DGAT-BLS downstream classifier for SEED DE features."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.empty(in_features, out_features))
        self.a = nn.Parameter(torch.empty(2 * out_features, 1))
        nn.init.xavier_uniform_(self.W, gain=1.414)
        nn.init.xavier_uniform_(self.a, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):
        h = torch.matmul(inp, self.W)
        batch, nodes = h.size(0), h.size(1)
        left = h.repeat(1, 1, nodes).view(batch, nodes * nodes, self.out_features)
        right = h.repeat(1, nodes, 1)
        a_input = torch.cat([left, right], dim=-1).view(
            batch, nodes, nodes, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        attention = torch.where(adj > 0, e, torch.full_like(e, -1e12))
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        if self.concat:
            return F.relu(h_prime)
        return h_prime


class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, dropout, alpha, n_heads):
        super().__init__()
        self.dropout = dropout
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True)
            for _ in range(n_heads)
        ])
        self.out_att = GraphAttentionLayer(
            n_hid * n_heads, n_class, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj1, adj2):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj1) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        return F.elu(self.out_att(x, adj2))


class ChannelAttention(nn.Module):
    def __init__(self, in_planes=62, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=1):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, in_channels=62, ratio=8):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return x


class SpearmanCorrelation(nn.Module):
    """Batch-wise electrode correlation block from the provided model code."""

    def forward(self, de_data):
        batch_size, electrodes, freq = de_data.size()
        ranks = torch.argsort(torch.argsort(de_data, dim=1), dim=1).float()
        ranks = ranks.permute(0, 2, 1)
        diff = torch.abs(ranks.unsqueeze(3) - ranks.unsqueeze(2))
        weight = 1.0 / (diff + 1.0)
        se = torch.sum(weight, dim=2, keepdim=True)
        corr = torch.matmul(weight.transpose(2, 3), weight)
        corr = corr - torch.matmul(se.transpose(2, 3), se) / max(electrodes - 1, 1)
        return corr.mean(dim=1)


class SENet(nn.Module):
    def __init__(self, in_channel, hidden_channel):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, hidden_channel, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_channel, in_channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        weights = self.fc(self.gap(x).flatten(1, 2))
        return torch.unsqueeze(weights, dim=2) * x


class SperChannel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(62, 62, 3, padding=1)
        self.conv2 = nn.Conv1d(62, 62, 3, padding=1)
        self.selu = nn.SELU()
        self.bn1 = nn.BatchNorm1d(62)
        self.bn2 = nn.BatchNorm1d(62)
        self.senet1 = SENet(62, 31)
        self.senet2 = SENet(62, 31)

    def forward(self, x):
        x1 = self.bn1(self.selu(self.conv1(x)))
        x2 = self.senet1(x1)
        x3 = x2 + x
        x4 = self.bn2(self.selu(self.conv2(x3)))
        x5 = self.senet2(x4)
        return x5 + x3


class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.depth_conv = nn.Conv1d(
            in_channels=in_channel,
            out_channels=in_channel,
            kernel_size=3,
            stride=1,
            padding=0,
            groups=in_channel,
        )
        self.point_conv = nn.Conv1d(in_channel, out_channel, kernel_size=1)

    def forward(self, x):
        return self.point_conv(self.depth_conv(x))


class CabamPro(nn.Module):
    def __init__(self):
        super().__init__()
        self.cbam1 = CBAM()
        self.cbam2 = CBAM()
        self.conv1 = nn.Conv1d(62, 62, 3, padding=0)
        self.conv2 = nn.Conv1d(62, 62, 3, padding=0)
        self.selu = nn.SELU()
        self.bn1 = nn.BatchNorm1d(62)
        self.bn2 = nn.BatchNorm1d(62)

    def forward(self, x):
        x1 = self.selu(self.bn1(self.conv1(x)))
        x2 = torch.cat((x, x1), dim=2)
        x3 = self.cbam1(x2)
        x4 = self.selu(self.bn2(self.conv2(x3)))
        x5 = torch.cat((x1, x4), dim=2)
        return self.cbam2(x5)


class DEProcess(nn.Module):
    def __init__(self):
        super().__init__()
        self.dwconv1 = DepthWiseConv(62, 62)
        self.dwconv2 = DepthWiseConv(62, 62)
        self.dwconv3 = DepthWiseConv(62, 62)
        self.selu = nn.SELU()
        self.bn1 = nn.BatchNorm1d(62)
        self.bn2 = nn.BatchNorm1d(62)
        self.bn3 = nn.BatchNorm1d(62)

    def forward(self, x):
        x1 = self.bn1(self.selu(self.dwconv1(x)))
        x2 = torch.cat((x, x1), dim=2)
        x3 = self.bn2(self.selu(self.dwconv2(x2)))
        x4 = torch.cat((x1, x3), dim=2)
        return self.bn3(self.selu(self.dwconv3(x4)))


class BLS(nn.Module):
    def __init__(self, in_nodes, feature_nodes, enhancement_nodes, out_nodes):
        super().__init__()
        self.feature_layers = nn.ModuleList([
            nn.Linear(in_nodes, feature_nodes) for _ in range(10)
        ])
        self.fc31 = nn.Linear(feature_nodes * 10, enhancement_nodes)
        self.fc32 = nn.Linear(feature_nodes * 10 + enhancement_nodes, out_nodes)

    def forward(self, x):
        feature_nodes = torch.cat(
            [torch.sigmoid(layer(x)) for layer in self.feature_layers], dim=1)
        enhancement_nodes = torch.sigmoid(self.fc31(feature_nodes))
        return self.fc32(torch.cat([feature_nodes, enhancement_nodes], dim=1))


def normalize_A(A, symmetry=False):
    A = F.relu(A)
    if symmetry:
        A = A + torch.transpose(A, 0, 1)
    d = torch.sum(A, 1)
    d = 1 / torch.sqrt(d + 1e-10)
    D = torch.diag_embed(d)
    return torch.matmul(torch.matmul(D, A), D)


class DGATBLSClassifier(nn.Module):
    """Provided DGAT-BLS model adapted to output class logits."""

    def __init__(self, nclass=3):
        super().__init__()
        self.BN1 = nn.BatchNorm1d(5)
        self.fc33 = nn.Linear(1458, 512)
        self.fc34 = nn.Linear(512, 128)
        self.fc35 = nn.Linear(128, nclass)
        self.dropout = nn.Dropout(p=0.1)
        self.sperman = SpearmanCorrelation()
        self.sperchannel = SperChannel()
        self.deprocess = DEProcess()
        self.cbam_pro = CabamPro()
        self.bls1 = BLS(3968, 200, 100, 1024)
        self.adj1 = nn.Parameter(torch.empty(62, 62))
        self.adj2 = nn.Parameter(torch.empty(62, 62))
        nn.init.xavier_normal_(self.adj1)
        nn.init.xavier_normal_(self.adj2)
        self.gatnet1 = GAT(71, 81, 64, 0.15, 0.1, 5)

    def forward(self, x):
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)
        corr = self.sperman(x)
        result1 = self.cbam_pro(x)
        result2 = self.sperchannel(corr)
        result3 = self.deprocess(x)
        resultx = torch.cat((result1, result2), dim=2)
        adj1 = normalize_A(self.adj1)
        adj2 = normalize_A(self.adj2)
        resulty = self.gatnet1(resultx, adj1, adj2)
        resulty = self.bls1(torch.flatten(resulty, 1, 2))
        result4 = torch.flatten(result3, 1, 2)
        result = torch.cat((resulty, result4), dim=1)
        result = self.dropout(F.selu(self.fc33(result)))
        result = self.dropout(F.selu(self.fc34(result)))
        return self.fc35(result)
