import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/work/shi/DeepLearning/models')
from fc_densenet_ggcn.layers_de import *

import numpy as np

from torch.nn import init

from fc_densenet_ggcn.ggcn import GatedGraphConvolutionNeuralNetwork
import scipy.sparse as sp

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class FCDenseNet(nn.Module):
    def __init__(self, in_channels=3, down_blocks=(5,5,5,5,5),
                 up_blocks=(5,5,5,5,5), bottleneck_layers=5,
                 growth_rate=16, out_chans_first_conv=48, n_classes=12):
        super().__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        cur_channels_count = 0
        skip_connection_channel_counts = []

        ## First Convolution ##

        self.add_module('firstconv', nn.Conv2d(in_channels=in_channels,
                  out_channels=out_chans_first_conv, kernel_size=3,
                  stride=1, padding=1, bias=True))
        cur_channels_count = out_chans_first_conv

        #####################
        # Downsampling path #
        #####################

        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(
                DenseBlock(cur_channels_count, growth_rate, down_blocks[i],dilation=1))
            cur_channels_count += (growth_rate*down_blocks[i])
            skip_connection_channel_counts.insert(0,cur_channels_count)
            self.transDownBlocks.append(TransitionDown(cur_channels_count))

        #####################
        #     Bottleneck    #
        #####################

        self.add_module('bottleneck',Bottleneck(cur_channels_count,
                                     growth_rate, bottleneck_layers))
        prev_block_channels = growth_rate*bottleneck_layers
        cur_channels_count += prev_block_channels

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks)-1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]

            self.denseBlocksUp.append(DenseBlock(
                cur_channels_count, growth_rate, up_blocks[i],
                    upsample=True))
            prev_block_channels = growth_rate*up_blocks[i]
            cur_channels_count += prev_block_channels

        self.transUpBlocks.append(TransitionUp(
            prev_block_channels, prev_block_channels))
        cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]

        self.denseBlocksUp.append(DenseBlock(
            cur_channels_count, growth_rate, up_blocks[-1],
                upsample=False))
        cur_channels_count += growth_rate*up_blocks[-1]


        
        self.denseBlocksUp2 = nn.ModuleList([])
        prev_block_channels= cur_channels_count
        self.transUpBlocks2=TransitionUp2(prev_block_channels, prev_block_channels)
            

 
        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
               out_channels=n_classes, kernel_size=1, stride=1,
                   padding=0, bias=True)
        self.finalConv2 = nn.Conv2d(in_channels=cur_channels_count,
               out_channels=n_classes, kernel_size=1, stride=1,
                   padding=0, bias=True)
        self.softmax = nn.LogSoftmax(dim=1)
        #self.GCN=GCN(nfeat=cur_channels_count,nhid1=100,nhid2=50,nhid3=25,nclass=n_classes,dropout=0.2)
        adj=sp.load_npz('/work/shi/DeepLearning/models/fc_densenet_ggcn/adjgcn7.npz')
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        adj=adj.cuda()
        self.ggcn=GatedGraphConvolutionNeuralNetwork(adj, hidden_size=cur_channels_count, num_edge_types=1,layer_timesteps=[1,1,1], residual_connections={2: [0]})
        #self.SupervisedGraphSage(num_classes=n_classes)
        #config =convcrf3.default_conf
        #config['filter_size'] = 7
        #config['col_feats']['use_bias'] = False
        #config['col_feats']['schan'] = 0.1
        #shape=(256, 256)
        #self.crf=convcrf3.GaussCRF(conf=config,shape=shape,nclasses=n_classes)
    def forward(self, x):
        out = self.firstconv(x)

        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)

        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)           


        scorea = self.finalConv(out)
        score = self.softmax(scorea)
        #scoreb=self.crf(unary=score, img=out)
        #scorec = torch.cat([score,out],1)
        b, f, h, w = out.shape[0:4]
        ps=h
        psa=ps*ps
        ws=7
        '''
        A=np.zeros((psa,psa))
        for i in range(0,psa):
         coly=i%ps
         rowy=i//ps

         colf=ps-coly
         rowf=ps-rowy
         if (colf>=ws):
          colf=ws
         if (rowf>=ws):
          rowf=ws
         for j in range(0,colf):
          for k in range(0,rowf):
           A[i,i+j+ps*k]=1
           A[i+j+ps*k,i]=1
        
        A=np.load('adjm3.npy')
        print('1')
        rowa,cola=np.where(A==1)
        adjc=np.ones(np.count_nonzero(A))
        print('2')
        adj = sp.coo_matrix((adjc, (rowa,cola)),shape=(b*h*w,b*h*w),dtype=np.float32)  
        print('3')
        '''
        #print('1')
        #adj_lists=np.load('my_filec.npy')
        #adj_list_type1 = AdjacencyList(node_num=b*h*w, adj_list=adj_lists, device=self.ggnn.device)
        #adj=sp.load_npz('adjgcn7.npz')
        features=out.view((b*h*w,f))
        #node_representations = self.ggnn.compute_node_representations(initial_node_representation=features)
        node_representations = self.ggcn(features)
        #adj = normalize(adj+ sp.eye(adj.shape[0]))
        #print('2')
        #adj = sparse_mx_to_torch_sparse_tensor(adj)
        #print('3')
        #adj_list=adj_list.cuda()
        #b, f, h, w = out.shape[0:4]
        #out=out.squeeze(0)

        #features=out.view((b*h*w,f))
        #out=torch.transpose(out, 0, 1)
        #scoreb=self.GCN(out,adj)
        '''
        agg1 = MeanAggregator(features, cuda=True)
        enc1 = Encoder(features, 1433, 128, adj_lists, agg1, gcn=True, cuda=True)
        agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=True)
        enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,base_model=enc1, gcn=True, cuda=True)
        enc1.num_samples = 5
        enc2.num_samples = 5
        graphsage=SupervisedGraphSage(11,enc2)
        #print('6')
        #print(graphsage)
        scoreb=graphsage.forward(np.arange(b*h*w))
        '''
        node_representations=node_representations.view((b,-1,h,w))
        scoreb=self.finalConv2(node_representations)
        scoreb=self.softmax(scoreb)
        #scoreb=scoreb.view((b,-1,h,w))
        #scoreb=torch.transpose(scoreb, 1, 3)
        #scoreb=torch.transpose(scoreb, 2, 3)
        #scoreb=scoreb.unsqueeze(0)
        scorec=score+scoreb
        return scorec

def FCDenseNet27_ggcn(n_classes):
    return FCDenseNet(
        in_channels=3, down_blocks=(1, 1, 1, 1, 1),
        up_blocks=(1, 1, 1, 1, 1), bottleneck_layers=1,
        growth_rate=12, out_chans_first_conv=48, n_classes=n_classes)
def FCDenseNet57_ggcn(n_classes):
    return FCDenseNet(
        in_channels=3, down_blocks=(4, 4, 4, 4, 4),
        up_blocks=(4, 4, 4, 4, 4), bottleneck_layers=4,
        growth_rate=12, out_chans_first_conv=48, n_classes=n_classes)


def FCDenseNet67_ggcn(n_classes):
    return FCDenseNet(
        in_channels=3, down_blocks=(5, 5, 5,5,5),
        up_blocks=(5, 5, 5,5,5), bottleneck_layers=5,
        growth_rate=16, out_chans_first_conv=48, n_classes=n_classes)


def FCDenseNet103_ggcn(n_classes):
    return FCDenseNet(
        in_channels=3, down_blocks=(4,5,7,10,12),
        up_blocks=(12,10,7,5,4), bottleneck_layers=15,
        growth_rate=16, out_chans_first_conv=48, n_classes=n_classes)
