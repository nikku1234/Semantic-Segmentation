import torch 
import torch.nn as nn
import torch.functional as F

class ModSegnet(nn.Module):
     """ModSegNet: Inspired from SegNet with small improvements
    Args:
        num_classes (int): number of classes to segment
        n_init_features (int): number of input features in the fist convolution
        drop_rate (float): dropout rate of the last two encoders
        filter_config (list of 5 ints): number of output features at each level 32->64->128->256
    """
    def __init__(self, num_classes,n_init_features=3, drop_rate = 0.5,filter_config=(32,64,128,256)):
        super(ModSegnet,self).__init__()

        self.encoder1 = Encoder(n_init_features, filter_config(0))
        self.encoder2 = _Encoder(filter_config[0], filter_config[1])
        self.encoder3 = _Encoder(filter_config[1], filter_config[2], drop_rate)
        self.encoder4 = _Encoder(filter_config[2], filter_config[3], drop_rate)

        self.decoder1 = _Decoder(filter_config[3], filter_config[2])
        self.decoder2 = _Decoder(filter_config[2], filter_config[1])
        self.decoder3 = _Decoder(filter_config[1], filter_config[0])
        self.decoder4 = _Decoder(filter_config[0], filter_config[0])

        # final classifier (equivalent to a fully connected layer)
        self.classifier = nn.Conv2d(filter_config[0], num_classes, 1)

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

    def forward(self, x):
        feat_encoder_1 = self.encoder1(x)
        size_2 = feat_encoder_1.size()
        feat_encoder_2, ind_2 = F.max_pool2d(feat_encoder_1, 2, 2,
                                             return_indices=True)

        feat_encoder_2 = self.encoder2(feat_encoder_2)
        size_3 = feat_encoder_2.size()
        feat_encoder_3, ind_3 = F.max_pool2d(feat_encoder_2, 2, 2,
                                             return_indices=True)

        feat_encoder_3 = self.encoder3(feat_encoder_3)
        size_4 = feat_encoder_3.size()
        feat_encoder_4, ind_4 = F.max_pool2d(feat_encoder_3, 2, 2,
                                             return_indices=True)

        feat_encoder_4 = self.encoder4(feat_encoder_4)
        size_5 = feat_encoder_4.size()
        feat_encoder_5, ind_5 = F.max_pool2d(feat_encoder_4, 2, 2,
                                             return_indices=True)

        feat_decoder = self.decoder1(feat_encoder_5, feat_encoder_4, ind_5, size_5)
        feat_decoder = self.decoder2(feat_decoder, feat_encoder_3, ind_4, size_4)
        feat_decoder = self.decoder3(feat_decoder, feat_encoder_2, ind_3, size_3)
        feat_decoder = self.decoder4(feat_decoder, feat_encoder_1, ind_2, size_2)

        return self.classifier(feat_decoder)


class Encoder(nn.Module):
    """Encoder layer encodes the features along the contracting path (left side).
    Args:
        num_input_features (int): number of input features
        num_output_features (int): number of output features
        drop_rate (float): dropout rate at the end of the block
    """
    def __init__(self,num_input_features,num_output_features,drop_rate=0):
        super(Encoder,self).__init__()

        layers = [nn.Conv2d(in_channels=num_input_features,out_channels=num_output_features,kernel_size=3,stride=1,1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=num_output_features,out_channels=num_output_features,kernel_size=3,stride=1,1),
                    nn.ReLU(inplace=True)]

        if drop_rate > 0 :
            layers = layers + [nn.Dropout(drop_rate)]
        
        self.features = nn.Sequential(*layers)
    
    def forward(self,x):
        return self.features(x)

class Decoder(nn.Module):
    """Decoder layer decodes the features by performing deconvolutions and
    concatenating the resulting features with cropped features from the
    corresponding encoder (skip-connections). Encoder features are cropped
    because convolution operations does not allow to recover the same
    resolution in the expansive path.
    Args:
        num_input_features (int): number of input features
        num_output_features (int): number of output features
    """
    def __init__(self,num_input_features,num_out_features):
        super(Decoder,self).__init__

        self.encoder = Encoder(num_input_features*2, num_output_features)
    
    def forward(self,x,feat_encoder, indices, size):
        unpooled = F.max_unpool2d(x, indices, 2, 2, 0, size)
        feat = torch.cat([unpooled, feat_encoder], 1)
        feat = self.encoder(feat)

        return feat
