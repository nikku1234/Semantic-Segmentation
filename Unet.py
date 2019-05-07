import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
'''
            Encodes the features along the contracting path, the left side of the architecture.Drop_rate used in the paper is 0.1

        Arguments used 
            num_input_features(int): number of input features
            num_output_features(int): number of output features
            drop_rate(float):the drop_rate at the end of the block
'''
    def __init__(self,num_input_features,num_output_features,drop_rate=0):
        super(Encoder,self).__init__()

        layers = [nn.Conv2d(num_input_features,num_input_features,kernel_size=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_output_features,num_output_features,kernel_size=3),
                nn.ReLU(inplace=True)]

        if drop_rate > 0 :
           layers = layers + [nn.Dropout(drop_rate)]
            
        self.features = nn.Sequential(*layers)
    
    def forward(self,x):
        return self.features(x)

    
class Decoder(nn.Module):
'''
    Decoder layer decodes the features by performing deconvolutions and concatinating the resulting features with cropped features from the corrosponding skip connections.
    This is the right side of the architecture.
    
    Arguments used 
            num_input_features(int): number of input features
            num_output_features(int): number of output features 

'''

    def __init__(self,num_input_features,num_output_features):
        super(Decoder,self).__init__()

        self.encoder = Encoder(num_input_features,num_output_features)
        self.decoder = nn.ConvTranspose2d(num_input_features,num_output_features,kernel_size=2,stride=2)
    
    def forward(self, x,feat_encoder):
        feat_decoder = F.relu(self.decoder(x),inplace=True)

        #Evaluation offset to allow cropping of the encoders features
        crop_size = feat_decoder.size(-1)
        offset = (feat_encoder.size(-1) - crop_size) // 2
        crop = feat_encoder[:, :, offset:offset + crop_size,offset:offset + crop_size]
        return self.encoder(torch.cat([feat_decoder, crop], 1))


class UNet(nn.Module):
    '''
    Arguments used in the code 
        num_classes(int) : number of classess used for the segmentation
        num_input_features(int) :the number of input features in the first convolution 
        drop_rate(float): the drop rate needed for the last two encoders
        filter_config(list of integers): the number of the output features in each level 64->128->256->512->1024
    '''

    def __init__(self,num_classes=2,num_input_features=1,drop_rate=1.5,filter_config=(64,128,256,512,1024)):
        super(UNet,self).__init__()

        self.encoder1 = Encoder(num_input_features,filter_config[0])
        self.encoder2 = Encoder(filter_config[0],filter_config[1])
        self.encoder3 = Encoder(filter_config[1],filter_config[2])
        self.encoder4 = Encoder(filter_config[2],filter_config[3],drop_rate)
        self.encoder5 = Encoder(filter_config[3],filter_config[4],drop_rate)

        self.decoder1 = Decoder(filter_config[4], filter_config[3])
        self.decoder2 = Decoder(filter_config[3], filter_config[2])
        self.decoder3 = Decoder(filter_config[2], filter_config[1])
        self.decoder4 = Decoder(filter_config[1], filter_config[0])

        # final classifier (equivalent to a fully connected layer)
        self.classifier = nn.Conv2d(filter_config[0], num_classes, 1)

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal(m.weight)

    def forward(self, x):
        feat_encoder_1 = self.encoder1(x)
        feat_encoder_2 = self.encoder2(F.max_pool2d(feat_encoder_1, 2))
        feat_encoder_3 = self.encoder3(F.max_pool2d(feat_encoder_2, 2))
        feat_encoder_4 = self.encoder4(F.max_pool2d(feat_encoder_3, 2))
        feat_encoder_5 = self.encoder5(F.max_pool2d(feat_encoder_4, 2))

        feat_decoder = self.decoder1(feat_encoder_5, feat_encoder_4)
        feat_decoder = self.decoder2(feat_decoder, feat_encoder_3)
        feat_decoder = self.decoder3(feat_decoder, feat_encoder_2)
        feat_decoder = self.decoder4(feat_decoder, feat_encoder_1)

        return self.classifier(feat_decoder)
