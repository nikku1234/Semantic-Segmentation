import torch 
import torch.nn as nn
import torch.functional as F

class Encoder(nn.Module):
    """Encoder layer follows VGG rules + keeps pooling indices
        Args:
            num_input_features (int): number of input features
            num_output_features (int): number of output features
            n_blocks (int): number of conv-batch-relu block inside the encoder
            drop_rate (float): dropout rate to use
        """
    def __init__(self,num_input_features,num_output_features,n_blocks=2,drop_rate=0.5):
        super(Encoder,self)__init__()

        layers = [nn.Conv2d(num_input_features,num_output_features,kernel_size=3,stride=1,1),
                nn.BatchNorm2d(num_output_features),
                nn.ReLU(inplace=True)]
        if n_blocks > 1 :
            if n_blocks == 3 :
                layers = layers +[nn.Conv2d(num_input_features,num_output_features,kernel_size=3,stride=1,1),
                                nn.BatchNorm2d(num_output_features),
                                nn.ReLU(inplace=True),
                                nn.Dropout(drop_rate)]
            else :
                layers = layers + [nn.Conv2d(num_output_features,num_output_features,kernel_size=3,stride=1,1)],
                                    nn.BatchNorm2d(num_output_features),
                                    nn.ReLU(inplace=True)]
            
        self.features = nn.Sequential(*layers)
    
    def forward(self,x,indices,size):
        output = self.features(x)
        return F.max_pool2d(output, 2, 2, return_indices=True), output.size()

    

class Decoder(nn.Module):
     """Decoder layer decodes the features by unpooling with respect to
    the pooling indices of the corresponding decoder part.
    Args:
        num_input_featurs (int): number of input features
        num_output_feaures (int): number of output features
        n_blocks (int): number of conv-batch-relu block inside the decoder
        drop_rate (float): dropout rate to use
    """

    def __init__(self,num_input_features,num_output_features,n_blocks=2,drop_rate=0.5):
        super(Decoder,self)__init__()

        layers = [nn.Conv2d(num_input_features,num_input_features,kernel_size=3,stride=1,1),
                    nn.BatchNorm2d(num_input_features),
                    nn.ReLU(inplace=True)]
        if n_blocks > 1:
            if n_blocks == 3 :
                layers = layers + [nn.Conv2d(num_input_features,num_output_features,kernel_size=3,stride=1,1),
                                    nn.BatchNorm2d(num_output_features),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(drop_rate)]
            else :
                layers = layers + [nn.Conv2d(num_input_features,num_input_features, kernel_size=3,stride=1,1),
                                    nn.BatchNorm2d(num_input_features),
                                    nn.ReLU(inplace=True)]
        
        self.features = nn.Sequential(*layers)
    
    def forward(nn.Module):
        unpooled = F.max_unpool2d(x, indices, 2, 2, 0, size)
        return self.features(unpooled)
            

class SegNet(nn.Module):
'''

    Arguments used 
        num_classes(int) : number of classess used for the segmentation
        num_input_features(int) :the number of input features in the first convolution 
        drop_rate(float): dropout rate of each encoder/decoder module
        filter_config(list of integers): the number of the output features in each level 64->128->256->512->512 

'''

    def __init__(self,num_classes,num_input_features,drop_rate,filter_config=(64,128,256,512,512)):
        super (SegNet,self).__init__()

        self.enocoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

         # setup number of conv-bn-relu blocks per module and number of filters

        encoder_n_layers = (2, 2, 3, 3, 3)
        encoder_filter_config = (n_init_features,) + filter_config
        decoder_n_layers = (3, 3, 3, 2, 1)
        decoder_filter_config = filter_config[::-1] + (filter_config[0],)

        for i in range(0, 5):
            # encoder architecture
            self.encoders.append(Encoder(encoder_filter_config[i],
                                          encoder_filter_config[i + 1],
                                          encoder_n_layers[i], drop_rate))

            # decoder architecture
            self.decoders.append(Decoder(decoder_filter_config[i],
                                          decoder_filter_config[i + 1],
                                          decoder_n_layers[i], drop_rate))

        # final classifier (equivalent to a fully connected layer)
        self.classifier = nn.Conv2d(filter_config[0], num_classes, 3, 1, 1)

    def forward(self, x):
        indices = []
        unpool_sizes = []
        feat = x

        # encoder path, keep track of pooling indices and features size
        for i in range(0, 5):
            (feat, ind), size = self.encoders[i](feat)
            indices.append(ind)
            unpool_sizes.append(size)

        # decoder path, upsampling with corresponding indices and size
        for i in range(0, 5):
            feat = self.decoders[i](feat, indices[4 - i], unpool_sizes[4 - i])

        return self.classifier(feat)





