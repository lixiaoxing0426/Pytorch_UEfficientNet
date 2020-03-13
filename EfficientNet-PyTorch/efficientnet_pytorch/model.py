import torch
from torch import nn
from torch.nn import functional as F

from .utils import (
    relu_fn,
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
)

class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = relu_fn(self._bn0(self._expand_conv(inputs)))
        x = relu_fn(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(relu_fn(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x


class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')

    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._dropout = self._global_params.dropout_rate
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)

        #self._conv2d = nn.Conv2d(2048,3,3)
        #self._upsamples = nn.Upsample(size=None, scale_factor=16.0, mode='nearest', align_corners=None)

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        x = relu_fn(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            #print(idx)     
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = relu_fn(self._bn1(self._conv_head(x)))

        return x

    def extract_features_up(self, inputs):
        """ Returns output of the final convolution layer """
        # Stem
        x = relu_fn(self._bn0(self._conv_stem(inputs)))
        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
        # Head
        x = relu_fn(self._bn1(self._conv_head(x)))
        x = self._conv2d(x) 
        x = self._upsamples(x)
        return x

    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """

        # Convolution layers
        x = self.extract_features(inputs)

        # Pooling and final linear layer
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        if self._dropout:
            x = F.dropout(x, p=self._dropout, training=self.training)

        #### 2019-12-13
        feats = x

        x = self._fc(x)

        #return x
        return x, feats

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return EfficientNet(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1000):
        model = EfficientNet.from_name(model_name, override_params={'num_classes': num_classes})
        load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000))
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name, also_need_pretrained_weights=False):
        """ Validates model name. None that pretrained weights are only available for
        the first four models (efficientnet-b{i} for i in 0,1,2,3) at the moment. """
        num_models = 4 if also_need_pretrained_weights else 8
        valid_models = ['efficientnet_b'+str(i) for i in range(num_models)]
        if model_name.replace('-','_') not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))


class UEfficientNet(nn.Module):

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # upsampling
        self._dropout = self._global_params.dropout_rate

        self._conv0 = nn.Conv2d(2048, 512, (1,1), stride=1, padding=0)
        self._bnc0 = nn.BatchNorm2d(512, momentum=bn_mom, eps=bn_eps)

        in_ch = [0, 512+512, 512+176, 256+64, 128+40, 64+24]
        out_ch= [0, 512, 256, 128, 64, 32]
        #self._up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self._up1 = nn.ConvTranspose2d(in_ch[1], in_ch[1], (4,4), stride=2, padding=1)
        self._up1_conv1 = nn.Conv2d(in_ch[1], out_ch[1], 3, padding=1)
        self._up1_bn1 = nn.BatchNorm2d(out_ch[1])
        self._up1_relu1 = nn.ReLU(inplace=True)
        self._up1_conv2 = nn.Conv2d(out_ch[1], out_ch[1], 3, padding=1)
        self._up1_bn2 = nn.BatchNorm2d(out_ch[1])
        self._up1_relu2 = nn.ReLU(inplace=True)

        #self._up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self._up2 = nn.ConvTranspose2d(in_ch[2], in_ch[2], (4,4), stride=2, padding=1)
        self._up2_conv1 = nn.Conv2d(in_ch[2], out_ch[2], 3, padding=1)
        self._up2_bn1 = nn.BatchNorm2d(out_ch[2])
        self._up2_relu1 = nn.ReLU(inplace=True)
        self._up2_conv2 = nn.Conv2d(out_ch[2], out_ch[2], 3, padding=1)
        self._up2_bn2 = nn.BatchNorm2d(out_ch[2])
        self._up2_relu2 = nn.ReLU(inplace=True)

        #self._up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self._up3 = nn.ConvTranspose2d(in_ch[3], in_ch[3], (4,4), stride=2, padding=1)
        self._up3_conv1 = nn.Conv2d(in_ch[3], out_ch[3], 3, padding=1)
        self._up3_bn1 = nn.BatchNorm2d(out_ch[3])
        self._up3_relu1 = nn.ReLU(inplace=True)
        self._up3_conv2 = nn.Conv2d(out_ch[3], out_ch[3], 3, padding=1)
        self._up3_bn2 = nn.BatchNorm2d(out_ch[3])
        self._up3_relu2 = nn.ReLU(inplace=True)

        #self._up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self._up4 = nn.ConvTranspose2d(in_ch[4], in_ch[4], (4,4), stride=2, padding=1)
        self._up4_conv1 = nn.Conv2d(in_ch[4], out_ch[4], 3, padding=1)
        self._up4_bn1 = nn.BatchNorm2d(out_ch[4])
        self._up4_relu1 = nn.ReLU(inplace=True)
        self._up4_conv2 = nn.Conv2d(out_ch[4], out_ch[4], 3, padding=1)
        self._up4_bn2 = nn.BatchNorm2d(out_ch[4])
        self._up4_relu2 = nn.ReLU(inplace=True)

        #self._up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self._up5 = nn.ConvTranspose2d(in_ch[5], in_ch[5], (4,4), stride=2, padding=1)
        self._up5_conv1 = nn.Conv2d(in_ch[5], out_ch[5], 3, padding=1)
        self._up5_bn1 = nn.BatchNorm2d(out_ch[5])
        self._up5_relu1 = nn.ReLU(inplace=True)
        self._up5_conv2 = nn.Conv2d(out_ch[5], out_ch[5], 3, padding=1)
        self._up5_bn2 = nn.BatchNorm2d(out_ch[5])
        self._up5_relu2 = nn.ReLU(inplace=True)

        self._outconv = nn.Conv2d(out_ch[5],1,1)
        #self._up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self._con1= nn.Conv2d(512+512, 512, (3,3), padding=1)
        #self._bnup1 = nn.BatchNorm2d(num_features=512, momentum=bn_mom, eps=bn_eps)
        #self._up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self._con2= nn.Conv2d(512+176, 256, (3,3), padding=1)
        #self._bnup2 = nn.BatchNorm2d(num_features=256, momentum=bn_mom, eps=bn_eps)
        #self._up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self._con3= nn.Conv2d(256+64, 128, (3,3), padding=1)
        #self._bnup3 = nn.BatchNorm2d(num_features=128, momentum=bn_mom, eps=bn_eps)
        #self._up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self._con4= nn.Conv2d(128+40, 64, (3,3), padding=1)
        #self._bnup4 = nn.BatchNorm2d(num_features=64, momentum=bn_mom, eps=bn_eps)
        #self._up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self._con5= nn.Conv2d(64+24, 1, (3,3), padding=1)

        #self._up1 = nn.ConvTranspose2d(512+512, 512, (4,4), stride=2, padding=1)
        #self._bnup1 = nn.BatchNorm2d(num_features=512, momentum=bn_mom, eps=bn_eps)
        #self._up2 = nn.ConvTranspose2d(512+176,  256, (4,4), stride=2, padding=1)
        #self._bnup2 = nn.BatchNorm2d(num_features=256, momentum=bn_mom, eps=bn_eps)
        #self._up3 = nn.ConvTranspose2d(256+64, 128, (4,4), stride=2, padding=1)
        #self._bnup3 = nn.BatchNorm2d(num_features=128, momentum=bn_mom, eps=bn_eps)
        #self._up4 = nn.ConvTranspose2d(128+40, 64, (4,4), stride=2, padding=1)
        #self._bnup4 = nn.BatchNorm2d(num_features=64, momentum=bn_mom, eps=bn_eps)
        #self._up5 = nn.ConvTranspose2d(64+24, 1, (4,4), stride=2, padding=1)


    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        x = relu_fn(self._bn0(self._conv_stem(inputs)))

        x_blocks = []
        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            x_blocks.append(x)
            
        # Head
        x = relu_fn(self._bn1(self._conv_head(x)))

        return x, x_blocks

    def up_features(self, x_features, inch, outch):

        x = relu_fn(self._bnup(self._up(x_features)))
        return x

    def x_concat(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        cat = torch.cat([x1, x2], dim=1)
        return cat

    def forward(self, inputs):

        x, x_blocks = self.extract_features(inputs)
       
        # Pooling and final linear layer
        out_size = x.shape[-1]
        x = F.adaptive_avg_pool2d(x,out_size)
        if self._dropout:
            x = F.dropout(x, p=self._dropout, training=self.training)

        x = relu_fn(self._bnc0(self._conv0(x)))
        x = self.x_concat(x, x_blocks[38].detach())

        x = self._up1(x) 
        x = self._up1_conv1(x) 
        x = self._up1_bn1(x)
        x = self._up1_relu1(x)    
        x = self._up1_conv2(x)
        x = self._up1_bn2(x)
        x = self._up1_relu2(x)    
        x = self.x_concat(x, x_blocks[26].detach())

        x = self._up2(x) 
        x = self._up2_conv1(x) 
        x = self._up2_bn1(x)
        x = self._up2_relu1(x)    
        x = self._up2_conv2(x)
        x = self._up2_bn2(x)
        x = self._up2_relu2(x)    
        x = self.x_concat(x, x_blocks[12].detach())

        x = self._up3(x) 
        x = self._up3_conv1(x) 
        x = self._up3_bn1(x)
        x = self._up3_relu1(x)    
        x = self._up3_conv2(x)
        x = self._up3_bn2(x)
        x = self._up3_relu2(x)    
        x = self.x_concat(x, x_blocks[7].detach())

        x = self._up4(x) 
        x = self._up4_conv1(x) 
        x = self._up4_bn1(x)
        x = self._up4_relu1(x)    
        x = self._up4_conv2(x)
        x = self._up4_bn2(x)
        x = self._up4_relu2(x)    
        x = self.x_concat(x, x_blocks[2].detach())
 
        x = self._up5(x) 
        x = self._up5_conv1(x) 
        x = self._up5_bn1(x)
        x = self._up5_relu1(x)    
        x = self._up5_conv2(x)
        x = self._up5_bn2(x)
        x = self._up5_relu2(x)    

        x = self._outconv(x)
        x = F.sigmoid(x)
        
        #if inputs.shape[-1]!=x.shape[-1] or inputs.shape[-2]!=x.shape[-2]
        #    x
      
        return x

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return UEfficientNet(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1000):
        model = UEfficientNet.from_name(model_name, override_params={'num_classes': num_classes})
        load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000))
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name, also_need_pretrained_weights=False):
        """ Validates model name. None that pretrained weights are only available for
        the first four models (efficientnet-b{i} for i in 0,1,2,3) at the moment. """
        num_models = 4 if also_need_pretrained_weights else 8
        valid_models = ['efficientnet_b'+str(i) for i in range(num_models)]
        if model_name.replace('-','_') not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))
