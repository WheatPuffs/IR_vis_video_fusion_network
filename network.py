from torch import nn, cat

channels = 32

# ====================================================================================
# Basic blocks
# ====================================================================================

class Input(nn.Module):
   def __init__(self, in_ch, out_ch, ker_sz = 3, strd = 1, padd = 0, refl_padd = [1, 1, 1, 1]):
        super(Input, self).__init__()
        self.sequence = nn.Sequential(
            nn.ReflectionPad2d(refl_padd),
            nn.Conv2d(
                in_channels  = in_ch, 
                out_channels = out_ch, 
                kernel_size  = ker_sz, 
                stride       = strd, 
                padding      = padd),
            nn.BatchNorm2d(num_features = out_ch),
            nn.PReLU()
            )
   def forward(self, x):
        return self.sequence(x)

class Conv_batch_prelu(nn.Module):
    def __init__(self, in_ch, out_ch, ker_sz = 3, strd = 1, padd = 1):
        super(Conv_batch_prelu, self).__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(
                in_channels  = in_ch, 
                out_channels = out_ch, 
                kernel_size  = ker_sz, 
                stride       = strd, 
                padding      = padd),
            nn.BatchNorm2d(num_features = out_ch),
            nn.PReLU()
            )
    def forward(self, x):
        return self.sequence(x)

class Conv_batch_tanh(nn.Module):
    def __init__(self, in_ch, out_ch, ker_sz = 3, strd = 1, padd = 1):
        super(Conv_batch_tanh, self).__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(
                in_channels  = in_ch, 
                out_channels = out_ch, 
                kernel_size  = ker_sz, 
                stride       = strd, 
                padding      = padd), 
            nn.BatchNorm2d(num_features = out_ch),
            nn.Tanh()
            )
    def forward(self, x):
        return self.sequence(x)

class Output(nn.Module):
    def __init__(self, in_ch, out_ch, ker_sz = 3, strd = 1, padd = 0, refl_padd = [1, 1, 1, 1]):
        super(Output, self).__init__()
        self.sequence = nn.Sequential(
            nn.ReflectionPad2d(refl_padd),
            nn.Conv2d(
                in_channels  = in_ch, 
                out_channels = out_ch, 
                kernel_size  = ker_sz, 
                stride       = strd, 
                padding      = padd), 
            nn.BatchNorm2d(num_features = out_ch),
            nn.Sigmoid()
            )
    def forward(self, x):
        return self.sequence(x)
        
# ====================================================================================
# Encoder and decoder
# ====================================================================================

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.input1=Input(
            in_ch     = 1, 
            out_ch    = channels // 4, 
            ker_sz    = 3, 
            strd      = 1, 
            padd      = 0,
            refl_padd = [1, 1, 1, 1])
        self.conv2=Conv_batch_prelu(
            in_ch  = channels // 4, 
            out_ch = channels // 4, 
            ker_sz = 3, 
            strd   = 1, 
            padd   = 1)
        self.conv3=Conv_batch_prelu(
            in_ch  = channels // 2, 
            out_ch = channels // 2, 
            ker_sz = 3, 
            strd   = 1, 
            padd   = 1)
        self.conv4 = Conv_batch_tanh(
            in_ch  = channels, 
            out_ch = channels, 
            ker_sz = 3, 
            strd   = 1, 
            padd   = 1)
        self.conv5 = Conv_batch_tanh(
            in_ch  = channels, 
            out_ch = channels, 
            ker_sz = 3, 
            strd   = 1, 
            padd   = 1)
        
    def forward(self, input_data):
        feature1 = self.input1(input_data)
        feature2 = cat([feature1,self.conv2(feature1)], 1)
        feature3 = cat([feature2,self.conv3(feature2)], 1)
        feature4 = self.conv4(feature3)
        feature5 = self.conv5(feature3)
        feature45 = cat([feature4, feature5], 1)
        return feature45
       
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv6 = Conv_batch_prelu(
            in_ch  = channels * 2, 
            out_ch = channels, 
            ker_sz = 3, 
            strd   = 1, 
            padd   = 1)
        self.conv7=Conv_batch_prelu(
            in_ch  = channels, 
            out_ch = channels // 2, 
            ker_sz = 3, 
            strd   = 1, 
            padd   = 1)
        self.conv8=Conv_batch_prelu(
            in_ch  = channels // 2, 
            out_ch = channels // 4, 
            ker_sz = 3, 
            strd   = 1, 
            padd   = 1)
        self.output9=Output(
            in_ch     = channels // 4, 
            out_ch    = 1, 
            ker_sz    = 3, 
            strd      = 1, 
            padd      = 0,
            refl_padd = [1, 1, 1, 1])

    def forward(self, feature45):
        feature6    = self.conv6(feature45)   
        feature7    = self.conv7(feature6) 
        feature8    = self.conv8(feature7)
        output_data = self.output9(feature8)
        return output_data

# ====================================================================================
# EOF
# ====================================================================================
