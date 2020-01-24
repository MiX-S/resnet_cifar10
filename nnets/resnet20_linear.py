import torch
from torch.nn import Conv2d, BatchNorm2d, ReLU, Linear

"""
My realisation of resnet20 without sequential
"""

class resnet20_explicit(torch.nn.Module):
    def __init__(self, use_batch_norm=True, use_drop_out=False, d_out_p=0.5):
        super(resnet20_explicit, self).__init__()
        # self.batch_norm0 = torch.nn.BatchNorm2d(3)

        self.conv1 = Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batch_norm1 = BatchNorm2d(16)
        self.act1  = ReLU()
        # dropout
        
                              ###############    Layer 1    ###############
        self.conv2 = Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batch_norm2 = BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.dropout = Dropout2d(p=0.5, inplace=False)

        self.conv3 = Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batch_norm3 = BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.dropout = Dropout2d(p=0.5, inplace=False)
        self.act2  = torch.nn.ReLU()


        self.conv4 = Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batch_norm4 = BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.dropout = Dropout2d(p=0.5, inplace=False)
        
        self.conv5 = Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batch_norm5 = BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.dropout = Dropout2d(p=0.5, inplace=False)
        self.act3  = torch.nn.ReLU()


        self.conv6 = Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batch_norm6 = BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.dropout = Dropout2d(p=0.5, inplace=False)
        
        self.conv7 = Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batch_norm7 = BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.dropout = Dropout2d(p=0.5, inplace=False)
        self.act4  = torch.nn.ReLU()
                              ###############    Layer 1    ###############


        
                              ###############    Layer 2    ###############
        self.conv8 = Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batch_norm8 = BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.dropout = Dropout2d(p=0.5, inplace=False)

        self.conv9 = Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batch_norm9 = BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.dropout = Dropout2d(p=0.5, inplace=False)
        self.act5  = torch.nn.ReLU()


        self.conv10 = Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batch_norm10 = BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.dropout = Dropout2d(p=0.5, inplace=False)

        self.conv11 = Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batch_norm11 = BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.dropout = Dropout2d(p=0.5, inplace=False)
        self.act6  = torch.nn.ReLU()


        self.conv12 = Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batch_norm12 = BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.dropout = Dropout2d(p=0.5, inplace=False)

        self.conv13 = Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batch_norm13 = BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.dropout = Dropout2d(p=0.5, inplace=False)
        self.act7  = torch.nn.ReLU()
                              ###############    Layer 2    ###############



                              ###############    Layer 3    ###############
        self.conv14 = Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batch_norm14 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.dropout = Dropout2d(p=0.5, inplace=False)

        self.conv15 = Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batch_norm15 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.dropout = Dropout2d(p=0.5, inplace=False)
        self.act8  = torch.nn.ReLU()


        self.conv16 = Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batch_norm16 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.dropout = Dropout2d(p=0.5, inplace=False)

        self.conv17 = Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batch_norm17 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.dropout = Dropout2d(p=0.5, inplace=False)
        self.act9  = torch.nn.ReLU()


        self.conv18 = Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batch_norm18 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.dropout = Dropout2d(p=0.5, inplace=False)

        self.conv19 = Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batch_norm19 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.dropout = Dropout2d(p=0.5, inplace=False)
        self.act10  = torch.nn.ReLU()
                              ###############    Layer 3    ###############                              

        self.pool1 = torch.nn.AvgPool2d(8, 8)
        self.fc1   = Linear(in_features=64, out_features=10, bias=True)


    
    def forward(self, x):

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.act1(x)
        
                              ###############    Layer 1    ###############
        x_res = x.clone()
        x = self.conv2(x_res)
        x = self.batch_norm2(x)
        # self.dropout

        x = self.conv3(x)
        x = self.batch_norm3(x)
        # self.dropout
        x = self.act2(x)


        x_res = x_res + x
        x = self.conv4(x_res)
        x = self.batch_norm4(x)
        # self.dropout

        x = self.conv5(x)
        x = self.batch_norm5(x)
        # self.dropout
        x = self.act3(x)


        x_res = x_res + x
        x = self.conv6(x_res)
        x = self.batch_norm6(x)
        # self.dropout

        x = self.conv7(x)
        x = self.batch_norm7(x)
        # self.dropout
        x = self.act4(x)
                              ###############    Layer 1    ###############


                              ###############    Layer 2    ###############
        x_res = x_res + x
        x = self.conv8(x_res)
        x = self.batch_norm8(x)
        # self.dropout

        x = self.conv9(x)
        x = self.batch_norm9(x)
        # self.dropout
        x = self.act5(x)


        x_res = x
        x = self.conv10(x_res)
        x = self.batch_norm10(x)
        # self.dropout

        x = self.conv11(x)
        x = self.batch_norm11(x)
        # self.dropout
        x = self.act6(x)


        x_res = x_res + x
        x = self.conv12(x_res)
        x = self.batch_norm12(x)
        # self.dropout

        x = self.conv13(x)
        x = self.batch_norm13(x)
        # self.dropout
        x = self.act7(x)
                              ###############    Layer 2    ###############


                              ###############    Layer 3    ###############
        x_res = x_res + x
        x = self.conv14(x_res)
        x = self.batch_norm14(x)
        # self.dropout

        x = self.conv15(x)
        x = self.batch_norm15(x)
        # self.dropout
        x = self.act8(x)


        x_res = x
        x = self.conv16(x_res)
        x = self.batch_norm16(x)
        # self.dropout

        x = self.conv17(x)
        x = self.batch_norm17(x)
        # self.dropout
        x = self.act9(x)


        x_res = x_res + x
        x = self.conv18(x_res)
        x = self.batch_norm18(x)
        # self.dropout

        x = self.conv19(x)
        x = self.batch_norm19(x)
        # self.dropout
        x = self.act10(x)
                              ###############    Layer 3    ###############
        
        x_res = x_res + x
        x = self.pool1(x)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        # print(x.shape)
        x = self.fc1(x)
        
        return x
