import torch
from torch import nn
class justCNN(nn.Module):
    def __init__(self,input_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.xConvBlock = nn.Sequential(nn.Conv1d(1 , 5 , 3)  , nn.ReLU()  ,nn.MaxPool1d(2 , 2),
                                        nn.Conv1d(5 , 10 , 3) , nn.ReLU() , nn.MaxPool1d(2 ,2))
        self.yConvBlock = nn.Sequential(nn.Conv1d(1 , 5 , 3)  , nn.ReLU()  ,nn.MaxPool1d(2 , 2),
                                        nn.Conv1d(5 , 10 , 3) , nn.ReLU() , nn.MaxPool1d(2 ,2))
        self.zConvBlock = nn.Sequential(nn.Conv1d(1 , 5 , 3)  , nn.ReLU()  ,nn.MaxPool1d(2 , 2),
                                        nn.Conv1d(5 , 10 , 3) , nn.ReLU() , nn.MaxPool1d(2 ,2))
        # self.denseBlock = nn.Sequential(nn.Linear(390 , 250) , nn.ReLU() ,
        #                                 nn.Linear(250 ,150) , nn.ReLU() ,
        #                                 nn.Linear(150 , 100) , nn.ReLU(),
        #                                 nn.Linear(100 , 50 ) , nn.ReLU(),
        #                                 nn.Linear(50 , 1)) # , nn.ReLU(),
                                        # nn.Linear(10 , 3) )
        temp_in = torch.unsqueeze(torch.zeros(input_size) , 0)
        temp_in =  temp_in.permute(1 , 0 , 2)
        # print(data.shape)
        # print(data.shape)
        xData = torch.unsqueeze(temp_in[0] , 1)#.permute(1 , 2 , 0)
        yData = torch.unsqueeze(temp_in[1] , 1)#.permute(1 , 2 , 0)
        zData = torch.unsqueeze(temp_in[2] , 1)#.permute(1 , 2 , 0)
        # print(xData.shape)
        xConvOutput = self.xConvBlock(xData)
        # print(xConvOutput.shape)
        yConvOutput = self.yConvBlock(yData)
        zConvOutput = self.zConvBlock(zData)
        xFlat =  torch.flatten(xConvOutput , 1)
        yFlat = torch.flatten(yConvOutput , 1)
        zFlat = torch.flatten(zConvOutput , 1)
        # print(xConvOutput.shape)
        catData = torch.cat((xFlat , yFlat , zFlat) , 1)
        self.denseBlock = nn.Sequential(nn.Linear(catData.shape[1] , 1))


    def forward(self ,data):
        # print(data.shape)
        # print(data.shape)
        data =  data.permute(1 , 0 , 2)
        # print(data.shape)
        # print(data.shape)
        xData = torch.unsqueeze(data[0] , 1)#.permute(1 , 2 , 0)
        yData = torch.unsqueeze(data[1] , 1)#.permute(1 , 2 , 0)
        zData = torch.unsqueeze(data[2] , 1)#.permute(1 , 2 , 0)
        # print(xData.shape)
        xConvOutput = self.xConvBlock(xData)
        # print(xConvOutput.shape)
        yConvOutput = self.yConvBlock(yData)
        zConvOutput = self.zConvBlock(zData)
        xFlat =  torch.flatten(xConvOutput , 1)
        yFlat = torch.flatten(yConvOutput , 1)
        zFlat = torch.flatten(zConvOutput , 1)
        # print(xConvOutput.shape)
        catData = torch.cat((xFlat , yFlat , zFlat) , 1)
        # print(catData.shape)
        catDataSqueezed = torch.squeeze(catData , 1)
        # print(catDataSqueezed.shape)
        output = self.denseBlock(catDataSqueezed)
        # print(output.shape)
        # return output.reshape(output.shape[0])
        return output

