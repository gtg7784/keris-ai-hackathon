# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from nsml import GPU_NUM

class CNNReg(nn.Module):
    def __init__(self, embedding_dim: int, max_length: int):
        super(CNNReg, self).__init__()
        self.character_size = 251
        self.kernel_size = [2, 3, 4, 5]
        self.channel_out = 10
        self.embedding = nn.Embedding(self.character_size, embedding_dim)
        self.conv1 = nn.ModuleList(
            [nn.Conv2d(1, self.channel_out, (k, embedding_dim)) for k in self.kernel_size])
        self.linear1 = nn.Linear(self.channel_out*len(self.kernel_size), 10)
        self.linear2 = nn.Linear(10, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        embed = self.embedding(x)   # (N, W, D)
        embed = embed.unsqueeze(1)  # (N,1,W,D), 1: channel_in

        # [(N,Channel_out,W,1), ...] * len(Kernel_size)
        feature_maps = [F.relu(conv(embed)) for conv in self.conv1]
        
        # [(N,Channel_out,W), ...] * len(Kernel_size)
        feature_maps = [feature_map.squeeze(3) for feature_map in feature_maps]

        # [(N, Channel_out), ...] * len(Kernel_size)
        pooled_output = [F.max_pool1d(feature_map, feature_map.size(2)) for feature_map in feature_maps]
        output = torch.cat(pooled_output, 1)
        output = output.view(output.size(0), -1)
        output = self.dropout(output)
        output = F.relu(self.linear1(output))
        output = self.dropout(output)
        output = self.linear2(output)
        return output

class Regression(nn.Module):
    """
    영화리뷰 예측을 위한 Regression 모델입니다.
    """
    def __init__(self, embedding_dim: int, max_length: int):
        """
        initializer
        :param embedding_dim: 데이터 임베딩의 크기입니다
        :param max_length: 인풋 벡터의 최대 길이입니다 (첫 번째 레이어의 노드 수에 연관)
        """
        super(Regression, self).__init__()
        self.embedding_dim = embedding_dim
        self.character_size = 251
        self.output_dim = 1  # Regression
        self.max_length = max_length
        self.kernel_size = [2, 3, 4, 5]
        self.channel_out = 10

        # 임베딩
        self.embeddings = nn.Embedding(self.character_size, self.embedding_dim)

        # 레이어
        # self.fc1 = nn.Linear(self.max_length * self.embedding_dim, 200)
        # self.fc2 = nn.Linear(200, 100)
        # self.fc3 = nn.Linear(100, 1)
        self.conv1 = nn.ModuleList(
            [nn.Conv2d(1, self.channel_out, (k, embedding_dim)) for k in self.kernel_size])
        self.linear1 = nn.Linear(self.channel_out*len(self.kernel_size), 10)
        self.linear2 = nn.Linear(10, 1)
        self.dropout = nn.Dropout()

    def forward(self, data: list):
        """
        :param data: 실제 입력값
        :return:
        """
        # 임베딩의 차원 변환을 위해 배치 사이즈를 구합니다.
        # batch_size = len(data)
        # list로 받은 데이터를 torch Variable로 변환합니다.
        data_in_torch = Variable(torch.from_numpy(np.array(data)).long())
        # 만약 gpu를 사용중이라면, 데이터를 gpu 메모리로 보냅니다.
        if GPU_NUM:
            data_in_torch = data_in_torch.cuda()
        # 뉴럴네트워크를 지나 결과를 출력합니다.
        # embeds = self.embeddings(data_in_torch)
        # hidden = self.fc1(embeds.view(batch_size, -1))
        # hidden = torch.relu(hidden)
        # hidden = self.fc2(hidden)
        # hidden = torch.relu(hidden)
        # output = self.fc3(hidden)
        # return output
        embed = self.embeddings(data_in_torch)   # (N, W, D)
        embed = embed.unsqueeze(1)  # (N,1,W,D), 1: channel_in

        # [(N,Channel_out,W,1), ...] * len(Kernel_size)
        feature_maps = [F.relu(conv(embed)) for conv in self.conv1]
        
        # [(N,Channel_out,W), ...] * len(Kernel_size)
        feature_maps = [feature_map.squeeze(3) for feature_map in feature_maps]

        # [(N, Channel_out), ...] * len(Kernel_size)
        pooled_output = [F.max_pool1d(feature_map, feature_map.size(2)) for feature_map in feature_maps]
        output = torch.cat(pooled_output, 1)
        output = output.view(output.size(0), -1)
        output = self.dropout(output)
        output = F.relu(self.linear1(output))
        output = self.dropout(output)
        output = self.linear2(output)
        return output

class Classification(nn.Module):
    """
    영화리뷰 예측을 위한 Classification 모델입니다.
    """
    def __init__(self, embedding_dim: int, max_length: int):
        """
        initializer
        :param embedding_dim: 데이터 임베딩의 크기입니다
        :param max_length: 인풋 벡터의 최대 길이입니다 (첫 번째 레이어의 노드 수에 연관)
        """
        super(Classification, self).__init__()
        self.embedding_dim = embedding_dim
        self.character_size = 251
        self.output_dim = 11  # Classification(0~10 범위의 lable)
        self.max_length = max_length

        # 임베딩
        self.embeddings = nn.Embedding(self.character_size, self.embedding_dim)

        # 레이어
        self.fc1 = nn.Linear(self.max_length * self.embedding_dim, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, self.output_dim)

    def forward(self, data: list):
        """
        :param data: 실제 입력값
        :return:
        """
        # 임베딩의 차원 변환을 위해 배치 사이즈를 구합니다.
        batch_size = len(data)
        # list로 받은 데이터를 torch Variable로 변환합니다.
        data_in_torch = Variable(torch.from_numpy(np.array(data)).long())
        # 만약 gpu를 사용중이라면, 데이터를 gpu 메모리로 보냅니다.
        if GPU_NUM:
            data_in_torch = data_in_torch.cuda()
        # 뉴럴네트워크를 지나 결과를 출력합니다.
        embeds = self.embeddings(data_in_torch)
        hidden = self.fc1(embeds.view(batch_size, -1))
        hidden = torch.relu(hidden)
        hidden = self.fc2(hidden)
        hidden = torch.relu(hidden)
        output = self.fc3(hidden)
        return output