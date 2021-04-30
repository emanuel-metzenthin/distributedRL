from typing import Text

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ImageDQN(nn.Module):
    BACKBONES = ['resnet18', 'resnet50']

    def __init__(self, backbone: Text = 'resnet50', num_actions: int = 9, num_history: int = 10):
        super().__init__()

        if backbone not in ImageDQN.BACKBONES:
            raise Exception(f'{backbone} not supported.')

        backbone_model = getattr(models, backbone)(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(backbone_model.children())[:-1])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.dqn = nn.Sequential(
            nn.Linear(backbone_model.fc.in_features + num_history * num_actions, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_actions)
        )

        self.dqn.apply(self.init_weights)
        self.num_actions = num_actions
        self.num_history = num_history

    def forward(self, X):
        images, histories = X

        if images.shape[1] != 3:
            images = images.permute([0, 3, 1, 2])
        histories = torch.reshape(histories, (-1, self.num_actions * self.num_history))

        features = self.feature_extractor(images).reshape(-1, self.dqn[0].in_features - self.num_actions * self.num_history)
        states = torch.cat((features, histories), dim=1)

        return self.dqn(states)

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


class ConvDuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvDuelingDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fc_input_dim = self.feature_size()

        self.value_stream = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512), nn.ReLU(), nn.Linear(512, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.output_dim),
        )

    def forward(self, state):
        features = self.conv(state)
        features = features.view(features.size(0), -1)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())

        return qvals

    def feature_size(self):
        return self.conv(torch.zeros(1, *self.input_dim)).view(1, -1).size(1)


class ConvDQN(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=256, init_w=3e-3):
        super(ConvDQN, self).__init__()
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.conv = nn.Sequential(
            nn.Conv2d(num_inputs[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fc_input_dim = self.feature_size()
        self.linear1 = nn.Linear(self.fc_input_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        features = self.conv(state)
        features = features.view(features.size(0), -1)

        x = F.relu(self.linear1(features))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

    def feature_size(self):
        return self.conv(torch.zeros(1, *self.num_inputs)).view(1, -1).size(1)


class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.feauture_layer = nn.Sequential(
            nn.Linear(self.input_dim[0], 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, self.output_dim)
        )

    def forward(self, state):
        features = self.feauture_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())

        return qvals
