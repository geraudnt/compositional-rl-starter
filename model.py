import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch_ac.model import ACModel, RecurrentACModel


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ACModel(nn.Module, RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory

        # Define image embedding
        k1,s1 = (2,1); p=2; k2,s2 = (2,1); k3,s3 = (2,1); c_out = 128
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=k1, stride=s1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=p, stride=p),
            nn.Conv2d(16, 32, kernel_size=k2, stride=s2),
            nn.ReLU(),
            nn.Conv2d(32, c_out, kernel_size=k3, stride=s3),
            nn.ReLU()
        )
        l = obs_space["image"][0]
        f = lambda l,k,s: (l-k)//s + 1
        self.image_embedding_size = f(f(f(l,k1,s1)//p,k2,s2),k3,s3)**2*c_out

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size
        
        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs_goal, memory):
        obs = obs_goal.image[:,:,:,:3]
        x = obs.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)
        
        embedding = x

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(embedding, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
                
        goal = obs_goal.image[:,:,:,3:]
        y = goal.transpose(1, 3).transpose(2, 3)
        y = self.image_conv(y)
        y = y.reshape(y.shape[0], -1)

        embedding *= y

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory
