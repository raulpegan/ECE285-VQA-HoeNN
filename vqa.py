import nntools as nt

import torch
import torch.nn as nn
import torchvision as tv

class NNClassifier(nt.NeuralNetwork):
    def __init__(self):
        super(NNClassifier, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
    def criterion(self, y, d):
        return self.cross_entropy_loss(y ,d)

class VQA(NNClassifier):
    def __init__(self, activation, dropout, combination, ans_vocab_size, qst_vocab_size, fine_tuning=False):
        super(VQA, self).__init__()
        
        # CNN Image Channel:
        # This channel provides an embedding for the image. We experiment with two embeddings 
        # 1)I: The  activations  from  the  last  hidden  layer  of  VG-GNet [48] are used as 4096-dim image embedding.
        # 2)norm I:These  are 2 normalized  activations  from  the last hidden layer of VGGNet [48]
        
        ##self.cnn = tv.models.vgg19_bn(pretrained=True)
        self.cnn = tv.models.vgg19_bn(pretrained=True)
        num_ftrs = self.cnn.classifier[-1].in_features
        for param in self.cnn.parameters():
            param.requires_grad = fine_tuning
        self.cnn.classifier[-1] = nn.Linear(num_ftrs, 1024)  # Embed size of 1024
        
        # Question encoder
        self.question_vec = nn.Embedding(qst_vocab_size, 300)
        #self.rnn = nn.LSTM(300, 512, 2)
        self.rnn = nn.GRU(300, 512, 2)
        self.fc_rnn = nn.Linear(2*2*512, 1024)
        
        # Rest
        self.combination = combination
        self.activation = activation
        self.dropout = dropout
        
        self.fc1 = nn.Linear(1024, ans_vocab_size)
        self.fc2 = nn.Linear(ans_vocab_size, ans_vocab_size)
        
        
        
    def forward(self, image, question):
        
        # CNN
        image_feat = self.cnn(image)
        # L2 Norm
        l2_norm = image_feat.norm(p=2, dim=1, keepdim=True).detach()
        image_feat = image_feat.div(l2_norm) 
        
        # Question Encoder
        quest_vec = self.question_vec(question)
        quest_vec = self.activation(quest_vec)
        quest_vec = quest_vec.transpose(0,1)
        #_, (hidden, cell) = self.rnn(quest_vec)
        _, hidden = self.rnn(quest_vec)
        quest_feat = torch.cat((hidden,hidden),2)
        quest_feat = quest_feat.transpose(0,1)
        quest_feat = quest_feat.reshape(quest_feat.size()[0], -1)
        quest_feat = self.activation(quest_feat)
        quest_feat = self.fc_rnn(quest_feat)
        
        h = self.combination(image_feat, quest_feat) 
        x = self.activation(h)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x