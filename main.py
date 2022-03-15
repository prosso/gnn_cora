# https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html

# conda activate transformers

from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='./data/Cora', name='Cora')

"""
Cora dataset:
#nodes: 2,708 (papers)
#edges: 10,556 (citation network)
#features: 1,433 (unique words)
#classes: 7 (Neural Networks, Case Based, Reinforcement Learning, Probabilistic Methods, Genetic Algorithms, Rule Learning, and
Theory)

The Cora dataset consists of 2708 scientific publications (NODES) classified into one of seven classes (CLASSES). The citation network consists of 5429 links (10858 if considering both directions AND the number of edges is reduced from 10858 to 10556 after removing duplicate edges to nodes). Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 1433 unique words (FEATURES).





print(dataset[0])
Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708]) --> the graph in the dataset contains 2708 nodes, each one having 1433 features. There are 10556/2 = 5278 undirected edges and the graph is assigned to 2708 target classes. train_mask contains the list of indexes of the examples used for train the model, val_mask contains the ones for evaluating the model, and test_mask the ones for testing the model.

x=[paper ID, sparse vector]
"""


#implementing a two-layer GCN
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    """
    NLLLoss is a loss function commonly used in multi-classes classification tasks. Its meaning is to take log the probability value after softmax and add the probability value of the correct answer to the average

    Planetoid was defined for multi-class classification problem on stratified labelled set

    how NLLLoss works: https://clay-atlas.com/us/blog/2021/05/25/nllloss-en-pytorch-loss-function/

    train_mask is used to get the indexes of train data examples in data
    """
    loss.backward()
    optimizer.step()


model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')
