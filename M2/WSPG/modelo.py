"""Implementación propia PointNet + ECC vectorial/GRU en PyG.

No se afirma equivalencia numérica con WSPG de Yang (sin código público).
Versión final: 32 canales, 10 pasos, MLP 8/16/32/32, 13 atributos de arista.
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import MessagePassing


class Contexto(MessagePassing):
    def __init__(self, hidden):
        super().__init__(aggr='mean')
        self.edge = nn.Sequential(nn.Linear(13,32),nn.ReLU(),nn.Linear(32,hidden),nn.Sigmoid())
        self.gru = nn.GRUCell(hidden,hidden)
        self.gate = nn.Linear(hidden,hidden)

    def forward(self,h,index,edge):
        weights = self.edge(edge)
        for _ in range(10):
            message = self.propagate(index,x=h,weights=weights)
            h = self.gru(message * torch.sigmoid(self.gate(h)),h)
        return h

    def message(self,x_j,weights):
        return x_j * weights


class WSPG(nn.Module):
    def __init__(self, clases=10):
        super().__init__()
        self.pointnet = nn.Sequential(nn.Linear(8,16),nn.ReLU(),nn.Linear(16,32),
            nn.ReLU(),nn.Linear(32,32),nn.ReLU())
        self.initial = nn.Sequential(nn.Linear(35,32),nn.ReLU())
        self.context = Contexto(32)
        # Conexión local conserva evidencia que la propagación puede diluir.
        self.classifier = nn.Linear(64,clases)

    def encode(self,p):
        return self.pointnet(p).amax(1)

    def forward(self,p,centers,enabled,index,edge):
        chunks = []
        for part in p.split(512):
            chunks.append(checkpoint(self.encode,part,use_reentrant=False) if self.training else self.encode(part))
        embedding = torch.cat(chunks)*enabled[:,None]
        h = self.initial(torch.cat([embedding,centers],1))
        return self.classifier(torch.cat([h,self.context(h,index,edge)],1))

