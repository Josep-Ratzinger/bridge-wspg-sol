"""Corrección residual aprendida sobre WSPG, con el respaldo congelado.

Ambas ramas (sin/con SOL) tienen la misma arquitectura y los mismos datos.
Ninguna clase se asigna mediante reglas geométricas después de predecir.
"""
import torch
from torch import nn
from .modelo import WSPG


class Ajuste(nn.Module):
    def __init__(self,limite=8.):
        super().__init__()
        self.limite=float(limite)
        self.local=nn.Sequential(nn.Linear(30,48),nn.ReLU(),nn.Linear(48,32),nn.ReLU())
        self.final=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,10))
        nn.init.zeros_(self.final[-1].weight);nn.init.zeros_(self.final[-1].bias)

    def forward(self,base,nodes,edges,raw):
        h=self.local(torch.cat([nodes,base.log_softmax(1).clamp(min=-12)],1))
        source,dest=edges
        weight=torch.exp(-((raw[source,0]-raw[dest,0])/.18).square()
                         -((raw[source,6]-raw[dest,6])/.25).square()).clamp_min(.01)
        agg=torch.zeros_like(h);den=h.new_zeros((len(h),1))
        agg.index_add_(0,dest,h[source]*weight[:,None]);den.index_add_(0,dest,weight[:,None])
        return self.limite*torch.tanh(self.final(torch.cat([h,agg/den.clamp_min(1e-5)],1)))


class WSPGRelacional(nn.Module):
    def __init__(self,strength=1.,base_scale=1.):
        super().__init__();self.base=WSPG(clases=10);self.ajuste=Ajuste();self.strength=float(strength);self.base_scale=float(base_scale)
        self.base.requires_grad_(False);self.register_buffer('nm',torch.zeros(20));self.register_buffer('ns',torch.ones(20))

    def train(self,mode=True):
        super().train(mode);self.base.eval();return self

    def forward(self,p,centers,enabled,index,edge,nodes):
        with torch.no_grad():base=self.base_scale*self.base(p,centers,enabled,index,edge)
        raw=nodes*self.ns+self.nm
        return base+self.strength*self.ajuste(base,nodes,index,raw)

