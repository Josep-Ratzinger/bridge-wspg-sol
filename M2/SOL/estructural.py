"""SOL experimental con relaciones explícitas y continuidad condicionada.

Penalización de entrenamiento, no recoloreado ni asignación por plantilla.
Solo se castiga evidencia observada contradictoria; una conexión ausente no
prueba que el componente sea incorrecto. Gravedad Z y tablero vial continuo.
"""
import torch
from ..estructura import costes


def preparar(features,edges,device):
    f=torch.as_tensor(features,dtype=torch.float32,device=device);e=torch.as_tensor(edges,dtype=torch.long,device=device)
    c=torch.as_tensor(costes(features),device=device);i,j=e
    above=(f[:,2]>.01).float();edge=((f[:,6]>.65)&(f[:,10]>.1)).float()
    # Smooth only compatible raised lateral fragments, not deck-to-rail edges.
    w=above[i]*above[j]*edge[i]*edge[j]*torch.exp(-((f[i,0]-f[j,0])/.12).square())
    vertical=f[:,12]*torch.clamp(f[:,3]*4,max=1)
    colw=vertical[i]*vertical[j]*(f[i,0]<-.05)*(f[j,0]<-.05)
    return f,e,c,w,colw


def perdida(logits,prepared):
    f,e,c,railw,colw=prepared;p=logits.softmax(1);i,j=e
    unary=(p*c).sum(1).mean()
    rail=((p[i,3]-p[j,3]).square()*railw).sum()/railw.sum().clamp_min(1)
    col=((p[i,0]-p[j,0]).square()*colw).sum()/colw.sum().clamp_min(1)
    inversion=logits.new_zeros(())
    height=f[i,0]-f[j,0]
    wrong=torch.relu(height-.035).clamp(max=1)*f[i,18]*f[j,18]
    # lower -> upper: pillar/cap/deck/railing, including inclined/curved spans.
    for lower,upper in [(0,1),(0,2),(1,2),(2,3),(2,5)]:
        inversion+=(p[i,lower]*p[j,upper]*wrong).mean()
    return unary+.2*rail+.1*col+.5*inversion

