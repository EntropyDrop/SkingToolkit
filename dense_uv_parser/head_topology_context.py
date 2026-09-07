"""Learn head continuity on actual cube neighbours without enforcing smoothness."""
import torch
from torch import nn
import torch.nn.functional as F


class HeadTopologyContext(nn.Module):
    def __init__(self,width,edges,other,geometry):
        super().__init__();n=len(other);face=geometry[:,5:11].argmax(1)
        left,right=edges.cpu();same=face[left]==face[right]
        matrices=[]
        for selected in (same,~same):
            a=torch.zeros(n,n);i,j=left[selected],right[selected]
            a[i,j]=1;a[other[i],other[j]]=1
            matrices.append(a/a.sum(1,keepdim=True).clamp_min(1))
        self.register_buffer('within_face',matrices[0],persistent=False)
        self.register_buffer('across_seam',matrices[1],persistent=False)
        self.message=nn.Sequential(nn.LayerNorm(width*3),nn.Linear(width*3,width),nn.GELU(),nn.Linear(width,width))
        # A v103 checkpoint initially has identical predictions. Continuity is
        # learned from data, not a rule filling holes or removing isolated cells.
        nn.init.zeros_(self.message[-1].weight);nn.init.zeros_(self.message[-1].bias)

    def forward(self,features):
        within=torch.matmul(self.within_face,features)
        across=torch.matmul(self.across_seam,features)
        return features+self.message(torch.cat([features,within,across],2))


def supervised_boundary_loss(probability,alpha,edges):
    """Teach both true contour changes and continuity, including cube seams."""
    left,right=edges
    different=(alpha[:,left]>.5)!=(alpha[:,right]>.5)
    p,q=probability[:,left],probability[:,right]
    mismatch=(p*(1-q)+(1-p)*q).clamp(1e-6,1-1e-6)
    loss=F.binary_cross_entropy(mismatch,different.float(),reduction='none')
    terms=[loss[m].mean() for m in (different,~different) if m.any()]
    return torch.stack(terms).mean() if terms else probability.sum()*0


def augment_connected_uv(base,model):
    """Train against missing patches and spurious patches on the cube surface."""
    b=len(base);adj=(model.topology_adapter.within_face+model.topology_adapter.across_seam)>0
    ids=model.ids[model.outer];outer_nodes=torch.where(model.outer)[0]
    region=torch.zeros(b,len(model.ids),device=base.device)
    seeds=outer_nodes[torch.randint(len(ids),(b,),device=base.device)]
    region[torch.arange(b,device=base.device),seeds]=1
    # Radius one or two crosses actual seams, never adjacent atlas islands.
    for _ in range(int(torch.randint(1,3,()).item())):
        region=((region@adj.float())>0).float().maximum(region)
    selected=region[:,model.outer].bool()
    uv=base.flatten(2);occupied=uv[:,3,ids]>.5
    add=torch.rand(b,1,device=base.device)<.5
    changed=selected&(occupied!=add)
    uv[:,3,ids]=torch.where(selected,add.float(),occupied.float())
    uv[:,:3,ids]=torch.where((changed&add)[:,None],uv[:,:3,ids-32],uv[:,:3,ids])
    uv[:,:3,ids]*=uv[:,3:4,ids]
    return base
