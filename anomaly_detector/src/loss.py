import torch
class anomaly_loss(torch.nn.Module):
    def __init__(self , radius):
        super().__init__()
        self.radius = radius
    
    def anomaly_score(self, emb_out):
        emb_out = emb_out.float()
        emb_norm = torch.norm(emb_out , dim=1)
        dist = (emb_norm - self.radius).relu()
        return dist
    def forward(self, emb_out , anomaly_label):
        dist = self.anomaly_score(emb_out)
        dist_anomaly = torch.abs(dist - anomaly_label)
        return dist_anomaly.mean()
    
