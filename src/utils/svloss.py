import torch
import torch.nn.functional as F

def generate_pseudo_enrollment_set(victim_model, dataloader, M=50):
    # 从数据集中随机提取M个干净语音特征作为伪注册集

    victim_model.eval()
    pseudo_embs = []

    with torch.no_grad():
        for batch in dataloader:
            batch_x = batch[0].cuda()
            embs = victim_model.compute_speaker_embedding(batch_x)
            pseudo_embs.append(embs)
            
            if sum(e.shape[0] for e in pseudo_embs) >= M:
                break
                
    pseudo_embs = torch.cat(pseudo_embs, dim=0)[:M]
    return pseudo_embs 

def compute_sv_loss(adv_embs, pseudo_embs, tau=0.8, beta=10.0):
    # 计算sv的loss

    adv_embs_norm = F.normalize(adv_embs, p=2, dim=-1)
    pseudo_embs_norm = F.normalize(pseudo_embs, p=2, dim=-1)

    s_matrix = torch.matmul(adv_embs_norm, pseudo_embs_norm.T)
    crossing_prob = torch.sigmoid(beta * (s_matrix - tau))

    loss_sv_risk = -torch.mean(crossing_prob)

    return loss_sv_risk