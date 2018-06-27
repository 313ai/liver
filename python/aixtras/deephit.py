import torch
import torch.nn.functional as F

def convex(x, y, sigma=1): 
    '''
    any convex function probably would work
    sigma here looks like a hyperparam
    '''
    return torch.exp((y-x)/sigma)

def deephit_loss(preds, evt_times, evt_types, t_max=None, num_evt_types=1, eps=1e-5):
    #import pdb; pdb.set_trace()
    if t_max is None:
        t_max = preds.shape[1]

    save_tmax = t_max
    save_pred_shape = preds.shape
    save_times = evt_times.detach().cpu().numpy()
    save_types = evt_types.detach().cpu().numpy()
    preds_cdf = torch.cat([
        torch.cumsum(preds[(i*t_max):((i+1)*t_max)], dim=1)
        for i in range(num_evt_types)
    ])
    save_preds_cdf = preds_cdf.detach().cpu().numpy()

    uncensored_idx = evt_types > 0
    save_unc_idx = uncensored_idx.detach().cpu().numpy()
    censored_idx = evt_types == 0
    save_cen_idx = censored_idx.detach().cpu().numpy()
    num_uncensored = uncensored_idx.sum()
    num_censored = censored_idx.sum()
    save_num_censored = num_censored.item()
    save_num_uncensored = num_uncensored.item()

    times_idx = (evt_types-1) * t_max + evt_times
    save_times_idx = times_idx.detach().cpu().numpy()

    if num_uncensored > 0:
        l1_part1_probs = preds[uncensored_idx, times_idx[uncensored_idx]]
        l1_part_1 = torch.sum(torch.log(l1_part1_probs))
    else:
        l1_part_1 = 0.0

    if num_censored > 0:
        l1_part_2 = torch.log(1 - preds_cdf[censored_idx, evt_times[censored_idx]] + eps).sum()
    else:
        l1_part_2 = 0.0

    l1_loss = -(l1_part_1 + l1_part_2)

    # grab low survival, first patient index, second patient index
    if num_uncensored > 2:
        assert(len(uncensored_idx) == len(preds_cdf))
        preds_cdf_uncensored = preds_cdf[uncensored_idx]
        assert(len(uncensored_idx) == len(evt_times))
        evt_times_uncensored = evt_times[uncensored_idx]

        pair_idxs = [] # (si, i, j)
        for idx_i, si in enumerate(evt_times_uncensored):
            for idx_j, sj in enumerate(evt_times_uncensored):
                if idx_i != idx_j:
                    if si < sj:
                        pair_idxs.append([si, idx_i, idx_j])

        # index to si cdf values for both paient
        pairwise_loss = 0
        for si, idx_i, idx_j in pair_idxs:
            Fki = preds_cdf_uncensored[idx_i, si]
            Fkj = preds_cdf_uncensored[idx_j, si]
            pairwise_loss += convex(Fki, Fkj)
    else:
        pairwise_loss = 0.0

    loss = l1_loss  + pairwise_loss
    #if torch.isnan(loss):
    #    import pdb; pdb.set_trace()

    return l1_loss, pairwise_loss
