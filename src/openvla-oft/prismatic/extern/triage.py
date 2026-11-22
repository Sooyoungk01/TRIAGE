import math

import torch
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama import modeling_llama
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

EPS = 1e-8
_ORIGINAL_FORWARD = None


def _rope(x, cos, sin):
    return (x * cos.unsqueeze(1)) + (rotate_half(x) * sin.unsqueeze(1))


def _qkv_weight(a):
    if not hasattr(a, "_triage_qkv"):
        a._triage_qkv = torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight], dim=0)
    return a._triage_qkv


def _layer(layer, h_rows, rows, keys, cos, sin, Kb, Vb, stat_rows=None):
    a = layer.self_attn
    bsz, n, _ = h_rows.shape
    hn = layer.input_layernorm(h_rows)
    q, k, v = nn.functional.linear(hn, _qkv_weight(a)).split(
        [a.num_heads * a.head_dim, a.num_key_value_heads * a.head_dim, a.num_key_value_heads * a.head_dim], dim=-1)
    q = q.view(bsz, n, a.num_heads, a.head_dim).transpose(1, 2)
    k = k.view(bsz, n, a.num_key_value_heads, a.head_dim).transpose(1, 2)
    v = v.view(bsz, n, a.num_key_value_heads, a.head_dim).transpose(1, 2)
    q, k = _rope(q, cos, sin), _rope(k, cos, sin)
    Kb[:, :, rows] = k
    Vb[:, :, rows] = v
    Kk = repeat_kv(Kb[:, :, keys], a.num_key_value_groups)
    Vk = repeat_kv(Vb[:, :, keys], a.num_key_value_groups)
    O = nn.functional.scaled_dot_product_attention(q, Kk, Vk, is_causal=False)
    att = None
    if stat_rows is not None:
        att = torch.softmax(torch.matmul(q[:, :, stat_rows], Kk.transpose(2, 3)) / math.sqrt(a.head_dim), dim=-1,
                            dtype=torch.float32)[0].mean(dim=(0, 1))
    O = O.transpose(1, 2).reshape(bsz, n, a.num_heads * a.head_dim)
    r1 = h_rows + a.o_proj(O)
    return r1 + layer.mlp(layer.post_attention_layernorm(r1)), att


def _propagate(obs, kept, hidden, knn):
    drop = torch.ones(obs.numel(), dtype=torch.bool, device=obs.device)
    drop[kept] = False
    if not bool(drop.any()):
        return obs.clone()
    hv = nn.functional.normalize(hidden.float(), dim=1)
    nb = torch.topk(hv[drop] @ hv[kept].t(), min(int(knn), kept.numel()), dim=1).indices
    est = obs.clone()
    est[drop] = obs[kept][nb].mean(dim=1)
    return est


def prefill(model, st, x, position_ids):
    layers = model.layers
    L, K, B = len(layers), st["prune_layer"], st["b_vis"]
    lo, hi = st["vis_start"], st["vis_end"]
    _, S, _ = x.shape
    act_hi = S - 1
    act_lo = act_hi - st["n_action"]
    dev = x.device
    pos = torch.arange(S, device=dev)
    if position_ids is None:
        position_ids = pos.unsqueeze(0)
    cos, sin = layers[0].self_attn.rotary_emb(x, position_ids)
    is_vis = (pos >= lo) & (pos < hi)
    always = pos[~is_vis]
    act = torch.arange(act_lo, act_hi, device=dev)
    prev = st["prev"]
    no_cache = prev is None or prev["K"][0].shape[2] != S
    a0 = layers[0].self_attn
    if no_cache:
        Kb = [torch.empty(1, a0.num_key_value_heads, S, a0.head_dim, device=dev, dtype=x.dtype) for _ in range(L)]
        Vb = [torch.empty_like(Kb[0]) for _ in range(L)]
    else:
        Kb, Vb = prev["K"], prev["V"]
    timed = dev.type == "cuda"
    if timed:
        t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t0.record()

    ref = None if no_cache else prev["ref"]
    first = ref is None
    h = x
    for j in range(K):
        h, att = _layer(layers[j], h, pos, pos, cos, sin, Kb[j], Vb[j],
                        stat_rows=pos[S - 1:] if j == 0 else (act if j == K - 1 else None))
        if j == 0:
            sal0 = att[lo:hi]
        if j == K - 1:
            a_full = att
    if not first:
        emb = x[0, lo:hi].float()
        sim = nn.functional.normalize(emb, dim=1) @ nn.functional.normalize(ref["emb"], dim=1).t()
        fc_now = ref["fc"][:, sim.argmax(dim=1)]
    H = h
    vis = pos[is_vis]

    if first:
        R_vis = vis[torch.topk(a_full[lo:hi], min(st["first_bvis"], vis.numel())).indices]
        prune = is_vis.clone()
        prune[R_vis] = False
    else:
        imp = a_full[lo:hi] + st["lam"] * fc_now.mean(dim=0)
        n_imp = min(st["n_important"], vis.numel())
        prev_kept = prev["kept"]
        order = torch.argsort(imp, descending=True)
        rank = torch.empty_like(order)
        rank[order] = torch.arange(order.numel(), device=dev)
        stay = torch.nonzero(prev_kept[lo:hi] & (rank < int(round(st["hysteresis"] * n_imp)))).squeeze(-1)
        if stay.numel() > n_imp:
            stay = stay[torch.argsort(rank[stay])[:n_imp]]
        rest = imp.clone()
        rest[stay] = float("-inf")
        imp_idx = vis[torch.cat([stay, torch.topk(rest, n_imp - stay.numel()).indices])]

        dh = torch.linalg.norm((H - prev["H"])[0].float(), dim=-1) / (torch.linalg.norm(H[0].float(), dim=-1) + EPS)
        impm = torch.zeros(S, dtype=torch.bool, device=dev)
        impm[imp_idx] = True
        old = prev["age"] + 1 > st["max_age"]
        Vh = repeat_kv(Vb[K - 1], a0.num_key_value_groups)[0].float()
        Ah = repeat_kv(prev["anchor"], a0.num_key_value_groups)[0].float()
        vdrift = (torch.linalg.norm(Vh - Ah, dim=-1) / (torch.linalg.norm(Vh, dim=-1) + EPS)).mean(dim=0)
        bad = old | ~prev["valid"] | (vdrift >= st["delta"])

        impv = torch.full((S,), float("-inf"), device=dev)
        impv[lo:hi] = imp
        nrm = bad | ~prev_kept
        nr, rr = imp_idx[nrm[imp_idx]], imp_idx[~nrm[imp_idx]]
        core = torch.cat([nr[torch.argsort(impv[nr], descending=True)],
                          rr[torch.argsort((dh * a_full)[rr], descending=True)]])
        mc = torch.nonzero(is_vis & ~impm & prev["valid"]).squeeze(-1)
        if mc.numel():
            mc = mc[torch.argsort(prev["age"][mc].float() * 1e4 + impv[mc].clamp(min=-1e3), descending=True)]
        mc = mc[:int(round(st["maint_ratio"] * B))]
        R_vis = torch.cat([mc, core[:min(B - int(mc.numel()), n_imp)]])
        inR = torch.zeros(S, dtype=torch.bool, device=dev)
        inR[R_vis] = True
        prune = is_vis & ~inR & bad
    rows = torch.cat([R_vis, always]).sort().values
    keys = pos[~prune]
    anchor = Vb[K - 1].clone() if no_cache else prev["anchor"]
    anchor[:, :, rows] = Vb[K - 1][:, :, rows]
    h = h[:, rows]

    fc_rows = torch.nonzero(torch.isin(rows, act)).squeeze(-1) if first else None
    cs = (cos[:, rows], sin[:, rows])
    for j in range(K, L):
        h, att = _layer(layers[j], h, rows, keys, *cs, Kb[j], Vb[j], stat_rows=fc_rows if j == L - 1 else None)
        if att is not None:
            a_deep = torch.zeros(S, device=dev)
            a_deep[keys] = att
    if timed:
        t1.record()
        torch.cuda.synchronize()
    h = model.norm(h)
    out = torch.zeros_like(x)
    out[:, rows] = h

    valid = torch.zeros(S, dtype=torch.bool, device=dev) if no_cache else prev["valid"].clone()
    valid[rows] = True
    age = torch.zeros(S, dtype=torch.long, device=dev) if no_cache else prev["age"] + 1
    age[rows] = 0
    if first:
        kept = torch.zeros(S, dtype=torch.bool, device=dev)
        kept[R_vis] = True
    else:
        kept = impm
    if first:
        fc31 = _propagate(a_deep[lo:hi].float(), R_vis - lo, H[0, lo:hi], 4)
        ref = {"emb": x[0, lo:hi].float(), "fc": torch.stack([sal0, fc31.to(sal0.dtype)])}
    st["out"] = {"K": Kb, "V": Vb, "anchor": anchor, "valid": valid, "H": H, "kept": kept, "age": age, "ref": ref}
    st["info"] = {"rows": int(rows.numel()), "seq_len": S,
                  "prefill_ms": t0.elapsed_time(t1) if timed else None}
    return out


def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None,
            use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, cache_position=None):
    st = getattr(self.config, "triage", None)
    if st is None:
        return _ORIGINAL_FORWARD(self, input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                                 past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
                                 output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                                 return_dict=return_dict, cache_position=cache_position)
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    if attention_mask is not None and not bool(attention_mask.all()):
        raise ValueError("TRIAGE assumes no padding.")
    if inputs_embeds.shape[0] != 1:
        raise ValueError("TRIAGE supports batch size 1 only.")
    h = prefill(self, st, inputs_embeds, position_ids)
    hidden = (h,) if output_hidden_states else None
    if not return_dict:
        return tuple(v for v in (h, None, hidden) if v is not None)
    return BaseModelOutputWithPast(last_hidden_state=h, past_key_values=DynamicCache(), hidden_states=hidden, attentions=None)


def install():
    global _ORIGINAL_FORWARD
    if _ORIGINAL_FORWARD is None:
        _ORIGINAL_FORWARD = modeling_llama.LlamaModel.forward
        modeling_llama.LlamaModel.forward = forward
