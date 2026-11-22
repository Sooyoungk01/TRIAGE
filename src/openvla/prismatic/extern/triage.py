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


def _qkv(layer, hn, cos, sin):
    a = layer.self_attn
    bsz, n, _ = hn.shape
    q, k, v = a.q_proj(hn), a.k_proj(hn), a.v_proj(hn)
    q = q.view(bsz, n, a.num_heads, a.head_dim).transpose(1, 2)
    k = k.view(bsz, n, a.num_key_value_heads, a.head_dim).transpose(1, 2)
    v = v.view(bsz, n, a.num_key_value_heads, a.head_dim).transpose(1, 2)
    return _rope(q, cos, sin), _rope(k, cos, sin), v


def _finish(layer, h_rows, O):
    a = layer.self_attn
    O = O.transpose(1, 2).reshape(h_rows.shape[0], h_rows.shape[1], a.num_heads * a.head_dim)
    r1 = h_rows + a.o_proj(O)
    return r1 + layer.mlp(layer.post_attention_layernorm(r1))


def _layer(layer, h_rows, rows, keys, cos, sin, Kb, Vb, mask, last_row=False):
    a = layer.self_attn
    q, k, v = _qkv(layer, layer.input_layernorm(h_rows), cos[:, rows], sin[:, rows])
    Kb[:, :, rows] = k
    Vb[:, :, rows] = v
    Kk = repeat_kv(Kb[:, :, keys], a.num_key_value_groups)
    Vk = repeat_kv(Vb[:, :, keys], a.num_key_value_groups)
    if rows.numel() == keys.numel() == Kb.shape[2]:
        O = nn.functional.scaled_dot_product_attention(q.contiguous(), Kk.contiguous(), Vk.contiguous(), is_causal=True)
    else:
        O = nn.functional.scaled_dot_product_attention(q, Kk, Vk, attn_mask=mask)
    att = None
    if last_row:
        s = torch.matmul(q[:, :, -1:], Kk.transpose(2, 3)) / math.sqrt(a.head_dim)
        s = s.masked_fill(~mask[-1:], float("-inf"))
        att = torch.softmax(s, dim=-1, dtype=torch.float32)[0].mean(dim=(0, 1))
    return _finish(layer, h_rows, O), att


def _scatter_row(att, keys, S):
    v = torch.zeros(S, device=att.device)
    v[keys] = att.mean(dim=(0, 1))
    return v


def prefill(model, st, x, position_ids):
    layers = model.layers
    L, K, B = len(layers), st["prune_layer"], st["b_vis"]
    ts = st["text_start"]
    _, S, _ = x.shape
    dev = x.device
    pos = torch.arange(S, device=dev)
    if position_ids is None:
        position_ids = pos.unsqueeze(0)
    cos, sin = layers[0].self_attn.rotary_emb(x, position_ids)
    text = pos[ts:]
    is_vis = (pos >= 1) & (pos < ts)
    causal = torch.ones(S, S, dtype=torch.bool, device=dev).tril()
    prev, ref = st["prev"], st["ref"]
    no_cache = prev is None or prev["K"][0].shape[2] != S
    dense = no_cache or ref is None
    first = dense and ref is None and no_cache
    if first:
        dense = False
    a0 = layers[0].self_attn
    if dense or first:
        Kb = [torch.empty(1, a0.num_key_value_heads, S, a0.head_dim, device=dev, dtype=x.dtype) for _ in range(L)]
        Vb = [torch.empty_like(Kb[0]) for _ in range(L)]
    else:
        Kb, Vb = prev["K"], prev["V"]
    frame = 0 if prev is None else prev["frame"] + 1
    last_comp = torch.full((S,), frame, dtype=torch.long, device=dev) if (dense or first) else prev["last_comp"]
    timed = dev.type == "cuda"
    if timed:
        t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t0.record()

    h = x
    for j in range(K):
        h, att = _layer(layers[j], h, pos, pos, cos, sin, Kb[j], Vb[j], causal, last_row=j in (0, K - 1))
        if j == 0:
            sal0 = att[1:ts]
        if j == K - 1:
            a_last = att
    if not dense and not first:
        emb = x[0, 1:ts].float()
        sim = nn.functional.normalize(emb, dim=1) @ nn.functional.normalize(ref["emb"], dim=1).t()
        fc_now = ref["fc"][:, sim.argmax(dim=1)]
    hK = h[:, pos].clone()
    vis = pos[is_vis]

    if dense:
        rows = keys = pos
        anchor = Vb[K - 1].clone()
    elif first:
        R_vis = vis[torch.topk(a_last[1:ts], min(st["first_bvis"], vis.numel())).indices]
        prune = is_vis.clone()
        prune[R_vis] = False
    else:
        imp = a_last[1:ts] + st["lam"] * fc_now.mean(dim=0)
        n_imp = min(st["n_important"], vis.numel())
        prev_kept = prev["kept"]
        order = torch.argsort(imp, descending=True)
        rank = torch.empty_like(order)
        rank[order] = torch.arange(order.numel(), device=dev)
        stay = torch.nonzero(prev_kept[1:ts] & (rank < int(round(st["hysteresis"] * n_imp)))).squeeze(-1)
        if stay.numel() > n_imp:
            stay = stay[torch.argsort(rank[stay])[:n_imp]]
        rest = imp.clone()
        rest[stay] = float("-inf")
        imp_idx = vis[torch.cat([stay, torch.topk(rest, n_imp - stay.numel()).indices])]

        dh = torch.linalg.norm((hK[0] - prev["hK"][0]).float(), dim=-1) / (torch.linalg.norm(hK[0].float(), dim=-1) + EPS)
        old = (frame - last_comp) > st["max_age"]
        Vh = repeat_kv(Vb[K - 1], a0.num_key_value_groups)[0].float()
        Ah = repeat_kv(prev["anchor"], a0.num_key_value_groups)[0].float()
        vdrift = (torch.linalg.norm(Vh - Ah, dim=-1) / (torch.linalg.norm(Vh, dim=-1) + EPS)).mean(dim=0)
        bad = old | (vdrift >= st["delta"]) | ~prev["valid"]
        impv = torch.full((S,), float("-inf"), device=dev)
        impv[1:ts] = imp
        chv = torch.zeros(S, device=dev)
        chv[imp_idx] = (dh * a_last)[imp_idx]

        nrm = bad | ~prev_kept
        nr, rr = imp_idx[nrm[imp_idx]], imp_idx[~nrm[imp_idx]]
        core = torch.cat([nr[torch.argsort(impv[nr], descending=True)], rr[torch.argsort(chv[rr], descending=True)]])
        impm = torch.zeros(S, dtype=torch.bool, device=dev)
        impm[imp_idx] = True
        mc = torch.nonzero(is_vis & ~impm & prev["valid"]).squeeze(-1)
        if mc.numel():
            mc = mc[torch.argsort((frame - last_comp)[mc].float() * 1e4 + impv[mc].clamp(min=-1e3), descending=True)]
        mc = mc[:int(round(st["maint_ratio"] * B))]
        R_vis = torch.cat([mc, core[:min(B - int(mc.numel()), n_imp)]])

        inR = torch.zeros(S, dtype=torch.bool, device=dev)
        inR[R_vis] = True
        prune = is_vis & ~inR & bad

    if not dense:
        rows = torch.cat([pos[:1]] * int(first) + [R_vis.sort().values, text])
        keys = pos[~prune]
        last_comp[rows] = frame
        anchor = Vb[K - 1].clone() if first else prev["anchor"]
        anchor[:, :, rows] = Vb[K - 1][:, :, rows]
        h = h[:, rows]
        mask = causal[rows][:, keys]
        for j in range(K, L):
            h, _ = _layer(layers[j], h, rows, keys, cos, sin, Kb[j], Vb[j], mask)
    else:
        for j in range(K, L):
            h, _ = _layer(layers[j], h, pos, pos, cos, sin, Kb[j], Vb[j], causal)

    if timed:
        t1.record()
        torch.cuda.synchronize()
    h = model.norm(h)
    out = torch.zeros_like(x)
    out[:, rows] = h
    cache = DynamicCache()
    for j in range(L):
        cache.update(Kb[j], Vb[j], j, {"cache_position": pos})

    if dense:
        last_comp[:] = frame
        valid, kept = torch.ones(S, dtype=torch.bool, device=dev), is_vis.clone()
    elif first:
        valid, kept = torch.zeros(S, dtype=torch.bool, device=dev), torch.zeros(S, dtype=torch.bool, device=dev)
        kept[R_vis] = True
    else:
        valid, kept = prev["valid"].clone(), impm
    valid[rows] = True
    st["decode"] = {"K": K, "S": S, "active": torch.ones(S, dtype=torch.bool, device=dev) if dense else ~prune,
                    "fc_layer": L - 1 if (dense or first) else None}
    st["ref_new"] = {"emb": x[0, 1:ts].float(), "fc0": sal0} if (dense or first) else None
    if first:
        st["ref_new"].update(kept=R_vis - 1, h2=hK[0, 1:ts].float())
    st["ref_dec"] = []
    st["out"] = {"K": Kb, "V": Vb, "anchor": anchor, "frame": frame, "last_comp": last_comp, "hK": hK,
                 "valid": valid, "kept": kept}
    st["info"] = {"dense": bool(dense), "rows": int(rows.numel()), "seq_len": S,
                  "prefill_ms": t0.elapsed_time(t1) if timed else None}
    return out, cache


def decode(model, st, x, position_ids, cache):
    d = st["decode"]
    K, S, active = d["K"], d["S"], d["active"]
    if position_ids is None:
        position_ids = torch.tensor([[cache.get_seq_length()]], device=x.device)
    cos, sin = model.layers[0].self_attn.rotary_emb(x, position_ids)
    h = x
    for j, layer in enumerate(model.layers):
        a = layer.self_attn
        q, k, v = _qkv(layer, layer.input_layernorm(h), cos, sin)
        Kf, Vf = cache.update(k, v, j)
        keep = torch.ones(Kf.shape[2], dtype=torch.bool, device=h.device)
        if j >= K:
            keep[:S] = active
        Kk = repeat_kv(Kf[:, :, keep], a.num_key_value_groups)
        Vk = repeat_kv(Vf[:, :, keep], a.num_key_value_groups)
        O = nn.functional.scaled_dot_product_attention(q, Kk, Vk)
        if j == d["fc_layer"]:
            p = torch.softmax(torch.matmul(q, Kk.transpose(2, 3)) / math.sqrt(a.head_dim), dim=-1, dtype=torch.float32)[0]
            st["ref_dec"].append(_scatter_row(p[:, :, :int(keep[:S].sum())], torch.nonzero(keep[:S]).squeeze(-1), S))
        h = _finish(layer, h, O)
    return model.norm(h), cache


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
    if inputs_embeds.shape[0] != 1:
        raise ValueError("TRIAGE supports batch size 1 only.")
    if inputs_embeds.shape[1] > 1:
        h, cache = prefill(self, st, inputs_embeds, position_ids)
    else:
        h, cache = decode(self, st, inputs_embeds, position_ids, past_key_values)
    if not return_dict:
        return (h, cache)
    return BaseModelOutputWithPast(last_hidden_state=h, past_key_values=cache, hidden_states=None, attentions=None)


def install():
    global _ORIGINAL_FORWARD
    if _ORIGINAL_FORWARD is None:
        _ORIGINAL_FORWARD = modeling_llama.LlamaModel.forward
        modeling_llama.LlamaModel.forward = forward
