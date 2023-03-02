import torch


def registed_att_hooks(model):
    """
    register forward hooks for all multimodal transformer layers
    so that the features are saved after every forward pass
    """

    for n, mod in enumerate(model.neural_visual_transformer.neural_state_blocks):
        mod.register_forward_hook(get_features(f'neural_state_block_{n}'))

    for n, mod in enumerate(model.neural_visual_transformer.neural_state_history_blocks):
        mod.register_forward_hook(get_features(f'neural_state_history_block_{n}'))

    for n, mod in enumerate(model.neural_visual_transformer.neural_state_history_self_attention):
        mod.register_forward_hook(get_features(f'neural_state_history_self_attention_{n}'))

    for n, mod in enumerate(model.neural_visual_transformer.neural_state_stimulus_blocks):
        mod.register_forward_hook(get_features(f'neural_state_stimulus_block_{n}'))

    for n, mod in enumerate(model.neural_visual_transformer.neural_state_stimulus_blocks):
        mod.attn.attn_drop.register_full_backward_hook(get_grads(f'neural_state_stimulus_block_{n}'))



# helper function for hook
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach().cpu()
    return hook

def get_grads(name):
    def hook(model, input, output):
        grads[name] = output.detach().cpu()
    return hook

def get_atts(name):
    def hook(model, input, output):
        attentions[name] = output
    return hook


def get_atts(model):
    attentions = {}

    for n, mod in enumerate(model.neural_visual_transformer.neural_state_blocks):
        attentions[f'neural_state_block_{n}'] = mod.attn.att.detach().cpu()

    for n, mod in enumerate(model.neural_visual_transformer.neural_state_history_blocks):
        attentions[f'neural_state_history_block_{n}'] = mod.attn.att.detach().cpu()

    for n, mod in enumerate(model.neural_visual_transformer.neural_state_stimulus_blocks):
        attentions[f'neural_stimulus_block_{n}'] = mod.attn.att.detach().cpu()
    
    return attentions

def get_grads(model):
    grads = {}
    
    for n, mod in enumerate(model.neural_visual_transformer.neural_state_stimulus_blocks):
        grads[f'neural_stimulus_block_{n}'] = mod.attn.att.detach().cpu()    
    return grads

def gradcam(atts, grads, clamp=True):
    common_keys = set(atts.keys()).intersection(set(grads.keys()))
    for key in common_keys:
        if clamp:
            grads[key] = grads[key].clamp(0)
        atts[key] = atts[key] * grads[key]
    return atts


def accum_atts(att_dict, stimulus, key=None):
    if key is None:
        att_keys = att_dict.keys()
    else:
        att_keys = [k for k in att_dict.keys() if key in k]
    atts = []
    for k in att_keys:
        att = att_dict[k]
        att = att.sum(-3).detach().cpu()
        reshape_c = att.shape[-1] // stimulus.shape[0]
        assert att.shape[-1] % stimulus.shape[0] == 0, "Attention shape does not match stimulus shape"
        att = att.view(att.shape[0], att.shape[-2], reshape_c, att.shape[-1] // reshape_c)
        att = att.sum(-2)
        atts.append(att)
    return torch.stack(atts)

def reshape_attentions(att_vis):
    n_id_block, n_vis_block = att_vis.shape[-2], att_vis.shape[-1]
    att_vis = att_vis.view(n_id_block, n_vis_block)
    reshape_c = att_vis.shape[-1] // stimulus.shape[0]
    assert att_vis.shape[-1] % stimulus.shape[0] == 0, "Attention shape does not match stimulus shape"
    att_vis = att_vis.view(att_vis.shape[0], reshape_c, att_vis.shape[1] // reshape_c)
    att_vis = att_vis.sum(-2)
    return att_vis


def all_device(data, device):
    device = torch.device(device)
    if isinstance(data, dict):
        return {k: all_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [all_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(all_device(v, device) for v in data)
    else:
        return data.to(device)


def cat_atts(attentions, layer_key):
    return torch.cat([attentions[k] for k in attentions.keys() if k.startswith(layer_key)])


def stack_atts(attentions, layer_key):
    # key: block name, value: B, L, N, H, W
    # out: B, L, N, H, W
    stacked_atts = torch.stack([attentions[k] for k in attentions.keys() if k.startswith(layer_key)])
    return stacked_atts.transpose(0, 1)
