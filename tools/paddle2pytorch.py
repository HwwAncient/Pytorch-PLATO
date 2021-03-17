import pickle
import os
import torch
import numpy as np
from collections import OrderedDict

# import paddle.fluid as fluid


optional_parameter = 'Model/UnifiedTransformer_0.predictor/FC_0.w_0'

# def paddle2numpy(input_pd, output_np):
#     state_dict_numpy = {}
#     if hasattr(fluid, "load_dygraph"):
#         # >= 1.6.0 compatible
#         state_dict_paddle, optimizers = fluid.load_dygraph(input_pd)
#     else:
#         state_dict_paddle, optimizers = fluid.dygraph.load_persistables(input_pd)
#
#     for name, param in state_dict_paddle.items():
#         if hasattr(param, "numpy"):
#             arr = param.numpy()
#         else:
#             value = param.value()
#             tensor = value.get_tensor()
#             arr = np.array(tensor)
#         state_dict_numpy[name] = arr
#
#     with open(output_np, 'wb') as fo:
#         pickle.dump(state_dict_numpy, fo)


def get_match_value(name, state_dict_numpy):
    """
    Need be overridden towards different models, here for UnifiedTransformer Model
    """
    if name == 'mask_embed':
        return state_dict_numpy['Model/UnifiedTransformer_0.mask_embed']
    elif name == 'latent_embeddings':
        return state_dict_numpy['Model/UnifiedTransformer_0.latent_embeddings']
    elif name == 'embedder.token_embedding.weight':
        return state_dict_numpy['Model/UnifiedTransformer_0/Embedder_0/Embedding_0.w_0']
    elif name == 'embedder.pos_embedding.weight':
        return state_dict_numpy['Model/UnifiedTransformer_0/Embedder_0/Embedding_1.w_0']
    elif name == 'embedder.type_embedding.weight':
        return state_dict_numpy['Model/UnifiedTransformer_0/Embedder_0/Embedding_2.w_0']
    elif name == 'embedder.turn_embedding.weight':
        return state_dict_numpy['Model/UnifiedTransformer_0/Embedder_0/Embedding_3.w_0']
    elif name == 'embed_layer_norm.weight':
        return state_dict_numpy['Model/UnifiedTransformer_0/LayerNorm_0.w_0']
    elif name == 'embed_layer_norm.bias':
        return state_dict_numpy['Model/UnifiedTransformer_0/LayerNorm_0.b_0']
    elif name == 'post_network.weight':
        return state_dict_numpy['Model/UnifiedTransformer_0.post_network/FC_0.w_0'].T
    elif name == 'discriminator.0.weight':
        return state_dict_numpy['Model/UnifiedTransformer_0.discriminator/FC_0.w_0'].T
    elif name == 'discriminator.0.bias':
        return state_dict_numpy['Model/UnifiedTransformer_0.discriminator/FC_0.b_0']
    elif name == 'bow_predictor.weight':
        return state_dict_numpy['Model/UnifiedTransformer_0.bow_predictor/FC_0.w_0'].T
    else:
        num = name.split('.')[1]
        assert num in [str(i) for i in range(12)]
        if name == f'layers.{num}.attn.linear_qkv.weight':
            return state_dict_numpy[f'Model/UnifiedTransformer_0/TransformerBlock_{num}/'
                                    f'MultiheadAttention_0/FC_0.w_0'].T
        elif name == f'layers.{num}.attn.linear_qkv.bias':
            return state_dict_numpy[f'Model/UnifiedTransformer_0/TransformerBlock_{num}/'
                                    f'MultiheadAttention_0/FC_0.b_0']
        elif name == f'layers.{num}.attn.linear_out.weight':
            return state_dict_numpy[f'Model/UnifiedTransformer_0/TransformerBlock_{num}/'
                                    f'MultiheadAttention_0/FC_1.w_0'].T
        elif name == f'layers.{num}.attn.linear_out.bias':
            return state_dict_numpy[f'Model/UnifiedTransformer_0/TransformerBlock_{num}/'
                                    f'MultiheadAttention_0/FC_1.b_0']
        elif name == f'layers.{num}.attn_norm.weight':
            return state_dict_numpy[f'Model/UnifiedTransformer_0/TransformerBlock_{num}/LayerNorm_0.w_0']
        elif name == f'layers.{num}.attn_norm.bias':
            return state_dict_numpy[f'Model/UnifiedTransformer_0/TransformerBlock_{num}/LayerNorm_0.b_0']
        elif name == f'layers.{num}.ff.linear_hidden.0.weight':
            return state_dict_numpy[f'Model/UnifiedTransformer_0/TransformerBlock_{num}/FeedForward_0/FC_0.w_0'].T
        elif name == f'layers.{num}.ff.linear_hidden.0.bias':
            return state_dict_numpy[f'Model/UnifiedTransformer_0/TransformerBlock_{num}/FeedForward_0/FC_0.b_0']
        elif name == f'layers.{num}.ff.linear_out.weight':
            return state_dict_numpy[f'Model/UnifiedTransformer_0/TransformerBlock_{num}/FeedForward_0/FC_1.w_0'].T
        elif name == f'layers.{num}.ff.linear_out.bias':
            return state_dict_numpy[f'Model/UnifiedTransformer_0/TransformerBlock_{num}/FeedForward_0/FC_1.b_0']
        elif name == f'layers.{num}.ff_norm.weight':
            return state_dict_numpy[f'Model/UnifiedTransformer_0/TransformerBlock_{num}/LayerNorm_1.w_0']
        elif name == f'layers.{num}.ff_norm.bias':
            return state_dict_numpy[f'Model/UnifiedTransformer_0/TransformerBlock_{num}/LayerNorm_1.b_0']
        else:
            raise ValueError('No matched name in state_dict_numpy!')


def numpy2pytorch(input_np, input_pt, output_pt):
    with open(input_np, 'rb') as fi:
        state_dict_numpy = pickle.load(fi)

    state_dict_pytorch = OrderedDict()
    state_dict_init_pytorch = torch.load(input_pt, map_location=lambda storage, loc: storage)
    for name, value in state_dict_init_pytorch.items():
        match_value = get_match_value(name, state_dict_numpy)
        assert match_value.shape == value.numpy().shape
        assert match_value.dtype == value.numpy().dtype
        dtype = value.dtype
        device = value.device
        state_dict_pytorch[name] = torch.tensor(match_value, dtype=dtype, device=device)

    torch.save(state_dict_pytorch, output_pt)


if __name__ == '__main__':
    input_paddle = '../model/PLATO'
    output_numpy = '../model/PLATO.np'
    input_numpy = output_numpy
    input_pytorch = '../model/PLATO_INIT.pt'
    output_pytorch = '../model/PLATO.pt'

    # 1. convert paddle state_dict to numpy state_dict
    # if not os.path.isfile(input_numpy):
    #     place = fluid.CPUPlace()
    #     with fluid.dygraph.guard(place):
    #         paddle2numpy(input_pd=input_paddle, output_np=output_numpy)

    # 2. convert numpy state_dict to pytorch state_dict
    numpy2pytorch(input_np=input_numpy, input_pt=input_pytorch, output_pt=output_pytorch)
