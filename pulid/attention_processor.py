# modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

NUM_ZERO = 0
ORTHO = False
ORTHO_v2 = False


class AttnProcessor(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        id_embedding=None,
        id_scale=1.0,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class IDAttnProcessor(nn.Module):
    r"""
    Attention processor for ID-Adapater.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
    """

    def __init__(self, hidden_size, cross_attention_dim=None):
        super().__init__()
        self.id_to_k = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.id_to_v = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        id_embedding=None,
        id_scale=1.0,
    ):  
        t_start = time.time()
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        cross_period = time.time() - t_start
        # # for id-adapter
        if id_embedding is not None:

            
            # 记录开始时间
            t_start = time.time()
            
            # 打印id_embedding信息
            # print(f"id_embedding dtype: {id_embedding.dtype}, device: {id_embedding.device}")
            
            if NUM_ZERO == 0:
                t1 = time.time()
                id_key = self.id_to_k(id_embedding)
                id_value = self.id_to_v(id_embedding)
                print(f"ID projection time: {time.time() - t1:.4f}s")
            else:
                t1 = time.time()
                zero_tensor = torch.zeros(
                    (id_embedding.size(0), NUM_ZERO, id_embedding.size(-1)),
                    dtype=id_embedding.dtype,
                    device=id_embedding.device,
                )
                id_key = self.id_to_k(torch.cat((id_embedding, zero_tensor), dim=1))
                id_value = self.id_to_v(torch.cat((id_embedding, zero_tensor), dim=1))
                print(f"ID projection with padding time: {time.time() - t1:.4f}s")

            t2 = time.time()
            id_key = attn.head_to_batch_dim(id_key).to(query.dtype)
            id_value = attn.head_to_batch_dim(id_value).to(query.dtype)
            # print(f"Dimension transform time: {time.time() - t2:.4f}s")

            t3 = time.time()
            id_attention_probs = attn.get_attention_scores(query, id_key, None)
            # print(f"Attention score time: {time.time() - t3:.4f}s")

            t4 = time.time()
            id_hidden_states = torch.bmm(id_attention_probs, id_value)
            id_hidden_states = attn.batch_to_head_dim(id_hidden_states)
            # print(f"BMM and transform time: {time.time() - t4:.4f}s")

            t5 = time.time()
            if not ORTHO:
                hidden_states = hidden_states + id_scale * id_hidden_states
            else:
                projection = (
                    torch.sum((hidden_states * id_hidden_states), dim=-2, keepdim=True)
                    / torch.sum((hidden_states * hidden_states), dim=-2, keepdim=True)
                    * hidden_states
                )
                orthogonal = id_hidden_states - projection
                hidden_states = hidden_states + id_scale * orthogonal
            # print(f"Final computation time: {time.time() - t5:.4f}s")
            
            print(f"Total ID attention time: {time.time() - t_start:.4f}s, cross_period: {cross_period:.4f}s, diff: {time.time() - t_start - cross_period:.4f}s")

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class AttnProcessor2_0(nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        id_embedding=None,
        id_scale=1.0,
    ):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        del query, key, value, residual
        return hidden_states


class IDAttnProcessor2_0(torch.nn.Module):
    _cache = {}
    _instance_count = 0
    _last_embedding_key = None  # 添加类变量来记录上一次的embedding特征

    def __init__(self, hidden_size, cross_attention_dim=None):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        
        # 为每个实例创建唯一标识符
        self.instance_id = IDAttnProcessor2_0._instance_count
        IDAttnProcessor2_0._instance_count += 1
        
        print("cross_attention_dim", cross_attention_dim,"hidden_size", hidden_size)
        self.id_to_k = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.id_to_v = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

    def _get_cache_key(self, id_embedding):
        # 创建缓存键
        return (self.instance_id, id_embedding.shape, id_embedding.sum().item())

    def _get_embedding_key(self, id_embedding):
        # 获取embedding的特征值
        return (id_embedding.shape, id_embedding.sum().item())

    def _compute_id_projections(self, id_embedding, query_dtype):
        # 检查id_embedding是否发生变化
        current_key = self._get_embedding_key(id_embedding)
        if IDAttnProcessor2_0._last_embedding_key is not None and current_key != IDAttnProcessor2_0._last_embedding_key:
            print("检测到新的id_embedding，清空缓存...")
            self._cache.clear()
        IDAttnProcessor2_0._last_embedding_key = current_key

        t_start = time.time()
        if NUM_ZERO == 0:
            # 检查缓存
            cache_key = self._get_cache_key(id_embedding)
            if cache_key in self._cache:
                print("Using cached ID projections")
                return self._cache[cache_key]

            id_key = self.id_to_k(id_embedding).to(query_dtype)
            id_value = self.id_to_v(id_embedding).to(query_dtype)
            
            # 直接存储在GPU上
            self._cache[cache_key] = (id_key.to(id_embedding.device), 
                                    id_value.to(id_embedding.device))
            print(f"ID projection time: {time.time() - t_start:.4f}s")
            return id_key, id_value
        else:
            zero_tensor = torch.zeros(
                (id_embedding.size(0), NUM_ZERO, id_embedding.size(-1)),
                dtype=id_embedding.dtype,
                device=id_embedding.device,
            )
            cache_key = self._get_cache_key(torch.cat((id_embedding, zero_tensor), dim=1))
            if cache_key in self._cache:
                print("Using cached ID projections with padding")
                return self._cache[cache_key]

            id_key = self.id_to_k(torch.cat((id_embedding, zero_tensor), dim=1)).to(query_dtype)
            id_value = self.id_to_v(torch.cat((id_embedding, zero_tensor), dim=1)).to(query_dtype)
            
            # 存入缓存
            self._cache[cache_key] = (id_key.to(id_embedding.device), 
                                    id_value.to(id_embedding.device))
            print(f"ID projection with padding time: {time.time() - t_start:.4f}s")
        
        return id_key, id_value

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        id_embedding=None,
        id_scale=1.0,
    ):
        t_start = time.time()
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        cross_period = time.time() - t_start  # 记录cross attention时间
        # for id embedding
        if id_embedding is not None:
            t_start = time.time()
            id_key, id_value = self._compute_id_projections(id_embedding, query.dtype)
            
            t2 = time.time()
            id_key = id_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            id_value = id_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            # print(f"Dimension transform time: {time.time() - t2:.4f}s, id_embedding dtype: {id_embedding.dtype},query.dtype: {query.dtype}")

            t3 = time.time()
            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            id_hidden_states = F.scaled_dot_product_attention(
                query, id_key, id_value, attn_mask=None, dropout_p=0.0, is_causal=False
            )

            id_hidden_states = id_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            id_hidden_states = id_hidden_states.to(query.dtype)
            # print(f"scaled_dot_product_attention time: {time.time() - t3:.4f}s")

            t5 = time.time()
            if not ORTHO and not ORTHO_v2:
                hidden_states = hidden_states + id_scale * id_hidden_states
            elif ORTHO_v2:
                orig_dtype = hidden_states.dtype
                hidden_states = hidden_states.to(torch.float32)
                id_hidden_states = id_hidden_states.to(torch.float32)
                attn_map = query @ id_key.transpose(-2, -1)
                attn_mean = attn_map.softmax(dim=-1).mean(dim=1)
                attn_mean = attn_mean[:, :, :5].sum(dim=-1, keepdim=True)
                projection = (
                    torch.sum((hidden_states * id_hidden_states), dim=-2, keepdim=True)
                    / torch.sum((hidden_states * hidden_states), dim=-2, keepdim=True)
                    * hidden_states
                )
                orthogonal = id_hidden_states + (attn_mean - 1) * projection
                hidden_states = hidden_states + id_scale * orthogonal
                hidden_states = hidden_states.to(orig_dtype)
            else:
                orig_dtype = hidden_states.dtype
                hidden_states = hidden_states.to(torch.float32)
                id_hidden_states = id_hidden_states.to(torch.float32)
                projection = (
                    torch.sum((hidden_states * id_hidden_states), dim=-2, keepdim=True)
                    / torch.sum((hidden_states * hidden_states), dim=-2, keepdim=True)
                    * hidden_states
                )
                orthogonal = id_hidden_states - projection
                hidden_states = hidden_states + id_scale * orthogonal
                hidden_states = hidden_states.to(orig_dtype)
            print(f"Total ID attention time: {time.time() - t_start:.4f}s, cross_period: {cross_period:.4f}s, diff: {time.time() - t_start - cross_period:.4f}s")
            del id_key, id_value, id_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        del query, key, value, residual
        return hidden_states
