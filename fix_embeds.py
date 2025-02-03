from transformer_lens.HookedTransformer import *


def input_to_embed_fix(
        self,
        input: Union[str, List[str], Int[torch.Tensor, "batch pos"]],
        prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
        padding_side: Optional[Union[Literal["left", "right"], None]] = USE_DEFAULT_VALUE,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv_cache: Optional[HookedTransformerKeyValueCache] = None,
) -> Tuple[
    Float[torch.Tensor, "batch pos d_model"],  # residual
    Optional[Int[torch.Tensor, "batch pos"]],  # tokens
    Optional[Float[torch.Tensor, "batch pos d_model"]],  # shortformer_pos_embed
    Optional[torch.Tensor],  # attention_mask [batch pos]
]:
    """Convert input to first residual stream.

    Args:
        input (Union[str, List[str], Int[torch.Tensor, "batch pos"]]): The input to the model.
        prepend_bos (bool, optional): Overrides self.cfg.default_prepend_bos. Whether to prepend
            the BOS token to the input (only applies when input is a string). Defaults to None,
            implying usage of self.cfg.default_prepend_bos which is set to True unless specified
            otherwise. Pass True or False to locally override the default.
        padding_side ([Literal["left", "right"], optional): Overrides
            self.tokenizer.padding_side. Specifies which side to pad when tokenizing
            multiple strings of different lengths.
        past_kv_cache (HookedTransformerKeyValueCache, optional): If passed, we're doing caching
            and attention_mask will be stored in the cache.
    """
    if isinstance(input, str) or isinstance(input, list):
        # If text, convert to tokens (batch_size=1)
        assert (
                self.tokenizer is not None
        ), "Must provide a tokenizer if passing a string to the model"
        # This is only intended to support passing in a single string
        tokens = self.to_tokens(input, prepend_bos=prepend_bos, padding_side=padding_side)
    else:
        tokens = input
    if len(tokens.shape) == 1:
        # If tokens are a rank 1 tensor, add a dummy batch dimension to avoid things breaking.
        tokens = tokens[None]
    if tokens.device.type != self.cfg.device:
        tokens = tokens.to(devices.get_device_for_block_index(0, self.cfg))

    if (
            (self.tokenizer and self.tokenizer.padding_side == "left")
            or attention_mask is not None
            or past_kv_cache is not None
    ):
        # This means we need to have an explicit attention mask.
        if attention_mask is None:
            # If the padding side is left or we are using caching, we need to compute the attention
            # mask for the adjustment of absolute positional embeddings and attention masking so
            # that pad tokens are not attended.
            if prepend_bos is USE_DEFAULT_VALUE:
                prepend_bos = self.cfg.default_prepend_bos
            attention_mask = utils.get_attention_mask(self.tokenizer, tokens, prepend_bos)

        assert attention_mask.shape == tokens.shape[:-1], (
            f"Attention mask shape {attention_mask.shape} does not match tokens shape "
            f"{tokens.shape}"
        )
        attention_mask = attention_mask.to(devices.get_device_for_block_index(0, self.cfg))
        if past_kv_cache is not None:
            # past_kv_cache is not None, so we're doing caching.
            # We need to extend the previous attention_mask.
            # Update the past_kv_cache with the new attention_mask (unless it's frozen)
            attention_mask = past_kv_cache.append_attention_mask(attention_mask)
    else:
        # We separate this case from for computational efficiency.
        attention_mask = None

    # If we're doing caching, then we reuse keys and values from previous runs, as that's the
    # only way that past activations will affect the final logits. The cache contains those so
    # we don't need to recompute them. This is useful for generating text. As we have absolute
    # positional encodings, to implement this we have a `pos_offset` variable, defaulting to
    # zero, which says to offset which positional encodings are used (cached keys and values
    # were calculated with their own positional encodings).
    if past_kv_cache is None:
        pos_offset = 0
    else:
        batch_size, ctx_length = tokens.shape
        (
            cached_batch_size,
            cache_ctx_length,
            num_heads_in_cache,
            d_head_in_cache,
        ) = past_kv_cache[0].past_keys.shape
        assert cached_batch_size == batch_size
        if self.cfg.n_key_value_heads is None:
            assert num_heads_in_cache == self.cfg.n_heads
        else:
            assert num_heads_in_cache == self.cfg.n_key_value_heads
        assert d_head_in_cache == self.cfg.d_head
        pos_offset = cache_ctx_length
    if self.cfg.use_hook_tokens:
        tokens = self.hook_tokens(tokens)
    embed = tokens#self.hook_embed(self.embed(tokens))  # [batch, pos, d_model] !Изменено
    if self.cfg.positional_embedding_type == "standard":
        pos_embed = self.hook_pos_embed(
            self.pos_embed(tokens, pos_offset, attention_mask)
        )  # [batch, pos, d_model]
        residual = embed + pos_embed  # [batch, pos, d_model]
        shortformer_pos_embed = None
    elif self.cfg.positional_embedding_type == "shortformer":
        # If we're using shortformer style attention, we don't add the positional embedding to
        # the residual stream. See HookedTransformerConfig for details
        pos_embed = self.hook_pos_embed(
            self.pos_embed(tokens, pos_offset, attention_mask)
        )  # [batch, pos, d_model]
        residual = embed
        shortformer_pos_embed = pos_embed
    elif self.cfg.positional_embedding_type == "rotary":
        # Rotary doesn't use positional embeddings, instead they're applied when dot producting
        # keys and queries. See HookedTransformerConfig for details
        residual = embed
        shortformer_pos_embed = None
    elif self.cfg.positional_embedding_type == "alibi":
        # ALiBi does not add positional embeddings to word embeddings,instead it biases QK attention scores.
        residual = embed
        shortformer_pos_embed = None
    else:
        raise ValueError(
            f"Invalid positional_embedding_type passed in {self.cfg.positional_embedding_type}"
        )
    return residual, tokens, shortformer_pos_embed, attention_mask

HookedTransformer.input_to_embed = input_to_embed_fix