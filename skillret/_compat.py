"""Compatibility patches for transformers 5.x breaking changes.

These monkey-patches allow custom model code (written for transformers 4.x)
to run under transformers 5.x without modifying any files inside .venv/.

This module is imported and applied once at startup by ``skillret.eval``.
"""

from __future__ import annotations

import importlib
import shutil
import sys
import types
from pathlib import Path

import torch


def patch_jina_v4_compat(model_path: str):
    """Patch jina-embeddings-v4's bundled qwen2_5_vl.py for transformers >=5.x.

    The model ships a copy of Qwen 2.5-VL source targeting transformers ~4.52.
    Versions >=5.x removed ``SlidingWindowCache`` and the ``'default'`` ROPE
    init function.  We apply minimal text patches to the cached module file so
    the original model architecture is preserved (important because PEFT
    adapters depend on exact layer structure).
    """
    cache_root = Path.home() / ".cache/huggingface/modules/transformers_modules"
    dst = cache_root / "jina_hyphen_embeddings_hyphen_v4"
    qwen_file = dst / "qwen2_5_vl.py"
    if not qwen_file.exists():
        from transformers import AutoConfig
        AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if not qwen_file.exists():
        return
    src = qwen_file.read_text()
    if "# PATCHED-FOR-TF5" in src:
        return
    patched = src.replace(
        "from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache",
        "from transformers.cache_utils import Cache, DynamicCache, StaticCache\n"
        "try:\n"
        "    from transformers.cache_utils import SlidingWindowCache\n"
        "except ImportError:\n"
        "    SlidingWindowCache = StaticCache  # transformers >=5.x compat",
    ).replace(
        "from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update",
        "from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update\n"
        "if 'default' not in ROPE_INIT_FUNCTIONS:\n"
        "    import torch as _t\n"
        "    def _default_rope(config, device=None, seq_len=None, **kw):\n"
        "        rd = getattr(config, 'rope_parameters', {}) or {}\n"
        "        base = rd.get('rope_theta', getattr(config, 'rope_theta', 10000.0))\n"
        "        head_dim = config.hidden_size // config.num_attention_heads\n"
        "        dim = int(head_dim * getattr(config, 'partial_rotary_factor', 1.0))\n"
        "        inv_freq = 1.0 / (base ** (_t.arange(0, dim, 2, dtype=_t.int64).float().to(device) / dim))\n"
        "        return inv_freq, 1.0\n"
        "    ROPE_INIT_FUNCTIONS['default'] = _default_rope",
    )
    patched = "# PATCHED-FOR-TF5\n" + patched
    qwen_file.write_text(patched)
    pycache = dst / "__pycache__"
    if pycache.exists():
        shutil.rmtree(pycache, ignore_errors=True)


def apply_transformers_compat_patches():
    """Apply all runtime compatibility patches for transformers 5.x."""

    # --- 1) rope_theta → rope_parameters migration ---------------------------
    from transformers import configuration_utils

    _orig_getattr = configuration_utils.PretrainedConfig.__getattribute__

    def _getattr_with_rope_fallback(self, key):
        if key == "rope_theta":
            try:
                return _orig_getattr(self, key)
            except AttributeError:
                rp = _orig_getattr(self, "rope_parameters") if hasattr(self, "rope_parameters") else None
                if isinstance(rp, dict) and "rope_theta" in rp:
                    return rp["rope_theta"]
                raise
        return _orig_getattr(self, key)

    configuration_utils.PretrainedConfig.__getattribute__ = _getattr_with_rope_fallback

    # --- 2) SlidingWindowCache removed in 5.x --------------------------------
    from transformers import cache_utils
    if not hasattr(cache_utils, "SlidingWindowCache"):
        cache_utils.SlidingWindowCache = cache_utils.DynamicCache

    # --- 2b) DynamicCache API changes in 5.x ----------------------------------
    if not hasattr(cache_utils.DynamicCache, "from_legacy_cache"):
        @classmethod
        def _from_legacy_cache(cls, past_key_values=None):
            cache = cls()
            if past_key_values is not None:
                for layer_idx, (key, value) in enumerate(past_key_values):
                    cache.update(key, value, layer_idx)
            return cache
        cache_utils.DynamicCache.from_legacy_cache = _from_legacy_cache

    if not hasattr(cache_utils.DynamicCache, "get_usable_length"):
        def _get_usable_length(self, new_seq_length=0, layer_idx=0):
            return self.get_seq_length(layer_idx)
        cache_utils.DynamicCache.get_usable_length = _get_usable_length

    if not hasattr(cache_utils.DynamicCache, "to_legacy_cache"):
        def _to_legacy_cache(self):
            return list(self)
        cache_utils.DynamicCache.to_legacy_cache = _to_legacy_cache

    # --- 3) tokenization_qwen2_fast renamed in 5.x ---------------------------
    _qwen2_tok_old = "transformers.models.qwen2.tokenization_qwen2_fast"
    if _qwen2_tok_old not in sys.modules:
        try:
            importlib.import_module(_qwen2_tok_old)
        except ModuleNotFoundError:
            _new_mod = importlib.import_module("transformers.models.qwen2.tokenization_qwen2")
            _shim = types.ModuleType(_qwen2_tok_old)
            _shim.Qwen2TokenizerFast = _new_mod.Qwen2Tokenizer
            _shim.Qwen2Tokenizer = _new_mod.Qwen2Tokenizer
            sys.modules[_qwen2_tok_old] = _shim

    # --- 4) _tied_weights_keys changed from list to dict in 5.x ---------------
    from transformers.modeling_utils import PreTrainedModel
    _orig_get_tied = PreTrainedModel.get_expanded_tied_weights_keys

    def _compat_get_tied(self, all_submodels=False):
        tw = self._tied_weights_keys
        if isinstance(tw, list):
            self._tied_weights_keys = {k: k for k in tw} if tw else None
        return _orig_get_tied(self, all_submodels=all_submodels)

    PreTrainedModel.get_expanded_tied_weights_keys = _compat_get_tied

    # --- 5) MISTRAL_INPUTS_DOCSTRING removed in 5.x (NV-Embed-v1) ------------
    from transformers.models.mistral import modeling_mistral as _mm
    if not hasattr(_mm, "MISTRAL_INPUTS_DOCSTRING"):
        _mm.MISTRAL_INPUTS_DOCSTRING = ""

    # --- 5a) Old custom code calls MistralDecoderLayer without position_embeddings
    _orig_mdl_fwd = _mm.MistralDecoderLayer.forward

    def _compat_mdl_forward(self, hidden_states, attention_mask=None,
                            position_ids=None, past_key_values=None,
                            use_cache=False, cache_position=None,
                            position_embeddings=None, **kwargs):
        if position_embeddings is None and position_ids is not None:
            rotary = getattr(self, "_compat_rotary_emb", None)
            if rotary is not None:
                position_embeddings = rotary(hidden_states, position_ids)
        return _orig_mdl_fwd(
            self, hidden_states, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            use_cache=use_cache, cache_position=cache_position,
            position_embeddings=position_embeddings, **kwargs,
        )

    _mm.MistralDecoderLayer.forward = _compat_mdl_forward

    _orig_mm_post_init = _mm.MistralModel.post_init

    def _patched_mm_post_init(self):
        _orig_mm_post_init(self)
        if hasattr(self, "rotary_emb") and hasattr(self, "layers"):
            for layer in self.layers:
                layer._compat_rotary_emb = self.rotary_emb
        if type(self).__name__ == "BidirectionalMistralModel":
            type(self).forward = _mm.MistralModel.forward

    _mm.MistralModel.post_init = _patched_mm_post_init

    # --- 5b) all_tied_weights_keys may not be set by post_init (NV-Embed) ----
    _orig_mark_tied = getattr(PreTrainedModel, "mark_tied_weights_as_initialized", None)
    if _orig_mark_tied is not None:
        def _safe_mark_tied(self, loading_info=None):
            if not hasattr(self, "all_tied_weights_keys"):
                try:
                    self.all_tied_weights_keys = self.get_expanded_tied_weights_keys(
                        all_submodels=False
                    )
                except Exception:
                    self.all_tied_weights_keys = {}
            return _orig_mark_tied(self, loading_info)
        PreTrainedModel.mark_tied_weights_as_initialized = _safe_mark_tied

    # --- 6) ProcessorMixin.get_attributes() misses inherited *_class attrs ----
    from transformers import processing_utils as _pu
    _orig_get_attrs = _pu.ProcessorMixin.get_attributes

    _modality_keys = list(_pu.MODALITY_TO_AUTOPROCESSOR_MAPPING.keys())

    @classmethod
    def _patched_get_attributes(cls):
        attrs = _orig_get_attrs.__func__(cls)
        if not attrs:
            import inspect
            found = []
            for klass in cls.__mro__:
                for attr_name, value in klass.__dict__.items():
                    if value is None or not attr_name.endswith("_class"):
                        continue
                    inferred = attr_name[: -len("_class")]
                    if inferred == "audio_tokenizer":
                        continue
                    if any(m in inferred for m in _modality_keys):
                        if inferred not in found:
                            found.append(inferred)
            if found:
                for klass in cls.__mro__:
                    if klass is cls:
                        continue
                    try:
                        params = list(inspect.signature(klass.__init__).parameters.keys())
                        if len(params) > 1 and params[0] == "self":
                            ordered = [p for p in params[1:] if p in found]
                            remaining = [f for f in found if f not in ordered]
                            return ordered + remaining
                    except (ValueError, TypeError):
                        continue
                return found
        return attrs

    _pu.ProcessorMixin.get_attributes = _patched_get_attributes

    # --- 6b) Fix meta-device params left after from_pretrained in 5.x --------
    from transformers import AutoModel as _AM
    _orig_am_from_pretrained = _AM.from_pretrained.__func__

    @classmethod
    def _patched_am_from_pretrained(cls, *args, **kwargs):
        model = _orig_am_from_pretrained(cls, *args, **kwargs)
        for name, param in list(model.named_parameters()):
            if param.device.type == "meta":
                parts = name.split(".")
                mod = model
                for part in parts[:-1]:
                    mod = getattr(mod, part)
                new_param = torch.nn.Parameter(
                    torch.zeros(param.shape, dtype=param.dtype, device="cpu"),
                    requires_grad=param.requires_grad,
                )
                setattr(mod, parts[-1], new_param)
        return model

    _AM.from_pretrained = _patched_am_from_pretrained

    # --- 7) ROPE_INIT_FUNCTIONS missing 'default' key in 5.x -----------------
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    if "default" not in ROPE_INIT_FUNCTIONS:
        def _rope_default(config, device, **kwargs):
            base = config.rope_theta
            dim = int(config.hidden_size // config.num_attention_heads)
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
            return inv_freq, 1.0
        ROPE_INIT_FUNCTIONS["default"] = _rope_default

    # --- 8) FlagEmbedding needs is_torch_fx_available (removed in 5.x) --------
    import transformers.utils.import_utils as _import_utils
    if not hasattr(_import_utils, "is_torch_fx_available"):
        _import_utils.is_torch_fx_available = lambda: False

    # --- 9) jina-reranker-v2 create_position_ids_from_input_ids ---------------
    import transformers.models.xlm_roberta.modeling_xlm_roberta as _xlm_mod
    if not hasattr(_xlm_mod, "create_position_ids_from_input_ids"):
        _xlm_mod.create_position_ids_from_input_ids = (
            _xlm_mod.XLMRobertaEmbeddings.create_position_ids_from_input_ids
        )

    # --- 10) FlagEmbedding needs tokenizer.prepare_for_model (removed in 5.x) -
    from transformers import PreTrainedTokenizerBase as _TokBase
    if not hasattr(_TokBase, "prepare_for_model"):
        def _prepare_for_model(
            self, ids, pair_ids=None, *,
            add_special_tokens=True, padding=False,
            truncation=False, max_length=None, **kwargs,
        ):
            trunc_strategy = kwargs.get("truncation_strategy", truncation)
            pair = pair_ids if pair_ids is not None else []
            cls_id = getattr(self, "cls_token_id", None) or getattr(self, "bos_token_id", 0)
            sep_id = getattr(self, "sep_token_id", None) or getattr(self, "eos_token_id", 2)
            if add_special_tokens:
                sequence = [cls_id] + ids + [sep_id] + pair + [sep_id]
                token_type_ids = [0] * (len(ids) + 2) + [1] * (len(pair) + 1)
            else:
                sequence = ids + pair
                token_type_ids = [0] * len(ids) + [1] * len(pair)
            if max_length and len(sequence) > max_length:
                if trunc_strategy == "only_second" and pair_ids is not None:
                    n_special = 3 if add_special_tokens else 0
                    max_pair = max(0, max_length - len(ids) - n_special)
                    pair_trunc = pair[:max_pair]
                    if add_special_tokens:
                        sequence = [cls_id] + ids + [sep_id] + pair_trunc + [sep_id]
                        token_type_ids = [0] * (len(ids) + 2) + [1] * (len(pair_trunc) + 1)
                    else:
                        sequence = ids + pair_trunc
                        token_type_ids = [0] * len(ids) + [1] * len(pair_trunc)
                else:
                    sequence = sequence[:max_length]
                    token_type_ids = token_type_ids[:max_length]
            attention_mask = [1] * len(sequence)
            from transformers import BatchEncoding
            return BatchEncoding({
                "input_ids": sequence,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            })
        _TokBase.prepare_for_model = _prepare_for_model


def fix_non_persistent_buffers(model):
    """Re-initialise non-persistent buffers left uninitialised by
    transformers 5.x meta-device loading (e.g. ``position_ids``, RoPE)."""
    fix_rotary_embeddings(model)
    for mod in model.modules():
        if hasattr(mod, "position_ids") and hasattr(mod, "max_position_embeddings"):
            pid = mod.position_ids
            if pid is not None and pid.numel() > 0 and pid.max().item() > mod.max_position_embeddings:
                correct = torch.arange(mod.max_position_embeddings, device=pid.device, dtype=pid.dtype)
                mod.register_buffer("position_ids", correct, persistent=False)
        elif hasattr(mod, "position_ids"):
            pid = mod.position_ids
            if pid is not None and pid.numel() > 0:
                expected = torch.arange(pid.numel(), device=pid.device, dtype=pid.dtype)
                if not torch.equal(pid, expected) and pid.max().item() > pid.numel():
                    mod.register_buffer("position_ids", expected, persistent=False)


def fix_rotary_embeddings(model):
    """Re-compute non-persistent RoPE buffers (inv_freq, cos_cached,
    sin_cached) that transformers 5.x meta-device loading leaves as
    uninitialized garbage."""
    for _name, mod in model.named_modules():
        if hasattr(mod, "inv_freq") and hasattr(mod, "dim") and hasattr(mod, "base"):
            correct = 1.0 / (
                mod.base ** (torch.arange(0, mod.dim, 2).float() / mod.dim)
            )
            mod.register_buffer("inv_freq", correct, persistent=False)
            max_pos = getattr(mod, "max_position_embeddings", 8192)
            sf = getattr(mod, "scaling_factor", 1.0)
            if hasattr(mod, "_set_cos_sin_cache"):
                mod._set_cos_sin_cache(
                    int(max_pos * sf), correct.device, torch.get_default_dtype()
                )


def reload_safetensors_weights(model, model_path: str):
    """Reload weights directly from safetensors, bypassing transformers 5.x
    loading bugs that corrupt custom-code models."""
    from safetensors import safe_open
    state_dict = {}
    for f in sorted(Path(model_path).glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as sf:
            for k in sf.keys():
                state_dict[k] = sf.get_tensor(k)
    model.load_state_dict(state_dict, strict=False)
