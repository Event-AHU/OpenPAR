from __future__ import annotations
from collections.abc import Iterable, Mapping
from typing import Any
_SURGERY_PREFIX = 'clip_model.visual.transformer.resblocks.'
_VISUAL_ALIAS_PREFIX = 'clip_model.visual.attn.'
_ATTENTION_KEY_MAP = {'.qkv.weight': '.in_proj_weight', '.qkv.bias': '.in_proj_bias', '.proj.weight': '.out_proj.weight', '.proj.bias': '.out_proj.bias'}

def adapt_lazy_visual_attention_state(state: Mapping[str, Any], target_keys: Iterable[str]) -> tuple[dict[str, Any], int, int]:
    targets = set(target_keys)
    adapted = dict(state)
    converted_count = 0
    for source_key in list(adapted):
        if not source_key.startswith(_SURGERY_PREFIX) or '.attn.' not in source_key:
            continue
        for source_suffix, target_suffix in _ATTENTION_KEY_MAP.items():
            if not source_key.endswith(source_suffix):
                continue
            target_key = source_key[:-len(source_suffix)] + target_suffix
            if source_key not in targets and target_key in targets:
                if target_key not in adapted:
                    adapted[target_key] = adapted[source_key]
                del adapted[source_key]
                converted_count += 1
            break
    dropped_count = 0
    for key in list(adapted):
        if key.startswith(_VISUAL_ALIAS_PREFIX) and key not in targets:
            del adapted[key]
            dropped_count += 1
    return (adapted, converted_count, dropped_count)
