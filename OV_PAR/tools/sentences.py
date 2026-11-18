from __future__ import annotations
from typing import Any, Sequence
import numpy as np
from dataset.generate_entences.process_sentences_rapov import generate_sentences_group as rapov_sentence_process
from dataset.generate_entences.process_sentences_rapov2 import generate_sentences_group2 as rapov2_merged_sentence_process
from dataset.generate_entences.process_sentences_rapov2_2 import generate_sentences_group2 as rapov2_separate_sentence_process
from dataset.generate_entences.processing_sentence_msp60k import generate_sentences_group2 as mspov_sentence_process

def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'numpy'):
        value = value.numpy()
    return np.asarray(value)

def build_training_group_sentences(dataset_name: str, labels: Any, base_group_sizes: Sequence[int], group_names: Sequence[str]) -> list[list[str]]:
    labels_array = _to_numpy(labels)
    if labels_array.ndim != 2 or labels_array.shape[1] != sum(base_group_sizes):
        raise ValueError('training-label width does not match base group sizes')
    if dataset_name == 'MSPOV':
        processor = mspov_sentence_process
    elif dataset_name == 'RAPOV':
        processor = rapov_sentence_process
    elif 'IdentityTraits' in group_names:
        processor = rapov2_merged_sentence_process
    else:
        processor = rapov2_separate_sentence_process
    group_sentences: list[list[str]] = []
    start = 0
    for group_size, group_name in zip(base_group_sizes, group_names):
        end = start + int(group_size)
        group_labels = labels_array[:, start:end].copy()
        if dataset_name == 'RAPOV':
            sentences = processor(group_labels, group_name)
        else:
            sentences = processor(group_labels, group_name, 'base')
        group_sentences.append(sum(sentences, []))
        start = end
    return group_sentences
