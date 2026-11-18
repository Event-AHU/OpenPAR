# Third-party notices

## OpenAI CLIP

The `clip/` implementation and tokenizer are based on [OpenAI CLIP](https://github.com/openai/CLIP).

Copyright (c) 2021 OpenAI. The MIT license is reproduced in [licenses/CLIP-MIT.txt](licenses/CLIP-MIT.txt).

The visual encoder also incorporates the attention design of [CLIP Surgery](https://github.com/xmed-lab/CLIP_Surgery).

## Logging utilities

`log_untils.py` contains logging utilities adapted and simplified from the supplied ByteDance utilities, which reference the [DETR utilities](https://github.com/facebookresearch/detr/blob/master/util/misc.py).

Copyright (c) ByteDance, Inc. and its affiliates. All rights reserved.

The original source notice refers to the LICENSE file of its source tree. The Apache 2.0 license used by [ByteDance iBOT](https://github.com/bytedance/ibot) is included in [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt).

## Learning-rate schedulers

The files in `solver/` are adapted from Ross Wightman's [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models).

Copyright 2020 Ross Wightman. The Apache 2.0 license is reproduced in [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt).

## Recover and Match (RAM)

The Sinkhorn iteration in `models/ot_solver.py` is adapted from [RAM](https://github.com/EricTan7/RAM), which credits [PythonOT/POT](https://github.com/PythonOT/POT). The multi-matching contrastive objective also follows RAM.

The SRENet implementation retains the full-model path and adapts the training orchestration, retrieval batching and dataset paths for this release.
