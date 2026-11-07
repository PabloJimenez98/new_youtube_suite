"""The CUDA stack sees this GPU: torch kernels, CT2 types, llama.cpp offload."""

import pytest

pytestmark = pytest.mark.gpu


def test_torch_has_kernels_for_this_gpu(prod_settings) -> None:
    import torch

    major, minor = torch.cuda.get_device_capability(0)
    assert f"sm_{major}{minor}" in torch.cuda.get_arch_list()
    x = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    assert torch.isfinite((x @ x).float()).all()


def test_ctranslate2_supports_the_production_compute_type(prod_settings) -> None:
    import ctranslate2

    assert prod_settings.whisper_device == "cuda", "production Whisper should run on the GPU"
    assert prod_settings.whisper_compute_type in ctranslate2.get_supported_compute_types("cuda")


def test_llama_cpp_is_a_cuda_build(prod_settings) -> None:
    from app.llm import cuda_libs

    cuda_libs.preload()
    import llama_cpp

    assert llama_cpp.llama_supports_gpu_offload(), "llama-cpp-python was built without CUDA (see README)"
