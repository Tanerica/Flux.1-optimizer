from huggingface_hub import login
login(token="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")
import torch
from diffusers import FluxTransformer2DModel
import torch.utils.benchmark as benchmark
from torchao.quantization import quantize_, int8_weight_only
from torchao.utils import unwrap_tensor_subclass
import torch._inductor

torch._inductor.config.mixed_mm_choice = "triton"


def get_example_inputs():
    example_inputs = torch.load("serialized_inputs.pt", weights_only=True)
    example_inputs = {k: v.to("cuda") for k, v in example_inputs.items()}
    example_inputs.update({"joint_attention_kwargs": None, "return_dict": False})
    return example_inputs


def benchmark_fn(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "f": f},
        num_threads=torch.get_num_threads(),
    )
    return f"{(t0.blocked_autorange().mean):.3f}"


@torch.no_grad()
def load_model():
    model = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", subfolder="transformer", torch_dtype=torch.bfloat16
    ).to("cuda")
    return model


def aot_compile(name, model, **sample_kwargs):
    path = f"./{name}.pt2"
    options = {
        "max_autotune": True,
        "triton.cudagraphs": True,
    }
    return torch._inductor.aoti_compile_and_package(
        torch.export.export(model, (), sample_kwargs),
        (),
        sample_kwargs,
        package_path=path,
        inductor_configs=options,
    )


def aot_load(path):
    return torch._inductor.aoti_load_package(path)


@torch.no_grad()
def f(model, **kwargs):
    return model(**kwargs)


if __name__ == "__main__":
    model = load_model()
    quantize_(model, int8_weight_only())
    inputs1 = get_example_inputs()
    unwrap_tensor_subclass(model)

    path = aot_compile("bs_1_1024", model, **inputs1)
    print(f"AoT compiled path {path}")

    compiled_func = aot_load(path)
    print(f"{compiled_func(**inputs1)[0].shape=}")

    for _ in range(5):
        _ = compiled_func(**inputs1)[0]

    time = benchmark_fn(f, compiled_func, **inputs1)
    print(f"{time=} seconds.")