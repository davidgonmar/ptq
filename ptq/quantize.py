from .symmetric import W8A8Linear, QuantizationMethod, QuantizationType
import transformers
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaAttention
import torch
import datasets
from tqdm import tqdm
import os

QM = QuantizationMethod
QT = QuantizationType


# for quantizing activations, we need to know the range of the activations.
@torch.no_grad()
def get_activation_range(model: torch.nn.Module, tokenizer, ds):
    # caches
    cache_path = "data/activation_cache_{}.pt".format(model.__class__.__name__)
    if os.path.exists(cache_path):
        return torch.load(cache_path)

    absmax_acts_outputs = {}
    absmax_acts_inputs = {}
    device = next(model.parameters()).device

    # for each layer, insert a hook to get the max activation value
    def hook_fn(name):
        def hook(module, input, output):
            if name not in absmax_acts_outputs:
                absmax_acts_outputs[name] = output.abs().max()
            else:
                absmax_acts_outputs[name] = max(
                    absmax_acts_outputs[name], output.abs().max()
                )
            assert len(input) == 1, "Only one input tensor is supported"
            if name not in absmax_acts_inputs:
                absmax_acts_inputs[name] = input[0].abs().max()
            else:
                absmax_acts_inputs[name] = max(
                    absmax_acts_inputs[name], input[0].abs().max()
                )

        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    # run the model on the dataset
    for batch in tqdm(ds, desc="Processing batches"):
        inputs = tokenizer(
            batch["text"], return_tensors="pt", truncation=True
        ).input_ids.to(device)

        model(inputs)

    # remove the hooks
    for hook in hooks:
        hook.remove()

    # save the cache
    torch.save((absmax_acts_inputs, absmax_acts_outputs), cache_path)
    return absmax_acts_inputs, absmax_acts_outputs


@torch.no_grad()
def quantize_llama(
    model: transformers.LlamaForCausalLM,
    tokenizer,
    n_bits: int = 8,
    quant_method: QM = QM.ABSMAX,
    quant_type: QT = QT.PER_TENSOR,
):
    # first we need to determine scales for the weights
    absmax_acts_inputs, absmax_acts_outputs = get_activation_range(
        model,
        tokenizer,
        ds=datasets.load_dataset(
            "Salesforce/wikitext", "wikitext-103-raw-v1", split="train[:100]"
        ),
    )
    qmax = 2 ** (n_bits - 1) - 1
    for name, m in model.named_modules():
        # continue if the module is not a linear layer
        if not isinstance(m, (LlamaMLP, LlamaAttention)):
            continue

        def get_input_scale(absmax):
            return absmax / qmax

        def _quant_linear(module, subname):
            return W8A8Linear.from_linear(
                module,
                quant_type,
                quant_method,
                n_bits,
                get_input_scale(absmax_acts_inputs[name + subname]),
                get_input_scale(absmax_acts_outputs[name + subname]),
            )

        if isinstance(m, LlamaMLP):
            m.gate_proj = _quant_linear(m.gate_proj, ".gate_proj")
            m.gate_down = _quant_linear(m.gate_proj, ".gate_down")
            m.gate_up = _quant_linear(m.gate_proj, ".gate_up")
        elif isinstance(m, LlamaAttention):
            m.q_proj = _quant_linear(m.q_proj, ".q_proj")
            m.k_proj = _quant_linear(m.k_proj, ".k_proj")
            m.v_proj = _quant_linear(m.v_proj, ".v_proj")
            m.o_proj = _quant_linear(m.o_proj, ".o_proj")
        return model
