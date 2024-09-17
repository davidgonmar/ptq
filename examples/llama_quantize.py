from ptq.quantize import quantize_llama, QuantizationTime
from transformers import LlamaForCausalLM, AutoTokenizer
import torch
import datasets
import tqdm
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tiny_llama = LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama_v1.1").to(device)

dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1")


class Evaluator:
    def __init__(self, dataset, tokenizer, device, n_samples=40):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        self.dataset = tokenizer(
            "\n\n".join(dataset["text"]), return_tensors="pt"
        ).input_ids.to(device)

        self.n_samples = n_samples

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        nlls = []
        for i in tqdm.tqdm(range(self.n_samples), desc="Evaluating..."):
            batch = self.dataset[:, (i * 2048) : ((i + 1) * 2048)].to(model.device)
            with torch.no_grad():
                lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = self.dataset[:, (i * 2048) : ((i + 1) * 2048)][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * 2048
            nlls.append(neg_log_likelihood)

        return torch.exp(torch.stack(nlls).sum() / (self.n_samples * 2048))


evaluator = Evaluator(dataset, tokenizer, device)

nll = evaluator.evaluate(tiny_llama)

quantized_llama_offline = quantize_llama(
    tiny_llama, tokenizer, n_bits=8, quant_time=QuantizationTime.OFFLINE
)

nll_quantized_offline = evaluator.evaluate(quantized_llama_offline)

del tiny_llama
del quantized_llama_offline

quantized_llama_online = quantize_llama(
    LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama_v1.1").to(device),
    tokenizer,
    n_bits=8,
    quant_time=QuantizationTime.ONLINE,
)

nll_quantized_online = evaluator.evaluate(quantized_llama_online)

print(f"TinyLlama ppl: {nll}")
print(f"Quantized TinyLlama ppl with offline activation quant: {nll_quantized_offline}")
print(f"Quantized TinyLlama ppl with online activation quant: {nll_quantized_online}")
