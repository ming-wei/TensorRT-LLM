import argparse
import time

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch import LLM
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm.bindings.executor import KvCacheConfig


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',
                        type=str,
                        default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--tp_size', type=int, default=1)
    parser.add_argument('--enable_overlap_scheduler',
                        default=False,
                        action='store_true')
    parser.add_argument('--enable_chunked_prefill',
                        default=False,
                        action='store_true')
    parser.add_argument('--enable_attention_dp',
                        default=False,
                        action='store_true')
    parser.add_argument('--kv_cache_enable_block_reuse',
                        default=False,
                        action='store_true')
    parser.add_argument('--kv_cache_dtype', type=str, default='auto')
    parser.add_argument('--moe_ep_size', type=int, default=-1)
    parser.add_argument('--moe_tp_size', type=int, default=-1)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    pytorch_config = PyTorchConfig(
        enable_overlap_scheduler=args.enable_overlap_scheduler,
        kv_cache_dtype=args.kv_cache_dtype)

    llm = LLM(model=args.model_dir,
              tensor_parallel_size=args.tp_size,
              max_batch_size=1,
              enable_build_cache=True,
              enable_chunked_prefill=args.enable_chunked_prefill,
              pytorch_backend_config=pytorch_config,
              moe_expert_parallel_size=args.moe_ep_size,
              moe_tensor_parallel_size=args.moe_tp_size,
              enable_attention_dp=args.enable_attention_dp,
              kv_cache_config=KvCacheConfig(
                  enable_block_reuse=args.kv_cache_enable_block_reuse))

    with open("/home/minwei/d/deepseek/prompt_96k_success.txt", "r") as f:
        prompts = [f.read()]

    prompts *= 10

    sampling_params = SamplingParams(max_tokens=32)

    start_time = time.time()  # Add timing
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()  # Add timing
    generation_time = end_time - start_time
    print(f"Generation time: {generation_time:.2f} seconds")

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")

    stats = llm.get_stats(timeout=600)
    print(stats)
    # Get stats asynchronously
    for stat in stats:
        prefill_time = stat.get('prefill_time', 0)
        print(f"Prefill time: {prefill_time} ms")

if __name__ == '__main__':
    main()
