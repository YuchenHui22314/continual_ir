from typing import List
import multiprocessing
import os
from vllm import LLM, SamplingParams

def _launch_llm_on_gpu(
    gpu_id: int,
    prompts_subset: List[str],
    model_path: str,
    tensor_parallel_size: int,
    max_model_len: int,
    temperature: float,
    max_tokens: int,
    return_queue: multiprocessing.Queue
):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(os.environ["CUDA_VISIBLE_DEVICES"])

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens
    )

    outputs = llm.generate(prompts_subset, sampling_params)
    texts = [output.outputs[0].text for output in outputs]

    # print the prompt and text pair
    for prompt, text in zip(prompts_subset, texts):
        print("==========================")
        print(f"Prompt: {prompt}\nText: {text}\n")

    return_queue.put((gpu_id, texts))

def generate_with_multi_gpu_vllm(
    prompts: List[str],
    num_gpus: int,
    model_path: str = "path_to_a_hugging_face_snapshot",
    temperature: float = 0.7,
    max_tokens: int = 2500,
    max_model_len: int = 20000,
    tensor_parallel_size: int = 1
) -> List[str]:
    """
    Generates responses for a list of prompts using multiple GPUs in parallel via vLLM,
    updating a progress bar for each completed prompt (fine-grained).
    """

    chunk_size = (len(prompts) + num_gpus - 1) // num_gpus
    prompt_chunks = [prompts[i * chunk_size:(i + 1) * chunk_size] for i in range(num_gpus)]

    procs = []
    manager = multiprocessing.Manager()
    return_queue = manager.Queue()

    for i in range(num_gpus):
        chunk = prompt_chunks[i]
        if not chunk:
            continue

        os.environ["CUDA_VISIBLE_DEVICES"] = str(i)
        print(os.environ["CUDA_VISIBLE_DEVICES"])

        p = multiprocessing.Process(
            target=_launch_llm_on_gpu,
            args=(i, chunk, model_path, tensor_parallel_size, max_model_len, temperature, max_tokens, return_queue)
        )
        p.start()
        procs.append(p)


    for p in procs:
        p.join()

     # Collect and sort results by GPU ID to maintain order
    outputs_by_gpu = {}
    while not return_queue.empty():
        gpu_id, texts = return_queue.get()
        outputs_by_gpu[gpu_id] = texts

    # Merge outputs in original prompt order
    outputs = []
    for i in range(num_gpus):
        if i in outputs_by_gpu:
            outputs.extend(outputs_by_gpu[i])

    return outputs