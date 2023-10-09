from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from tqdm.auto import tqdm
import json
import multiprocessing

parser = ArgumentParser()
parser.add_argument("--model_path", type=str, default="./Qwen-14B-Chat")
parser.add_argument("--top_p", type=float, default=0.9)
parser.add_argument("--temperature", type=float, default=0.1)
parser.add_argument("--max_length", type=int, default=2048)
parser.add_argument("--data_path", nargs="+", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--gpu_ids", type=str, default="0,1,2,3,4,5,6,7")
parser.add_argument("--max_sample", type=int, default=-1)
args = parser.parse_args()

N_GPU = len(args.gpu_ids.split(","))


def load_model(device_id: int):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, device_map=f"cuda:{device_id}"
    ).eval()

    model.generation_config = GenerationConfig.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    model.generation_config.top_p = args.top_p
    model.generation_config.temperature = args.temperature
    model.generation_config.max_length = args.max_length

    return model


def load_data_set():
    data_set = []
    for data_path in args.data_path:
        with open(data_path, "r", encoding="utf-8") as f:
            data_set += json.load(f)

    if args.max_sample > 0:
        data_set = data_set[: args.max_sample]

    return data_set


def worker(resource):
    device_id, data = resource
    model = load_model(device_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    for item in tqdm(data):
        instruction = item["conversations"][0]["value"]

        prompt = instruction

        response, _ = model.chat(tokenizer, prompt, history=None)

        if response.startswith('"'):
            response = response[1:]
        if response.endswith('"'):
            response = response[:-1]

        # important
        item["conversations"][1]["value"] = response

    print(f"Process {device_id} finished")

    return data


def main():
    data_set = load_data_set()
    N = len(data_set)

    device_ids = list(range(N_GPU))
    data_chunks = [
        data_set[i * N // N_GPU : (i + 1) * N // N_GPU] for i in range(N_GPU)
    ]
    resource = list(zip(device_ids, data_chunks))

    pool = multiprocessing.Pool(processes=N_GPU)
    res = pool.map(worker, resource)
    pool.close()
    pool.join()

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
