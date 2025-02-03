from eval import *
from gen import *
from pertubation import *
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    # Perturbation parameter
    parser.add_argument("--type", type=str, required=False, default="default")
    parser.add_argument("--split", type=int, required=False, default=20)
    parser.add_argument("--bitflip_max", type=int, required=False, default=5)

    # Generation parameter
    parser.add_argument("--model", type=str, required=False, default="EleutherAI/pythia-70m")
    parser.add_argument("--iter", type=int, required=False, default=10)
    parser.add_argument("--tokenizer", type=str, required=False, default="EleutherAI/pythia-70m")

    # Evaluation parameter
    parser.add_argument("--task", type=str, required=False, default="completion")
    parser.add_argument("--metric", type=str, required=False, default="ncd")
    parser.add_argument("--threshold", type=int, required=False, default=0)
    parser.add_argument("--result", type=str, default="./result.json")
    
    args = parser.parse_args()
    file, out, p_type, s_ratio, b_max, = args.file, args.output, args.type, args.split, args.bitflip_max
    model, iter, task, metric, threshold, rs, tokenizer = args.model, args.iter, args.task, args.metric, args.threshold, args.result, args.tokenizer

    perturbation(file, out, p_type, s_ratio, b_max)
    generation(model, out, task, iter, tokenizer)
    evaluation(out, threshold, task, metric, rs)
