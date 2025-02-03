import json, os, textdistance, argparse
import numpy as np
from rouge_score import rouge_scorer
from tqdm import tqdm

def get_data(file : str):
    with open(file, 'r') as j:
         data = json.loads(j.read())
    return data

def mean_perf(gen, ref : str, metric : str = 'ncd', encode : bool = False):
    rs = []
    for g in gen:
        if metric == 'levenshtein' :
            rs.append(textdistance.levenshtein.normalized_similarity(g, ref))
        elif metric == 'cosine':
            rs.append(textdistance.cosine.normalized_similarity(g, ref))
        elif metric == 'ncd':
            rs.append(textdistance.zlib_ncd.normalized_similarity(g, ref))
        elif metric == 'rouge':
            scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
            scores = scorer.score(ref, g)
            rs.append(scores['rouge1'].precision)
    return np.mean(np.asarray(rs))

def bitflip_eval(data, metric : str = 'ncd'):
    ref, rs = data['ref'] if 'ref' in data.keys() else data['bitflip'][0]['ref'], []
    for exp in data['bitflip'] :
        gen = [el.replace(exp['prompt'], '') for el in exp['gen']]
        rs.append(mean_perf(gen, ref, metric))
    return rs

def summary_eval(data, metric : str = 'rouge'):
    ref, rs = data['ref'] if 'ref' in data.keys() else data['bitflip'][0]['gen'][0], []
    for exp in data['bitflip'] :
        gen = [el.replace(exp['prompt'], '') for el in exp['gen']]
        rs.append(mean_perf(gen, ref, metric))
    return rs

def sensitivity_eval(X):
    drop = []
    for idx in range(len(X)-1):
        drop.append(abs(X[idx] - X[idx + 1])) 
    return np.max(drop)

def evaluation(folder : str, threshold : float, task : str, metric : str, output : str):
    files = [pos_json for pos_json in os.listdir(folder) if pos_json.endswith('.json')]
    result = []
    print(f"Evaluation in progress ...\nNumber of files : {len(files)}")
    for file in tqdm(files):
        with open(f'./{folder}/{file}', 'r') as j:
            data = json.loads(j.read())
        eval = bitflip_eval(data, metric) if task == 'completion' else summary_eval(data, metric)
        result.append({"file" : file, "sensitivity" : sensitivity_eval(eval), "memorized" : True if sensitivity_eval(eval) > threshold else False})

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w') as f:
        json.dump(result, f)
    print("Evaluation finished")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--task", type=str, required=False, default="completion")
    parser.add_argument("--metric", type=str, required=False, default="ncd")
    parser.add_argument("--threshold", type=int, required=False, default=0)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()
    folder, task, metric, threshold, out = args.folder, args.task, args.metric, args.threshold, args.output

    evaluation(folder, threshold, task, metric, out)