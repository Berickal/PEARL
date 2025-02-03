from openai import OpenAI
from tqdm import tqdm
import json, logging, torch, os
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

OPENAI_Key = '#'
LOGGER = logging.getLogger()

class GPT_4o:
    def __init__(self, folder : str, task : str = "completion", iter : int = 10) -> None:
        self.openai = OpenAI(api_key=OPENAI_Key)
        self.task = task
        self.folder = folder
        self.files = [pos_json for pos_json in os.listdir(folder) if pos_json.endswith('.json')]
        self.iter = iter

    def get_instruct(self, param : list) -> str:
        match self.task:
            case "completion" : #Completion_Text
                return f'Finish the text below. You will provide as output the full text for a length of approximately {len(param[0].split())} words.'
            case "completion_code" : #Completion_Code
                return 'Complete the following code by providing as output only the full code.'
            case "summary" : #Summary_Text
                return 'Provide as output only the summary of the provide text in 100 words max.'
            case "summary_code" : # Code_summary:
                return 'Provide a concise, one-paragraph summary explaining the main purpose of the following code.'
            case -1:
                return None

    def gen_single(self, prmpt : str, param):
        instruct = self.get_instruct([param])
        completion = self.openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": instruct},
                {"role": "user", "content": f"{prmpt}" }
            ], temperature= 0.3 )
        return completion.choices[0].message.content
    
    def generation(self):
        print(f"Generation in progress ...\nNumber of files : {len(self.files)}")
        for file in tqdm(self.files):
            with open(f'./{self.folder}/{file}', 'r') as j:
                data = json.loads(j.read())
        
            for idx, prt in enumerate(data["bitflip"]):
                rs = []
                for iter in range(self.iter):
                    if self.task == "completion":
                        rs.append(self.gen_single(prt["prompt"], data["text"]))
                    elif self.task == "summary":
                        rs.append(self.gen_single(prt["prompt"], None))
                data['bitflip'][idx]['gen'] = rs
            with open(f'./{self.folder}/{file}', 'w') as f:
                json.dump(data, f)
        print("Generation finished")



class Pythia:
    def __init__(self, model : str, tokenizer : str, folder : str, iter : int = 10):
        self.model = AutoModelForCausalLM.from_pretrained(model, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.folder = folder
        self.files = [pos_json for pos_json in os.listdir(folder) if pos_json.endswith('.json')]
        self.iter = iter

    def generation(self):
        print(f"Generation in progress ...\nNumber of files : {len(self.files)}")
        for file in tqdm(self.files):
            with open(f'./{self.folder}/{file}', 'r') as j:
                data = json.loads(j.read())
            
            for idx, prt in enumerate(data['bitflip']):
                rs = []
                for iter in range(self.iter):
                    input = self.tokenizer(prt['prompt'].replace('[MASK]', ''), return_tensors="pt").to(self.model.device)
                    output = self.model.generate(**input, max_new_tokens=100, pad_token_id=self.tokenizer.eos_token_id)
                    rs.append(self.tokenizer.decode(output[0], skip_special_tokens=True))
                
                data['bitflip'][idx]['gen'] = rs
            with open(f'./{self.folder}/{file}', 'w') as f:
                json.dump(data, f)
        print("Generation finished")


def generation(model : str, folder : str, task : str = None, iter : int = 10, tokenizer : str = "EleutherAI/pythia-70m"):
    gen = GPT_4o(folder, task, iter) if model == "gpt_4o" else Pythia(model, tokenizer, folder, iter)
    gen.generation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--model", type=str, required=False, default="EleutherAI/pythia-70m")
    parser.add_argument("--tokenizer", type=str, required=False, default="EleutherAI/pythia-70m")
    parser.add_argument("--task", type=str, required=False, default="completion")
    parser.add_argument("--iter", type=int, required=False, default=10)

    args = parser.parse_args()
    folder, model, task, iter = args.folder, args.model, args.task, args.iter

    generation(model, folder, task, iter)