import tiktoken
import random
from tqdm import tqdm
from random import shuffle
import argparse
import json
import os

class BitFlip:
    def __init__(self, input):
        self.input = input
        self.enc = tiktoken.encoding_for_model("gpt-4o")
        self.tokens = self.enc.encode(input)
        self.bin = []
        for tok in self.tokens:
            self.bin.append(format(tok, 'b'))

    def bit_flip(self, bit):
        return 0 if bit == '1' else 1

    def flip(self, prob : list[int]):
        bin_flip = []
        for tok in self.bin:
            n_tok = -1
            while (n_tok > self.enc.max_token_value) or (n_tok < 0):
                rd_ = random.choices([0, 1], weights = prob, k = len(tok))
                fl = ''
                for idx, b in enumerate(rd_):
                    fl += str(tok[idx]) if b == 0 else str(self.bit_flip(tok[idx]))
                n_tok = int(fl, 2)
            bin_flip.append(n_tok)
        return bin_flip, self.enc.decode(bin_flip)
    

class Truncation:
    def __init__(self, input : str, code : bool = False) -> None:
        self.input = input
        self.code = code

    def from_end(self, tr : int) -> tuple[str, str]:
        """
            This function allow to truncate the text from the end.
            e.g : This dog is lazy -> This dog [MASK] with tr = 50
            @tr -> truncation rate in %
        """
        words = self.input.splitlines() if self.code else self.input.split()
        lim = len(words) - round(len(words)*tr/100)
        prompt = '\n'.join(words[0:lim]) + '\n[MASK]' if self.code else ' '.join(words[0:lim]) + ' [MASK]'
        ref = '\n'.join(words[lim:]) if self.code else ' '.join(words[lim:])
        return prompt, ref
    
    def from_mid(self, tr : int) -> tuple[str, str]:
        """
            This function allow to truncate the text from the middle.
            e.g : This dog is lazy -> This [MASK] lazy with tr = 50
            @tr -> truncation rate in %
        """
        words = self.input.splitlines() if self.code else self.input.split()
        lim = round(len(words)*tr/100)
        start, end = int((len(words) - lim) / 2), int((len(words) + lim)/2)
        prompt = '\n'.join(words[0:start]) + '\n[MASK]\n' + '\n'.join(words[end:]) if self.code else ' '.join(words[0:start]) + ' [MASK] ' + ' '.join(words[end:])
        ref = '\n'.join(words[start:end]) if self.code else ' '.join(words[start:end])
        return prompt, ref
    
    def from_start(self, tr : int) -> tuple[str, str]:
        """
            This function allow to truncate the text from the start.
            e.g : This dog is lazy -> [MASK] is lazy with tr = 50
            @tr -> truncation rate in %
        """
        words = self.input.splitlines() if self.code else self.input.split()
        lim = round(len(words)*tr/100)
        prompt = '[MASK]\n' + '\n'.join(words[lim:]) if self.code else '[MASK] ' + ' '.join(words[lim:])
        ref = '\n'.join(words[0:lim]) if self.code else ' '.join(words[0:lim])
        return prompt, ref
    
    def random_token(self, tr : int) -> tuple[str, str]:
        words = self.input.split(' ')
        idx_, rm_words = list(range(len(words))), round(len(words)*tr/100)
        shuffle(idx_)


def perturbation(file: str, out: str, p_type: str = "default", s_ratio: int = 20, b_max: int = 5):
    with open(file, 'r') as j:
        texts = json.loads(j.read())
    
    print('Pertubation process ...')
    if(p_type == "default"):
        for idx, txt in tqdm(enumerate(texts)):
            # Truncate X% of the given text -> prefix (100-X) - suffixe (X)
            data, bf, trunc = {}, [], Truncation(txt)
            prt, data['ref'] = trunc.from_end(s_ratio)
            data['text'] = txt

            # Progressively alter the prefix
            for i in range(b_max):
                flip, bf_ = BitFlip(prt.replace("[MASK]", "")), {}
                tok_, txt_ = flip.flip([1000-i, i])
                bf_['noise'], bf_['prompt'] = i/10, txt_ + "[MASK]"
                bf.append(bf_)
            
            data['bitflip'] = bf
            os.makedirs(os.path.dirname(f'./{out}/sample_{idx}.json'), exist_ok=True)
            with open(f'./{out}/sample_{idx}.json', 'w') as f:
                json.dump(data, f)
    
    elif(p_type == "bitflip"):
        for idx, txt in tqdm(enumerate(texts)):
            data, bf = {'text', txt}, []
        
        for i in range(b_max):
            flip, bf_ = BitFlip(txt), {}
            tok_, txt_ = flip.flip([1000-i, i])
            bf_['noise'], bf_['prompt'] = i/10, txt_
            bf.append(bf_)
            
            data['bitflip'] = bf
            os.makedirs(os.path.dirname(f'./{out}/sample_{idx}.json'), exist_ok=True)
            with open(f'./{out}/sample_{idx}.json', 'w') as f:
                json.dump(data, f)

    print('Pertubation process end.')
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--type", type=str, required=False, default="default")
    parser.add_argument("--split", type=int, required=False, default=20)
    parser.add_argument("--bitflip_max", type=int, required=False, default=5)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()
    file, p_type, s_ratio, b_max, out = args.file, args.type, args.split, args.bitflip_max, args.output

    perturbation(file, out, p_type, s_ratio, b_max)
