# Memorization or Interpolation? Detecting LLM Memorization through Input Perturbation Analysis

## Pearl default

```bash
#install requirements
pip install requirements.txt
# Evaluate datasets sensitivity with Pearl
python pearl.py --file data/test.json --output output/test
```
That will considered a completion task by spliting each text in data/test.json into 80% - 20% and progressively alter the input obtained with BitFlip method (from 0% to 5%). Each perturbed input is then submitted to a model (Pythia_70m by default) to generated 10 outputs. The sensitivity is then compute and a report is generate in result.json file.

## GPT_4o

This repository propose an evaluation of GPT_4o following several tasks :
- Text completion task : --task completion
- Code completion task : --task completion_code
- Text summary task : --task summary
- Code description task : --task summary_code

You can run this evaluation withe the following command : 
```bash
# Evaluate GPT_4o
python pearl.py --file data/test.json --output output/test --model gpt_4o --task completion

# You can also controlled the number of output generated per prompt with --iter
python pearl.py --file data/test.json --output output/test --model gpt_4o --task completion --iter 2
```

## Threshold
You can setup the sensitivity threshold with the following command
```bash
# Evaluate datasets sensitivity with Pearl
python pearl.py --file data/test.json --output output/test --threshold 1.1
```


## Independent Tasks

You can perform the different steps of Pearl independantly 

For the pertubation :
```bash
# Evaluate datasets sensitivity with Pearl
python perturbation.py --file data/test.json --output output/test

#With optional options
python perturbation.py --file data/test.json --output output/test --bitflip_max 10 --split 10
```

For the generation :
```bash
# Evaluate datasets sensitivity with Pearl
python gen.py --model EleutherAI/pythia-70m --folder output/test/

# With optional options
python gen.py --model gpt_4o --folder output/test/ --task completion --iter 2
```

For the evaluation :
```bash
# Evaluate datasets sensitivity with Pearl
python evaluation.py --folder output/test/ --output ./result.json

#With optional options
python evaluation.py --folder output/test/ --output ./result.json --threshold 1.1 --task completion --metric ncd
```
