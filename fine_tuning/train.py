import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
import wandb
from tqdm import tqdm
import bitsandbytes as bnb
from datasets import load_dataset
from transformers import AutoTokenizer, GPTNeoXForCausalLM

torch.set_float32_matmul_precision('high')

class Trainer:
    def __init__(self, num_epochs=100, batch_size=4, max_seq_length=512):
        self.max_seq_length = max_seq_length
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.grad_accumulation_steps = 64
        self.step = 0

        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.dataset = self.prepare_dataset()
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        self.model = GPTNeoXForCausalLM.from_pretrained(
            "EleutherAI/pythia-410m",
            torch_dtype=torch.float16,
        ).cuda()

        self.opt = bnb.optim.Lion(
            params=self.model.parameters(),
            lr=1e-5,
            weight_decay=1e-1,
            betas=(0.9, 0.95),
            optim_bits=8,
        )

        self.loss_fn = nn.CrossEntropyLoss()

        self.show_params()

    def prepare_dataset(self):
        """Load and tokenize the dataset."""
        with open('./../../data/pile.json', 'r') as j:
            dataset = json.loads(j.read())

        def tokenize_function(examples):
            return self.tokenizer(
                examples,
                truncation=True,
                padding="max_length",
                max_length=self.max_seq_length,
            )

        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, remove_columns=["text"]
        )
        tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        return tokenized_dataset

    def show_params(self):
        """Display model parameter counts."""
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {params}")

    def train_step(self, batch):
        """Perform a single training step."""
        input_ids, attention_mask = batch["input_ids"].cuda(), batch["attention_mask"].cuda()
        labels = input_ids.clone()

        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        return loss

    def train(self):
        """Train the model."""
        wandb.init(project="Pythia_ft", entity="#")
        self.step = 0 
    
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            prog = tqdm(self.loader)
            self.opt.zero_grad()
    
            for i, batch in enumerate(prog):
                self.step += 1
                loss = self.train_step(batch)
                prog.set_description(f"Loss: {loss.item():.4f}")
                wandb.log({"loss": loss.item()}, step=self.step)
    
                if (i + 1) % self.grad_accumulation_steps == 0:
                    self.opt.step()
                    self.opt.zero_grad()
    
            self.save_checkpoint(epoch)
            wandb.log({"epoch_loss": loss.item()}, step=self.step)
    

    def save_checkpoint(self, epoch):
        """Save model checkpoint."""
        self.model.save_pretrained(f"./models/pythia_410m_ft_{epoch + 1}", max_shard_size="500MB")
        wandb.log({"epoch_loss": epoch}, step=epoch)


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()

