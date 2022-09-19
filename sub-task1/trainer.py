import os
from pickletools import optimize
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch import optim
from models.scheduler import LinearWarmupScheduler
from dotenv import load_dotenv

class Trainer :

    def __init__(self, args, device, model, train_dataloader, eval_dataloader) :
        self._args = args
        self._model = model
        self._device = device
        self._train_dataloader = train_dataloader
        self._eval_dataloader = eval_dataloader

        self._total_steps = len(train_dataloader) * args.epochs if args.max_steps == -1 else args.max_steps
        self._warmup_steps = int(args.warmup_ratio * self._total_steps)
        self._optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self._scheduler = LinearWarmupScheduler(self._optimizer, self._total_steps, self._warmup_steps)

    def train(self,) :

        load_dotenv(dotenv_path="wandb.env")
        WANDB_AUTH_KEY = os.getenv("WANDB_AUTH_KEY")
        wandb.login(key=WANDB_AUTH_KEY)

        name = f"EP:{self._args.epochs}_BS:{self._args.batch_size}_LR:{self._args.learning_rate}_WD:{self._args.weight_decay}"
        wandb.init(
            entity="sangha0411",
            project="fashion-how",
            group=f"sub-task1",
            name=name
        )

        training_args = {"epochs": self._args.epochs, 
            "total_steps" : self._total_steps,
            "warmup_steps" : self._warmup_steps,
            "batch_size": self._args.batch_size, 
            "learning_rate": self._args.learning_rate, 
            "weight_decay": self._args.weight_decay, 
        }
        wandb.config.update(training_args)

        if not os.path.exists(self._args.save_path) :
            os.mkdir(self._args.save_path)
        
        checkpoints_dir_path = os.path.join(self._args.save_path, f"model{self._args.num_model}")
        if not os.path.exists(checkpoints_dir_path) :
            os.mkdir(checkpoints_dir_path)

        print("\nTraining")        
        train_data_iterator = iter(self._train_dataloader)
        for step in tqdm(range(self._total_steps), desc="Training") :
            try :
                data = next(train_data_iterator)
            except StopIteration :
                train_data_iterator = iter(self._train_dataloader)
                data = next(train_data_iterator)

            self._optimizer.zero_grad()

            img = data["image"].to(self._device)
            daily = data["daily"].to(self._device)
            gender = data["gender"].to(self._device)
            emb = data["embellishment"].to(self._device)

            daily_logit, gender_logit, emb_logit = self._model(img)
            
            daily_loss = self.softmax_loss(daily_logit, daily)
            gender_loss = self.softmax_loss(gender_logit, gender)
            emb_loss = self.softmax_loss(emb_logit, emb)

            loss = daily_loss + gender_loss + emb_loss
            loss.backward()
            self._optimizer.step()
            self._scheduler.step()

            if step > 0 and step % self._args.logging_steps == 0 :
                info = {"train/learning_rate" : self._scheduler.get_last_lr()[0], 
                    "train/daily_loss": daily_loss.item(), 
                    "train/gender_loss" : gender_loss.item(), 
                    "train/embellishment_loss": emb_loss.item(),
                    "train/step" : step,
                }
                wandb.log(info)
                print(info)

            if step > 0 and step % self._args.save_steps == 0 :
                if self._args.do_eval :
                    self.evaluate(step)
                
                if self._args.do_eval == False :
                    model_path = f"checkpoint-{step}.pt"
                    self.save(self._model, checkpoints_dir_path, model_path)
                   

        if self._args.do_eval :
            self.evaluate(self._total_steps)

        if self._args.do_eval == False :
            model_path = f"checkpoint-{self._total_steps}.pt"
            self.save(self._model, checkpoints_dir_path, model_path)

        wandb.finish()

    def evaluate(self, step) :
        eval_size = len(self._eval_dataloader) * self._args.eval_batch_size
        print("\nEvaluating model at %d step" %step)
        with torch.no_grad() :
            self._model.eval()
            daily_acc, gender_acc, emb_acc = 0.0, 0.0, 0.0

            for eval_data in tqdm(self._eval_dataloader, desc= "Evaluating") :

                img = eval_data["image"].to(self._device)
                daily = eval_data["daily"].to(self._device)
                gender = eval_data["gender"].to(self._device)
                emb = eval_data["embellishment"].to(self._device)

                daily_logit, gender_logit, emb_logit = self._model(img)

                daily_acc += self.acc(daily_logit, daily)
                gender_acc += self.acc(gender_logit, gender)
                emb_acc += self.acc(emb_logit, emb)

            daily_acc /= eval_size
            gender_acc /= eval_size
            emb_acc /= eval_size

            acc = (daily_acc + gender_acc + emb_acc) / 3
                
            info = {"eval/daily_acc": daily_acc, 
                "eval/gender_acc" : gender_acc, 
                "eval/embellishment_acc": emb_acc,
                "eval/acc" : acc,
                "eval/step" : step,
            }
            wandb.log(info)  
            print(info) 
            self._model.train()

    def softmax_loss(self, logit, label) :
        log_softmax = -F.log_softmax(logit, dim=-1)
        loss = log_softmax * label
        loss_per_data = torch.mean(loss, dim=-1)
        mean_loss = torch.mean(loss_per_data)
        return mean_loss

    def acc(self, logit, label) :
        acc = 0.0
        logit = logit.detach().cpu().numpy()
        label = label.detach().cpu().numpy()

        for j in range(len(logit)) :
            if logit[j].argmax() == label[j].argmax() :
                acc += 1.0

        return acc

    def save(self, model, dir_path, model_name) :
        if not os.path.exists(dir_path) :
            os.mkdir(dir_path)
        model_path = os.path.join(dir_path, model_name)        
        torch.save(model.state_dict(), model_path)
