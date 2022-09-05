import os
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
        warmup_steps = int(args.warmup_ratio * self._total_steps)
        self._optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self._scheduler = LinearWarmupScheduler(self._optimizer, self._total_steps, warmup_steps)

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
            "batch_size": self._args.batch_size, 
            "learning_rate": self._args.learning_rate, 
            "weight_decay": self._args.weight_decay, 
        }
        wandb.config.update(training_args)

        loss_kl = nn.KLDivLoss(reduction='batchmean')

        print("\nTraining")
        train_data_iterator = iter(self._train_dataloader)
        for step in tqdm(range(self._total_steps)) :
            try :
                data = next(train_data_iterator)
            except StopIteration :
                train_data_iterator = iter(self._train_dataloader)
                data = next(train_data_iterator)

            self._optimizer.zero_grad()

            img = data["image"].to(self._device)
            batch_size = img.shape[0]
            img_concat = torch.cat([img, img], dim=0)

            daily = data["daily"].to(self._device)
            gender = data["gender"].to(self._device)
            emb = data["embellishment"].to(self._device)

            daily_logit, gender_logit, emb_logit = self._model(img_concat)
            daily_logit1, daily_logit2 = daily_logit[:batch_size], daily_logit[batch_size:]
            gender_logit1, gender_logit2 = gender_logit[:batch_size], gender_logit[batch_size:]
            emb_logit1, emb_logit2 = emb_logit[:batch_size], emb_logit[batch_size:]
            
            daily_loss = self.rdrop_loss(daily_logit1, daily_logit2, daily)
            gender_loss = self.rdrop_loss(gender_logit1, gender_logit2, gender)
            emb_loss = self.rdrop_loss(emb_logit1, emb_logit2, emb)
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
                    path = os.path.join(self._args.save_path, f"model{self._args.num_model}", f"checkpoint-{step}.pt")
                    torch.save(self._model.state_dict(), path)

        wandb.finish()


    def evaluate(self, step) :
        eval_size = len(self._eval_dataloader) * self._args.eval_batch_size
        print("\nEvaluating model at %d step" %step)
        with torch.no_grad() :
            self._model.eval()
            daily_acc, gender_acc, emb_acc = 0.0, 0.0, 0.0

            for eval_data in tqdm(self._eval_dataloader) :

                img = eval_data["image"].to(self._device)
                daily = eval_data["daily"].to(self._device)
                gender = eval_data["gender"].to(self._device)
                emb = eval_data["embellishment"].to(self._device)

                daily_logit, gender_logit, emb_logit = self._model(img)

                daily_acc += self.acc_fn(daily_logit, daily)
                gender_acc += self.acc_fn(gender_logit, gender)
                emb_acc += self.acc_fn(emb_logit, emb)

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

    def loss_fn(self, logit, label) :

        log_softmax = -F.log_softmax(logit, dim=-1)
        loss = log_softmax * label
        loss_per_data = torch.mean(loss, dim=-1)
        mean_loss = torch.mean(loss_per_data)
        return mean_loss

    def acc_fn(self, logit, label) :

        acc = 0.0
        logit = logit.detach().cpu().numpy()
        label = label.detach().cpu().numpy()

        for j in range(len(logit)) :
            if logit[j].argmax() == label[j].argmax() :
                acc += 1.0

        return acc

    def rdrop_loss(self, logit1, logit2, label) :
        ce_loss1 = self.loss_fn(logit1, label)
        ce_loss2 = self.loss_fn(logit2, label)

        kl_loss1 = F.kl_div(F.log_softmax(logit1, dim=-1), F.softmax(logit2, dim=-1), reduction='batchmean')
        kl_loss2 = F.kl_div(F.log_softmax(logit2, dim=-1), F.softmax(logit1, dim=-1), reduction='batchmean')

        loss = (ce_loss1 + ce_loss2) + 0.1 * (kl_loss1 + kl_loss2)
        return loss