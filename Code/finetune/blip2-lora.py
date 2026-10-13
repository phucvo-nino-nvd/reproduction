import os

import requests
from transformers import BlipProcessor, BlipForQuestionAnswering
from transformers import Blip2Processor, AutoProcessor, Blip2ForConditionalGeneration, \
    AutoModelForVisualQuestionAnswering
from datasets import load_dataset
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
from peft import LoraConfig, get_peft_model

model_dir = 'Salesforce/'
processor = Blip2Processor.from_pretrained(f"{model_dir}blip2-opt-2.7b", revision="51572668da0eb669e01a189dc22abe6088589a24")
image_dir = '/lfs/skampere1/0/truongvo/projects/icd-agent/experiments/mas_icd/SpatialMQA/Dataset/COCO2017/test2017/'

train_ds = load_dataset("json", data_files="/lfs/skampere1/0/truongvo/projects/icd-agent/experiments/mas_icd/SpatialMQA/Dataset/blip2_train/train_3780.jsonl",
                        split="train[:100%]")
eval_ds = load_dataset("json", data_files="/lfs/skampere1/0/truongvo/projects/icd-agent/experiments/mas_icd/SpatialMQA/Dataset/blip2_train/dev_536.jsonl",
                       split="train[:100%]")
print("Training sets: {} - Validating set: {}".format(len(train_ds), len(eval_ds)))

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = Blip2ForConditionalGeneration.from_pretrained(f"{model_dir}blip2-opt-2.7b", revision="51572668da0eb669e01a189dc22abe6088589a24", device_map="cuda:0",
                                                      load_in_8bit=True)

# Let's define the LoraConfig
config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj"]
)

model = get_peft_model(model, config).to(device)

torch.cuda.empty_cache()
torch.manual_seed(42)


class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # get image + text
        question = self.dataset[idx]['question']
        answer = str(self.dataset[idx]['answer'])
        image_id = self.dataset[idx]['image']
        image_path = f"/lfs/skampere1/0/truongvo/projects/icd-agent/experiments/mas_icd/SpatialMQA/Dataset/COCO2017/test2017/{image_id}"
        image = Image.open(image_path).convert("RGB")

        encoding = self.processor(images=image, text=f"{question} {answer}\n", padding="max_length",
                                  max_length=128, truncation=True, return_tensors="pt")
        question_len = len(self.processor.tokenizer(question).input_ids)
        labels = encoding["input_ids"].clone()
        labels[:, :question_len] = -100
        labels[encoding["attention_mask"] == 0] = -100
        encoding["labels"] = labels

        # remove batch dimension
        for k, v in encoding.items():  encoding[k] = v.squeeze()
        return encoding


train_dataset = ImageCaptioningDataset(dataset=train_ds, processor=processor)
valid_dataset = ImageCaptioningDataset(dataset=eval_ds, processor=processor)

batch_size = 8
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1, verbose=False)

cal_num = 1
num_epochs = 30
patience = 5
min_eval_loss = float("inf")
early_stopping_hook = 0
tracking_information = []
scaler = torch.cuda.amp.GradScaler()
criterion = nn.CrossEntropyLoss(ignore_index=1)

for epoch in range(num_epochs):
    epoch_loss = 0
    cal_loss = 0
    model.train()
    for idx, batch in zip(tqdm(range(len(train_dataloader)), desc='Training batch: ...'), train_dataloader):
        input_ids = batch.pop('input_ids').to(device)
        pixel_values = batch.pop('pixel_values').to(device)
        attention_mask = batch.pop('attention_mask').to(device)
        labels = batch.pop('labels').to(device)

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = model(input_ids=input_ids,
                         pixel_values=pixel_values,
                         attention_mask=attention_mask,
                         labels=labels).loss
        epoch_loss += loss.item()
        optimizer.zero_grad()

        cal_loss += loss
        if (idx + 1) % cal_num == 0 or idx == len(train_dataloader) - 1:
            if (idx + 1) % cal_num == 0:
                cal_loss = cal_loss / cal_num
            else:
                cal_loss = cal_loss / ((idx + 1) % cal_num)
            print('loss:', cal_loss)
            scaler.scale(cal_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            cal_loss = 0

    model.eval()
    eval_loss = 0
    for idx, batch in zip(tqdm(range(len(valid_dataloader)), desc='Validating batch: ...'), valid_dataloader):
        input_ids = batch.pop('input_ids').to(device)
        pixel_values = batch.pop('pixel_values').to(device)
        attention_mask = batch.pop('attention_mask').to(device)
        labels = batch.pop('labels').to(device)
        # print('labels:',labels)
        # outputs = model(input_ids=input_ids,
        #             pixel_values=pixel_values,
        #             attention_mask=attention_mask,
        #             labels=labels)
        # loss = outputs.loss

        # labels_mask = batch.pop('labels_mask').to(device)
        loss = model(input_ids=input_ids,
                     pixel_values=pixel_values,
                     attention_mask=attention_mask,
                     labels=labels).loss

        eval_loss += loss.item()
        # break

    # break

    tracking_information.append(
        (epoch_loss / len(train_dataloader), eval_loss / len(valid_dataloader), optimizer.param_groups[0]["lr"]))
    print("Epoch: {} - Training loss: {} - Eval Loss: {} - LR: {}".format(epoch + 1, epoch_loss / len(train_dataloader),
                                                                          eval_loss / len(valid_dataloader),
                                                                          optimizer.param_groups[0]["lr"]))
    scheduler.step()
    if eval_loss < min_eval_loss:
        model.save_pretrained(f"/lfs/skampere1/0/truongvo/projects/icd-agent/experiments/mas_icd/SpatialMQA/finetune_models/models_arg/blip2_lora_20240531/epoch-{epoch}",
                              from_pt=True)
        print(f"Saved model to /lfs/skampere1/0/truongvo/projects/icd-agent/experiments/mas_icd/SpatialMQA/finetune_models/models_arg/blip2_lora_20240531")
        min_eval_loss = eval_loss
        early_stopping_hook = 0
    else:
        early_stopping_hook += 1
        if early_stopping_hook > patience:
            break

pickle.dump(tracking_information, open("tracking_information.pkl", "wb"))
print("The finetuning process has done!")
