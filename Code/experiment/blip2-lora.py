import os
import requests
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from peft import PeftModel, PeftConfig
import torch

RESULT_FILE_PATH = '/lfs/skampere1/0/truongvo/projects/icd-agent/experiments/mas_icd/SpatialMQA/output/'
FILE_PATH = '/lfs/skampere1/0/truongvo/projects/icd-agent/experiments/mas_icd/SpatialMQA/Dataset/dataset/'
DEVICE_INDEX = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_dir = 'Salesforce/'

processor = AutoProcessor.from_pretrained(f"{model_dir}blip2-opt-2.7b", revision="51572668da0eb669e01a189dc22abe6088589a24")
model = Blip2ForConditionalGeneration.from_pretrained(f"{model_dir}blip2-opt-2.7b", revision="51572668da0eb669e01a189dc22abe6088589a24", device_map="cuda:0", load_in_8bit=True)
lora_dir = '/lfs/skampere1/0/truongvo/projects/icd-agent/experiments/mas_icd/SpatialMQA/finetune_models/models_arg/blip2_lora_20240531'
lora_point = f"{lora_dir}/epoch-{max(int(d.split('-')[1]) for d in os.listdir(lora_dir))}"

model = PeftModel.from_pretrained(model, lora_point).to(DEVICE_INDEX)
import json
image_dir = '/lfs/skampere1/0/truongvo/projects/icd-agent/experiments/mas_icd/SpatialMQA/Dataset/COCO2017/test2017/'
count = 0
right_count = 0
with open(f'{FILE_PATH}test.jsonl', 'r', encoding="utf-8") as f,open(f'{RESULT_FILE_PATH}blip2_lora_20240531.jsonl', 'w+', encoding="utf-8") as fout:
    for id, line in enumerate(f):
        data = json.loads(line)
        question = data['question']
        options = data['options']
        image_name = data['image']
        image_filepath = image_dir + image_name
        image = Image.open(image_filepath)
        question = f'You are currently a senior expert in spatial relation reasoning. \n Given an Image, a Question, and Options, your task is to answer the correct spatial relation. Note that you only need to choose one option from all options without explaining any reason. \n Input: Image: <image>, Question: {question}, Options: {"; ".join(options)}. \n Output:'
        inputs = processor(images=image, text=question, return_tensors="pt").to(DEVICE_INDEX)
        predictions = model.generate(**inputs)
        print("predictions: ", predictions)
        output = processor.batch_decode(predictions, skip_special_tokens=True)[0].strip().rstrip('.')
        count += 1
        if len(output) == 0:
            output = '--'
        if output.lower() in data['answer']:
            result_json = {'id':id,'result':1,"output":output.lower(),"answer":data['answer']}
            fout.write(json.dumps(result_json,ensure_ascii=False)+'\n')
            right_count += 1
        elif data['answer'] in output.lower():
            result_json = {'id':id,'result':1,"output":output.lower(),"answer":data['answer']}
            fout.write(json.dumps(result_json,ensure_ascii=False)+'\n')
            right_count += 1
        else:
            result_json = {'id':id,'result':0,"output":output.lower(),"answer":data['answer']}
            fout.write(json.dumps(result_json,ensure_ascii=False)+'\n')
        print(f'{output.lower()}')
        print(f"{data['answer']}")
        print(f'right_count: {right_count}')
        print(f'count: {count}')
        print(f'accuracy: {right_count/count}')

accuracy = right_count/count
print(f'accuracy: {accuracy}')
