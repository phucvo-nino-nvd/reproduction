from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests
import json


RESULT_FILE_PATH = '/lfs/skampere1/0/truongvo/projects/icd-agent/experiments/mas_icd/SpatialMQA/output/'
FILE_PATH = '/lfs/skampere1/0/truongvo/projects/icd-agent/experiments/mas_icd/SpatialMQA/Dataset/dataset/'
DEVICE_INDEX = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl", revision="a24b501a4104cd714544c3d354873d44f6ceabba").to(DEVICE_INDEX)
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl", revision="a24b501a4104cd714544c3d354873d44f6ceabba")

image_dir = '/lfs/skampere1/0/truongvo/projects/icd-agent/experiments/mas_icd/SpatialMQA/Dataset/COCO2017/test2017/'
count = 0
right_count = 0
with open(f'{FILE_PATH}test.jsonl', 'r', encoding="utf-8") as f,open(f'{RESULT_FILE_PATH}instrutblip_20240601.jsonl', 'w+', encoding="utf-8") as fout:
    for id, line in enumerate(f):
        data = json.loads(line)
        question = data['question']
        options = data['options']
        image_name = data['image']
        image_filepath = image_dir + image_name
        image = Image.open(image_filepath)
        question = f'You are currently a senior expert in spatial relation reasoning. \n Given an Image, a Question and Options, your task is to answer the correct spatial relation. Note that you only need to choose one option from the all options without explaining any reason. \n Input: Image: <image>, Question: {question}, Options: {"; ".join(options)}. \n Output:'
        inputs = processor(images=image, text=question, return_tensors="pt").to(DEVICE_INDEX)
        outputs = model.generate(
                **inputs,
                do_sample=False,
                num_beams=5,
                max_length=256,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1,
        )
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip().rstrip('.')
        print(generated_text)
        output = generated_text
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
