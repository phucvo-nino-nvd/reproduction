from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.tokenization_mplug_owl import MplugOwlTokenizer
from mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
from peft import PeftModel, PeftConfig
import torch
from PIL import Image
import json


FILE_PATH = '/lfs/skampere1/0/truongvo/projects/icd-agent/experiments/mas_icd/SpatialMQA/Dataset/dataset/'
RESULT_FILE_PATH = '/lfs/skampere1/0/truongvo/projects/icd-agent/experiments/mas_icd/SpatialMQA/output/m10/'
pretrained_ckpt = 'MAGAer13/mplug-owl-llama-7b-pt'
device="cuda:0"
peft_model_id = '/lfs/skampere1/0/truongvo/projects/icd-agent/experiments/mas_icd/SpatialMQA/finetune_models/models_arg/mplug_lora_20240530'
model = MplugOwlForConditionalGeneration.from_pretrained(
    pretrained_ckpt,
    torch_dtype=torch.bfloat16,
).to(device)
model = PeftModel.from_pretrained(model, peft_model_id).to(device)
image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
tokenizer = MplugOwlTokenizer.from_pretrained('MAGAer13/mplug-owl-llama-7b-pt')
processor = MplugOwlProcessor(image_processor, tokenizer)

generate_kwargs = {
    'do_sample': True,
    'top_k': 0,
    'max_length': 512,
    'temperature': 0.1
}

image_dir = '/lfs/skampere1/0/truongvo/projects/icd-agent/experiments/mas_icd/SpatialMQA/Dataset/COCO2017/test2017/'
count = 0
right_count = 0
with open(f'{FILE_PATH}test.jsonl', 'r', encoding="utf-8") as f,open(f'{RESULT_FILE_PATH}mplug_lora_0.jsonl', 'w+', encoding="utf-8") as fout:
    for id, line in enumerate(f):
        data = json.loads(line)
        question = data['question']
        options = data['options']
        image_name = data['image']
        question = f'You are currently a senior expert in spatial relation reasoning. \n Given an Image, a Question and Options, your task is to answer the correct spatial relation. Note that you only need to choose one option from the all options without explaining any reason. \n Input: Question: {question}, Options: {"; ".join(options)}. \n Output:'
        if not count:
            print(f'question:{question}')
        image_filepath = image_dir + image_name
        images = [Image.open(image_filepath)]
        prompts = [
            f'The following is a conversation between a curious human and AI assistant.\nHuman: <image>\nHuman: {question}\nAI: '
        ]
        inputs = processor(text=prompts, images=images, return_tensors='pt')
        inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            res = model.generate(**inputs, **generate_kwargs)
        sentence = tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
        print(sentence)
        count += 1
        output = sentence.strip().rstrip('.')
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
