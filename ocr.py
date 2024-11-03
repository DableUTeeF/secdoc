import os
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
import pandas as pd
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('year', type=str)
    parser.add_argument('quarter', type=str)
    parser.add_argument('file', type=str)
    args = parser.parse_args()

    src = '/project/lt200203-aimedi/palm/sec_doc_sum/data/images/'
    fdst = '/project/lt200203-aimedi/palm/sec_doc_sum/data/features/'
    odst = '/project/lt200203-aimedi/palm/sec_doc_sum/data/ocr/'

    model_id = "/project/lt200203-aimedi/palm/huggingface/Llama-3.2-11B-Vision-Instruct"
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    folder = f'{args.year}_Q{args.quarter}'
    os.makedirs(os.path.join(fdst, folder, args.file), exist_ok=True)
    os.makedirs(os.path.join(odst, folder, args.file), exist_ok=True)
    for file in os.listdir(os.path.join(src, folder, args.file)):
        image = Image.open(os.path.join(src, folder, args.file, file))
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "อ่านข้อความจากรูป"},
            ]},
        ]
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)
        output = model.generate(**inputs, max_new_tokens=1024)
        with open(os.path.join(odst, folder, args.file, file.replace('jpg', 'txt')), 'w') as wr:
            wr.write(processor.decode(output[0])[len(input_text)-1:])
        with torch.no_grad():
            y = model(**inputs, output_hidden_states=True, return_dict=True)
        torch.save(
            y['hidden_states'],
            os.path.join(fdst, folder, args.file, file.replace('jpg', 'pth'))
        )
