from torch import nn
from transformers import AutoModel
import math
import torch


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained('/project/lt200203-aimedi/palm/huggingface/wangchanberta-base-att-spm-uncased')
        self.mapool = nn.AdaptiveMaxPool1d(512)

    def forward(self, inputs):
        outputs = None
        l = math.ceil(inputs['input_ids'].size(1) / 256)
        with torch.no_grad():
            for i in range(l):
                input_ids = inputs['input_ids'][:, i * 256:(i + 2) * 256-5]
                attention_mask = inputs['attention_mask'][:, i * 256:(i + 2) * 256-5]
                output = self.encoder(input_ids, attention_mask)
                if outputs is None:
                    outputs = output['last_hidden_state']
                else:
                    outputs = torch.cat((outputs, output['last_hidden_state']), 1)
        outputs = outputs.permute(0, 2, 1)
        outputs = self.mapool(outputs).permute(0, 2, 1)
        return outputs


class Estimater(nn.Module):
    def __init__(self):
        super().__init__()
        # self.encoder = AutoModel.from_pretrained('/project/lt200203-aimedi/palm/huggingface/wangchanberta-base-att-spm-uncased')
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        self.decoder = nn.Linear(768, 768)
        self.gelu = nn.GELU()
        self.adpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, 13)

    def forward(self, outputs):
        reduced = self.decoder(outputs)
        reduced = self.gelu(reduced)
        reduced = self.adpool(reduced).view(-1, 512)
        return self.fc(reduced)


if __name__ == '__main__':
    from transformers import AutoTokenizer
    from dataset import TXTData

    dset = TXTData('/media/palm/Data/กลต/ocr/', '/media/palm/Data/กลต/Dataset/Training_dataset.xlsx')
    tokenizer = AutoTokenizer.from_pretrained('airesearch/wangchanberta-base-att-spm-uncased')
    data = dset[0]
    inputs = tokenizer(data[0], return_tensors='pt')
    model = Estimater()
    model(inputs)
