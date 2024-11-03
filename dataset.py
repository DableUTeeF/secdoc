from torch.utils.data import Dataset
import os
import glob
import pandas as pd
import numpy as np


class TXTData(Dataset):
    def __init__(self, directory, labels):
        df = pd.read_excel(labels)
        self.data = []
        for i, row in df.iterrows():
            if row['Filename'].startswith('III'):
                continue
            fol = f'{row["Year"]}_Q{row["Quarter"]}'
            length = len(os.listdir(os.path.join(directory, fol, row['Filename'])))
            text = ''
            for j in range(length):
                with open(os.path.join(directory, fol, row['Filename'], f'{j}.txt')) as wr:
                    text += wr.read()
                    text += '\n'
            self.data.append((text, row))
        # self.qual_map = {
        #     'A': np.array([1, 0, 0, 0]),
        #     'B': np.array([0, 1, 0, 0]),
        #     'C': np.array([0, 0, 1, 0]),
        #     'D': np.array([1, 0, 0, 1]),
        # }
        self.qual_map = {
            'A': 0,
            'B': 1,
            'C': 2,
            'D': 3,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        txt = data[0]
        row = data[1]
        binaries = row.values[4:]
        bin1 = binaries[:5]
        a1 = binaries[5] if binaries[5] in ['Y', 'N'] else binaries[7]
        a2 = binaries[6] if binaries[6] in ['Y', 'N'] else binaries[8]
        bin2 = np.stack((a1, a2))
        bin3 = binaries[-2:]
        binaries = np.concatenate((bin1, bin2, bin3)) == 'Y'
        qual = self.qual_map[row.Report_quality]
        return txt, qual, binaries.tolist()


if __name__ == '__main__':
    x = TXTData('/media/palm/Data/กลต/ocr/', '/media/palm/Data/กลต/Dataset/Training_dataset.xlsx')
    x[0]
