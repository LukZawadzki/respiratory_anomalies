from torch.utils.data import Dataset, DataLoader

import os
import pandas as pd

class RespiratoryDataset(Dataset):
    def __init__(self, test=False):

        self.transforms = None
        self.recordings = []
        self.data = pd.DataFrame(columns=[
            'recording_id',
            'patient number',
            'recording index',
            'chest location',
            'acquisition mode',
            'recording equipment',
            # 'crackles',
            # 'wheezes'
        ])
        if not test:
            self.diagnosis = self._read_diagnosis('ICBHI_Challenge_diagnosis.txt')

        data_path = 'ICBHI_final_database'
        file_list = os.listdir(data_path)
        for file in file_list:
            file_name = file[:-3]
            if f"{file_name}.txt" in file_list and f"{file_name}.wav" in file_list:
                self.recordings.append(self._read_wav(os.path.join(data_path, f"{file_name}.wav")))
                self.data.loc[len(self.data.index)] = self._parse_file_name(file_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data.iloc[idx]
        recording = self.recordings[idx]
        diagnosis = self.diagnosis[self.diagnosis['patient number'] == data['patient number']]['diagnosis']
        return data, recording, diagnosis

    def _parse_file_name(self, file_name):
        columns_keys = [
            'patient number',
            'recording index',
            'chest location',
            'acquisition mode',
            'recording equipment'
        ]
        values = file_name.split('_')
        return {key: value for key, value in zip(columns_keys, values)}

    def _read_wav(self, path):
        ## TODO - wczytywanie i przetwarzanie nagran
        pass

    def _read_txt(self):
        ## TODO - wczytywanie i przetwarzanie danych o cyklach oddechowych obecnych w narganiu
        pass

    def _read_diagnosis(self, diagnosis_file):
        df_diagnosis = pd.DataFrame(columns=['patient number', 'diagnosis'])
        with open(diagnosis_file, 'r') as file:
            for line in file.readlines():
                idx, diagnosis = line.split('\t')
                new_row = {'patient number': idx, 'diagnosis': diagnosis[:-2]}
                df_diagnosis.loc[len(df_diagnosis.index)] = new_row
        return df_diagnosis