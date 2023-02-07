import itertools
from utils import clean_text

import os
import pandas as pd

from pprint import pprint

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts

from pytorch_lightning import LightningModule, Trainer, seed_everything

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import re
import emoji
from soynlp.normalizer import repeat_normalize

# import kss

args = {
    'random_seed': 42, # Random Seed
    'pretrained_model': 'beomi/KcELECTRA-base',  # Transformers PLM name
    'pretrained_tokenizer': '',  # Optional, Transformers Tokenizer Name. Overrides `pretrained_model`
    'batch_size': 32,
    'lr': 5e-6,  # Starting Learning Rate
    'epochs': 300,  # Max Epochs
    'max_length': 64,  # Max Length input size
    'train_data_path': '/content/drive/MyDrive/train_6000_2.csv',  # Train Dataset file 
    'val_data_path': '/content/drive/MyDrive/valid_6000_2.csv',  # Validation Dataset file 
    'test_mode': False,  # Test Mode enables `fast_dev_run`
    'optimizer': 'AdamW',  # AdamW vs AdamP
    'lr_scheduler': 'exp',  # ExponentialLR vs CosineAnnealingWarmRestarts
    'fp16': True,  # Enable train on FP16(if GPU)
    'tpu_cores': 0,  # Enable TPU with 1 core or 8 cores
    # 'cpu_workers': os.cpu_count(),
    'cpu_workers': 0
}

accuracy_list = []
loss_list = []
class Model(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters() # 이 부분에서 self.hparams에 위 kwargs가 저장된다.
        
        self.clsfier = AutoModelForSequenceClassification.from_pretrained(self.hparams.pretrained_model, num_labels = 5)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.pretrained_tokenizer
            if self.hparams.pretrained_tokenizer
            else self.hparams.pretrained_model
        )

    def forward(self, **kwargs):
        return self.clsfier(**kwargs)

    def step(self, batch, batch_idx):
        data, labels = batch
        output = self(input_ids=data, labels=labels)

        # Transformers 4.0.0+
        loss = output.loss
        logits = output.logits

        preds = logits.argmax(dim=-1)

        y_true = list(labels.cpu().numpy())
        y_pred = list(preds.cpu().numpy())

        return {
            'loss': loss,
            'y_true': y_true,
            'y_pred': y_pred,
        }

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def epoch_end(self, outputs, state='train'):
        loss = torch.tensor(0, dtype=torch.float)
        for i in outputs:
            loss += i['loss'].cpu().detach()
        loss = loss / len(outputs)

        y_true = []
        y_pred = []
        for i in outputs:
            y_true += i['y_true']
            y_pred += i['y_pred']
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average = 'micro')
        rec = recall_score(y_true, y_pred, average = 'micro')
        f1 = f1_score(y_true, y_pred, average = 'micro')

        self.log(state+'_loss', float(loss), on_epoch=True, prog_bar=True)
        self.log(state+'_acc', acc, on_epoch=True, prog_bar=True)
        self.log(state+'_precision', prec, on_epoch=True, prog_bar=True)
        self.log(state+'_recall', rec, on_epoch=True, prog_bar=True)
        self.log(state+'_f1', f1, on_epoch=True, prog_bar=True)
    
        accuracy_list.append(acc)
        loss_list.append(loss)
        print(f'[Epoch {self.trainer.current_epoch} {state.upper()}] Loss: {loss}, Acc: {acc}, Prec: {prec}, Rec: {rec}, F1: {f1}')
        return {'loss': loss}
    
    def training_epoch_end(self, outputs):
        self.epoch_end(outputs, state='train')

    def validation_epoch_end(self, outputs):
        self.epoch_end(outputs, state='val')

    def configure_optimizers(self):
        if self.hparams.optimizer == 'AdamW':
            optimizer = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay = 0.3)
            # optimizer = AdamW(self.parameters(), lr=0)#0에 가까운 아주 작은 값을 입력해야함
        elif self.hparams.optimizer == 'AdamP':
            from adamp import AdamP
            optimizer = AdamP(self.parameters(), lr=self.hparams.lr)
        else:
            raise NotImplementedError('Only AdamW and AdamP is Supported!')
        if self.hparams.lr_scheduler == 'cos':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
        elif self.hparams.lr_scheduler == 'exp':
            scheduler = ExponentialLR(optimizer, gamma=0.5)
        elif self.hparams.lr_scheduler == 'cosup':
            scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=10, T_mult=2, eta_max=0.1,  T_up=10, gamma=0.5)
        else:
            raise NotImplementedError('Only cos and exp lr scheduler is Supported!')
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }

    def read_data(self, path):
        if path.endswith('xlsx'):
            return pd.read_excel(path)
        elif path.endswith('csv'):
            return pd.read_csv(path)
        elif path.endswith('tsv') or path.endswith('txt'):
            return pd.read_csv(path, sep='\t')
        else:
            raise NotImplementedError('Only Excel(xlsx)/Csv/Tsv(txt) are Supported')

    def clean(self, x):
        emojis = ''.join(emoji.UNICODE_EMOJI.keys())
        pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
        url_pattern = re.compile(
            r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
        x = pattern.sub(' ', x)
        x = url_pattern.sub('', x)
        x = x.strip()
        x = repeat_normalize(x, num_repeats=2)
        return x

    def encode(self, x, **kwargs):
        return self.tokenizer.encode(
            self.clean(str(x)),
            padding='max_length',
            max_length=self.hparams.max_length,
            truncation=True,
            **kwargs,
        )

    def preprocess_dataframe(self, df):
        df['document'] = df['document'].map(self.encode)
        return df

    def dataloader(self, path, shuffle=False):
        df = self.read_data(path)
        df = self.preprocess_dataframe(df)

        dataset = TensorDataset(
            torch.tensor(df['document'].to_list(), dtype=torch.long),
            torch.tensor(df['label'].to_list(), dtype=torch.long),
        )
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size * 1 if not self.hparams.tpu_cores else self.hparams.tpu_cores,
            shuffle=shuffle,
            num_workers=self.hparams.cpu_workers,
        )

    def train_dataloader(self):
        return self.dataloader(self.hparams.train_data_path, shuffle=True)

    def val_dataloader(self):
        return self.dataloader(self.hparams.val_data_path, shuffle=False)
        
class ModelHandler:
    def __init__(self):
        self.id2label = {0: 'negative', 1: 'positive'}

    def _clean_text(self, text):
        model_input = []
        if isinstance(text, str):
            cleaned_text = clean_text(text)
            model_input.append(cleaned_text)
        elif isinstance(text, (list, tuple)) and len(text) > 0 and (all(isinstance(t, str) for t in text)):
            cleaned_text = itertools.chain((clean_text(t) for t in text))
            model_input.extend(cleaned_text)
        else:
            model_input.append('')
        return model_input


class DLModelHandler(ModelHandler):
    def __init__(self):
        super().__init__()
        self.initialize()

    def initialize(self):
        # Loading tokenizer and De-serializing model
        #모델 불러오기
        self.model = torch.load('kcelectra_total_model2.pt')
        self.model.eval()

    def preprocess(self, text):
        # cleansing raw text
        sp_text = kss.split_sentences(text)
        return sp_text
        
        # vectorizing cleaned text
        ...

    def inference(self, text):
        # get predictions from model as probabilities
        return torch.softmax(self.model(**self.model.tokenizer(text, return_tensors='pt')).logits, dim=-1)
        
    def postprocess(self, sp_text):
        # process predictions to predicted label and output format
        #일기 데이터 한 문장씩 모델로 돌리기
        #방법 1. 감정별 리스트 만들고 / 문장 돌아갈때 마다 각 감정에 해당하는 값들 추가 -> 문장이 상대적이어서 패스
        #방법 2. 각 문장의 최대 감정을 구함(max) -> 구한 최대 감정들을 count해서 가장 많이나온 감정!
        import numpy as np
        def emotion_count(li):
            counts = {}
            for x in li:
                if x in counts:
                    counts[x] += 1
                else:
                    counts[x] = 1
            return counts

        def top_3(count_dict, n = 3):
            return sorted(count_dict.items(), reverse = True, key = lambda x:x[1])[:n]

        def neu_emo(text):
            sec_emotions = []
            for sent in sp_text:
                single_text = self.inference(sent)

                #single_text에 대한 softmax값이 담길 lo리스트
                lo = []
                #tensor형태 -> 1차원 list로 변경
                for i in single_text:
                    a = i.detach().numpy()
                    a = a.tolist()# float type
                    lo.append(a)
                    lo = np.concatenate(lo).tolist()#2차원 배열 -> 1차원 배열로
                    # print(lo)
                    lo_sor = sorted(lo)
                    sec_max_emo = lo.index(lo_sor[-2])#두번째로 큰 값의 인덱스 저장
                    # print(sec_max_emo)

                    if sec_max_emo == 0:
                        sec_emotions.append("행복")
                    elif sec_max_emo == 1:
                        sec_emotions.append("중립")
                    elif sec_max_emo == 2:
                        sec_emotions.append("불안")
                    elif sec_max_emo == 3:
                        sec_emotions.append("슬픔")
                    elif sec_max_emo == 4:
                        sec_emotions.append("분노")    

                # print("max emotions list : ", max_emotions)

            print("max emotions list : ", sec_emotions)
            counts = emotion_count(sec_emotions)
            print("emotion counts: ", counts)

            top = top_3(counts, n = 3)
            return top



        #최대 감정들이 담길 리스트
        max_emotions = []

        #문장별로 infer()돌리기
        for sent in sp_text:
            single_text = self.inference(sent)

            #single_text에 대한 softmax값이 담길 lo리스트
            lo = []
            #tensor형태 -> 1차원 list로 변경
            for i in single_text:
                a = i.detach().numpy()
                a = a.tolist()# float type
                lo.append(a)
                lo = np.concatenate(lo).tolist()#2차원 배열 -> 1차원 배열로
                # print(lo)
                max_emo = lo.index(max(lo))#최대값 = 최대감정 인덱스
                # print(max_emo)

                if max_emo == 0:
                    max_emotions.append("행복")
                elif max_emo == 1:
                    max_emotions.append("중립")
                elif max_emo == 2:
                    max_emotions.append("불안")
                elif max_emo == 3:
                    max_emotions.append("슬픔")
                elif max_emo == 4:
                    max_emotions.append("분노")    

            # print("max emotions list : ", max_emotions)

        # print("max emotions list : ", max_emotions)
        counts = emotion_count(max_emotions)
        # print("emotion counts: ", counts)

        top = top_3(counts, n = 3)
        # print(top)

        first_emo_cnt = top[0][0]
        if len(top) == 1:
            first_emo_cnt == '중립'#top_3에 중립만 나올 때 -> 문장 별 2순위 감정까지 넣기
            top = neu_emo(sp_text)#top 재설정해주기
            # print(top)
        else: 
            if len(top) == 2:
                second_emo_cnt = top[1][0]
                return first_emo_cnt, second_emo_cnt
            elif len(top) == 3:
                second_emo_cnt = top[1][0]
                third_emo_cnt = top[2][0]
                return first_emo_cnt, second_emo_cnt, third_emo_cnt
            # print('감정1: ', first_emo_cnt)
            # print('감정2: ', second_emo_cnt)
            # print('감정3: ', third_emo_cnt)

        # for cnt in range(len(top)):
        #     emo = top[cnt][0]
        #     print("{0}순위 감정은 {1}입니다.".format(cnt+1, emo))

        # #감정개수가 같을 때
        # if first_emo_cnt == second_emo_cnt:
        #     # print("1,2 순위 감정은 공통 감정입니다.")
        #     return()
        # elif first_emo_cnt == second_emo_cnt == third_emo_cnt:
        #     # print("1,2,3 순위 감정은 공통 감정입니다.")
        

    def handle(self, data):
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        return self.postprocess(model_output)
