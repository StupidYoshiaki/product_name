import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pprint import pprint

import datetime
import os
import sys

from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification
from transformers import Trainer, TrainingArguments

from datasets import Dataset
from datasets import DatasetDict

from function import seed_everything, to_dict, create_label2id, preprocess_data


if __name__ == "__main__":
  # 引数を取得
  args = sys.argv
  data_range = str(args[1])
  
  # 乱数シードを設定する
  seed_everything()
  
  # configを設定
  config = {
    'aug': False,     # データ拡張を行うかどうか
    'revise': True, # 人手によるデータ修正を行うかどうか
    'select': True  # 価格ドットコムによるデータ選択を行うかどうか
  }
  
  # データ閣僚を行うかどうか
  if config['aug']:
    aug = ''
  else:
    aug = '_without_aug'
  
  # データ修正を行うかどうか
  if config['revise']:
    revised = '_revised'
  else:
    revised = ''
    
  # データ選択を行うかどうか
  if config['select']:
    selected = '_selected'
  else:
    selected = ''
  
  # データの読み込み
  dir_name_train = data_range.split('_')[0]
  dir_name_valid = dir_name_train
  dir_name_test = data_range.split('_')[1]
  dir_name = dir_name_train + '_' + dir_name_test
  dir_name_train = f'./data/train/{dir_name_train}/ne_train{aug}{revised}{selected}.csv'
  dir_name_valid = f'./data/valid/{dir_name_valid}/ne_valid{aug}{revised}{selected}.csv'
  df_train = pd.read_csv(dir_name_train, encoding='utf-16')
  df_valid = pd.read_csv(dir_name_valid, encoding='utf-16')
  
  print('train_path:', dir_name_train)
  print('valid_path:', dir_name_valid)
  
  # DatasetDictを作成する
  df_train['entities'] = df_train['entities'].apply(to_dict)
  df_valid['entities'] = df_valid['entities'].apply(to_dict)
  dataset_train = Dataset.from_pandas(df_train)
  dataset_valid = Dataset.from_pandas(df_valid)
  dataset = DatasetDict({
    "train": dataset_train,
    "validation": dataset_valid
  })

  # トークナイザを読み込む
  model_name = "cl-tohoku/bert-base-japanese-v3"
  tokenizer = AutoTokenizer.from_pretrained(model_name)

  # ラベルとIDを紐付けるdictを作成する
  label2id = create_label2id(dataset["train"]["entities"])
  id2label = {v: k for k, v in label2id.items()}
  pprint(id2label)

  # データセットに対して前処理を行う
  train_dataset = dataset["train"].map(
    preprocess_data,
    fn_kwargs={
        "tokenizer": tokenizer,
        "label2id": label2id,
    },
    remove_columns=dataset["train"].column_names,
  )
  validation_dataset = dataset["validation"].map(
    preprocess_data,
    fn_kwargs={
        "tokenizer": tokenizer,
        "label2id": label2id,
    },
    remove_columns=dataset["validation"].column_names,
  )

  # モデルを読み込む
  model = AutoModelForTokenClassification.from_pretrained(
    model_name, label2id=label2id, id2label=id2label
  )

  # collate関数にDataCollatorForTokenClassificationを用いる
  data_collator = DataCollatorForTokenClassification(tokenizer)
  
  # 現在日を取得する
  dt = str(datetime.datetime.now())
  dt = dt.split()[0]
  dt = dt.split('-')
  dt = ''.join(dt)
  
  # 結果を保存するフォルダを作成する
  DIR_NAME = f'./result/{dir_name}/{dt}'
  if not os.path.exists(DIR_NAME):
    os.mkdir(DIR_NAME)
    DIR_NAME = DIR_NAME + '/v1'
    os.mkdir(DIR_NAME)
  else:
    files = os.listdir(DIR_NAME)
    DIR_NAME = DIR_NAME + '/v' + str(len(files)+1)
    os.mkdir(DIR_NAME)

  # Trainerに渡す引数を初期化する
  training_args = TrainingArguments(
    output_dir=DIR_NAME+"/output_bert_ner", # 結果の保存フォルダ
    per_device_train_batch_size=32, # 訓練時のバッチサイズ
    per_device_eval_batch_size=32, # 評価時のバッチサイズ
    learning_rate=1e-4, # 学習率
    lr_scheduler_type="linear", # 学習率スケジューラ
    warmup_ratio=0.1, # 学習率のウォームアップ
    num_train_epochs=5, # 訓練エポック数
    evaluation_strategy="epoch", # 評価タイミング
    save_strategy="epoch", # チェックポイントの保存タイミング
    logging_strategy="epoch", # ロギングのタイミング
    fp16=True, # 自動混合精度演算の有効化
  )

  # Trainerを初期化する
  trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=data_collator,
    args=training_args,
  )

  # 訓練する
  trainer.train()
  
  # configの内容を保存する
  with open(DIR_NAME + '/config.txt', 'w') as f:
    f.write(str(config))