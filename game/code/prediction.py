from glob import glob
import sys
import os
import re
import ast

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer
from seqeval.metrics import classification_report
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification

from datasets import Dataset
from datasets import DatasetDict

from tqdm import tqdm
from torch.utils.data import DataLoader

from function import (seed_everything, to_dict, 
                      create_label2id, preprocess_data, run_prediction, 
                      extract_entities, compute_scores, convert_results_to_labels,
                      find_error_results, output_text_with_label, find_success_results)


if __name__ == "__main__":
  # 引数を取得
  args = sys.argv
  data_range = str(args[1])
  dt = str(args[2])
  version = str(args[3])
  
  # シード値の固定
  seed_everything()
  
  # ディレクトリパスを設定
  DIR_NAME = f'./result/{data_range}/{dt}/{version}'
  
  # DIR_NAMEにあるconfigを読み込む
  with open(DIR_NAME + '/config.txt', 'r') as f:
    config = ast.literal_eval(f.read())
  
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
  df_valid = pd.read_csv(f'./data/valid/{dir_name_valid}/ne_valid{aug}{revised}{selected}.csv', encoding='utf-16')
  df_test = pd.read_csv(f'./data/test/2012/ne_test_2012.csv', encoding='utf-16')
  
  # DatasetDictを作成する
  df_valid['entities'] = df_valid['entities'].apply(to_dict)
  df_test['entities'] = df_test['entities'].apply(to_dict)
  dataset_valid = Dataset.from_pandas(df_valid)
  dataset_test = Dataset.from_pandas(df_test)
  dataset = DatasetDict({
    "validation": dataset_valid,
    "test": dataset_test
  })
  
  # トークナイザを読み込む
  model_name = "cl-tohoku/bert-base-japanese-v3"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  data_collator = DataCollatorForTokenClassification(tokenizer)

  # ラベルとIDを紐付けるdictを作成する
  label2id = create_label2id(dataset["validation"]["entities"])
  id2label = {v: k for k, v in label2id.items()}
  
  # データセットに対して前処理を行う
  validation_dataset = dataset["validation"].map(
    preprocess_data,
    fn_kwargs={
        "tokenizer": tokenizer,
        "label2id": label2id,
    },
    remove_columns=dataset["validation"].column_names,
  )
  test_dataset = dataset["test"].map(
    preprocess_data,
    fn_kwargs={
        "tokenizer": tokenizer,
        "label2id": label2id,
    },
    remove_columns=dataset["test"].column_names,
  )
  
  # データローダーを作成する
  validation_dataloader = DataLoader(
			validation_dataset,
			batch_size=32,
			shuffle=False,
			collate_fn=data_collator,
	)
  test_dataloader = DataLoader(
			test_dataset,
			batch_size=32,
			shuffle=False,
			collate_fn=data_collator,
	)
  
  # 最良モデルを読み込む
  best_score = 0
  for checkpoint in sorted(glob(DIR_NAME + '/output_bert_ner/checkpoint-*')): # 各チェックポイントでの処理
    model = AutoModelForTokenClassification.from_pretrained(checkpoint) # モデルの読み込み
    model.to("cuda") # GPUに転送
    predictions = run_prediction(validation_dataloader, model) # 固有表現ラベルの予測
    results = extract_entities(
      predictions, dataset["validation"], tokenizer, id2label
    ) # 固有表現ラベルの抽出
    true_labels, pred_labels = convert_results_to_labels(results) # 正解データと予測データのラベルのlistを作成する
    scores = compute_scores(true_labels, pred_labels, "micro") # 評価スコアの計算
    if best_score < scores["f1-score"]:
      best_model = model # 最良モデルの更新
  best_model = best_model.to("cuda:0") # GPUに転送
  
  # 固有表現ラベルを予測する
  predictions = run_prediction(test_dataloader, best_model)
  
  # 固有表現ラベルを抽出する
  results = extract_entities(
    predictions, dataset["test"], tokenizer, id2label
  )
  
  # 正解データと予測データのラベルのlistを作成する
  true_labels, pred_labels = convert_results_to_labels(results)
  
  # scoreからはじまるtextファイルを取得する
  # dir_len = len(glob.glob(DIR_NAME + "/score*.txt"))
  
  # 評価結果を出力する
  with open(DIR_NAME + "/score.txt", "w") as w:
    repo = classification_report(true_labels, pred_labels, digits=4)
    w.write(repo)
    print(repo)
  
  # エラー事例を保存する
  error_results = find_error_results(results)
  with open(DIR_NAME + "/error.txt", "w") as w:
    for result in error_results:
      idx = result["idx"]
      true_text = output_text_with_label(result, "entities")
      pred_text = output_text_with_label(result, "pred_entities")
      w.write(f"事例{idx}の正解: {true_text}\n")
      w.write(f"事例{idx}の予測: {pred_text}\n")
      
  # 成功事例を保存する
  success_results = find_success_results(results)
  with open(DIR_NAME + "/success.txt", "w") as w:
    for result in success_results:
      idx = result["idx"]
      true_text = output_text_with_label(result, "entities")
      pred_text = output_text_with_label(result, "pred_entities")
      w.write(f"事例{idx}の正解: {true_text}\n")
      w.write(f"事例{idx}の予測: {pred_text}\n")