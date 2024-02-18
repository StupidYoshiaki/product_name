import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from transformers import PreTrainedTokenizer
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import set_seed

from spacy_alignments.tokenizations import get_alignments
from seqeval.metrics.sequence_labeling import get_entities
from seqeval.metrics import f1_score, precision_score, recall_score

from datasets import Dataset
from datasets import DatasetDict

from tqdm import tqdm
import random
import ast
from pprint import pprint

# from torchcrf import CRF
from transformers import BertForTokenClassification, PretrainedConfig
from transformers.modeling_outputs import TokenClassifierOutput


# 乱数シードの固定
def seed_everything(seed=42):
  random.seed(seed)
  torch.manual_seed(seed)
  np.random.seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  generator = torch.Generator()
  generator.manual_seed(seed)


def to_dict(df):
  return ast.literal_eval(df)


def output_tokens_and_labels(text, entities, tokenizer):
  """トークンのlistとラベルのlistを出力"""
  # 文字のlistとトークンのlistのアライメントをとる
  characters = list(text)
  tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
  char_to_token_indices, _ = get_alignments(characters, tokens)

  # "O"のラベルで初期化したラベルのlistを作成する
  labels = ["O"] * len(tokens)
  for entity in entities: # 各固有表現で処理する
    entity_span, entity_type = entity["span"], entity["type"]
    # print(char_to_token_indices[entity_span[1] - 1])
    start = char_to_token_indices[entity_span[0]][0]
    end = char_to_token_indices[entity_span[1] - 1][0]
    # 固有表現の開始トークンの位置に"B-"のラベルを設定する
    labels[start] = f"B-{entity_type}"
    # 固有表現の開始トークン以外の位置に"I-"のラベルを設定する
    for idx in range(start + 1, end + 1):
      labels[idx] = f"I-{entity_type}"
  # 特殊トークンの位置にはラベルを設定しない
  labels[0] = "-"
  labels[-1] = "-"
  return tokens, labels


def create_character_labels(text, entities):
    """文字ベースでラベルのlistを作成"""
    # "O"のラベルで初期化したラベルのlistを作成する
    labels = ["O"] * len(text)
    for entity in entities: # 各固有表現を処理する
        entity_span, entity_type = entity["span"], entity["type"]
        # 固有表現の開始文字の位置に"B-"のラベルを設定する
        labels[entity_span[0]] = f"B-{entity_type}"
        # 固有表現の開始文字以外の位置に"I-"のラベルを設定する
        for i in range(entity_span[0] + 1, entity_span[1]):
            labels[i] = f"I-{entity_type}"
    return labels


def convert_results_to_labels(results):
    """正解データと予測データのラベルのlistを作成"""
    true_labels, pred_labels = [], []
    for result in results: # 各事例を処理する
        # 文字ベースでラベルのリストを作成してlistに加える
        true_labels.append(
            create_character_labels(result["text"], result["entities"])
        )
        pred_labels.append(
            create_character_labels(result["text"], result["pred_entities"])
        )
    return true_labels, pred_labels
  
  
def compute_scores(true_labels, pred_labels, average):
    """適合率、再現率、F値を算出"""
    scores = {
        "precision": precision_score(true_labels, pred_labels, average=average),
        "recall": recall_score(true_labels, pred_labels, average=average),
        "f1-score": f1_score(true_labels, pred_labels, average=average),
    }
    return scores


def create_label2id(entities_list):
  """ラベルとIDを紐付けるdictを作成"""
  # "O"のIDには0を割り当てる
  label2id = {"O": 0}
  # 固有表現タイプのsetを獲得して並び替える
  entity_types = set(
    [e["type"] for entities in entities_list for e in entities]
  )
  entity_types = sorted(entity_types)
  for i, entity_type in enumerate(entity_types):
    # "B-"のIDには奇数番号を割り当てる
    label2id[f"B-{entity_type}"] = i * 2 + 1
    # "I-"のIDには偶数番号を割り当てる
    label2id[f"I-{entity_type}"] = i * 2 + 2
  return label2id


def preprocess_data(data, tokenizer, label2id):
  """データの前処理"""
  # テキストのトークナイゼーションを行う
  inputs = tokenizer(
    data["text"],
    return_tensors="pt",
    return_special_tokens_mask=True,
  )
  inputs = {k: v.squeeze(0) for k, v in inputs.items()}

  # 文字のlistとトークンのlistのアライメントをとる
  characters = list(data["text"])
  tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"])
  char_to_token_indices, _ = get_alignments(characters, tokens)

  # "O"のIDのlistを作成する
  labels = torch.zeros_like(inputs["input_ids"])
  for entity in data["entities"]: # 各固有表現を処理する
    start_token_indices = char_to_token_indices[entity["span"][0]]
    end_token_indices = char_to_token_indices[
      entity["span"][1] - 1
    ]
    # 文字に対応するトークンが存在しなければスキップする
    if (len(start_token_indices) == 0 or len(end_token_indices) == 0):
      continue
    start, end = start_token_indices[0], end_token_indices[0]
    entity_type = entity["type"]
    # 固有表現の開始トークンの位置に"B-"のIDを設定する
    labels[start] = label2id[f"B-{entity_type}"]
    # 固有表現の開始トークン以外の位置に"I-"のIDを設定する
    if start != end:
      labels[start + 1 : end + 1] = label2id[f"I-{entity_type}"]
  # 特殊トークンの位置のIDは-100とする
  labels[torch.where(inputs["special_tokens_mask"])] = -100
  inputs["labels"] = labels
  return inputs


def convert_list_dict_to_dict_list(list_dict):
  """ミニバッチのデータを事例単位のlistに変換"""
  dict_list = []
  # dictのキーのlistを作成する
  keys = list(list_dict.keys())
  for idx in range(len(list_dict[keys[0]])): # 各事例で処理する
    # dictの各キーからデータを取り出してlistに追加する
    dict_list.append({key: list_dict[key][idx] for key in keys})
  return dict_list


def run_prediction(dataloader, model):
  """予測スコアに基づき固有表現ラベルを予測"""
  predictions = []
  for batch in tqdm(dataloader): # 各ミニバッチを処理する
    inputs = {
        k: v.to(model.device)
        for k, v in batch.items()
        if k != "special_tokens_mask"
    }
    # 予測スコアを取得する
    logits = model(**inputs).logits
    # 最もスコアの高いIDを取得する
    batch["pred_label_ids"] = logits.argmax(-1)
    batch = {k: v.cpu().tolist() for k, v in batch.items()}
    # ミニバッチのデータを事例単位のlistに変換する
    predictions += convert_list_dict_to_dict_list(batch)
  return predictions


def extract_entities(predictions, dataset, tokenizer, id2label):
    """固有表現を抽出"""
    results = []
    for prediction, data in zip(predictions, dataset):
        # 文字のlistを取得する
        characters = list(data["text"])

        # 特殊トークンを除いたトークンのlistと予測ラベルのlistを取得する
        tokens, pred_labels = [], []
        all_tokens = tokenizer.convert_ids_to_tokens(
            prediction["input_ids"]
        )
        for token, label_id in zip(
            all_tokens, prediction["pred_label_ids"]
        ):
            # 特殊トークン以外をlistに追加する
            if token not in tokenizer.all_special_tokens:
                tokens.append(token)
                pred_labels.append(id2label[label_id])

        # 文字のlistとトークンのlistのアライメントをとる
        _, token_to_char_indices = get_alignments(characters, tokens)

        # 予測ラベルのlistから固有表現タイプと、
        # トークン単位の開始位置と終了位置を取得して、
        # それらを正解データと同じ形式に変換する
        pred_entities = []
        for entity in get_entities(pred_labels):
            entity_type, token_start, token_end = entity
            # 文字単位の開始位置を取得する
            try:
              char_start = token_to_char_indices[token_start][0]
            except IndexError:
              # print(token_start)
              # print(token_to_char_indices)
              # print(len(token_to_char_indices))
              # print(data["text"][token_start])
              continue
            # 文字単位の終了位置を取得する
            try:
              char_end = token_to_char_indices[token_end][-1] + 1
            except IndexError:
              # print(token_end)
              # print(token_to_char_indices)
              # print(len(token_to_char_indices))
              # print(data["text"][token_end])
              continue
            pred_entity = {
                "name": "".join(characters[char_start:char_end]),
                "span": [char_start, char_end],
                "type": entity_type,
            }
            pred_entities.append(pred_entity)
        data["pred_entities"] = pred_entities
        results.append(data)
    return results
  
  
def find_error_results(results):
		"""エラー事例を発見"""
		error_results = []
		for idx, result in enumerate(results): # 各事例を処理する
				result["idx"] = idx
				# 正解データと予測データが異なるならばlistに加える
				if result["entities"] != result["pred_entities"]:
						error_results.append(result)
		return error_results


def find_success_results(results):
		"""エラー事例を発見"""
		success_results = []
		for idx, result in enumerate(results): # 各事例を処理する
				result["idx"] = idx
				# 正解データと予測データが異なるならばlistに加える
				if result["entities"] == result["pred_entities"]:
						success_results.append(result)
		return success_results


def output_text_with_label(result, entity_column):
		"""固有表現ラベル付きテキストを出力"""
		text_with_label = ""
		entity_count = 0
		for i, char in enumerate(result["text"]): # 各文字を処理する
				# 出力に加えていない固有表現の有無を判定する
				if entity_count < len(result[entity_column]):
						entity = result[entity_column][entity_count]
						# 固有表現の先頭の処理を行う
						if i == entity["span"][0]:
								entity_type = entity["type"]
								text_with_label += f" [({entity_type}) "
						text_with_label += char
						# 固有表現の末尾の処理を行う
						if i == entity["span"][1] - 1:
								text_with_label += "] "
								entity_count += 1
				else:
						text_with_label += char
		return text_with_label