import daachorse
import json
import unicodedata
import re
import pandas as pd
import time
import numpy as np
import random
import sys
import os
from sklearn.model_selection import train_test_split


# 乱数のシードを固定
def seed_everything(seed=42):
  random.seed(seed)
  np.random.seed(seed)
  

# パターンマッチングにより固有表現を抽出
def create_entities(text, pma):
  ne = []
  entities = pma.find(text) # [(0, 1, 2), (0, 2, 1), (1, 4, 0)] -> [(start, end, pattern_id), ...]
  if len(entities) == 0:
    return np.nan
  else:
    for entity in entities:
      start, end, pattern_id = entity
      ne.append({
        'name': text[start:end],
        'span': [start, end],
        'type': 'GAME'
      })
    return ne
  
  
# カテゴリ情報を製品名に変換
def category_to_product(text, pma, products):
  entities = pma.find(text)
  if len(entities) == 0:
    return text
  else:
    text_converted = ''
    pos = 0
    for entity in entities:
      start, end, pattern_id = entity
      text_converted = text_converted + text[pos:start] + random.choice(products)
      pos = end
    text_converted = text_converted + text[pos:]
    return text_converted
  

# カテゴリ情報を製品名に変換したデータを取得
# def get_data_augmentation(df, products):
#   # 用いるカテゴリ名
#   categories = ['ゲーム', 'ゲームソフト', 'ゲームソフトウェア', 'ゲームタイトル', 'ゲームアプリ', 'ビデオゲーム']
#   # 文字数の多い順にソート
#   categories.sort(key=lambda x: len(x), reverse=True)
#   # パターンマッチング
#   pma2 = daachorse.Automaton(categories)
#   # カテゴリ名を製品名に変換
#   df['aug'] = 0
#   df[['text', 'aug']] = df.apply(lambda x: category_to_product(x['text'], pma2, products), axis=1, result_type='expand')
#   return df


# ゲームのジャンルを製品名に変換したデータを取得
def get_data_augmentation(df, products):
  # 用いるカテゴリ名
  categories = ['アクションゲーム', '格闘ゲーム', '格ゲー', '音楽ゲーム', '音ゲー', 'レースゲーム', 'LDゲーム', 'シューティングゲーム', 
                'FPS', 'バトルロイヤルゲーム', 'ロールプレイングゲーム', 'RPG', 'アクションRPG', 'シミュレーションRPG', 
                'ローグライクゲーム', 'シミュレーションゲーム', 'アドベンチャーゲーム', 'ノベルゲーム', 'ホラーゲーム', 'ホラゲー', 
                '脱出ゲーム', '美少女ゲーム', 'ギャルゲー', 'BLゲーム', '乙女ゲー', 'パズルゲーム', 'ミニゲーム', '落ちゲー', 
                '恋愛シミュレーションゲーム', '育成ゲーム', 'リズムゲーム']
  
  # categoriesの中から「ゲーム」で終わっている単語の「ム」を削除したものをcategoriesに追加
  categories2 = []
  for category in categories:
    if category.endswith('ゲーム'):
      categories2.append(category[:-1])
  categories = categories + categories2
  
  # 文字数の多い順にソート
  categories.sort(key=lambda x: len(x), reverse=True)
  
  # パターンマッチング
  pma2 = daachorse.Automaton(categories, daachorse.MATCH_KIND_LEFTMOST_LONGEST)
  
  # カテゴリ名を製品名に変換
  df['text'] = df['text'].apply(lambda x: category_to_product(x, pma2, products))
  
  return df


def preprocess_data(df):  
  # リンクやハッシュタグが含まれるツイートを削除
  df = df[~df['text'].str.contains('http')]
  df = df[~df['text'].str.contains('#')]
  df = df[~df['text'].str.contains('@')]

  # 【】, []が含まれるツイートを削除
  df = df[~df['text'].str.contains('【')]
  df = df[~df['text'].str.contains('】')]
  df = df[~df['text'].str.contains('\[')]
  df = df[~df['text'].str.contains('\]')]
  
  return df


def preprocess_product_list(products):
  # あとで'XI', 'ギャラガ'も追加する
  removes = ['リング', 'X', 'たまらん', '蚊', 'ONE', '麻雀', 'キーパー', 'ゴルフ', 'マスク', 'イース',
            'es', '日常', '怒', '将軍', 'カードゲーム', '忍', 'パチンコ', 'ICO', 'ダウンロード', 
            '銀河', 'ガジェット', '侍', 'ウイルス', 'プリプリ', 'ステルス', 'ルーン', '戦乱', '神業',
            '人生ゲーム', '豆しば', 'ラビリンス', 'ヴァンパイア', 'RAGE', 'フィスト', 'へべれけ', 'みてはいけない',
            'エキスパート', '1942', 'ソロモン', '空戦', 'グランドスラム', 'タイムリープ', 'ドクロ', 'ALIVE', 'ブレイクスルー', 'バツグン',
            'レイマン', 'ピンボール', 'SWITCH', 'WRC', 'パラノイア', 'HOOK', 'アクアパッツァ', '剣豪', '大旋風', 'ぶっつぶし', 'タマラン',
            'テーマパーク', 'クライマー', '自動', '定期', '月間', 'フォロー', '仲良', 'ボルト', 'スケート', 'シーズン']
  
  products = set(products)
  removes = set(removes)
  products = products - removes
  products = list(products)
    
  return products


def train(config):
  # 処理時間の計測
  time1 = time.perf_counter()
  
  # 乱数のシードを固定
  seed_everything(42)
  
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
    
  # データ修正も選択も行わない場合、preprocessedのデータを使う
  if revised == '' and selected == '':
    preprocessed = '_preprocessed'
  else:
    preprocessed = ''
  
  # 学習データの年
  # years = ['2011']
  years = ['2007', '2008', '2009', '2010', '2011']
  
  # ツイートデータの読み込み
  print('-- ツイートデータの読み込み')
  df = pd.read_csv(f'./data/raw/tweet_2011.csv', engine='python', encoding='utf-16')
  all_num = len(df)
  
  # ツイートデータの正規化
  df['text'] = df['text'].str.normalize('NFKC')
    
  # 欠損値の削除
  df = df.dropna(subset=['text'])

  # 商品名データの読み込み
  dict_path = f'./data/json/product_game{preprocessed}{selected}.json'
  with open(dict_path, 'r') as f:
    d = json.load(f)
    
  # 商品名のリストを作成
  if config['all_period']:
    # 2011年以前のデータを全て含める場合
    products = []
    for year, product_list in d.items():
      if int(year) <= int(years[-1]):
        for product in product_list:
          # 文字の正規化
          product = unicodedata.normalize('NFKC', product)
          # ()や<>の除去
          product = re.sub(r'[\(].*[\)]', '', product)
          product = re.sub(r'[\<].*[\>]', '', product)
          # 末尾の空白の除去
          product = product.rstrip()
          if product == '':
            continue
          products.append(product)
    # yearsを変更
    years = [list(d.keys())[0], years[-1]]
  else:  
    # 2007~2011年のデータのみを含める場合
    products = []
    for year in years:
      for product in d[year]:
        # 文字の正規化
        product = unicodedata.normalize('NFKC', product)
        # ()や<>の除去
        product = re.sub(r'[\(].*[\)]', '', product)
        product = re.sub(r'[\<].*[\>]', '', product)
        # 末尾の空白の除去
        product = product.rstrip()
        if product == '':
          continue
        products.append(product)

  # 重複を削除
  products = list(set(products))
  
  # 製品名の処理
  if config['revise']:
    print('-- ゲームタイトル名辞書・教師データの修正')
    df = preprocess_data(df)
    products = preprocess_product_list(products)
  
  # 文字数の多い順にソート
  products.sort(key=lambda x: len(x))
  
  # パターンマッチング
  print('-- パターンマッチング')
  pma = daachorse.Automaton(products, daachorse.MATCH_KIND_LEFTMOST_LONGEST)
  df['entities'] = df['text'].apply(lambda x: create_entities(x, pma))
  
  # entitiesが空の行を削除
  df_product = df.dropna(subset=['entities'])
  
  # entitiesが空の行を抽出
  df_aug = df[df['entities'].isnull()]
  
  # データ拡張
  if config['aug']:
    df_aug = get_data_augmentation(df_aug, products)
    # パターンマッチング
    df_aug['entities'] = df_aug['text'].apply(lambda x: create_entities(x, pma))
    
  # entitiesが空の行を削除
  df_aug = df_aug.dropna(subset=['entities'])
    
  # データを結合
  df = pd.concat([df_product, df_aug], axis=0)
  
  # シャッフル
  df = df.sample(frac=1, random_state=42).reset_index(drop=True)
  
  # ツイートデータの保存
  df_train, df_valid = train_test_split(df, test_size=0.2, random_state=42)
  dir_name = f'{years[0]}-{years[-1]}'
  
  # 結果を保存するフォルダを作成する
  DIR_NAME_train = f'./data/train/{dir_name}/processing'
  if not os.path.exists(DIR_NAME_train):
    os.mkdir(DIR_NAME_train)
  DIR_NAME_valid = f'./data/valid/{dir_name}/processing'
  if not os.path.exists(DIR_NAME_valid):
    os.mkdir(DIR_NAME_valid)
  
  # データを保存
  print('-- データの保存')
  if config['aug']:
    df_train.to_csv(f'{DIR_NAME_train}/ne_train{revised}{selected}.csv', index=False, encoding='utf-16')
    df_valid.to_csv(f'{DIR_NAME_valid}/ne_valid{revised}{selected}.csv', index=False, encoding='utf-16')
  else:
    df_train.to_csv(f'{DIR_NAME_train}/ne_train_without_aug{revised}{selected}.csv', index=False, encoding='utf-16')
    df_valid.to_csv(f'{DIR_NAME_valid}/ne_valid_without_aug{revised}{selected}.csv', index=False, encoding='utf-16')
    
  time2 = time.perf_counter()

  print(f'years: {years[0]}~{years[-1]}')
  print('製品名の数:', len(products))
  print('商品名データ数:', len(df_product))
  print('カテゴリデータ数:', len(df_aug))
  print('データの合計:', len(df))
  print('元データの総数:', all_num)
  print('処理時間:', time2 - time1)
  

if __name__ == '__main__':
  # configを設定
  config = {
    'aug': False,     # データ拡張を行うかどうか
    'revise': True, # 人手によるデータ修正を行うかどうか
    'select': True,   # 価格ドットコムによるデータ選択を行うかどうか
    'all_period': False # 2007年以前のデータを全て含めるかどうか
  }
  
  # 学習データ、テストデータの作成
  train(config)
