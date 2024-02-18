import urllib.parse
import json
import sys
import daachorse

import requests
from bs4 import BeautifulSoup


product_dict = {}


# 商品名データの読み込み
print('-- 商品名データの読み込み')
with open('./data/json/product_game_preprocessed.json', 'r') as f:
  d = json.load(f)
  

# カテゴリーの設定
category = ['ゲーム', 'ソフト', 'ソフトウェア']
pma = daachorse.Automaton(category)

# 商品名の検索
print('-- 商品名の検索')
for year in d.keys():
  print(f'   -- {year}')
  for product in d[year]:
    # 商品名をURLエンコード
    try:
      product_encoded = urllib.parse.quote(product, encoding='shift-jis')
    except UnicodeEncodeError:
      continue
    
    # htmlの取得
    url = f'https://kakaku.com/search_results/{product_encoded}/?act=Input&lid=pc_ksearch_searchbutton_top'
    r = requests.get(url)
    soup = BeautifulSoup(r.content, "html.parser")
    
    p_tags = soup.find_all('p', class_='p-item_category')
    
    # 検索結果がなかった場合、次の商品へ
    if len(p_tags) == 0:
      continue
    
    # 検索結果があった場合、ソフトウェアのものかどうかを判定
    p_tags = p_tags[:min(5, len(p_tags))]
    for p_tag in p_tags:
      flag = True if len(pma.find(p_tag.text)) > 0 else False
      if flag:
        if year not in product_dict.keys():
          product_dict[year] = []
        product_dict[year].append(product)
        break


# データセットの保存
print('-- データセットの保存')
with open('./data/json/product_game_selected.json', 'w') as f:
  json.dump(product_dict, f, indent=2, ensure_ascii=False)
