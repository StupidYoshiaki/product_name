import json
import unicodedata
import re


product_dict = {}


if __name__ == '__main__':  
  # 商品名データの読み込み
  print('-- 商品名データの読み込み')
  with open('./data/json/product_game_jpgames.json', 'r') as f:
    d = json.load(f)
    
  # 商品名jsonの作成
  for year in d.keys():
    # 2023年以降の年は無視
    try:
      tmp = int(year)
    except ValueError:
      continue
    
    for product in d[year]:
      # 文字の正規化
      product = unicodedata.normalize('NFKC', product)
      # ()や<>の除去
      product = re.sub(r'[\(].*[\)]', '', product)
      product = re.sub(r'[\<].*[\>]', '', product)
      # 右端の空白の除去
      product = product.rstrip()
      # 左端の空白の除去
      product = product.lstrip()
      # yearの\nの除去
      year = year.replace('\n', '')
      # 辞書に追加
      if year not in product_dict.keys():
        product_dict[year] = []
      product_dict[year].append(product)
  
  # key順にソート
  product_dict = dict(sorted(product_dict.items()))
      
  # 商品名のリストをjsonで保存
  print('-- 商品名のリストを保存')
  with open('./data/json/product_game_preprocessed.json', 'w') as f:
    json.dump(product_dict, f, indent=2, ensure_ascii=False)
  