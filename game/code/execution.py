import subprocess

with open('./result/2011-2011_2012/processing/result.txt', mode='w') as f:
  f.write('\n')
  f.write('カテゴリデータの比率が与える影響の検証\n')
  f.write('\n')


interval = 300
loop_num = 3000 + interval

for i in range(0, loop_num, interval):
  print('i:', i)
  # ファイルに書き込み
  with open('./result/2011-2011_2012/processing/result.txt', mode='a') as f:
    f.write('--------------------------------------------------------------------------------\n')
    f.write('\n')
  
  # NERプロセスの実行
  print('-- NER')
  subprocess.run(['python', r"./code/processing_2011/ner_2011.py", str(i)])
  
  # 学習プロセスの実行
  print('-- 学習')
  subprocess.run(['python', r"./code/processing_2011/trainer_2011.py"])
  
  # 予測プロセスの実行
  print('-- 予測')
  subprocess.run(['python', r"./code/processing_2011/prediction_2011.py", str(i)])
  
  print()
