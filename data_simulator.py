import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# --- 設定パラメータ ---
START_TIME = datetime(2025, 10, 16, 0, 0, 0) # シミュレーション開始時刻
NUM_DAYS = 2  # シミュレートする日数
INTERVAL_MINUTES = 10 # サンプリング間隔: 10分ごと
TOTAL_POINTS = NUM_DAYS * 24 * (60 // INTERVAL_MINUTES) # 2 * 24 * 6 = 288 行
ANOMALY_COUNT = 8  # ランダムに挿入する異常データ点の数 (5-10 の範囲)
OUTPUT_FILE = 'simulated_power_data.csv'

# --- 通常の消費電力パターンの基準 (kW/h) ---
# 家庭/デバイスの時間帯別の基準消費電力を模擬
BASE_POWER_LEVELS = {
    # 深夜（低消費電力モード）
    (0, 6): 0.2,
    # 早朝（やや上昇。例: 湯沸かし、人が起床）
    (6, 9): 0.8,
    # 日中（作業モード。例: PC、冷蔵庫）
    (9, 17): 0.6,
    # 夕方/ピーク（調理、娯楽。高消費）
    (17, 22): 1.5,
    # 夜間（徐々に低下）
    (22, 24): 0.4,
}

# --- 1. 通常の時系列データを生成 ---

# タイムスタンプ列を作成
timestamps = [START_TIME + timedelta(minutes=i * INTERVAL_MINUTES) for i in range(TOTAL_POINTS)]

# 時間帯に応じて基準電力量を算出
power_data = []
for ts in timestamps:
    hour = ts.hour
    base_power = 0.0
    # 現在の時間帯の基準電力量を探索
    for (start, end), level in BASE_POWER_LEVELS.items():
        if start <= hour < end:
            base_power = level
            break
            
    # ランダムな揺らぎを加える（機器の起動、外気温の変化などを模擬）
    noise = np.random.normal(0, 0.1) # 平均0、標準偏差0.1の乱数
    final_power = max(0.0, base_power + noise) # 電力量が負にならないようにする
    power_data.append(final_power)

# DataFrame に変換
df = pd.DataFrame({'timestamp': timestamps, 'power_kW': power_data})

# --- 2. 異常データ点をランダムに挿入 ---

# 挿入位置をランダムに選択（先頭と末尾は避ける）
anomaly_indices = random.sample(range(1, TOTAL_POINTS - 1), ANOMALY_COUNT)

# 異常値の設定
# 異常値は当該時間帯の通常値より十分高く設定
# [3.0, 6.0] kW/h の範囲でランダムに設定し、故障や未停止の大電力機器を模擬
for index in anomaly_indices:
    anomaly_value = np.random.uniform(3.0, 6.0)
    df.loc[index, 'power_kW'] = anomaly_value
    # 当該行を異常点としてフラグ付け（後続のモデル検証のため）
    df.loc[index, 'is_anomaly'] = 1

# --- 3. データクリーニングと保存 ---

# 'is_anomaly' 列の NaN（非異常点）を 0 で埋める
df['is_anomaly'] = df['is_anomaly'].fillna(0).astype(int)

# timestamp をインデックスに設定
df.set_index('timestamp', inplace=True)

# CSV ファイルへ保存
 df.to_csv(OUTPUT_FILE)

print(f"✅ シミュレーションデータの生成に成功！")
print(f"ファイル名: {OUTPUT_FILE}")
print(f"総レコード数: {len(df)}")
print(f"異常点数: {df['is_anomaly'].sum()}")
print("\n先頭5行のプレビュー:")
print(df.head())
print("\n異常データ点のプレビュー:")
print(df[df['is_anomaly'] == 1])