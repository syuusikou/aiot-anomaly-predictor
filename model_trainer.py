import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

# --- 設定パラメータ ---
INPUT_FILE = 'simulated_power_data.csv'
MODEL_OUTPUT_FILE = 'anomaly_detector.pkl'
# IsolationForest モデルのパラメータ:
# 'auto' は異常点の割合を推定するが、デモでは保守的に設定
CONTAMINATION_RATE = 0.05 
# 注意: 異常点は 8 個、総数 288、割合は約 2.7%。5% は安全な範囲。

def train_and_save_model():
    """データを読み込み、IsolationForest モデルを学習し、モデルを保存する。"""
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ エラー: 入力ファイル '{INPUT_FILE}' が見つかりません。先に data_simulator.py を実行してください。")
        return

    # 1. データの読み込みと準備
    print(f"1. データ読込: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, index_col='timestamp', parse_dates=True)
    
    # 'power_kW' のみを用いて異常検知
    X = df[['power_kW']]
    
    # 真のラベル（モデル検証用。実運用では存在しない）
    y_true = df['is_anomaly']

    # 2. モデル学習
    print(f"2. IsolationForest モデルを学習中...")
    
    # モデル初期化。random_state を固定し再現性を確保
    model = IsolationForest(
        contamination=CONTAMINATION_RATE, 
        random_state=42, 
        n_estimators=100  # 決定木の数
    )
    
    # 学習
    model.fit(X)
    print("✅ 学習完了。")

    # 3. モデル検証（有効性の確認）
    print("3. モデル検証を実施中...")
    
    # predictions: -1 は異常（Anomaly）、1 は正常（Normal）
    predictions = model.predict(X)
    
    # 予測結果を y_true（is_anomaly）に合わせ 0/1 へ変換
    # 予測 -1（異常）は 1（is_anomaly）に対応
    y_pred = np.where(predictions == -1, 1, 0)

    # モデルで異常とされた点を抽出
    anomalies_detected_df = df[y_pred == 1]
    
    # 集計
    total_anomalies_true = y_true.sum()
    total_anomalies_detected = y_pred.sum()
    
    # モデルが事前に設定した異常点を検出できたか確認
    # 予測が異常で、実際にも異常である点の数（True Positives）
    correctly_identified_count = np.sum((y_true == 1) & (y_pred == 1))
    
    print("--- 検証結果 ---")
    print(f"   事前設定の異常総数 (True): {total_anomalies_true}")
    print(f"   モデルが検出した異常総数: {total_anomalies_detected}")
    print(f"   ✔️ 正しく識別された異常点 (True Positives): {correctly_identified_count} / {total_anomalies_true}")

    # モデルが識別できなかった異常点（偽陰性: 漏報）
    false_negatives_df = df[(y_true == 1) & (y_pred == 0)]
    if not false_negatives_df.empty:
        print("\n   🚨 警告: 以下の異常点はモデルが識別できませんでした（漏報）:")
        print(false_negatives_df[['power_kW']])
    
    # モデルが誤って異常とした正常点（偽陽性: 誤報）
    false_positives_df = df[(y_true == 0) & (y_pred == 1)]
    if not false_positives_df.empty:
        print(f"\n   ⚠️ 注意: 以下の正常点が誤って異常と判定されました（誤報、計 {len(false_positives_df)} 件）:")
        print(false_positives_df[['power_kW']].head())

    # 4. モデルを保存
    joblib.dump(model, MODEL_OUTPUT_FILE)
    print(f"\n4. ✅ モデルを '{MODEL_OUTPUT_FILE}' に保存しました。")
    print("   このファイルは FastAPI サービスでリアルタイム予測に使用されます。")


if __name__ == '__main__':
    train_and_save_model()