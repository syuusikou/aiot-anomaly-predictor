from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, conlist
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime


# FastAPI アプリの初期化
app = FastAPI(
    title="AIoT デバイス稼働状態監視 API",
    description="Isolation Forest モデルに基づき、時系列の電力データの異常をリアルタイムで監視します。",
    version="1.0.0"
)

# 許可する正確なオリジンを定義（Vite フロントエンドの開発サーバー）
# ブラウザの解釈差異による CORS エラーを防ぐため、localhost と 127.0.0.1 の両方を列挙
ALLOWED_ORIGINS = [
    "http://localhost:5173",    # Vite フロントエンドのアドレス
    "http://127.0.0.1:5173",    # 代替 IP アドレス
    # 理論上は不要だがテストとして許可
    "http://127.0.0.1:8000",

]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,      # 許可するドメイン
    allow_credentials=True,             # cookie の送信を許可
    allow_methods=["*"],                # すべての HTTP メソッドを許可 (POST, GET など)
    allow_headers=["*"],                # すべてのヘッダーを許可
)

# --- 1. 設定とモデルの読み込み ---
MODEL_PATH = 'anomaly_detector.pkl'

# モデルファイルの存在を確認
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ エラー: モデルファイル '{MODEL_PATH}' が見つかりません。先に model_trainer.py を実行してください。")

# アプリ起動時にモデルを読み込み
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ モデル '{MODEL_PATH}' の読み込みに成功しました。")
except Exception as e:
    raise RuntimeError(f"❌ エラー: モデルの読み込みに失敗しました。ファイル破損の可能性があります。詳細: {e}")

# --- 2. Pydantic 入力データモデルの定義 ---

# 単一タイムポイントのデータ構造を定義
class DataPoint(BaseModel):
    """単一のセンサーデータ点を定義: タイムスタンプと電力値。"""
    timestamp: datetime = Field(
        ..., 
        description="データ取得の時点（ISO 8601 形式。例: 2025-10-16T08:00:00）"
    )
    power_kW: float = Field(
        ..., 
        ge=0,  # 電力値が 0 以上であることを保証
        description="当該時点の消費電力（単位: kW）。"
    )

# API が受け取る完全なデータパッケージ構造を定義
class TimeSeriesData(BaseModel):
    """API が受け取る時系列データのリストを定義。"""
    # conlist は入力がリストであり、少なくとも1つの DataPoint を含むことを厳格に制限
    time_series: conlist(DataPoint, min_length=1) = Field(
        ..., 
        description="時系列順に並んだセンサーデータ点のリスト。"
    )

# リスト内の単一データポイントの構造を定義
class PreviewDataPoint(BaseModel):
    timestamp: datetime = Field(..., description="データポイントのタイムスタンプ。")
    power_kW: float = Field(..., description="元の消費電力値 (kW)。")
    anomaly_score: float = Field(..., description="Isolation Forest の異常スコア。")
    is_anomaly: int = Field(..., description="予測ラベル (1: 正常, -1: 異常)。")

# API 応答の構造を定義
class PredictionResponse(BaseModel):
    """異常予測 API の応答構造を定義。"""
    status: str = Field(..., description="予測ステータス: Normal（正常）または Warning（警告）。")
    average_anomaly_score: float = Field(..., description="今回の入力データの平均異常スコア。値が低いほど異常。")
    message: str = Field(..., description="呼び出し側への簡潔な説明。")
    # 👇 新しいフィールド：フロントエンドのリストプレビュー用データ
    submitted_data_preview: list[PreviewDataPoint] = Field(
        ..., 
        description="今回送信されたすべてのデータポイントおよびその予測結果のリスト。"
    )

# --- 3. コア予測 API ルート ---

# しきい値の定義: 平均異常スコアの判定に使用。モデルの学習状況に応じて調整してください。
# IsolationForest の decision_function スコア: 値が低いほど異常。
# ここでは 0.05 未満の平均スコアを潜在的な異常と見なします。
ANOMALY_SCORE_THRESHOLD = 0.05 

@app.post("/predict_anomaly", response_model=PredictionResponse)
def predict_anomaly(data: TimeSeriesData):
    """
    一連の時系列電力データを受け取り、前処理後に AI モデルで異常状態を予測します。
    """
    
    # 1. データ変換と前処理（Pandas）
    try:
        # Pydantic 入力データを Pandas DataFrame に変換
        df_input = pd.DataFrame([p.model_dump() for p in data.time_series])
        
        # データを時系列順に並べ替える
        df_input['timestamp'] = pd.to_datetime(df_input['timestamp'])
        df_input.set_index('timestamp', inplace=True)
        
        # モデルが必要とする特徴量列を抽出（ここでは power_kW のみ）
        X_input = df_input[['power_kW']]

    except Exception as e:
        # データ変換中に発生したエラーを捕捉
        raise HTTPException(status_code=400, detail=f"データ前処理に失敗しました。データ形式を確認してください。詳細: {e}")


    # 2. モデル推論を実行し、異常スコアを計算
    try:
        # 読み込んだモデルで推論を実行。
        # decision_function は異常スコアを返す: 低いほど異常、高いほど正常。
        anomaly_scores = model.decision_function(X_input)
        
        # 今回の入力データ点の平均異常スコアを算出
        avg_score = anomaly_scores.mean()

        # モデル予測: -1 は異常、1 は正常
        predictions = model.predict(X_input)

    except Exception as e:
        # モデル推論中に発生したエラーを捕捉
        raise HTTPException(status_code=500, detail=f"内部サーバーエラー: モデル推論に失敗しました。システム管理者に連絡してください。{e}")

    # 3. **【新規】submitted_data_preview リストを構築** # zip を使用して原始データ、スコア、予測ラベルを組み合わせる
    submitted_data_preview_list = []
    
    # 原始データ点のリスト (DataPoint Pydantic オブジェクト)
    raw_data_points = data.time_series
    
    # zip を使用して原始データ、スコア、予測ラベルを組み合わせ、PreviewDataPoint 要求の辞書形式にまとめる
    for raw_point, score, prediction in zip(raw_data_points, anomaly_scores, predictions):
        submitted_data_preview_list.append({
            "timestamp": raw_point.timestamp, 
            "power_kW": raw_point.power_kW,
            "anomaly_score": float(score),       # 標準の float に変換
            "is_anomaly": int(prediction)        # 変換を int (-1 または 1)
        })

    # 4. ステータス判定と応答の構築
    
    # ロジック: いずれかの点が異常（-1）または平均スコアがしきい値未満の場合は Warning
    is_hard_anomaly = -1 in predictions
    is_low_score_warning = avg_score < ANOMALY_SCORE_THRESHOLD

    if is_hard_anomaly or is_low_score_warning:
        status = "Warning"
        message = "⚠️ 異常な消費電力パターンを検出。関連機器の即時確認を推奨します。"
    else:
        status = "Normal"
        message = "✔️ 消費電力パターンは安定。異常は検出されませんでした。"

    # JSON レスポンスを返す
    return PredictionResponse(
        status=status,
        average_anomaly_score=float(avg_score),
        message=message,
        submitted_data_preview=submitted_data_preview_list  # <--- 最終的な割り当て
    )