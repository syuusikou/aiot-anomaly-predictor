# 🤖 AIoT 予知保全システム (異常検知予測)

## 概要 (Overview)

本プロジェクトは、製造業や設備管理における**予知保全 (Predictive Maintenance / PdM)** を実現するためのAIoTシステムです。時系列センサーデータをリアルタイムで分析し、設備の異常が発生する前に早期に検知・予測することで、ダウンタイムの最小化と運用コストの削減に貢献します。

-----

## 解決するビジネス課題 (Project Value: Business Pain Points)

このシステムは、特に以下のビジネス課題を解決します。

### 1\. 予期せぬダウンタイムの防止

従来の事後保全では、機器の故障後に修理が必要となり、生産ラインの停止（ダウンタイム）が発生していました。本システムは、**異常の予兆を事前に検知**することで、計画的なメンテナンスを可能にし、予期せぬライン停止を大幅に削減します。

### 2\. コスト最適化

異常検知AIモデル（Isolation Forest）を使用することで、人による監視や定期的な過剰保全を削減します。これにより、**メンテナンスコスト**と**スペアパーツの在庫コスト**を最適化します。

### 3\. データドリブンな意思決定

リアルタイムのデータシミュレーションと予測結果をフロントエンドで可視化することで、保守担当者が**データに基づいた迅速な意思決定**を行うことを可能にします。

-----

## 技術スタック (Technology Stack)

本プロジェクトは、以下のモダンな技術スタックで構成されており、高い性能と拡張性を実現しています。

| コンポーネント | 技術 | 役割 |
| :--- | :--- | :--- |
| **バックエンド (API)** | **Python / FastAPI** | 高速な非同期APIサーバーの構築。データの受信とAIモデルへの受け渡しを担います。 |
| **AI/機械学習** | **Scikit-learn** (Isolation Forest) | 時系列センサーデータのパターンを学習し、リアルタイムで異常度を予測する中核技術です。 |
| **コンテナ化** | **Docker** | バックエンド環境をコンテナ化し、依存関係を排除したクロスプラットフォームなデプロイを可能にします。 |
| **フロントエンド (UI)** | **React (Vite)** | センサーデータのシミュレーションとAPI予測結果をリアルタイムで可視化するユーザーインターフェースです。 |

-----

## 💻 ローカルでの実行方法 (Local Setup Guide)

このプロジェクトは、**Docker** を使用して最も簡単に起動できます。

### 前提条件 (Prerequisites)

  * [Docker Desktop](https://www.docker.com/products/docker-desktop/) がインストールされていること。

### 1\. プロジェクトのクローンとビルド

まず、GitHubからプロジェクトをローカルにクローンし、Dockerイメージをビルドします。

```bash
# GitHubからクローン
git clone https://github.com/syuusikou/aiot-anomaly-predictor.git
cd aiot-anomaly-predictor

# Dockerイメージをビルド
docker build -t aiot-predictor-api .
```

### 2\. Dockerコンテナの起動

ビルドしたイメージからコンテナを起動します。FastAPI はポート **8000** で公開されます。

```bash
docker run -d -p 8000:8000 --name aiot-api-server aiot-predictor-api
```

### 3\. フロントエンドの起動

フロントエンドフォルダに移動し、依存関係をインストールしてから開発サーバーを起動します。

```bash
cd frontend
# 依存関係のインストール
npm install

# 開発サーバーの起動 (localhost:5173 などで起動します)
npm run dev
```

-----

## 🔌 API呼び出しとテスト (API Testing)

バックエンドAPIは `http://127.0.0.1:8000/predict_anomaly` にて稼働しています。

### API エンドポイント

| メソッド | URL | 役割 |
| :--- | :--- | :--- |
| `POST` | `/predict_anomaly` | センサーデータを受け取り、異常スコアを返します。 |

### cURLでのテスト例

以下のコマンドをターミナルで実行し、APIの動作を確認できます。

```bash
curl -X POST "http://127.0.0.1:8000/predict_anomaly" \
     -H "Content-Type: application/json" \
     -d '{
       "sensor_data": [
         {"timestamp": 1678886400, "temperature": 45.1, "vibration": 1.2},
         {"timestamp": 1678886460, "temperature": 45.2, "vibration": 1.3}
       ]
     }'
```

#### 応答例 (Success Response)

```json
{
  "prediction": [
    {
      "temperature_score": 0.05,
      "vibration_score": 0.15,
      "is_anomaly": 0
    },
    {
      "temperature_score": 0.06,
      "vibration_score": 0.16,
      "is_anomaly": 0
    }
  ]
}
```
