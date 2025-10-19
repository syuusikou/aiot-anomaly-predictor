# 1. ベースイメージ：公式の Python 3.11 軽量版（Slim）イメージを使用
# フル版イメージより小さく、起動が速い。
FROM python:3.11-slim

# 2. 作業ディレクトリの設定：後続のすべての操作はこのディレクトリで実行
WORKDIR /app

# 3. 依存ファイルのコピーとインストール
# まず依存リストをコピーし、Dockerのキャッシュメカニズムを活用、依存関係が変わらない場合は再インストール不要
COPY requirements.txt .

# Python依存関係のインストール
# --no-cache-dir イメージサイズを削減；-r ファイルからインストールを指定
RUN pip install --no-cache-dir -r requirements.txt

# 4. アプリケーションコードとAIモデルのコピー
# FastAPIアプリケーションファイルとモデルファイルをコピー
COPY app.py .
COPY anomaly_detector.pkl .

# 5. ポートの公開：FastAPIはデフォルトで8000ポートを使用
EXPOSE 8000

# 6. コンテナ起動コマンド：Uvicornを使用してFastAPIアプリケーションを起動
# --host 0.0.0.0 外部アクセスを許可
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]