"""APIサーバー起動スクリプト

使い方:
    python run.py

ブラウザで http://localhost:8000 にアクセスしてください。
APIドキュメントは http://localhost:8000/docs で確認できます。
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.api_server:app",
        host="127.0.0.1",  # 自分のPCからのみ接続を受け付ける
        port=8000,
        reload=True,
    )
