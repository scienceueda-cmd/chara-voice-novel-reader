# キャラ分けAIボイス小説リーダー

「小説家になろう」の作品を、登場人物ごとに声を分けて読み上げるローカル用Webアプリ。

1. なろうのURL / Nコードから目次・本文を取得
2. Gemini が登場人物と各行の話者・感情を解析
3. VOICEVOX でキャラ別ボイスを合成し、ブラウザで再生（mp3 書き出しも可）

- 解析は50行ずつ並列に投げ、終わった部分から再生を始める（1話目は約36秒で再生開始）
- 再生中に次の話を先に解析しておくので、2話目以降は約1.5秒で始まる
- 感情に合わせて VOICEVOX のスタイル（喜び・怒り・ささやき など）を切り替える

## 必要なもの

- Python 3.10 以上
- [VOICEVOX](https://voicevox.hiroshiba.jp/)（起動しておく。既定は `http://127.0.0.1:50021`）
- [ffmpeg](https://ffmpeg.org/)（PATH に通す。pydub の mp3 書き出しに必要）
- Gemini API キー（[Google AI Studio](https://aistudio.google.com/)）

## セットアップ

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

`.env` に `GEMINI_API_KEY` を設定する。

## 起動

```bash
python run.py
```

- アプリ: http://localhost:8000
- APIドキュメント: http://localhost:8000/docs

## 構成

```
run.py                  uvicorn 起動スクリプト
src/
  api_server.py         FastAPI（REST + WebSocket /ws/synthesize）
  narou_fetcher.py      なろうAPI・本文スクレイピング（ルビ対応）
  text_preprocessor.py  行ID付与・括弧判定・チャンク分割
  llm_parser.py         Gemini 2パス解析（キャラ抽出 / 行ごとの話者・感情）
  tts_engine.py         VOICEVOX 連携（感情→スタイル・速度・抑揚）
  audio_builder.py      pydub で音声結合・書き出し
  schemas.py            Pydantic モデル
  config.py             設定（.env 読み込み）
frontend/               index.html + style.css（素のJS）
data/                   実行時データ（git 管理外）
  character_db.json     作品ごとのキャラ→ボイス設定
  audio/                書き出した音声
  tmp_audio/            行ごとの合成キャッシュ（削除してよい）
tests/test_e2e.py       Playwright による E2E テスト
```

## E2E テスト

サーバーと VOICEVOX を起動した状態で:

```bash
pip install playwright
playwright install chromium
python tests/test_e2e.py
```

スクリーンショットは `tests/screenshots/` に保存される。

## ライセンス

ポートフォリオとして公開しています。ライセンスは付けていないため、コードの転用・再配布はご遠慮ください。
