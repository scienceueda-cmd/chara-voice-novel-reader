"""設定管理モジュール"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.parent

# データ保存ディレクトリ
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

AUDIO_DIR = DATA_DIR / "audio"
AUDIO_DIR.mkdir(exist_ok=True)

SCRIPTS_DIR = DATA_DIR / "scripts"
SCRIPTS_DIR.mkdir(exist_ok=True)

TMP_AUDIO_DIR = DATA_DIR / "tmp_audio"
TMP_AUDIO_DIR.mkdir(exist_ok=True)

# Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# VOICEVOX
VOICEVOX_HOST = os.getenv("VOICEVOX_HOST", "http://127.0.0.1:50021")

# スクレイピング設定
NAROU_BASE_URL = "https://ncode.syosetu.com"
NAROU_API_URL = "https://api.syosetu.com/novelapi/api/"
FETCH_INTERVAL = 1.5  # リクエスト間隔（秒）

# LLM解析設定
CHUNK_SIZE = 50  # 1チャンクあたりの行数
CONTEXT_OVERLAP = 8  # 各チャンクに文脈として渡す直前の行数
ANALYSIS_CONCURRENCY = 4  # 同時に解析するチャンク数（無料枠のレート制限に配慮）

# 音声設定
SILENCE_BETWEEN_LINES_MS = 300  # セリフ間の無音（ミリ秒）
SILENCE_SCENE_BREAK_MS = 1500  # シーン区切りの無音（ミリ秒）
