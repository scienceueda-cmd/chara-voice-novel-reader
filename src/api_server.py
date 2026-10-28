"""APIサーバー

FastAPIバックエンド。
フロントエンドの静的ファイル配信とREST APIを提供する。

起動方法:
    uvicorn src.api_server:app --reload --port 8000
    または python run.py
"""
import asyncio
import json
import logging
import traceback
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import config
from src.schemas import (
    NovelMetadata,
    Episode,
    CharacterProfile,
    ScriptLine,
    VoiceInfo,
    CharacterVoiceMapping,
    POVType,
    AnalysisResult,
    TOCChapter,
)
from src.narou_fetcher import (
    extract_ncode,
    get_novel_metadata,
    fetch_episode,
    fetch_table_of_contents,
)
from src.text_preprocessor import add_line_ids, format_for_llm, chunk_lines
from src.llm_parser import extract_characters, analyze_full_episode, fallback_script_line
from src.tts_engine import VoicevoxEngine
from src.audio_builder import build_audio, export_audio, generate_output_filename

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# FastAPI アプリケーション
# ============================================================

app = FastAPI(
    title="キャラ分けAIボイス小説リーダー",
    version="0.1.0",
    description="小説家になろうの作品をキャラ別AIボイスで読み上げるアプリケーション",
)

# フロントエンドは同一オリジン（このサーバー）から配信するため、CORSは許可しない

# TTSエンジン（シングルトン）
tts_engine = VoicevoxEngine()

# フロントエンドディレクトリ
FRONTEND_DIR = config.PROJECT_ROOT / "frontend"


# ============================================================
# リクエスト/レスポンス モデル
# ============================================================

class FetchNovelRequest(BaseModel):
    """小説取得リクエスト"""
    url_or_ncode: str = Field(description="なろうのURLまたはNコード")
    page: int = Field(default=1, description="目次のページ番号")


class FetchNovelResponse(BaseModel):
    """小説取得レスポンス"""
    metadata: NovelMetadata
    table_of_contents: list[TOCChapter]
    current_page: int
    total_pages: int


class FetchEpisodeRequest(BaseModel):
    """エピソード取得リクエスト"""
    ncode: str = Field(description="Nコード")
    episode_no: int = Field(description="話数（短編の場合は0）")


class AnalyzeCharactersRequest(BaseModel):
    """キャラクター解析リクエスト"""
    text: str = Field(description="解析対象テキスト")
    ncode: str = Field(default="", description="Nコード")
    known_characters: list[CharacterProfile] = Field(default_factory=list, description="既知のキャラクターリスト")


class AnalyzeCharactersResponse(BaseModel):
    """キャラクター解析レスポンス"""
    characters: list[CharacterProfile]
    pov_type: POVType
    pov_character: str


class AnalyzeScriptRequest(BaseModel):
    """台本解析リクエスト"""
    paragraphs: list[str] = Field(description="段落リスト")
    ruby_paragraphs: list[str] = Field(default_factory=list, description="ルビ読み適用済みの段落リスト")
    characters: list[CharacterProfile] = Field(description="キャラクターリスト")
    pov_type: POVType = Field(description="視点タイプ")
    pov_character: str = Field(description="視点キャラクター名")


class VoicePreviewRequest(BaseModel):
    """ボイスプレビューリクエスト"""
    text: str = Field(default="こんにちは、私の声はこんな感じです。", description="プレビュー用テキスト")
    voice_id: int = Field(description="話者ID（スタイルID）")


class VoiceStatusResponse(BaseModel):
    """VOICEVOX状態レスポンス"""
    available: bool
    version: str | None = None
    speaker_count: int = 0


class SynthesizeRequest(BaseModel):
    """音声合成リクエスト (REST版)"""
    ncode: str = Field(default="unknown", description="Nコード（ファイル名用）")
    episode_no: int = Field(default=0, description="話数（ファイル名用）")
    script_lines: list[ScriptLine] = Field(description="台本行リスト")
    character_voice_mapping: list[CharacterVoiceMapping] = Field(description="キャラ→ボイス対応リスト")
    output_format: str = Field(default="mp3", description="出力フォーマット (mp3 / wav)")
    speed_scale: float = Field(default=1.0, description="読み上げ速度スケール")


class CombineRequest(BaseModel):
    ncode: str = Field(description="Nコード")
    episode_no: int = Field(description="話数")
    script_lines: list[ScriptLine] = Field(description="台本行リスト")
    output_format: str = Field(default="mp3", description="出力フォーマット (mp3 / wav)")

class SynthesizeResponse(BaseModel):
    """音声合成レスポンス"""
    filename: str
    duration_seconds: float
    download_url: str


class ErrorResponse(BaseModel):
    """エラーレスポンス"""
    detail: str


# ============================================================
# スタートアップ/シャットダウン イベント
# ============================================================

@app.on_event("startup")
async def startup_event() -> None:
    """起動時にVOICEVOXの状態を確認する"""
    logger.info("🚀 キャラ分けAIボイス小説リーダー APIサーバー起動中...")
    logger.info(f"   プロジェクトルート: {config.PROJECT_ROOT}")
    logger.info(f"   音声出力先: {config.AUDIO_DIR}")
    logger.info(f"   フロントエンド: {FRONTEND_DIR}")

    # VOICEVOX接続チェック
    if tts_engine.is_available():
        version = tts_engine.get_version()
        voices = tts_engine.get_available_voices()
        logger.info(f"✅ VOICEVOX 接続成功 (v{version})")
        logger.info(f"   利用可能ボイス数: {len(voices)}")

        # 話者サマリーをログ出力
        speakers = tts_engine.get_speakers_summary()
        for speaker in speakers:
            styles = [s["name"] for s in speaker.get("styles", [])]
            logger.info(f"   🎤 {speaker['name']}: {', '.join(styles)}")
    else:
        logger.warning("⚠️ VOICEVOX に接続できません。VOICEVOXを起動してください。")
        logger.warning(f"   接続先: {config.VOICEVOX_HOST}")


# ============================================================
# 1. 小説管理 API
# ============================================================

@app.post(
    "/api/novel/fetch",
    response_model=FetchNovelResponse,
    summary="小説メタデータ・目次取得",
    tags=["小説管理"],
)
async def api_fetch_novel(request: FetchNovelRequest) -> FetchNovelResponse:
    """URLまたはNコードから小説のメタデータと目次を取得する"""
    try:
        ncode = extract_ncode(request.url_or_ncode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        # メタデータと目次を並行取得（両方ともI/Oバウンド）
        loop = asyncio.get_event_loop()
        metadata, toc = await asyncio.gather(
            loop.run_in_executor(None, get_novel_metadata, ncode),
            loop.run_in_executor(None, fetch_table_of_contents, ncode, request.page),
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"小説取得エラー: {e}")
        raise HTTPException(status_code=500, detail="小説情報の取得に失敗しました。")

    import math
    total_pages = max(1, math.ceil(metadata.total_episodes / 100))

    return FetchNovelResponse(
        metadata=metadata,
        table_of_contents=toc,
        current_page=request.page,
        total_pages=total_pages
    )


@app.post(
    "/api/novel/episode",
    response_model=Episode,
    summary="エピソード本文取得",
    tags=["小説管理"],
)
async def api_fetch_episode(request: FetchEpisodeRequest) -> Episode:
    """指定話の本文テキストを取得する"""
    try:
        loop = asyncio.get_event_loop()
        episode = await loop.run_in_executor(
            None, fetch_episode, request.ncode, request.episode_no
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"エピソード取得エラー: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"エピソードの取得に失敗しました: {e}",
        )

    return episode


# ============================================================
# 2. キャラクター解析 API
# ============================================================

@app.post(
    "/api/analyze/characters",
    response_model=AnalyzeCharactersResponse,
    summary="キャラクター抽出（Pass 1）",
    tags=["キャラクター解析"],
)
async def api_analyze_characters(
    request: AnalyzeCharactersRequest,
) -> AnalyzeCharactersResponse:
    """小説テキストからキャラクターを自動抽出する（LLM Pass 1）"""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="解析テキストが空です")

    try:
        characters, pov_type, pov_character = await extract_characters(request.text, request.known_characters)
    except Exception as e:
        logger.error(f"キャラクター解析エラー: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"キャラクター解析に失敗しました: {e}",
        )

    return AnalyzeCharactersResponse(
        characters=characters,
        pov_type=pov_type,
        pov_character=pov_character,
    )


@app.post(
    "/api/analyze/script",
    response_model=list[ScriptLine],
    summary="行単位解析（Pass 2）",
    tags=["キャラクター解析"],
)
async def api_analyze_script(request: AnalyzeScriptRequest) -> list[ScriptLine]:
    """段落テキストを行単位で話者・感情解析する（LLM Pass 2）

    前処理→チャンク分割→LLM解析を一括実行する。
    """
    if not request.paragraphs:
        raise HTTPException(status_code=400, detail="段落リストが空です")

    try:
        # テキスト前処理
        processed_lines = add_line_ids(request.paragraphs, request.ruby_paragraphs)

        # 全チャンクを解析
        script_lines = await analyze_full_episode(
            processed_lines=processed_lines,
            characters=request.characters,
            pov_type=request.pov_type,
            pov_character=request.pov_character,
            chunk_size=config.CHUNK_SIZE,
            context_overlap=config.CONTEXT_OVERLAP,
        )
    except Exception as e:
        logger.error(f"台本解析エラー: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"台本解析に失敗しました: {e}",
        )

    return script_lines


@app.websocket("/ws/analyze")
async def ws_analyze(websocket: WebSocket) -> None:
    """台本解析をチャンク単位でストリーミングする（LLM Pass 2）

    クライアントは接続後に AnalyzeScriptRequest と同じJSONを送信する。

    サーバーは次のメッセージを送る:
    {"type": "skeleton", "lines": [...]}  全行の仮データ（行ID・本文は確定、話者等は予備判定）
    {"type": "chunk", "lines": [...]}     解析が終わったチャンク（終わった順）
    {"type": "done"}
    {"type": "error", "detail": "..."}

    skeleton で全行の並びを先に確定させるので、クライアントは
    解析が終わっていない位置へのシークも行番号で扱える。
    """
    await websocket.accept()
    try:
        request = AnalyzeScriptRequest(**json.loads(await websocket.receive_text()))
        if not request.paragraphs:
            await websocket.send_json({"type": "error", "detail": "段落リストが空です"})
            return

        processed_lines = add_line_ids(request.paragraphs, request.ruby_paragraphs)
        await websocket.send_json({
            "type": "skeleton",
            "lines": [fallback_script_line(line).model_dump(mode="json") for line in processed_lines],
        })

        async def send_chunk(lines: list[ScriptLine]) -> None:
            await websocket.send_json({
                "type": "chunk",
                "lines": [line.model_dump(mode="json") for line in lines],
            })

        async def wait_for_disconnect() -> None:
            while (await websocket.receive())["type"] != "websocket.disconnect":
                pass

        # 解析中にクライアントが切断（別の話へ移動など）したら、
        # 次のチャンクの完了を待たずにすぐ解析を止める
        analysis = asyncio.create_task(analyze_full_episode(
            processed_lines=processed_lines,
            characters=request.characters,
            pov_type=request.pov_type,
            pov_character=request.pov_character,
            on_chunk=send_chunk,
        ))
        watcher = asyncio.create_task(wait_for_disconnect())
        await asyncio.wait({analysis, watcher}, return_when=asyncio.FIRST_COMPLETED)
        if not analysis.done():
            analysis.cancel()
            logger.info("WebSocket切断のため台本解析を中止: /ws/analyze")
            return
        watcher.cancel()
        analysis.result()  # 解析中の例外はここで送出
        await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        logger.info("WebSocket切断: /ws/analyze")
    except Exception as e:
        logger.error(f"台本解析エラー: {e}\n{traceback.format_exc()}")
        try:
            await websocket.send_json({"type": "error", "detail": f"台本解析に失敗しました: {e}"})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ============================================================
# 3. ボイス管理 API
# ============================================================

@app.get("/api/character_db", summary="キャラボイス保存DBの取得", tags=["キャラ図鑑"])
async def get_character_db() -> dict[str, dict[str, Any]]:
    db_path = config.DATA_DIR / "character_db.json"
    if db_path.exists():
        import json
        try:
            data = json.loads(db_path.read_text(encoding="utf-8"))
            if data and isinstance(next(iter(data.values())), int):
                return {}
            return data
        except Exception:
            return {}
    return {}

@app.post("/api/character_db", summary="キャラボイス保存DBの更新", tags=["キャラ図鑑"])
async def update_character_db(update: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    db_path = config.DATA_DIR / "character_db.json"
    db = {}
    import json
    if db_path.exists():
        try:
            data = json.loads(db_path.read_text(encoding="utf-8"))
            if not (data and isinstance(next(iter(data.values())), int)):
                db = data
        except Exception:
            pass
    db.update(update)
    db_path.write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8")
    return db

@app.get(
    "/api/voices",
    response_model=list[VoiceInfo],
    summary="利用可能ボイス一覧",
    tags=["ボイス管理"],
)
async def api_get_voices() -> list[VoiceInfo]:
    """VOICEVOX の利用可能な話者・スタイル一覧を返す"""
    if not tts_engine.is_available():
        raise HTTPException(
            status_code=503,
            detail="VOICEVOXに接続できません。VOICEVOXを起動してください。",
        )

    try:
        voices = tts_engine.get_available_voices()
    except Exception as e:
        logger.error(f"ボイス一覧取得エラー: {e}")
        raise HTTPException(status_code=502, detail=f"ボイス一覧の取得に失敗しました: {e}")

    return voices


@app.get(
    "/api/voices/status",
    response_model=VoiceStatusResponse,
    summary="VOICEVOX接続状態",
    tags=["ボイス管理"],
)
async def api_voices_status() -> VoiceStatusResponse:
    """VOICEVOXの接続状態を返す"""
    available = tts_engine.is_available()
    version: str | None = None
    speaker_count = 0

    if available:
        try:
            version = tts_engine.get_version()
            voices = tts_engine.get_available_voices()
            speaker_count = len(voices)
        except Exception:
            pass

    return VoiceStatusResponse(
        available=available,
        version=version,
        speaker_count=speaker_count,
    )


@app.post(
    "/api/voices/preview",
    summary="ボイスプレビュー",
    tags=["ボイス管理"],
)
async def api_voice_preview(request: VoicePreviewRequest) -> FileResponse:
    """指定ボイスで短いプレビュー音声を生成して返す"""
    if not tts_engine.is_available():
        raise HTTPException(
            status_code=503,
            detail="VOICEVOXに接続できません。",
        )

    try:
        loop = asyncio.get_event_loop()
        wav_bytes = await loop.run_in_executor(
            None,
            tts_engine.synthesize,
            request.text,
            request.voice_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"プレビュー生成エラー: {e}")
        raise HTTPException(status_code=500, detail=f"プレビュー音声の生成に失敗しました: {e}")

    # 一時ファイルに書き出して返す
    preview_path = config.AUDIO_DIR / "_preview.wav"
    preview_path.write_bytes(wav_bytes)

    return FileResponse(
        path=str(preview_path),
        media_type="audio/wav",
        filename="preview.wav",
    )


# ============================================================
# 4. 音声合成 API（REST）
# ============================================================

@app.post(
    "/api/synthesize",
    response_model=SynthesizeResponse,
    summary="音声合成（REST）",
    tags=["音声合成"],
)
async def api_synthesize(request: SynthesizeRequest) -> SynthesizeResponse:
    """台本全体の音声を合成してMP3/WAVファイルとして保存する

    進捗のリアルタイム通知が必要な場合は WebSocket版 `/ws/synthesize` を使用してください。
    """
    if not request.script_lines:
        raise HTTPException(status_code=400, detail="台本行が空です")

    if not tts_engine.is_available():
        raise HTTPException(
            status_code=503,
            detail="VOICEVOXに接続できません。",
        )

    # キャラ→ボイスマッピングの辞書を構築
    voice_map = _build_voice_map(request.character_voice_mapping)

    try:
        # 各行の音声を生成
        audio_segments = await _synthesize_all_lines(
            script_lines=request.script_lines,
            voice_map=voice_map,
            speed_scale=request.speed_scale,
        )

        # 音声結合
        loop = asyncio.get_event_loop()
        combined = await loop.run_in_executor(
            None, build_audio, request.script_lines, audio_segments
        )

        # エクスポート
        output_path = generate_output_filename(
            request.ncode, request.episode_no, request.output_format
        )
        await loop.run_in_executor(
            None, export_audio, combined, output_path, request.output_format
        )

        duration = combined.duration_seconds

    except Exception as e:
        logger.error(f"音声合成エラー: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"音声合成に失敗しました: {e}")

    filename = output_path.name
    return SynthesizeResponse(
        filename=filename,
        duration_seconds=round(duration, 2),
        download_url=f"/api/audio/{filename}",
    )


@app.get(
    "/api/audio/{filename:path}",
    summary="音声ファイルダウンロード",
    tags=["音声合成"],
)
async def api_get_audio(filename: str) -> FileResponse:
    """生成済み音声ファイルを返す"""
    file_path = config.AUDIO_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"音声ファイルが見つかりません: {filename}")

    # セキュリティチェック: パストラバーサル防止
    try:
        file_path.resolve().relative_to(config.AUDIO_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="不正なファイルパスです")

    # MIME タイプ判定
    media_type = "audio/mpeg" if file_path.suffix == ".mp3" else "audio/wav"

    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=filename,
    )

@app.post(
    "/api/audio/combine",
    response_model=SynthesizeResponse,
    summary="一時音声ファイルの結合",
    tags=["音声合成"],
)
async def api_combine_audio(request: CombineRequest) -> SynthesizeResponse:
    """フロントエンドで生成された各行の一時音声ファイルを結合し、最終ファイルを出力する"""
    audio_segments: dict[str, bytes] = {}
    
    # 1. 既存のtmp_audioファイルを読み込む
    for line in request.script_lines:
        if line.type == "scene_break":
            continue
        tmp_file = config.TMP_AUDIO_DIR / f"{request.ncode}_{request.episode_no}_{line.line_id}.wav"
        if tmp_file.exists():
            audio_segments[line.line_id] = tmp_file.read_bytes()
        else:
            audio_segments[line.line_id] = b""
            
    # 2. 音声結合
    loop = asyncio.get_event_loop()
    combined = await loop.run_in_executor(
        None, build_audio, request.script_lines, audio_segments
    )
    
    # 3. エクスポート
    output_path = generate_output_filename(
        request.ncode, request.episode_no, request.output_format
    )
    await loop.run_in_executor(
        None, export_audio, combined, output_path, request.output_format
    )
    
    return SynthesizeResponse(
        filename=output_path.name,
        duration_seconds=round(combined.duration_seconds, 2),
        download_url=f"/api/audio/{output_path.name}",
    )


# ============================================================
# 5. WebSocket 音声合成（進捗ストリーミング）
# ============================================================

@app.websocket("/ws/synthesize")
async def ws_synthesize(websocket: WebSocket) -> None:
    """WebSocketで音声合成の進捗をリアルタイム送信する

    クライアントは接続後に以下のJSONを送信する:
    {
        "ncode": "n1234ab",
        "episode_no": 1,
        "script_lines": [...],
        "character_voice_mapping": [...],
        "output_format": "mp3"
    }

    サーバーは進捗メッセージをストリーミング送信する:
    {"type": "progress", "current": 1, "total": 50, "line_id": "L001", "message": "..."}
    {"type": "complete", "filename": "...", "duration_seconds": 123.4, "download_url": "..."}
    {"type": "error", "detail": "..."}
    """
    await websocket.accept()
    logger.info("WebSocket接続: /ws/synthesize")

    try:
        # クライアントからリクエストデータを受信
        raw_data = await websocket.receive_text()
        data = json.loads(raw_data)
        request = SynthesizeRequest(**data)

        if not request.script_lines:
            await websocket.send_json({"type": "error", "detail": "台本行が空です"})
            await websocket.close()
            return

        if not tts_engine.is_available():
            await websocket.send_json({
                "type": "error",
                "detail": "VOICEVOXに接続できません。",
            })
            await websocket.close()
            return

        # キャラ→ボイスマッピング構築
        voice_map = _build_voice_map(request.character_voice_mapping)

        total_lines = len(request.script_lines)
        audio_segments: dict[str, bytes] = {}

        # 各行の音声を生成（進捗送信付き）
        for i, line in enumerate(request.script_lines):
            # シーン区切りはスキップ
            if line.type == "scene_break":
                await websocket.send_json({
                    "type": "progress",
                    "current": i + 1,
                    "total": total_lines,
                    "line_id": line.line_id,
                    "message": f"シーン区切り ({line.line_id})",
                })
                continue

            # 読み上げテキストがない行はスキップ
            speak_text = line.text_to_speak if line.text_to_speak else line.text
            if not speak_text.strip():
                continue

            # ボイス情報を取得
            mapping = voice_map.get(line.speaker)
            if mapping is None:
                # マッピングがない場合はナレーターのマッピングを試す
                mapping = voice_map.get("ナレーター")
            if mapping is None:
                # それでもない場合は最初のマッピングを使用
                mapping = next(iter(voice_map.values()), None)

            if mapping is None:
                await websocket.send_json({
                    "type": "error",
                    "detail": "ボイスマッピングが設定されていません",
                })
                await websocket.close()
                return

            try:
                loop = asyncio.get_event_loop()
                wav_bytes = await loop.run_in_executor(
                    None,
                    tts_engine.synthesize_script_line,
                    line,
                    mapping.voice_info.voice_id,
                    mapping.emotion_style_map if mapping.emotion_style_map else None,
                    request.speed_scale,
                )
                audio_segments[line.line_id] = wav_bytes
                
                if wav_bytes:
                    tmp_file = config.TMP_AUDIO_DIR / f"{request.ncode}_{request.episode_no}_{line.line_id}.wav"
                    tmp_file.write_bytes(wav_bytes)
                    await websocket.send_json({
                        "type": "line_audio",
                        "line_id": line.line_id,
                        "url": f"/api/tmp_audio/{tmp_file.name}",
                    })
            except Exception as e:
                logger.warning(f"音声生成スキップ ({line.line_id}): {e}")
                # エラーでも継続
                audio_segments[line.line_id] = b""

            # 進捗送信
            await websocket.send_json({
                "type": "progress",
                "current": i + 1,
                "total": total_lines,
                "line_id": line.line_id,
                "message": f"音声生成中: {line.speaker}「{speak_text[:20]}...」" if len(speak_text) > 20 else f"音声生成中: {line.speaker}「{speak_text}」",
            })

        # すべての生成が終わったらストリーム完了通知
        await websocket.send_json({
            "type": "stream_done",
        })

    except WebSocketDisconnect:
        logger.info("WebSocket切断: /ws/synthesize")
    except json.JSONDecodeError as e:
        logger.error(f"WebSocket JSONパースエラー: {e}")
        try:
            await websocket.send_json({"type": "error", "detail": f"JSONの形式が不正です: {e}"})
        except Exception:
            pass
    except Exception as e:
        logger.error(f"WebSocket音声合成エラー: {e}\n{traceback.format_exc()}")
        try:
            await websocket.send_json({"type": "error", "detail": f"音声合成エラー: {e}"})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ============================================================
# 6. フロントエンド配信
# ============================================================

@app.get("/", response_class=HTMLResponse, tags=["フロントエンド"])
async def root() -> HTMLResponse:
    """フロントエンドのindex.htmlを返す"""
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse(
            content="""
            <html>
            <head><title>キャラ分けAIボイス小説リーダー</title></head>
            <body>
                <h1>🎤 キャラ分けAIボイス小説リーダー</h1>
                <p>APIサーバーは正常に動作しています。</p>
                <p>フロントエンドがまだ配置されていません。</p>
                <p><code>frontend/index.html</code> を作成してください。</p>
                <hr>
                <p><a href="/docs">📖 API ドキュメント (Swagger UI)</a></p>
                <p><a href="/redoc">📄 API ドキュメント (ReDoc)</a></p>
            </body>
            </html>
            """,
            status_code=200,
        )

    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


# 一時音声ファイルのサービング
app.mount("/api/tmp_audio", StaticFiles(directory=str(config.TMP_AUDIO_DIR)), name="tmp_audio")

# フロントエンド静的ファイル（APIルートより後にマウント）
if FRONTEND_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(FRONTEND_DIR)),
        name="frontend_static",
    )


# ============================================================
# ヘルパー関数
# ============================================================

def _build_voice_map(
    mappings: list[CharacterVoiceMapping],
) -> dict[str, CharacterVoiceMapping]:
    """キャラクター名→CharacterVoiceMapping の辞書を構築する"""
    voice_map: dict[str, CharacterVoiceMapping] = {}
    for mapping in mappings:
        voice_map[mapping.character_name] = mapping
    return voice_map


async def _synthesize_all_lines(
    script_lines: list[ScriptLine],
    voice_map: dict[str, CharacterVoiceMapping],
    speed_scale: float = 1.0,
) -> dict[str, bytes]:
    """全行の音声を合成する（REST版、進捗通知なし）

    Args:
        script_lines: 台本行リスト
        voice_map: キャラ名→ボイスマッピング辞書

    Returns:
        {line_id: WAVバイト} の辞書
    """
    audio_segments: dict[str, bytes] = {}
    loop = asyncio.get_event_loop()

    for line in script_lines:
        # シーン区切りはスキップ
        if line.type == "scene_break":
            continue

        # 読み上げテキスト判定
        speak_text = line.text_to_speak if line.text_to_speak else line.text
        if not speak_text.strip():
            continue

        # ボイスマッピング解決
        mapping = voice_map.get(line.speaker)
        if mapping is None:
            mapping = voice_map.get("ナレーター")
        if mapping is None:
            mapping = next(iter(voice_map.values()), None)

        if mapping is None:
            logger.warning(f"ボイスマッピングなし、スキップ: {line.line_id}")
            continue

        try:
            wav_bytes = await loop.run_in_executor(
                None,
                tts_engine.synthesize_script_line,
                line,
                mapping.voice_info.voice_id,
                mapping.emotion_style_map if mapping.emotion_style_map else None,
                speed_scale,
            )
            audio_segments[line.line_id] = wav_bytes
        except Exception as e:
            logger.warning(f"音声生成スキップ ({line.line_id}): {e}")
            audio_segments[line.line_id] = b""

    return audio_segments
