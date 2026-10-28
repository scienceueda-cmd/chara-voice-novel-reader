"""LLM台本パーサーモジュール

Gemini APIを使用して小説テキストから：
- キャラクターの自動抽出（Pass 1）
- 行単位の話者・感情解析（Pass 2）
を行う。
"""
import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from google import genai
from google.genai import errors, types

from src.config import GEMINI_API_KEY, GEMINI_MODEL, CHUNK_SIZE, CONTEXT_OVERLAP, ANALYSIS_CONCURRENCY
from src.schemas import (
    CharacterProfile, ScriptLine, LineType, Emotion, POVType
)
from src.text_preprocessor import ProcessedLine, format_for_llm, chunk_lines

logger = logging.getLogger(__name__)

# Geminiクライアントの初期化
client = genai.Client(api_key=GEMINI_API_KEY)

# リトライ対象のHTTPステータス（レート制限・一時的な過負荷）
RETRYABLE_STATUS_CODES = {429, 500, 503}


async def call_gemini_with_retry(
    model: str,
    contents: str,
    config: types.GenerateContentConfig,
    max_retries: int = 5,
) -> types.GenerateContentResponse:
    """Gemini APIを非同期で呼び出す（一時的なエラーは指数バックオフで再試行）

    非同期クライアントと asyncio.sleep を使うため、
    待機中もイベントループ（他のリクエスト）をブロックしない。
    """
    for attempt in range(max_retries):
        try:
            return await client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except errors.APIError as e:
            if e.code not in RETRYABLE_STATUS_CODES or attempt == max_retries - 1:
                raise
            wait = 2 ** attempt * 3
            logger.warning(f"Gemini API error ({e.code}) on attempt {attempt+1}/{max_retries}, retrying in {wait}s...")
            await asyncio.sleep(wait)
    raise RuntimeError("unreachable")



# ============================================================
# Pass 1: キャラクター抽出
# ============================================================

CHARACTER_EXTRACTION_PROMPT = """あなたは日本語小説の音声朗読用台本を作成する専門家AIです。
以下の小説テキストを読み、登場人物と作品の視点を分析してください。

【分析ルール】
1. 登場人物を全員抽出し、それぞれの「名前」「別名・あだ名」「一人称」「口調の特徴」「性別」「役割」「簡単な説明」を記述すること。
2. 作品の視点タイプを判定すること。
3. 視点人物を特定すること。
4. ナレーター（地の文担当）も登場人物リストに含めること。
5. 【利用可能なボイス一覧】から、そのキャラクターの性別・年齢・性格に最も合致するボイスID（style_id）を選び、recommended_voice_idとして出力すること。

【利用可能なボイス一覧】
{voices_list}

【既に声が決まっているキャラクター（推測不要）】
{known_chars_str}

【小説テキスト】
{novel_text}
"""

CHARACTER_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "pov_type": {
            "type": "string",
            "enum": ["first_person", "third_person_limited", "third_person_omniscient"]
        },
        "pov_character": {
            "type": "string",
            "description": "視点人物の名前"
        },
        "characters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                    "first_person": {"type": "string"},
                    "speech_style": {"type": "string"},
                    "gender": {"type": "string", "enum": ["male", "female", "unknown"]},
                    "role": {"type": "string", "enum": ["protagonist", "heroine", "supporting", "mob", "narrator"]},
                    "description": {"type": "string"},
                    "recommended_voice_id": {"type": "integer", "description": "キャラクターに最も合うボイススタイルID"}
                },
                "required": ["name", "gender", "role", "description"]
            }
        }
    },
    "required": ["pov_type", "pov_character", "characters"]
}


async def extract_characters(novel_text: str, known_characters: list[CharacterProfile] = None) -> tuple[list[CharacterProfile], POVType, str]:
    """小説テキストからキャラクターを抽出する（Pass 1）

    Args:
        novel_text: 小説のテキスト（最初の1章または数千文字）

    Returns:
        tuple[list[CharacterProfile], POVType, str]:
            - キャラクターリスト
            - 視点タイプ
            - 視点キャラクター名
    """
    from src.tts_engine import VoicevoxEngine
    
    tts = VoicevoxEngine()
    voices = await asyncio.to_thread(tts.get_available_voices)
    
    # 短くするために代表的なスタイルのみにする
    selected_voices = [v for v in voices if "ノーマル" in v.style_name or v.name in ["ナースロボ＿タイプＴ", "春日部つむぎ", "青山龍星", "玄野武宏", "白上虎太郎", "剣崎雌雄"]]
    voices_str = "\n".join([f"- ID {v.style_id}: {v.name} ({v.style_name})" for v in selected_voices])

    known_chars_str = "なし"
    if known_characters:
        known_chars_str = "\n".join([f"- {c.name} (役割: {c.role})" for c in known_characters])
    prompt = CHARACTER_EXTRACTION_PROMPT.format(novel_text=novel_text, voices_list=voices_str, known_chars_str=known_chars_str)

    response = await call_gemini_with_retry(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CHARACTER_EXTRACTION_SCHEMA,
            temperature=0.3,
        ),
    )

    result = json.loads(response.text)

    # CharacterProfileに変換
    characters = []
    for char_data in result.get("characters", []):
        characters.append(CharacterProfile(
            name=char_data.get("name", ""),
            aliases=char_data.get("aliases", []),
            first_person=char_data.get("first_person", ""),
            speech_style=char_data.get("speech_style", ""),
            gender=char_data.get("gender", "unknown"),
            role=char_data.get("role", "supporting"),
            description=char_data.get("description", ""),
            voice_id=char_data.get("recommended_voice_id"),
            voice_style_id=char_data.get("recommended_voice_id"),
        ))
        
    # 既知キャラクターが結果に含まれていなければマージする
    if known_characters:
        for kc in known_characters:
            if not any(c.name == kc.name for c in characters):
                characters.append(kc)


    # ナレーターがない場合は追加
    narrator_exists = any(c.role == "narrator" for c in characters)
    if not narrator_exists:
        characters.append(CharacterProfile(
            name="ナレーター",
            aliases=[],
            first_person="",
            speech_style="落ち着いた語り口",
            gender="unknown",
            role="narrator",
            description="地の文・状況描写を担当するナレーター",
        ))

    pov_type = POVType(result.get("pov_type", "first_person"))
    pov_character = result.get("pov_character", "")

    logger.info(f"キャラクター抽出完了: {len(characters)}人, 視点: {pov_type.value}, 視点人物: {pov_character}")

    return characters, pov_type, pov_character


# ============================================================
# Pass 2: 行単位の話者・感情解析
# ============================================================

LINE_ANALYSIS_PROMPT = """あなたは日本語小説の音声朗読用台本を作成する音声ディレクターAIです。
以下の【登場人物リスト】と【直前の会話文脈】を踏まえ、【対象テキスト】の各行を解析して指定のJSON形式で出力してください。

【登場人物リスト】
{characters_info}

【視点情報】
- 視点タイプ: {pov_type}
- 視点人物: {pov_character}

【解析ルール】
1. 各行のline_idに対応する話者(speaker)、種別(type)、感情(emotion)、感情強度(intensity: 1-3)を判定すること。
2. type は "dialogue"(発話), "internal_monologue"(心の声), "narration"(地の文), "system"(アナウンス・効果音), "scene_break"(シーン区切り) のいずれか。
3. 「」で囲まれた行は dialogue。『』や（）で囲まれた行は internal_monologue か dialogue か文脈で判断。
4. 一人称視点の地の文で、主人公の主観的な内面描写は internal_monologue、客観的な状況描写は narration とする。
5. narration の speaker は「ナレーター」とする。
6. emotion は neutral, happy, angry, sad, surprised, fear, disgust, whisper, shout, tender, smug, crying のいずれか。
7. シーン区切り（"---シーン区切り---"）は type="scene_break", speaker="ナレーター", emotion="neutral", intensity=1 とする。
8. 原文テキストは絶対に改変・省略しないこと。

【直前の会話文脈】
{previous_context}

【対象テキスト】
{target_text}
"""

LINE_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "line_id": {"type": "string"},
                    "text": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": ["dialogue", "narration", "internal_monologue", "system", "scene_break"]
                    },
                    "speaker": {"type": "string"},
                    "emotion": {
                        "type": "string",
                        "enum": ["neutral", "happy", "angry", "sad", "surprised", "fear", "disgust", "whisper", "shout", "tender", "smug", "crying"]
                    },
                    "intensity": {"type": "integer"}
                },
                "required": ["line_id", "text", "type", "speaker", "emotion", "intensity"]
            }
        }
    },
    "required": ["lines"]
}


def _format_characters_for_prompt(characters: list[CharacterProfile]) -> str:
    """キャラクターリストをプロンプト用にフォーマットする"""
    lines = []
    for char in characters:
        parts = [f"- {char.name}"]
        if char.aliases:
            parts.append(f"別名: {'、'.join(char.aliases)}")
        if char.first_person:
            parts.append(f"一人称「{char.first_person}」")
        if char.speech_style:
            parts.append(f"口調「{char.speech_style}」")
        parts.append(f"性別: {char.gender}")
        parts.append(f"役割: {char.role}")
        if char.description:
            parts.append(f"説明: {char.description}")
        lines.append(", ".join(parts))
    return "\n".join(lines)


def _format_context(context_lines: list[ProcessedLine]) -> str:
    """直前の文脈（原文）をプロンプト用にフォーマットする"""
    if not context_lines:
        return "（なし — これが最初のチャンクです）"
    return format_for_llm(context_lines)


async def analyze_lines(
    processed_lines: list[ProcessedLine],
    characters: list[CharacterProfile],
    pov_type: POVType,
    pov_character: str,
    context_lines: list[ProcessedLine] | None = None,
) -> list[ScriptLine]:
    """行単位の話者・感情解析を行う（Pass 2）

    Args:
        processed_lines: 前処理済みの行リスト（1チャンク分）
        characters: キャラクターリスト
        pov_type: 視点タイプ
        pov_character: 視点キャラクター名
        context_lines: チャンク直前の原文の行（文脈引き継ぎ用）

    Returns:
        ScriptLine のリスト
    """
    characters_info = _format_characters_for_prompt(characters)
    target_text = format_for_llm(processed_lines)
    context_text = _format_context(context_lines or [])

    prompt = LINE_ANALYSIS_PROMPT.format(
        characters_info=characters_info,
        pov_type=pov_type.value,
        pov_character=pov_character,
        previous_context=context_text,
        target_text=target_text,
    )

    response = await call_gemini_with_retry(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=LINE_ANALYSIS_SCHEMA,
            temperature=0.2,
        ),
    )

    result = json.loads(response.text)

    # LLMの判定結果をline_idで引けるようにする
    # （チャンク外のline_idや重複はここで捨てる）
    analyzed: dict[str, dict] = {}
    for line_data in result.get("lines", []):
        line_id = line_data.get("line_id", "")
        if line_id not in analyzed:
            analyzed[line_id] = line_data

    # 元の行を基準にScriptLineを組み立てる
    # テキストは常に原文を使い、LLMが書き換えた本文は採用しない
    script_lines: list[ScriptLine] = []
    missing_ids: list[str] = []

    for original in processed_lines:
        line_data = analyzed.get(original.line_id)
        if line_data is None:
            # LLMが行を落とした場合は前処理の予備判定でフォールバック
            missing_ids.append(original.line_id)
            script_lines.append(fallback_script_line(original))
            continue

        # 感情のバリデーション
        try:
            emotion = Emotion(line_data.get("emotion", "neutral"))
        except ValueError:
            emotion = Emotion.NEUTRAL

        # 種別のバリデーション
        try:
            line_type = LineType(line_data.get("type", "narration"))
        except ValueError:
            line_type = LineType.NARRATION

        script_lines.append(ScriptLine(
            line_id=original.line_id,
            text=original.original_text,
            text_to_speak=original.text_to_speak,
            type=line_type,
            speaker=line_data.get("speaker") or "ナレーター",
            emotion=emotion,
            intensity=max(1, min(3, line_data.get("intensity", 1))),
        ))

    if missing_ids:
        logger.warning(f"LLMの出力に含まれなかった行（予備判定で補完）: {', '.join(missing_ids)}")
    logger.info(f"行解析完了: {len(script_lines)}行処理")
    return script_lines


def fallback_script_line(original: ProcessedLine) -> ScriptLine:
    """LLMが判定しなかった行を、前処理の予備判定だけでScriptLineにする"""
    if original.is_scene_break:
        line_type = LineType.SCENE_BREAK
    elif original.preliminary_type == "dialogue":
        line_type = LineType.DIALOGUE
    elif original.preliminary_type in ("monologue", "inner_or_quote"):
        line_type = LineType.INTERNAL_MONOLOGUE
    elif original.preliminary_type == "system":
        line_type = LineType.SYSTEM
    else:
        line_type = LineType.NARRATION

    # 話者が分からないのでナレーターに読ませる
    return ScriptLine(
        line_id=original.line_id,
        text=original.original_text,
        text_to_speak=original.text_to_speak,
        type=line_type,
        speaker="ナレーター",
    )


async def analyze_full_episode(
    processed_lines: list[ProcessedLine],
    characters: list[CharacterProfile],
    pov_type: POVType,
    pov_character: str,
    chunk_size: int = CHUNK_SIZE,
    context_overlap: int = CONTEXT_OVERLAP,
    max_concurrency: int = ANALYSIS_CONCURRENCY,
    on_chunk: Callable[[list[ScriptLine]], Awaitable[None]] | None = None,
) -> list[ScriptLine]:
    """エピソード全体をチャンク分割して並列に解析する

    チャンクは重ならないように分け、各チャンクには直前の原文
    context_overlap 行を文脈として渡す。前のチャンクの解析結果を
    待たないので、全チャンクを同時に解析できる。

    Args:
        processed_lines: 前処理済み行リスト（エピソード全体）
        characters: キャラクターリスト
        pov_type: 視点タイプ
        pov_character: 視点キャラクター名
        chunk_size: 1チャンクの行数
        context_overlap: 文脈として渡す直前の行数
        max_concurrency: 同時にGeminiへ投げるチャンク数の上限
        on_chunk: チャンクの解析が終わるたびに（終わった順に）呼ばれる

    Returns:
        エピソード全体のScriptLineリスト（元の行順）

    Raises:
        全チャンクの解析に失敗した場合は最後の例外を送出する。
        一部だけ失敗した場合は、そのチャンクを予備判定で補完して続行する。
    """
    chunks = chunk_lines(processed_lines, chunk_size, overlap=0)
    total_chunks = len(chunks)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def analyze_chunk(index: int, start: int, chunk: list[ProcessedLine]) -> tuple[int, list[ScriptLine], Exception | None]:
        async with semaphore:
            logger.info(f"チャンク {index+1}/{total_chunks} を解析中...")
            try:
                result = await analyze_lines(
                    processed_lines=chunk,
                    characters=characters,
                    pov_type=pov_type,
                    pov_character=pov_character,
                    context_lines=processed_lines[max(0, start - context_overlap):start],
                )
                return index, result, None
            except Exception as e:
                logger.warning(f"チャンク {index+1}/{total_chunks} の解析に失敗、予備判定で補完: {e}")
                return index, [fallback_script_line(line) for line in chunk], e

    starts = [sum(len(c) for c in chunks[:i]) for i in range(total_chunks)]
    tasks = [
        asyncio.create_task(analyze_chunk(i, start, chunk))
        for i, (start, chunk) in enumerate(zip(starts, chunks))
    ]

    results: list[list[ScriptLine]] = [[] for _ in chunks]
    failures: list[Exception] = []
    try:
        for finished in asyncio.as_completed(tasks):
            index, lines, error = await finished
            results[index] = lines
            if error:
                failures.append(error)
            if on_chunk:
                await on_chunk(lines)
    finally:
        # 呼び出し側の切断などで中断した場合、残りのGemini呼び出しを止める
        unfinished = [task for task in tasks if not task.done()]
        for task in unfinished:
            task.cancel()
        if unfinished:
            logger.info(f"未完了のチャンク {len(unfinished)}/{total_chunks} 件の解析を中止")

    if failures and len(failures) == total_chunks:
        raise failures[-1]

    all_script_lines = [line for lines in results for line in lines]
    logger.info(f"エピソード全体の解析完了: {len(all_script_lines)}行（失敗チャンク {len(failures)}）")
    return all_script_lines
