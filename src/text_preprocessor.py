"""テキスト前処理モジュール

取得した小説テキストを、LLM解析に適した形式に変換する。
- 行番号の付与
- シーン区切りの検出
- 括弧種別の判定
- 読み上げ用テキストの生成
"""
import re
from dataclasses import dataclass
from src.schemas import LineType


# シーン区切りのパターン
SCENE_BREAK_PATTERNS = [
    re.compile(r'^[◆◇◇＊\*◆◇＊★☆●○■□▲△▽▼]{2,}$'),
    re.compile(r'^[―ー\-=＝]{3,}$'),
    re.compile(r'^[\s　]*$'),  # 空行
    re.compile(r'^\s*[・]{3,}\s*$'),
]

# 括弧の種類と意味
BRACKET_TYPES = {
    ('「', '」'): 'dialogue',           # 通常の発話
    ('『', '』'): 'inner_or_quote',     # 心の声・引用・念話
    ('（', '）'): 'monologue',          # モノローグ・心の声
    ('(', ')'): 'monologue',           # モノローグ（半角）
    ('【', '】'): 'system',             # システムメッセージ
    ('《', '》'): 'system',             # スキル名等（ルビと区別注意）
}


@dataclass
class ProcessedLine:
    """前処理済みの行データ"""
    line_id: str           # 行ID (例: "L001")
    original_text: str     # 原文
    text_to_speak: str     # 読み上げ用テキスト（括弧除去等）
    preliminary_type: str  # 予備的な種別判定
    is_scene_break: bool   # シーン区切りか


def add_line_ids(
    paragraphs: list[str],
    ruby_paragraphs: list[str] | None = None,
) -> list[ProcessedLine]:
    """段落リストに行番号を付与し、前処理を行う

    Args:
        paragraphs: 段落テキストのリスト
        ruby_paragraphs: ルビ読み適用済みの段落リスト（読み上げ用）。
            paragraphs と件数が一致しない場合は無視する。

    Returns:
        ProcessedLine のリスト
    """
    processed_lines: list[ProcessedLine] = []
    line_counter = 1
    if not ruby_paragraphs or len(ruby_paragraphs) != len(paragraphs):
        ruby_paragraphs = paragraphs

    for paragraph, ruby_paragraph in zip(paragraphs, ruby_paragraphs):
        # 空行チェック
        stripped = paragraph.strip()
        if not stripped:
            # 空行はシーン区切りの候補
            processed_lines.append(ProcessedLine(
                line_id=f"L{line_counter:03d}",
                original_text="",
                text_to_speak="",
                preliminary_type="scene_break",
                is_scene_break=True,
            ))
            line_counter += 1
            continue
        
        # シーン区切り判定
        is_break = any(pattern.match(stripped) for pattern in SCENE_BREAK_PATTERNS)
        if is_break:
            processed_lines.append(ProcessedLine(
                line_id=f"L{line_counter:03d}",
                original_text=stripped,
                text_to_speak="",
                preliminary_type="scene_break",
                is_scene_break=True,
            ))
            line_counter += 1
            continue
        
        # 括弧種別による予備判定
        preliminary_type = _detect_bracket_type(stripped)
        
        # 読み上げ用テキスト生成
        text_to_speak = _generate_speakable_text(ruby_paragraph.strip() or stripped, preliminary_type)
        
        processed_lines.append(ProcessedLine(
            line_id=f"L{line_counter:03d}",
            original_text=stripped,
            text_to_speak=text_to_speak,
            preliminary_type=preliminary_type,
            is_scene_break=False,
        ))
        line_counter += 1
    
    return processed_lines


def _detect_bracket_type(text: str) -> str:
    """テキストの括弧種別を判定する
    
    行頭の括弧で予備的な種別を判定。
    最終的な判定はLLMが行う。
    """
    stripped = text.strip()
    
    for (open_bracket, close_bracket), bracket_type in BRACKET_TYPES.items():
        if stripped.startswith(open_bracket):
            return bracket_type
    
    return "narration"  # デフォルトは地の文


def _generate_speakable_text(text: str, bracket_type: str) -> str:
    """読み上げ用テキストを生成する
    
    - 鍵括弧を除去
    - 余分な記号を除去
    - ただし原文は保持（ScriptLine.text に残す）
    """
    result = text
    
    # 行頭・行末の括弧を除去
    bracket_pairs = [('「', '」'), ('『', '』'), ('（', '）'), ('(', ')'), ('【', '】')]
    for open_b, close_b in bracket_pairs:
        if result.startswith(open_b) and result.endswith(close_b):
            result = result[1:-1]
            break
    
    # 読み上げに不要な記号を処理
    # 「……」→ 短い無音に変換（TTSが処理するのでそのまま残す）
    # 「――」→ そのまま（TTSが適切に処理する）
    
    return result.strip()


def format_for_llm(processed_lines: list[ProcessedLine]) -> str:
    """LLMに渡す形式にフォーマットする
    
    例:
    [L001] 俺は朝目を覚ました。
    [L002] 「おはよう」
    [L003] と、隣にいたアクアが声をかけてきた。
    """
    formatted_lines = []
    for line in processed_lines:
        if line.is_scene_break:
            formatted_lines.append(f"[{line.line_id}] ---シーン区切り---")
        else:
            formatted_lines.append(f"[{line.line_id}] {line.original_text}")
    
    return "\n".join(formatted_lines)


def chunk_lines(
    processed_lines: list[ProcessedLine],
    chunk_size: int = 50,
    overlap: int = 8,
) -> list[list[ProcessedLine]]:
    """行リストをチャンクに分割する（スライディングウィンドウ）
    
    Args:
        processed_lines: 前処理済み行リスト
        chunk_size: 1チャンクの行数
        overlap: チャンク間のオーバーラップ行数（文脈引き継ぎ用）
    
    Returns:
        チャンクのリスト
    """
    # シーン区切りでない行だけをカウント対象にする
    non_break_lines = [l for l in processed_lines if not l.is_scene_break]
    
    if len(non_break_lines) <= chunk_size:
        return [processed_lines]  # 全体が1チャンクに収まる場合
    
    chunks: list[list[ProcessedLine]] = []
    step = chunk_size - overlap
    
    for start in range(0, len(processed_lines), step):
        chunk = processed_lines[start:start + chunk_size]
        if chunk:
            chunks.append(chunk)
        
        # 最後のチャンクに到達
        if start + chunk_size >= len(processed_lines):
            break
    
    return chunks
