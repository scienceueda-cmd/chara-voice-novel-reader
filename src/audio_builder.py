"""音声結合モジュール

個別の音声セグメント（WAV）を結合し、
適切な無音挿入とともに1つの音声ファイルにまとめる。
"""
import io
import logging
from pathlib import Path
from typing import Callable

from pydub import AudioSegment

from src.config import SILENCE_BETWEEN_LINES_MS, SILENCE_SCENE_BREAK_MS, AUDIO_DIR
from src.schemas import ScriptLine, LineType

logger = logging.getLogger(__name__)


def create_silence(duration_ms: int, sample_rate: int = 24000) -> AudioSegment:
    """指定ミリ秒の無音を生成する"""
    return AudioSegment.silent(duration=duration_ms, frame_rate=sample_rate)


def wav_bytes_to_segment(wav_bytes: bytes) -> AudioSegment:
    """WAVバイト列をAudioSegmentに変換する"""
    return AudioSegment.from_wav(io.BytesIO(wav_bytes))


def build_audio(
    script_lines: list[ScriptLine],
    audio_segments: dict[str, bytes],
    silence_between_lines_ms: int = SILENCE_BETWEEN_LINES_MS,
    silence_scene_break_ms: int = SILENCE_SCENE_BREAK_MS,
    progress_callback: Callable[[int, int], None] | None = None,
) -> AudioSegment:
    """台本と音声セグメントから完成音声を構築する
    
    Args:
        script_lines: 台本行のリスト
        audio_segments: {line_id: WAVバイト} の辞書
        silence_between_lines_ms: セリフ間の無音（ミリ秒）
        silence_scene_break_ms: シーン区切りの無音（ミリ秒）
        progress_callback: 進捗コールバック (current, total)
    
    Returns:
        結合されたAudioSegment
    """
    combined = AudioSegment.empty()
    total = len(script_lines)
    
    for i, line in enumerate(script_lines):
        if progress_callback:
            progress_callback(i + 1, total)
        
        # シーン区切り
        if line.type == LineType.SCENE_BREAK:
            combined += create_silence(silence_scene_break_ms)
            continue
        
        # 音声セグメントを追加
        if line.line_id in audio_segments:
            wav_bytes = audio_segments[line.line_id]
            if wav_bytes:
                try:
                    segment = wav_bytes_to_segment(wav_bytes)
                    combined += segment
                except Exception as e:
                    logger.error(f"音声セグメント変換エラー ({line.line_id}): {e}")
        
        # セリフ間の無音挿入
        if i < total - 1:  # 最後の行の後には無音を入れない
            next_line = script_lines[i + 1]
            if next_line.type == LineType.SCENE_BREAK:
                pass  # シーン区切りの無音は別途処理
            elif line.speaker != next_line.speaker:
                # 話者が変わる場合は少し長めの無音
                combined += create_silence(int(silence_between_lines_ms * 1.5))
            else:
                combined += create_silence(silence_between_lines_ms)
    
    return combined


def export_audio(
    audio: AudioSegment,
    output_path: Path | str,
    format: str = "mp3",
    bitrate: str = "192k",
) -> Path:
    """音声をファイルにエクスポートする
    
    Args:
        audio: AudioSegment
        output_path: 出力ファイルパス
        format: 出力フォーマット（mp3, wav）
        bitrate: MP3のビットレート
    
    Returns:
        出力ファイルのPath
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "mp3":
        audio.export(str(output_path), format="mp3", bitrate=bitrate)
    elif format == "wav":
        audio.export(str(output_path), format="wav")
    else:
        raise ValueError(f"サポートされていないフォーマット: {format}")
    
    logger.info(f"音声エクスポート完了: {output_path} ({audio.duration_seconds:.1f}秒)")
    return output_path


def generate_output_filename(ncode: str, episode_no: int, format: str = "mp3") -> Path:
    """出力ファイル名を生成する"""
    return AUDIO_DIR / f"{ncode}_ep{episode_no:03d}.{format}"
