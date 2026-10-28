"""音声合成エンジンモジュール

VOICEVOX REST API と連携して音声を生成する。
将来的にStyle-Bert-VITS2やGPT-SoVITSにも対応可能な
抽象化された設計。
"""
import io
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import requests

from src.config import VOICEVOX_HOST
from src.schemas import VoiceInfo, ScriptLine, Emotion

logger = logging.getLogger(__name__)


class TTSEngine(ABC):
    """TTSエンジンの抽象基底クラス"""
    
    @abstractmethod
    def get_available_voices(self) -> list[VoiceInfo]:
        """利用可能な話者リストを取得する"""
        pass
    
    @abstractmethod
    def synthesize(self, text: str, voice_id: int, style_id: int | None = None, 
                   speed: float = 1.0, pitch: float = 0.0, intonation: float = 1.0) -> bytes:
        """テキストから音声（WAVバイト）を生成する"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """エンジンが利用可能か確認する"""
        pass


class VoicevoxEngine(TTSEngine):
    """VOICEVOX REST API を使用した音声合成エンジン
    
    VOICEVOXが localhost:50021 で起動している必要がある。
    API仕様: https://voicevox.github.io/voicevox_engine/api/
    """
    
    def __init__(self, host: str = VOICEVOX_HOST):
        self.host = host.rstrip("/")
        self._speakers_cache: list[dict] | None = None
    
    def is_available(self) -> bool:
        """VOICEVOXエンジンが起動しているか確認する"""
        try:
            response = requests.get(f"{self.host}/version", timeout=3)
            return response.status_code == 200
        except requests.ConnectionError:
            return False
    
    def get_version(self) -> str:
        """VOICEVOXのバージョンを取得する"""
        response = requests.get(f"{self.host}/version", timeout=5)
        response.raise_for_status()
        return response.json()
    
    def get_available_voices(self) -> list[VoiceInfo]:
        """利用可能な話者・スタイルの一覧を取得する
        
        Returns:
            VoiceInfo のリスト。各スタイルが個別のエントリとして返される。
            例: ずんだもん(ノーマル), ずんだもん(あまあま), ずんだもん(ツンツン)...
        """
        speakers = self._get_speakers()
        voices: list[VoiceInfo] = []
        
        for speaker in speakers:
            speaker_name = speaker["name"]
            for style in speaker.get("styles", []):
                voices.append(VoiceInfo(
                    voice_id=style["id"],
                    name=speaker_name,
                    style_name=style["name"],
                    style_id=style["id"],
                    engine="voicevox",
                ))
        
        return voices
    
    def get_speakers_summary(self) -> list[dict]:
        """話者の概要一覧を取得する（UIのプルダウン用）
        
        Returns:
            [{"name": "ずんだもん", "styles": [{"name": "ノーマル", "id": 3}, ...]}]
        """
        return self._get_speakers()
    
    def synthesize(
        self,
        text: str,
        voice_id: int,
        style_id: int | None = None,
        speed: float = 1.0,
        pitch: float = 0.0,
        intonation: float = 1.0,
    ) -> bytes:
        """テキストからWAV音声を生成する
        
        VOICEVOXの2ステップ合成:
        1. /audio_query でクエリ（音素・アクセント情報）を生成
        2. /synthesis でWAV音声を生成
        
        Args:
            text: 読み上げるテキスト
            voice_id: 話者ID（スタイルID）
            style_id: スタイルID（voice_idと同じ場合は省略可）
            speed: 話速スケール（0.5〜2.0、デフォルト1.0）
            pitch: ピッチ（-0.15〜0.15、デフォルト0.0）
            intonation: 抑揚スケール（0.0〜2.0、デフォルト1.0）
        
        Returns:
            WAV形式の音声データ（bytes）
        
        Raises:
            requests.RequestException: API通信エラー
            ValueError: テキストが空の場合
        """
        if not text or not text.strip():
            raise ValueError("読み上げテキストが空です")
        
        speaker = style_id if style_id is not None else voice_id
        
        # Step 1: Audio Query の生成
        query_response = requests.post(
            f"{self.host}/audio_query",
            params={"text": text, "speaker": speaker},
            timeout=30,
        )
        query_response.raise_for_status()
        audio_query = query_response.json()
        
        # パラメータの調整
        audio_query["speedScale"] = speed
        audio_query["pitchScale"] = pitch
        audio_query["intonationScale"] = intonation
        
        # Step 2: 音声合成
        synth_response = requests.post(
            f"{self.host}/synthesis",
            params={"speaker": speaker},
            json=audio_query,
            timeout=60,
        )
        synth_response.raise_for_status()
        
        return synth_response.content
    
    def synthesize_script_line(
        self,
        line: ScriptLine,
        voice_id: int,
        emotion_style_map: dict[str, int] | None = None,
        speed: float = 1.0,
    ) -> bytes:
        """ScriptLineから音声を生成する
        
        感情に応じてスタイルIDを切り替え、
        強度に応じてパラメータを調整する。
        
        Args:
            line: 台本の行データ
            voice_id: デフォルトの話者ID
            emotion_style_map: 感情→スタイルIDのマッピング。
                None の場合は voice_id のキャラのスタイルから自動で作る。

        Returns:
            WAV形式の音声データ
        """
        # 読み上げテキストの決定
        text = line.text_to_speak if line.text_to_speak else line.text
        if not text.strip():
            return b""

        # 感情に応じたスタイルIDの選択
        if emotion_style_map is None:
            emotion_style_map = self.get_emotion_style_map(voice_id)
        style_id = emotion_style_map.get(line.emotion.value, voice_id)
        
        # 感情強度に応じたパラメータ調整
        # 引数の speed をベースとして計算
        final_speed = speed
        pitch = 0.0
        intonation = 1.0
        
        if line.emotion == Emotion.WHISPER:
            final_speed *= 0.85
            intonation = 0.7
        elif line.emotion == Emotion.SHOUT:
            final_speed *= 1.1
            pitch = 0.05
            intonation = 1.5
        elif line.emotion == Emotion.SAD:
            final_speed *= 0.9
            intonation = 0.8
        elif line.emotion == Emotion.HAPPY:
            final_speed *= 1.05
            intonation = 1.3
        elif line.emotion == Emotion.ANGRY:
            final_speed *= 1.1
            pitch = 0.03
            intonation = 1.4
        
        # 強度に応じてパラメータを増幅
        if line.intensity >= 2:
            intonation *= 1.15
        if line.intensity >= 3:
            intonation *= 1.15
            final_speed *= 1.05

        # スタイルが感情を表現している場合は、調整を半分に弱めて過剰な演技を防ぐ
        if style_id != voice_id:
            final_speed = speed * (1 + (final_speed / speed - 1) * 0.5)
            pitch *= 0.5
            intonation = 1 + (intonation - 1) * 0.5

        return self.synthesize(
            text=text,
            voice_id=voice_id,
            style_id=style_id,
            speed=final_speed,
            pitch=pitch,
            intonation=intonation,
        )
    
    def get_emotion_style_map(self, base_style_id: int) -> dict[str, int]:
        """選ばれた声に対応する感情→スタイルIDの対応を返す"""
        return build_emotion_style_map(base_style_id, self._get_speakers())

    def _get_speakers(self) -> list[dict]:
        """話者情報をキャッシュ付きで取得する"""
        if self._speakers_cache is None:
            response = requests.get(f"{self.host}/speakers", timeout=10)
            response.raise_for_status()
            self._speakers_cache = response.json()
        return self._speakers_cache


# --- Emotion -> VOICEVOX Style マッピング ---

# 感情ごとに使うスタイル名の候補（先頭ほど優先）
# VOICEVOX 0.25 の全127スタイルを確認して作成。
# セクシー・クイーン・アナウンスなど、感情と結びつかないスタイルは使わない。
EMOTION_STYLE_CANDIDATES: dict[str, list[str]] = {
    "happy": ["喜び", "あまあま", "甘々", "たのしい", "わーい", "うきうき", "明るい", "元気", "楽々"],
    "tender": ["あまあま", "甘々", "しっとり", "おちつき"],
    "angry": ["怒り", "ツンギレ", "おこ", "人間（怒り）ver.", "不機嫌", "ツンツン"],
    "smug": ["ツンツン", "呆れ"],
    "disgust": ["不機嫌", "呆れ", "ツンツン"],
    "sad": ["悲しみ", "哀しみ", "かなしみ", "かなしい", "なみだめ", "泣き", "絶望と敗北"],
    "crying": ["びえーん", "泣き", "なみだめ", "悲しみ", "哀しみ", "かなしみ", "かなしい"],
    "whisper": ["ささやき", "囁き", "ヒソヒソ", "内緒話"],
    "fear": ["恐怖", "こわがり", "びくびく", "おどおど", "人見知り"],
    "surprised": ["おどろき"],
    "shout": ["熱血", "つよつよ", "覚醒"],
}


def build_emotion_style_map(base_style_id: int, speakers: list[dict]) -> dict[str, int]:
    """選ばれた声（スタイルID）と同じキャラのスタイルから、感情→スタイルIDの対応を作る

    例: ずんだもん(ノーマル: 3) の場合
    {"happy": 1(あまあま), "sad": 76(なみだめ), "whisper": 22(ささやき), ...}

    対応するスタイルがない感情は含めない（元の声のまま読む）。
    """
    for speaker in speakers:
        styles = {style["name"]: style["id"] for style in speaker.get("styles", [])}
        if base_style_id not in styles.values():
            continue

        style_map: dict[str, int] = {}
        for emotion, candidates in EMOTION_STYLE_CANDIDATES.items():
            for name in candidates:
                if name in styles:
                    style_map[emotion] = styles[name]
                    break
        return style_map

    return {}
