"""データスキーマ定義

小説テキスト、キャラクタープロファイル、台本行などの
構造化データモデルを定義する。
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from enum import Enum


class LineType(str, Enum):
    """行の種別"""
    DIALOGUE = "dialogue"              # 発話（セリフ）
    NARRATION = "narration"            # 地の文（ナレーション）
    INTERNAL_MONOLOGUE = "internal_monologue"  # 心の声
    SYSTEM = "system"                  # システムメッセージ・効果音
    SCENE_BREAK = "scene_break"        # シーン区切り


class Emotion(str, Enum):
    """感情カテゴリ"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    ANGRY = "angry"
    SAD = "sad"
    SURPRISED = "surprised"
    FEAR = "fear"
    DISGUST = "disgust"
    WHISPER = "whisper"
    SHOUT = "shout"
    TENDER = "tender"
    SMUG = "smug"
    CRYING = "crying"


class POVType(str, Enum):
    """視点の種類"""
    FIRST_PERSON = "first_person"                  # 一人称
    THIRD_PERSON_LIMITED = "third_person_limited"    # 三人称限定
    THIRD_PERSON_OMNISCIENT = "third_person_omniscient"  # 三人称全知


class CharacterProfile(BaseModel):
    """キャラクタープロファイル
    
    LLMが小説テキストから抽出するキャラクター情報。
    ユーザーがボイスを割り当てる際の基本データになる。
    """
    name: str = Field(description="キャラクター名")
    aliases: list[str] = Field(default_factory=list, description="別名・あだ名のリスト")
    first_person: str = Field(default="", description="一人称（俺、私、僕 等）")
    speech_style: str = Field(default="", description="口調の特徴（〜だぜ、〜ですわ 等）")
    gender: str = Field(default="unknown", description="性別: male / female / unknown")
    role: str = Field(default="supporting", description="役割: protagonist / heroine / supporting / mob / narrator")
    description: str = Field(default="", description="キャラクターの簡単な説明")
    voice_id: int | None = Field(default=None, description="割り当てられたTTS話者ID")
    voice_style_id: int | None = Field(default=None, description="割り当てられたTTSスタイルID")


class ScriptLine(BaseModel):
    """台本の1行
    
    LLMが解析した行単位の情報。
    この情報をもとにTTSエンジンが音声を生成する。
    """
    line_id: str = Field(description="行ID（例: L001）")
    text: str = Field(description="原文テキスト（改変なし）")
    text_to_speak: str = Field(default="", description="読み上げ用テキスト（括弧除去・ルビ適用済み）")
    type: LineType = Field(description="行の種別")
    speaker: str = Field(description="話者名（キャラ名 or 'ナレーター'）")
    emotion: Emotion = Field(default=Emotion.NEUTRAL, description="感情")
    intensity: int = Field(default=1, ge=1, le=3, description="感情の強さ（1-3）")


class NovelMetadata(BaseModel):
    """小説のメタデータ"""
    ncode: str = Field(description="作品Nコード")
    title: str = Field(description="作品タイトル")
    writer: str = Field(description="作者名")
    story: str = Field(default="", description="あらすじ")
    total_episodes: int = Field(default=0, description="全話数")
    novel_type: int = Field(default=1, description="1: 連載, 2: 短編")
    conversation_rate: int = Field(default=0, description="会話率(%)")
    total_chars: int = Field(default=0, description="総文字数")


class Episode(BaseModel):
    """1話分のデータ"""
    episode_no: int = Field(description="話数")
    subtitle: str = Field(default="", description="サブタイトル")
    raw_text: str = Field(default="", description="生テキスト")
    paragraphs: list[str] = Field(default_factory=list, description="段落リスト")
    ruby_paragraphs: list[str] = Field(default_factory=list, description="ルビ読み適用済みの段落リスト（paragraphsと同じ並び）")


class TOCEpisode(BaseModel):
    """目次の話情報"""
    episode_no: int = Field(description="話数")
    subtitle: str = Field(description="サブタイトル")
    url: str = Field(default="", description="URL")
    date: str = Field(default="", description="投稿日時")


class TOCChapter(BaseModel):
    """目次の章情報"""
    chapter_title: str = Field(default="", description="章タイトル")
    episodes: list[TOCEpisode] = Field(default_factory=list, description="話リスト")


class AnalysisResult(BaseModel):
    """1話分の解析結果"""
    metadata: NovelMetadata
    episode: Episode
    characters: list[CharacterProfile] = Field(default_factory=list)
    pov_type: POVType = Field(default=POVType.FIRST_PERSON)
    pov_character: str = Field(default="")
    script_lines: list[ScriptLine] = Field(default_factory=list)


class VoiceInfo(BaseModel):
    """TTS話者情報"""
    voice_id: int = Field(description="話者ID")
    name: str = Field(description="話者名")
    style_name: str = Field(default="ノーマル", description="スタイル名")
    style_id: int = Field(default=0, description="スタイルID")
    engine: str = Field(default="voicevox", description="TTSエンジン名")


class CharacterVoiceMapping(BaseModel):
    """キャラクターとボイスの対応"""
    character_name: str = Field(description="キャラクター名")
    voice_info: VoiceInfo = Field(description="割り当てられたボイス情報")
    emotion_style_map: dict[str, int] = Field(
        default_factory=dict,
        description="感情→スタイルIDのマッピング（例: {'happy': 3, 'angry': 5}）"
    )


class AnalyzeCharactersRequest(BaseModel):
    text: str = Field(..., description="エピソード本文")
    ncode: str = Field("", description="Nコード")
    known_characters: list[CharacterProfile] = Field(default_factory=list, description="既知のキャラクター")
