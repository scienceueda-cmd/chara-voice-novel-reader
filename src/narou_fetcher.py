"""なろう小説テキスト取得モジュール

小説家になろう（https://ncode.syosetu.com）から
小説のメタデータと本文テキストを取得する。

使用API:
- 公式なろう小説API（メタデータのみ）
- Webスクレイピング（本文取得）
"""
import re
import time
import logging
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, Tag

from src.config import (
    NAROU_BASE_URL, NAROU_API_URL, FETCH_INTERVAL
)
from src.schemas import NovelMetadata, Episode, TOCEpisode, TOCChapter

logger = logging.getLogger(__name__)

# User-Agentヘッダー
HEADERS = {
    "User-Agent": "KyaraWakeAIVoiceNovelReader/1.0 (Personal Use)"
}


def extract_ncode(url_or_ncode: str) -> str:
    """URLまたはNコードからNコードを抽出する
    
    Args:
        url_or_ncode: なろうのURL（例: https://ncode.syosetu.com/n1234ab/）
                      またはNコード（例: n1234ab）
    
    Returns:
        小文字のNコード（例: n1234ab）
    
    Raises:
        ValueError: Nコードを抽出できない場合
    """
    # URLからNコードを抽出
    match = re.search(r'ncode\.syosetu\.com/([Nn]\d+[A-Za-z]+)', url_or_ncode)
    if match:
        return match.group(1).lower()
    
    # 直接Nコードが渡された場合
    match = re.match(r'^([Nn]\d+[A-Za-z]+)$', url_or_ncode.strip())
    if match:
        return match.group(1).lower()
    
    raise ValueError(f"Nコードを抽出できません: {url_or_ncode}")


def get_novel_metadata(ncode: str) -> NovelMetadata:
    """公式APIから小説のメタデータを取得する
    
    Args:
        ncode: 作品Nコード
    
    Returns:
        NovelMetadata オブジェクト
    
    Raises:
        ValueError: 作品が見つからない場合
        requests.RequestException: API通信エラー
    """
    params = {
        "ncode": ncode,
        "out": "json",
        "of": "t-n-u-w-s-ga-l-nt-e-k-gf",  # title, ncode, userid, writer, story, general_all_no, length, novel_type, end, keyword, kaiwaritu (gf is not the right key)
        "lim": 1,
    }
    
    response = requests.get(NAROU_API_URL, params=params, headers=HEADERS, timeout=10)
    response.raise_for_status()
    data = response.json()
    
    if data[0]["allcount"] == 0:
        raise ValueError(f"作品が見つかりません: {ncode}")
    
    novel = data[1]
    
    return NovelMetadata(
        ncode=novel.get("ncode", ncode).lower(),
        title=novel.get("title", ""),
        writer=novel.get("writer", ""),
        story=novel.get("story", ""),
        total_episodes=novel.get("general_all_no", 0),
        novel_type=novel.get("novel_type", 1),
        conversation_rate=novel.get("kaiwaritu", 0),
        total_chars=novel.get("length", 0),
    )


def _process_ruby(element: Tag) -> str:
    """rubyタグを処理してルビ読みを優先するテキストに変換する
    
    <ruby>妖精<rp>(</rp><rt>フェアリー</rt><rp>)</rp></ruby>
    → "フェアリー" （ルビ読みを優先）
    
    ルビがない場合はそのままテキストを返す。
    """
    result_parts = []
    
    for child in element.children:
        if isinstance(child, Tag):
            if child.name == "ruby":
                # <rt>タグの中身（ルビ）を取得
                rt = child.find("rt")
                if rt:
                    result_parts.append(rt.get_text())
                else:
                    # rtがない場合は元テキスト
                    result_parts.append(child.get_text())
            elif child.name in ("rp", "rt"):
                # rp, rt単体は無視（rubyの外にある場合）
                continue
            else:
                # その他のタグは再帰処理
                result_parts.append(_process_ruby(child))
        else:
            # テキストノード
            result_parts.append(str(child))
    
    return "".join(result_parts)


def _get_text_preserving_ruby(element: Tag) -> tuple[str, str]:
    """要素からテキストを取得する（ルビ処理あり/なし両方）
    
    Returns:
        tuple[str, str]: (原文テキスト, ルビ適用済みテキスト)
            原文テキスト: ルビ部分は漢字のまま
            ルビ適用済み: ルビ部分はひらがな/カタカナに置換済み
    """
    # 原文（ルビ除去）: rubyタグから<rt><rp>を除去
    original_parts = []
    ruby_parts = []
    
    for child in element.children:
        if isinstance(child, Tag):
            if child.name == "ruby":
                # 原文: 漢字部分のみ
                texts = []
                for sub in child.children:
                    if isinstance(sub, Tag):
                        if sub.name in ("rt", "rp"):
                            continue
                        texts.append(sub.get_text())
                    else:
                        texts.append(str(sub))
                original_parts.append("".join(texts))
                
                # ルビ適用: <rt>の中身
                rt = child.find("rt")
                if rt:
                    ruby_parts.append(rt.get_text())
                else:
                    ruby_parts.append("".join(texts))
            elif child.name == "br":
                original_parts.append("\n")
                ruby_parts.append("\n")
            else:
                text = child.get_text()
                original_parts.append(text)
                ruby_parts.append(text)
        else:
            text = str(child)
            original_parts.append(text)
            ruby_parts.append(text)
    
    return "".join(original_parts), "".join(ruby_parts)


def fetch_episode(ncode: str, episode_no: int) -> Episode:
    """指定話の本文テキストを取得する
    
    Args:
        ncode: 作品Nコード
        episode_no: 話数（短編の場合は0を指定）
    
    Returns:
        Episode オブジェクト
    
    Raises:
        ValueError: 本文が見つからない場合
        requests.RequestException: 通信エラー
    """
    if episode_no == 0:
        url = f"{NAROU_BASE_URL}/{ncode}/"
    else:
        url = f"{NAROU_BASE_URL}/{ncode}/{episode_no}/"
    
    logger.info(f"本文取得中: {url}")
    
    response = requests.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status()
    response.encoding = "utf-8"
    
    soup = BeautifulSoup(response.text, "lxml")
    
    # サブタイトル
    subtitle_elem = soup.select_one(".novel_subtitle, .p-novel__subtitle")
    subtitle = subtitle_elem.get_text(strip=True) if subtitle_elem else ""
    
    # 本文
    honbun_elem = soup.select_one("#novel_honbun, .p-novel__body")
    if not honbun_elem:
        raise ValueError(f"本文が見つかりません: {url}")
    
    # 段落ごとにテキスト抽出（ルビ処理あり）
    paragraphs = []
    raw_lines = []
    
    for p in honbun_elem.find_all("p"):
        original_text, ruby_text = _get_text_preserving_ruby(p)
        original_text = original_text.strip()
        ruby_text = ruby_text.strip()
        
        if original_text:  # 空段落はスキップ
            paragraphs.append(original_text)
            raw_lines.append(ruby_text)

    return Episode(
        episode_no=episode_no,
        subtitle=subtitle,
        raw_text="\n".join(paragraphs),
        paragraphs=paragraphs,
        ruby_paragraphs=raw_lines,
    )


def fetch_table_of_contents(ncode: str, page: int = 1) -> list[TOCChapter]:
    """目次（章構成・全話リスト）を取得する

    Args:
        ncode: 作品Nコード
        page: 取得するページ番号

    Returns:
        TOCChapter のリスト
    """
    url = f"{NAROU_BASE_URL}/{ncode}/"
    if page > 1:
        url += f"?p={page}"
        
    logger.info(f"目次取得中: {url}")

    response = requests.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status()
    response.encoding = "utf-8"
    
    soup = BeautifulSoup(response.text, "lxml")
    
    chapters: list[TOCChapter] = []
    current_chapter = TOCChapter(chapter_title="", episodes=[])
    
    # 目次要素を探す
    toc_area = soup.select_one(".index_box, .p-eplist")
    if not toc_area:
        # 短編の場合は目次なし
        return []
    
    for element in toc_area.children:
        if not isinstance(element, Tag):
            continue
        
        # 章タイトル
        element_classes = str(element.get("class", []))
        if "chapter_title" in element_classes or "p-eplist__chapter-title" in element_classes:
            # 前の章を保存
            if current_chapter.episodes:
                chapters.append(current_chapter)
            current_chapter = TOCChapter(
                chapter_title=element.get_text(strip=True),
                episodes=[]
            )
        
        # 各話リンク
        elif element.name == "dl" or (element.name == "div" and "p-eplist__sublist" in str(element.get("class", []))):
            subtitle_link = element.select_one("a")
            if subtitle_link:
                href = subtitle_link.get("href", "")
                ep_match = re.search(r'/(\d+)/?$', href)
                ep_no = int(ep_match.group(1)) if ep_match else 0
                
                date_elem = element.select_one("dt, .p-eplist__update")
                date_str = date_elem.get_text(strip=True) if date_elem else ""
                
                current_chapter.episodes.append(TOCEpisode(
                    episode_no=ep_no,
                    subtitle=subtitle_link.get_text(strip=True),
                    url=href,
                    date=date_str,
                ))
    
    # 最後の章を追加
    if current_chapter.episodes:
        chapters.append(current_chapter)
    
    return chapters


def fetch_multiple_episodes(
    ncode: str,
    episode_numbers: list[int],
    interval: float = FETCH_INTERVAL,
) -> list[Episode]:
    """複数話をインターバルを空けて取得する
    
    Args:
        ncode: 作品Nコード
        episode_numbers: 取得する話数のリスト
        interval: リクエスト間隔（秒）
    
    Returns:
        Episode のリスト
    """
    episodes = []
    
    for i, ep_no in enumerate(episode_numbers):
        if i > 0:
            time.sleep(interval)  # レートリミット
        
        try:
            episode = fetch_episode(ncode, ep_no)
            episodes.append(episode)
            logger.info(f"  取得完了: 第{ep_no}話 '{episode.subtitle}' ({len(episode.paragraphs)}段落)")
        except Exception as e:
            logger.error(f"  取得失敗: 第{ep_no}話 - {e}")
    
    return episodes
