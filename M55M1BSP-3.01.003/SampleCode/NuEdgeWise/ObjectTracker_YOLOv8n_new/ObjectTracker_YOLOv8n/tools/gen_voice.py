"""
Generate Chinese TTS voice clips for the EdgeEye guide-vision project.

Output: 86 WAV files (16 kHz, 16-bit, mono PCM) in ./voice/
  - dir_0..2.wav      directions (left / center / right)
  - act_caution/danger/unknown.wav
  - cls_0..79.wav     COCO classes (aligned with Labels.cpp index)

Requirements:
  pip install edge-tts
  ffmpeg in PATH   (https://ffmpeg.org/download.html)

Usage:
  python gen_voice.py
Then copy the generated `voice/` folder to the SD card root (so it lives at 0:/voice/).
"""

import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

import edge_tts

VOICE = "zh-TW-HsiaoChenNeural"   # 台灣女聲，清晰、語速適中
OUT_DIR = Path(__file__).resolve().parent.parent / "voice"
TMP_DIR = OUT_DIR / "_mp3"

DIRECTIONS = ["左前方", "正前方", "右前方"]

ACTIONS = {
    "act_caution": "有",
    "act_danger":  "正在接近",
    "act_unknown": "有障礙物",
}

CLASSES = [
    "人", "腳踏車", "汽車", "機車", "飛機", "公車", "火車", "卡車", "船", "紅綠燈",
    "消防栓", "停止標誌", "停車計時器", "長椅", "鳥", "貓", "狗", "馬", "羊", "牛",
    "大象", "熊", "斑馬", "長頸鹿", "背包", "雨傘", "手提包", "領帶", "行李箱", "飛盤",
    "滑雪板", "雪板", "球", "風箏", "球棒", "棒球手套", "滑板", "衝浪板", "網球拍", "瓶子",
    "酒杯", "杯子", "叉子", "刀子", "湯匙", "碗", "香蕉", "蘋果", "三明治", "橘子",
    "花椰菜", "紅蘿蔔", "熱狗", "披薩", "甜甜圈", "蛋糕", "椅子", "沙發", "盆栽", "床",
    "餐桌", "馬桶", "電視", "筆電", "滑鼠", "遙控器", "鍵盤", "手機", "微波爐", "烤箱",
    "烤麵包機", "水槽", "冰箱", "書", "時鐘", "花瓶", "剪刀", "玩偶", "吹風機", "牙刷",
]
assert len(CLASSES) == 80


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        sys.exit("[error] ffmpeg not found in PATH. Install it first.")


async def tts(text: str, mp3_path: Path) -> None:
    comm = edge_tts.Communicate(text, VOICE, rate="+0%")
    await comm.save(str(mp3_path))


def to_wav(mp3_path: Path, wav_path: Path) -> None:
    # 16 kHz / 16-bit / mono PCM
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(mp3_path),
            "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
            str(wav_path),
        ],
        check=True,
    )


async def make_one(text: str, name: str) -> None:
    mp3 = TMP_DIR / f"{name}.mp3"
    wav = OUT_DIR / f"{name}.wav"
    await tts(text, mp3)
    to_wav(mp3, wav)
    print(f"  {name:12s}  {text}")


async def main() -> None:
    check_ffmpeg()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[info] voice: {VOICE}")
    print(f"[info] output: {OUT_DIR}")

    print("[directions]")
    for i, t in enumerate(DIRECTIONS):
        await make_one(t, f"dir_{i}")

    print("[actions]")
    for name, t in ACTIONS.items():
        await make_one(t, name)

    print("[classes]")
    for i, t in enumerate(CLASSES):
        await make_one(t, f"cls_{i}")

    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print(f"\n[done] 86 WAV files written to {OUT_DIR}")
    print("       Copy this folder to the SD card so it appears as 0:/voice/")


if __name__ == "__main__":
    asyncio.run(main())
