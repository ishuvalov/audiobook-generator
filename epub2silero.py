#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["torch>=2.0", "numpy", "imageio-ffmpeg"]
#
# [tool.uv]
# # На Linux torch с PyPI тянет библиотеки CUDA на несколько гигабайт, поэтому там
# # берём версию для процессора. На Windows и macOS PyPI и так даёт версию для процессора.
# # Для видеокарты NVIDIA: замените в url "cpu" на "cu128" и удалите marker.
# sources = { torch = [{ index = "pytorch", marker = "sys_platform == 'linux'" }] }
#
# [[tool.uv.index]]
# name = "pytorch"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""
epub2silero.py - озвучка книги EPUB голосами Silero TTS напрямую, без SAPI5.

Запуск через uv (сам поставит Python, torch и ffmpeg в отдельное окружение):
    uv run epub2silero.py книга.epub                    (аудио - в audiobooks/книга рядом со скриптом)
    uv run epub2silero.py книга.epub папка_для_аудио
    uv run epub2silero.py книга.epub --list             (только показать главы)

Модель Silero скачивается при первом запуске (один файл в папке models рядом со скриптом).
Прерванную озвучку можно продолжить той же командой: готовые главы пропускаются.
Все настройки - константы ниже.
"""

# ============================== НАСТРОЙКИ ==============================

# --- Модель и голос ---
MODEL_URL = "https://models.silero.ai/models/tts/ru/v5_5_ru.pt"
MODEL_FILE = ""          # путь к файлу модели; "" = в папке models рядом со скриптом
SPEAKER = "eugene"        # aidar, baya, kseniya, xenia, eugene
SAMPLE_RATE = 48000      # 8000, 24000 или 48000
DEVICE = "cpu"           # "cpu" или "cuda"
NUM_THREADS = 4          # потоки процессора; больше 4-6 обычно не ускоряет
# Параметры apply_tts: ударения, буква ё, омографы. Для моделей старше v5
# оставьте только put_accent и put_yo.
TTS_OPTIONS = dict(put_accent=True, put_yo=True,
                   put_stress_homo=True, put_yo_homo=True)

# --- Выходные файлы ---
OUTPUT_FORMAT = "mp3"    # "mp3", "m4a" (нужен ffmpeg) или "wav"
BITRATE = "96k"          # для mp3 и m4a
FFMPEG = ""              # путь к ffmpeg; "" = из пакета imageio-ffmpeg, иначе системный
FILE_NAME = "{num} {title}"   # шаблон имени файла главы
SKIP_EXISTING = True     # не озвучивать главу заново, если её файл уже есть
WRITE_TAGS = True        # название, автор, номер главы в тегах mp3/m4a

# --- Разбиение на главы ---
# "auto"     - оглавление EPUB; если его нет - заголовки; если и их нет - файлы
# "toc"      - только по оглавлению EPUB
# "headings" - по заголовкам в тексте (см. HEADING_LEVELS)
# "files"    - каждый внутренний файл книги отдельной главой
SPLIT_MODE = "auto"
TOC_MAX_DEPTH = 0        # сколько уровней оглавления учитывать; 0 = все
HEADING_LEVELS = (1, 2)  # для режима "headings": h1 и h2
MIN_CHAPTER_CHARS = 200  # более короткие куски (титул, эпиграф) приклеиваются
                         # к следующей главе; 0 = не склеивать
SKIP_TITLES = (r"^(содержание|оглавление)$",)   # главы с такими названиями пропускаются
ONLY_CHAPTERS = ""       # озвучить часть глав, например "1-3,7"; "" = все
READ_CHAPTER_TITLE = True   # произносить название главы, если его нет в тексте

# --- Текст ---
MAX_CHUNK_CHARS = 600    # сколько символов отдавать модели за один раз
SKIP_FOOTNOTES = True    # не читать номера сносок и блоки сносок
LATIN = "translit"       # латиница: "translit" (читать по буквам-звукам) или "drop"
ROMAN_NUMERALS = True    # читать римские числа из I, V, X как обычные

# --- Паузы, миллисекунды ---
PAUSE_CHAPTER_START = 400
PAUSE_HEADING = 900      # до и после заголовка
PAUSE_PARAGRAPH = 400
PAUSE_LINE = 200         # между строками, разделёнными <br> (стихи)
PAUSE_CHUNK = 250        # между кусками одного длинного абзаца
PAUSE_CHAPTER_END = 1200

# ========================== КОНЕЦ НАСТРОЕК ==========================

import argparse
import ctypes
import posixpath
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import warnings
import wave
import xml.etree.ElementTree as ET
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import NamedTuple
from urllib.parse import unquote

torch = None   # импортируется в load_model(): для --list torch не нужен


# ------------------------------ EPUB ------------------------------

class Block(NamedTuple):
    text: str
    kind: str    # "h" заголовок, "p" абзац, "l" строка после <br>
    level: int   # уровень заголовка 1-6, иначе 0


class Chapter(NamedTuple):
    title: str
    blocks: list


BLOCK_TAGS = {
    "p", "div", "li", "blockquote", "section", "article", "body", "table",
    "tr", "td", "th", "ul", "ol", "dl", "dt", "dd", "pre", "hr", "header",
    "footer", "aside", "nav", "figure", "figcaption", "caption", "address",
}
HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
VOID_TAGS = {"br", "hr", "img", "meta", "link", "input", "col", "wbr", "area", "base"}
SILENT_TAGS = {"script", "style", "svg", "math", "template"}
NOTE_TYPES = {"noteref", "footnote", "endnote", "rearnote", "footnotes", "endnotes", "rearnotes"}
NOTE_MARK = re.compile(r"^\s*[\[({]?\s*(\d{1,4}|\*+)\s*[\])}]?\s*$")


class ContentParser(HTMLParser):
    """Превращает XHTML-файл книги в список блоков текста и запоминает якоря (id)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks = []
        self.anchors = {}      # id -> номер блока, с которого начинается якорь
        self.title = ""
        self._buf = []
        self._heading = 0
        self._break = "p"
        self._skip_tag = None
        self._skip_depth = 0
        self._in_title = False
        self._link_start = None

    def _flush(self):
        text = " ".join("".join(self._buf).split())
        self._buf = []
        self._link_start = None
        if text:
            kind = "h" if self._heading else self._break
            self.blocks.append(Block(text, kind, self._heading))
        self._break = "p"

    def handle_starttag(self, tag, attrs):
        if self._skip_tag:
            if tag == self._skip_tag:
                self._skip_depth += 1
            return
        attrs = dict(attrs)
        is_block = tag in BLOCK_TAGS or tag in HEADING_TAGS
        if is_block:
            self._flush()
        # Якорь указывает на блок, который сейчас начнётся или уже набирается.
        for key in ("id", "name") if tag == "a" else ("id",):
            if attrs.get(key):
                self.anchors.setdefault(attrs[key], len(self.blocks))

        is_note = SKIP_FOOTNOTES and (
            tag == "sup" or NOTE_TYPES & set((attrs.get("epub:type") or "").split()))
        if tag in SILENT_TAGS or is_note:
            if tag not in VOID_TAGS:
                self._skip_tag, self._skip_depth = tag, 1
        elif tag == "title":
            self._in_title = True
        elif tag in HEADING_TAGS:
            self._heading = HEADING_TAGS[tag]
        elif tag == "br":
            had_text = bool("".join(self._buf).strip())
            self._flush()
            self._break = "l" if had_text else "p"
        elif tag == "a" and attrs.get("href"):
            self._link_start = len(self._buf)

    def handle_endtag(self, tag):
        if self._skip_tag:
            if tag == self._skip_tag:
                self._skip_depth -= 1
                if self._skip_depth == 0:
                    self._skip_tag = None
            return
        if tag == "title":
            self._in_title = False
        elif tag in HEADING_TAGS:
            self._flush()
            self._heading = 0
        elif tag in BLOCK_TAGS:
            self._flush()
        elif tag == "a" and self._link_start is not None:
            # Ссылка, весь текст которой - "[12]" или "*", это номер сноски.
            if SKIP_FOOTNOTES and NOTE_MARK.match("".join(self._buf[self._link_start:])):
                del self._buf[self._link_start:]
            self._link_start = None

    def handle_data(self, data):
        if self._skip_tag:
            return
        if self._in_title:
            self.title += data
        else:
            self._buf.append(data)

    def close(self):
        super().close()
        self._flush()


class NavParser(HTMLParser):
    """Оглавление EPUB 3: ссылки внутри <nav epub:type="toc">."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.navs = []         # [(epub:type, [(depth, title, href), ...]), ...]
        self._depth = 0
        self._href = None
        self._text = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "nav":
            self.navs.append((attrs.get("epub:type") or "", []))
            self._depth = 0
        elif not self.navs:
            return
        elif tag in ("ol", "ul"):
            self._depth += 1
        elif tag == "a" and attrs.get("href"):
            self._href, self._text = attrs["href"], []

    def handle_endtag(self, tag):
        if not self.navs:
            return
        if tag in ("ol", "ul"):
            self._depth -= 1
        elif tag == "a" and self._href is not None:
            title = " ".join("".join(self._text).split())
            self.navs[-1][1].append((max(self._depth, 1), title, self._href))
            self._href = None

    def handle_data(self, data):
        if self._href is not None:
            self._text.append(data)

    def entries(self):
        for nav_type, items in self.navs:
            if "toc" in nav_type.split() and items:
                return items
        return next((items for _, items in self.navs if items), [])


def decode_markup(data):
    if data[:3] == b"\xef\xbb\xbf":
        return data[3:].decode("utf-8", "replace")
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16", "replace")
    found = (re.search(rb"""encoding=["']([\w.-]+)["']""", data[:200])
             or re.search(rb"""charset=["']?([\w.-]+)""", data[:2048]))
    try:
        return data.decode(found.group(1).decode("ascii") if found else "utf-8", "replace")
    except LookupError:
        return data.decode("utf-8", "replace")


class Book:
    """Текст книги одной лентой блоков плюс всё, что нужно для деления на главы."""

    def __init__(self, path):
        self.blocks = []
        self.doc_start = {}    # файл -> номер первого блока
        self.doc_title = {}    # файл -> <title>
        self.anchors = {}      # (файл, id) -> номер блока
        self.toc = []          # [(depth, title, файл, id), ...]
        with zipfile.ZipFile(path) as archive:
            names = {name.lower(): name for name in archive.namelist()}

            def read(name):
                real = names.get(name.lower())
                return archive.read(real) if real else None
            self._load(read)

    @staticmethod
    def _resolve(base_file, href):
        target, _, fragment = href.partition("#")
        if not target:
            return base_file, unquote(fragment)
        path = posixpath.normpath(posixpath.join(posixpath.dirname(base_file), unquote(target)))
        return path, unquote(fragment)

    def _load(self, read):
        container = read("META-INF/container.xml")
        if container is None:
            raise ValueError("это не EPUB: нет META-INF/container.xml")
        opf_path = ET.fromstring(container).find(".//{*}rootfile").get("full-path")
        opf = ET.fromstring(read(opf_path))

        def meta(tag):
            node = opf.find(".//{*}metadata/{*}" + tag)
            return " ".join((node.text or "").split()) if node is not None else ""
        self.title, self.author = meta("title"), meta("creator")

        manifest, nav_path = {}, None
        for item in opf.iterfind(".//{*}manifest/{*}item"):
            item_path = self._resolve(opf_path, item.get("href", ""))[0]
            manifest[item.get("id")] = (item_path, item.get("media-type", ""))
            if "nav" in (item.get("properties") or "").split():
                nav_path = item_path

        spine = opf.find(".//{*}spine")
        if spine is None:
            raise ValueError("в книге нет раздела spine")
        for itemref in spine.iterfind("{*}itemref"):
            doc_path, media_type = manifest.get(itemref.get("idref"), ("", ""))
            is_text = "html" in media_type or doc_path.lower().endswith((".xhtml", ".html", ".htm"))
            data = read(doc_path) if is_text and doc_path != nav_path else None
            if data is None:
                continue
            parser = ContentParser()
            parser.feed(decode_markup(data))
            parser.close()
            offset = len(self.blocks)
            self.doc_start[doc_path] = offset
            self.doc_title[doc_path] = " ".join(parser.title.split())
            for anchor, index in parser.anchors.items():
                self.anchors[(doc_path, anchor)] = offset + index
            self.blocks.extend(parser.blocks)

        if nav_path and read(nav_path) is not None:
            parser = NavParser()
            parser.feed(decode_markup(read(nav_path)))
            parser.close()
            self.toc = [(depth, title, *self._resolve(nav_path, href))
                        for depth, title, href in parser.entries()]
        ncx_path = manifest.get(spine.get("toc"), ("", ""))[0]
        if not self.toc and ncx_path and read(ncx_path) is not None:
            self.toc = self._read_ncx(read(ncx_path), ncx_path)

    def _read_ncx(self, data, ncx_path):
        entries = []

        def walk(node, depth):
            for point in node.iterfind("{*}navPoint"):
                label = point.find("{*}navLabel")
                content = point.find("{*}content")
                title = " ".join("".join(label.itertext()).split()) if label is not None else ""
                if content is not None and content.get("src"):
                    entries.append((depth, title, *self._resolve(ncx_path, content.get("src"))))
                walk(point, depth + 1)

        nav_map = ET.fromstring(data).find("{*}navMap")
        if nav_map is not None:
            walk(nav_map, 1)
        return entries

    # --- границы глав: списки (номер блока, название) ---

    def bounds_toc(self):
        bounds = []
        for depth, title, doc_path, anchor in self.toc:
            if TOC_MAX_DEPTH and depth > TOC_MAX_DEPTH:
                continue
            position = self.anchors.get((doc_path, anchor), self.doc_start.get(doc_path))
            if position is not None:
                bounds.append((position, title))
        return bounds

    def bounds_headings(self):
        return [(index, block.text) for index, block in enumerate(self.blocks)
                if block.kind == "h" and block.level in HEADING_LEVELS]

    def bounds_files(self):
        starts = sorted(self.doc_start.items(), key=lambda item: item[1])
        bounds = []
        for number, (doc_path, start) in enumerate(starts):
            end = starts[number + 1][1] if number + 1 < len(starts) else len(self.blocks)
            heading = next((b.text for b in self.blocks[start:end] if b.kind == "h"), "")
            bounds.append((start, heading or self.doc_title.get(doc_path, "")))
        return bounds


def plain(text):
    return re.sub(r"[\W_]+", "", text.lower())


def build_chapters(book):
    """Возвращает (способ разбиения, [Chapter, ...])."""
    modes = {"toc": book.bounds_toc, "headings": book.bounds_headings, "files": book.bounds_files}
    if SPLIT_MODE in modes:
        mode, bounds = SPLIT_MODE, modes[SPLIT_MODE]()
    else:
        for mode in ("toc", "headings", "files"):
            bounds = modes[mode]()
            if len({position for position, _ in bounds}) >= 2:
                break

    # Несколько пунктов оглавления могут указывать в одну точку (часть и её первая глава).
    starts = {}
    for position, title in sorted(bounds, key=lambda bound: bound[0]):
        starts.setdefault(position, [])
        if title and title not in starts[position]:
            starts[position].append(title)
    has_front = 0 not in starts       # текст до первой главы: титул, аннотация
    positions = sorted(starts.keys() | {0}) + [len(book.blocks)]

    chapters = []
    for start, end in zip(positions, positions[1:]):
        titles, blocks = starts.get(start, []), book.blocks[start:end]
        title = ". ".join(name.rstrip(". ") for name in titles) or ("Начало" if has_front and start == 0 else "")
        if any(re.search(pattern, title.strip(), re.I) for pattern in SKIP_TITLES):
            continue
        if not any(speakable(block.text) for block in blocks):
            continue
        if READ_CHAPTER_TITLE:
            # Название произносим, только если текст главы сам с него не начинается.
            opening, first = plain("".join(block.text for block in blocks[:3])), plain(blocks[0].text)
            missing = [name for name in titles if speakable(name) and not (
                opening.startswith(plain(name)) or first and plain(name).startswith(first))]
            blocks = [Block(name, "h", 1) for name in missing] + blocks
        chapters.append(Chapter(title, blocks))

    # Слишком короткие куски приклеиваются к следующей главе (последний - к предыдущей).
    merged, pending = [], None
    for chapter in chapters:
        if pending:
            chapter = Chapter(chapter.title or pending.title, pending.blocks + chapter.blocks)
            pending = None
        if sum(len(block.text) for block in chapter.blocks) < MIN_CHAPTER_CHARS:
            pending = chapter
        else:
            merged.append(chapter)
    if pending and merged:
        merged[-1] = Chapter(merged[-1].title, merged[-1].blocks + pending.blocks)
    elif pending:
        merged.append(pending)
    return mode, merged


# ----------------------------- ТЕКСТ -----------------------------

ONES = ("ноль один два три четыре пять шесть семь восемь девять десять одиннадцать "
        "двенадцать тринадцать четырнадцать пятнадцать шестнадцать семнадцать "
        "восемнадцать девятнадцать").split()
TENS = "- - двадцать тридцать сорок пятьдесят шестьдесят семьдесят восемьдесят девяносто".split()
HUNDREDS = "- сто двести триста четыреста пятьсот шестьсот семьсот восемьсот девятьсот".split()
SCALES = (None, ("тысяча", "тысячи", "тысяч"), ("миллион", "миллиона", "миллионов"),
          ("миллиард", "миллиарда", "миллиардов"), ("триллион", "триллиона", "триллионов"))


def number_words(digits):
    """Число прописью (количественное, именительный падеж): '1905' -> 'тысяча девятьсот пять'."""
    if len(digits) > 15 or (len(digits) > 1 and digits[0] == "0"):
        return " ".join(ONES[int(d)] for d in digits)      # номера и коды - по цифрам
    number = int(digits)
    if number == 0:
        return ONES[0]
    words, scale = [], 0
    while number:
        number, group = divmod(number, 1000)
        if group:
            part = [HUNDREDS[group // 100]] if group >= 100 else []
            rest = group % 100
            if rest >= 20:
                part.append(TENS[rest // 10])
                rest %= 10
            if rest:
                part.append({1: "одна", 2: "две"}.get(rest, ONES[rest]) if scale == 1 else ONES[rest])
            if scale == 1 and group == 1:
                part = []                                   # "тысяча", а не "одна тысяча"
            if scale:
                last2, last1 = group % 100, group % 10
                form = 2 if 11 <= last2 <= 14 or last1 == 0 or last1 >= 5 else (0 if last1 == 1 else 1)
                part.append(SCALES[scale][form])
            words = part + words
        scale += 1
    return " ".join(words)


ROMAN = re.compile(r"\b(?=[IVX])(X{0,3})(IX|IV|V?I{0,3})\b")
ROMAN_VALUES = {"": 0, "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9}

LATIN_PAIRS = (("shch", "щ"), ("sch", "ш"), ("ch", "ч"), ("sh", "ш"), ("th", "т"), ("ph", "ф"),
               ("zh", "ж"), ("kh", "х"), ("ck", "к"), ("qu", "кв"), ("ya", "я"), ("yu", "ю"),
               ("ee", "и"), ("oo", "у"))
LATIN_SINGLE = dict(zip("abcdefghijklmnopqrstuvwxyz",
                        "а б к д е ф г х и дж к л м н о п к р с т у в в кс и з".split()))


def transliterate(match):
    word = match.group(0).lower()
    for latin, cyrillic in LATIN_PAIRS:
        word = word.replace(latin, cyrillic)
    return "".join(LATIN_SINGLE.get(char, char) for char in word)


ABBREVIATIONS = ((r"\bт\.\s?е\.", "то есть"), (r"\bт\.\s?к\.", "так как"),
                 (r"\bи\s+т\.\s?д\.", "и так далее"), (r"\bи\s+т\.\s?п\.", "и тому подобное"),
                 (r"\bи\s+др\.", "и другие"))

DASH = ","      # чем заменять тире; уточняется по алфавиту модели в load_model()


def speakable(text):
    return bool(re.search(r"[а-яё0-9]" if LATIN == "drop" else r"[а-яёa-z0-9]", text, re.I))


def normalize(text):
    """Готовит предложение для Silero: числа, латиница, знаки препинания."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\u00ad\u200b-\u200d\u2060\ufeff]", "", text)
    text = re.sub(r"[\u00a0\u2000-\u200a\u202f]", " ", text)
    # '+' в Silero - знак ударения перед гласной. Такие плюсы оставляем, остальные читаем.
    text = re.sub(r"\+(?![аеёиоуыэюя])", " плюс ", text, flags=re.I)
    text = re.sub(r"([аеёиоуыэюя])\u0301", r"+\1", text, flags=re.I)    # а́ -> +а
    text = re.sub(r"[\u0300-\u036f]", "", text)
    for symbol, word in (("№", " номер "), ("%", " процентов "), ("§", " параграф "),
                         ("&", " и "), ("=", " равно ")):
        text = text.replace(symbol, word)
    for pattern, words in ABBREVIATIONS:
        text = re.sub(pattern, words, text, flags=re.I)

    if ROMAN_NUMERALS:
        text = ROMAN.sub(lambda m: str(10 * len(m.group(1)) + ROMAN_VALUES[m.group(2)]), text)
    text = re.sub(r"(\d)-(?:ый|ой|ая|ое|ые|ых|го|му|ми|й|я|е|м|х|ю)\b", r"\1", text)    # 5-й -> 5
    text = re.sub(r"(?<![\d.,])(\d{1,3}(?: \d{3})+)(?![\d])", lambda m: m.group(1).replace(" ", ""), text)
    text = re.sub(r"(\d)[.:/](?=\d)", r"\1 ", text)              # 12:30, 01.02.2003
    text = re.sub(r"(\d),(?=\d)", r"\1 и ", text)                # 3,5 -> три и пять
    text = re.sub(r"(\d)\s*[-–—]\s*(?=\d)", r"\1 — ", text)      # 1941-1945
    text = re.sub(r"\d+", lambda m: " " + number_words(m.group(0)) + " ", text)

    text = re.sub(r"[A-Za-z]+", transliterate if LATIN == "translit" else " ", text)

    text = re.sub(r"[(\[{]", ", ", text)
    text = re.sub(r"[)\]}]", ",", text)
    text = text.replace("...", "…")
    text = re.sub(r"[^а-яё+ .,!?…:;–—-]", "", text, flags=re.I)   # всё, чего модель не читает
    text = re.sub(r"(^|\s)[-–—]+(?=\s|$)", r"\1" + DASH, text)    # тире, но не дефис в слове
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?…])", r"\1", text)
    text = re.sub(r",(\s*,)+", ",", text)                        # ", ," -> ","
    text = re.sub(r",\s*([.;:!?…])", r"\1", text)                # ",." -> "."
    text = re.sub(r"([.;:!?…])\s*,", r"\1", text)                # ".," -> "."
    text = text.strip(" ,;:")
    if text and text[-1] not in ".!?…":
        text += "."
    return text


def split_sentences(text):
    """Режет абзац на предложения по . ! ? … перед заглавной буквой, цифрой, кавычкой, тире."""
    sentences, start = [], 0
    for match in re.finditer(r"[.!?…]+[»”\")\]]*\s+(?=[«„“\"(\[—–-]*\s*[А-ЯЁA-Z\d])", text):
        before = text[start:match.start()]
        if match.group(0).strip() == "." and re.search(r"(?:^|[\s(«\"])[А-ЯЁA-Z]$", before):
            continue                                             # инициал: "А. С. Пушкин"
        sentences.append(text[start:match.end()].strip())
        start = match.end()
    sentences.append(text[start:].strip())
    return [sentence for sentence in sentences if sentence]


def split_long(text, limit):
    """Делит слишком длинное предложение: по ; : , – ближе к пределу, иначе по пробелу."""
    parts = []
    while len(text) > limit:
        window = text[:limit]
        cut = max(window.rfind(mark) for mark in ("; ", ": ", ", ", " – ", " — "))
        if cut < limit * 0.3:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = limit - 1
        parts.append(text[:cut + 1].strip())
        text = text[cut + 1:].strip()
    return parts + [text] if text else parts


def text_chunks(text):
    """Абзац -> куски для модели: целые предложения, вместе не длиннее MAX_CHUNK_CHARS."""
    chunks, current = [], ""
    for sentence in split_sentences(text):
        sentence = normalize(sentence)
        if not speakable(sentence):
            continue
        for piece in split_long(sentence, MAX_CHUNK_CHARS):
            if current and len(current) + 1 + len(piece) > MAX_CHUNK_CHARS:
                chunks.append(current)
                current = ""
            current = f"{current} {piece}".strip()
    return chunks + [current] if current else chunks


# ----------------------------- СИНТЕЗ -----------------------------

def load_model():
    global DASH, torch
    import torch

    path = (Path(MODEL_FILE) if MODEL_FILE
            else Path(__file__).resolve().parent / "models" / posixpath.basename(MODEL_URL))
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Скачиваю модель в {path} ...")
        partial = path.with_name(path.name + ".download")
        try:
            torch.hub.download_url_to_file(MODEL_URL, str(partial), progress=True)
        except Exception as error:
            sys.exit(f"Не удалось скачать модель: {error}\n"
                     f"Скачайте её браузером: {MODEL_URL}\nи положите сюда: {path}")
        partial.replace(path)

    torch.set_num_threads(NUM_THREADS)
    # Код внутри файла модели написан небрежно, и Python 3.12+ ругается на него при загрузке.
    warnings.filterwarnings("ignore", category=SyntaxWarning, message="invalid escape sequence")
    # Файл открываем сами и не закрываем: так torch не спотыкается о русские буквы в пути.
    model = torch.package.PackageImporter(open(path, "rb")).load_pickle("tts_models", "model")
    model.to(torch.device(DEVICE))

    speakers = getattr(model, "speakers", None)
    if speakers and SPEAKER not in speakers:
        sys.exit(f"Голоса '{SPEAKER}' в модели нет. Доступны: {', '.join(speakers)}")
    symbols = getattr(model, "symbols", "")
    if isinstance(symbols, str):
        DASH = next((dash for dash in "—–" if dash in symbols), ",")

    try:
        synthesize(model, "Проверка связи.", retry=False)
    except Exception as error:
        sys.exit(f"Модель не смогла озвучить пробную фразу: {error!r}\n"
                 "Проверьте SAMPLE_RATE и TTS_OPTIONS в начале скрипта.")
    return model


def synthesize(model, text, retry=True):
    """Текст -> список кусков звука (PCM, 16 бит, моно)."""
    try:
        audio = model.apply_tts(text=text, speaker=SPEAKER, sample_rate=SAMPLE_RATE, **TTS_OPTIONS)
    except Exception as error:
        if not retry:
            raise
        # Обычная причина - слишком длинный кусок: делим пополам и пробуем ещё раз.
        halves = split_long(text, max(len(text) // 2 + 1, 40)) if len(text) > 80 else []
        if len(halves) < 2:
            print(f"\n  ! пропущен фрагмент «{text[:50]}…»: {error}")
            return []
        return [pcm for half in halves for pcm in synthesize(model, half)]
    samples = (audio.detach().to("cpu").clamp(-1.0, 1.0) * 32767.0).to(torch.int16).contiguous()
    return [ctypes.string_at(samples.data_ptr(), samples.numel() * 2)]


def silence(milliseconds):
    return b"\x00\x00" * (SAMPLE_RATE * milliseconds // 1000)


def render_chapter(model, chapter, wav_path):
    """Озвучивает главу в WAV, возвращает длительность в секундах."""
    total = sum(len(block.text) for block in chapter.blocks) or 1
    done, previous = 0, None
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframesraw(silence(PAUSE_CHAPTER_START))
        for block in chapter.blocks:
            chunks = text_chunks(block.text)
            if chunks and previous:
                near_heading = "h" in (previous.kind, block.kind)
                wav.writeframesraw(silence(PAUSE_HEADING if near_heading
                                        else PAUSE_LINE if block.kind == "l" else PAUSE_PARAGRAPH))
            for number, chunk in enumerate(chunks):
                if number:
                    wav.writeframesraw(silence(PAUSE_CHUNK))
                for pcm in synthesize(model, chunk):
                    wav.writeframesraw(pcm)
            if chunks:
                previous = block
            done += len(block.text)
            print(f"\r  {done * 100 // total:3d}%", end="", flush=True)
        wav.writeframesraw(silence(PAUSE_CHAPTER_END))
        return wav.getnframes() / SAMPLE_RATE


def find_ffmpeg():
    """FFMPEG из настроек, иначе ffmpeg из пакета imageio-ffmpeg, иначе системный."""
    if FFMPEG:
        return shutil.which(FFMPEG)
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return shutil.which("ffmpeg")


def encode(wav_path, target, tags):
    """WAV -> mp3/m4a через ffmpeg. Пишет во временный файл и переименовывает."""
    codec = {"mp3": ["-codec:a", "libmp3lame", "-id3v2_version", "3"], "m4a": ["-codec:a", "aac"]}[OUTPUT_FORMAT]
    partial = target.with_name(target.stem + ".part" + target.suffix)
    command = [find_ffmpeg(), "-y", "-loglevel", "error", "-i", str(wav_path), *codec, "-b:a", BITRATE]
    if WRITE_TAGS:
        for key, value in tags.items():
            if value:
                command += ["-metadata", f"{key}={value}"]
    try:
        subprocess.run(command + [str(partial)], check=True)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


# ----------------------------- ЗАПУСК -----------------------------

def parse_ranges(spec, count):
    """'1-3,7' -> {1, 2, 3, 7}; пустая строка -> все главы."""
    if not spec.strip():
        return set(range(1, count + 1))
    chosen = set()
    for part in spec.split(","):
        first, _, last = part.strip().partition("-")
        chosen.update(range(int(first), int(last or first) + 1))
    return chosen


def safe_name(title):
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", title)
    return " ".join(name.split())[:80].rstrip(" .") or "Без названия"


def clock(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")   # редкие символы в названиях не уронят вывод
    arguments = argparse.ArgumentParser(description="EPUB -> аудиокнига голосами Silero TTS")
    arguments.add_argument("book", help="файл книги .epub")
    arguments.add_argument("out_dir", nargs="?",
                           help="папка для аудиофайлов (по умолчанию audiobooks/<имя файла книги> рядом со скриптом)")
    arguments.add_argument("--list", action="store_true", help="показать главы и выйти")
    args = arguments.parse_args()

    if OUTPUT_FORMAT not in ("mp3", "m4a", "wav"):
        sys.exit(f"Неизвестный OUTPUT_FORMAT: {OUTPUT_FORMAT}")
    try:
        book = Book(args.book)
    except (OSError, ValueError, zipfile.BadZipFile, ET.ParseError) as error:
        sys.exit(f"Не удалось прочитать книгу: {error}")
    mode, chapters = build_chapters(book)
    if not chapters:
        sys.exit("В книге не нашлось текста для озвучки.")
    try:
        selected = parse_ranges(ONLY_CHAPTERS, len(chapters))
    except ValueError:
        sys.exit(f'Не понял ONLY_CHAPTERS = "{ONLY_CHAPTERS}". Пример: "1-3,7"')
    width = max(2, len(str(len(chapters))))

    how = {"toc": "по оглавлению EPUB", "headings": "по заголовкам в тексте", "files": "по файлам книги"}[mode]
    print(f"{book.author + '. ' if book.author else ''}{book.title or Path(args.book).stem}")
    print(f"Глав: {len(chapters)} (разбиение {how})")
    for number, chapter in enumerate(chapters, 1):
        size = sum(len(block.text) for block in chapter.blocks)
        mark = " " if number in selected else "-"
        print(f" {mark}{number:>{width}}. {chapter.title or 'Без названия'}  [{size} зн.]")
    if args.list:
        return

    if OUTPUT_FORMAT != "wav" and not find_ffmpeg():
        sys.exit(f"Не найден ffmpeg. Запускайте скрипт через uv run, укажите путь в FFMPEG "
                 f"или поставьте OUTPUT_FORMAT = \"wav\".")
    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent / "audiobooks" / Path(args.book).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    model = None
    audio_total, started = 0.0, time.time()

    for number, chapter in enumerate(chapters, 1):
        if number not in selected:
            continue
        name = FILE_NAME.format(num=str(number).zfill(width), title=safe_name(chapter.title))
        target = out_dir / f"{name}.{OUTPUT_FORMAT}"
        print(f"[{number}/{len(chapters)}] {target.name}")
        if SKIP_EXISTING and target.is_file() and target.stat().st_size > 0:
            print("  уже есть, пропускаю")
            continue
        if model is None:
            model = load_model()

        wav_path = out_dir / f"{name}.part.wav"
        chapter_started = time.time()
        try:
            duration = render_chapter(model, chapter, wav_path)
            if OUTPUT_FORMAT == "wav":
                wav_path.replace(target)
            else:
                tags = {"title": chapter.title, "album": book.title, "artist": book.author,
                        "track": f"{number}/{len(chapters)}", "genre": "Audiobook"}
                encode(wav_path, target, tags)
        finally:
            wav_path.unlink(missing_ok=True)
        elapsed = max(time.time() - chapter_started, 0.001)
        audio_total += duration
        print(f"\r  {clock(duration)} звука за {clock(elapsed)} (x{duration / elapsed:.0f})")

    if model is None:
        print("Все выбранные главы уже озвучены.")
    else:
        print(f"Готово: {clock(audio_total)} звука за {clock(time.time() - started)}. Папка: {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nПрервано. Запустите ту же команду ещё раз: готовые главы будут пропущены.")