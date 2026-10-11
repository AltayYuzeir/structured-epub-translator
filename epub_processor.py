from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from lxml import etree

LANGUAGE_CODES = {
    "Arabic": "ar", "Chinese": "zh", "Czech": "cs", "Danish": "da",
    "Dutch": "nl", "English": "en", "Finnish": "fi", "French": "fr",
    "German": "de", "Greek": "el", "Hebrew": "he", "Hungarian": "hu",
    "Indonesian": "id", "Italian": "it", "Japanese": "ja", "Korean": "ko",
    "Norwegian": "no", "Polish": "pl", "Portuguese": "pt", "Romanian": "ro",
    "Russian": "ru", "Slovak": "sk", "Spanish": "es", "Swedish": "sv",
    "Thai": "th", "Turkish": "tr", "Ukrainian": "uk", "Vietnamese": "vi",
}

def update_epub_language(work_dir: Path, target_language: str) -> None:
    """Update the dc:language tag in all OPF files to the target language code."""
    target_code = LANGUAGE_CODES.get(target_language, target_language.lower()[:2])
    lang_tag_qname = "{http://purl.org/dc/elements/1.1/}language"

    for opf in find_opf_files(work_dir):
        original = opf.read_bytes()
        parser = etree.XMLParser(
            remove_blank_text=False,
            resolve_entities=False,
            no_network=True,
            recover=False,
            huge_tree=True,
        )
        tree = etree.ElementTree(etree.fromstring(original, parser))
        
        updated = False
        for element in tree.getroot().iter():
            if element.tag == lang_tag_qname:
                element.text = target_code
                updated = True

        if updated:
            opf.write_bytes(serialize_xhtml(tree, original))


HTML_SUFFIXES = {".xhtml", ".html", ".htm"}
SKIP_ANCESTORS = {"script", "style", "pre", "code", "svg", "math"}
XML_NS = "http://www.w3.org/XML/1998/namespace"


@dataclass
class TextSlot:
    """A mutable text/tail location in an lxml tree."""

    element: etree._Element
    is_tail: bool

    @property
    def text(self) -> str:
        return self.element.tail if self.is_tail else (self.element.text or "")

    @text.setter
    def text(self, value: str) -> None:
        if self.is_tail:
            self.element.tail = value
        else:
            self.element.text = value


@dataclass
class TranslationUnit:
    """A group of text slots translated together."""

    unit_id: str
    slots: list[TextSlot]
    source_texts: list[str]

    @property
    def combined_text(self) -> str:
        parts = []
        for i, text in enumerate(self.source_texts):
            parts.append(f"⟦SEG_{i}⟧{text}⟦/SEG_{i}⟧")
        return "\n".join(parts)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_filename(name: str) -> str:
    name = re.sub(r"[^\w.\-]+", "_", name, flags=re.UNICODE)
    return name.strip("._") or "translated.epub"


def _local_name(element: etree._Element) -> str:
    return etree.QName(element).localname.lower()


def _has_skipped_ancestor(element: etree._Element) -> bool:
    current = element
    while current is not None:
        if _local_name(current) in SKIP_ANCESTORS:
            return True
        current = current.getparent()
    return False


def _is_translatable(text: str) -> bool:
    if not text or not text.strip():
        return False
    # Don't feed isolated XML/HTML-looking fragments to the model.
    if text.strip().startswith("<") and text.strip().endswith(">"):
        return False
    return True


def parse_xhtml(data: bytes, path: Path) -> etree._ElementTree:
    parser = etree.XMLParser(
        remove_blank_text=False,
        resolve_entities=False,
        no_network=True,
        recover=False,
        strip_cdata=False,
        huge_tree=True,
    )
    try:
        return etree.ElementTree(etree.fromstring(data, parser))
    except etree.XMLSyntaxError as exc:
        raise ValueError(
            f"Could not parse EPUB XHTML/XML safely: {path.name}. "
            f"The file is not well-formed XML. Error: {exc}"
        ) from exc


def serialize_xhtml(tree: etree._ElementTree, original: bytes) -> bytes:
    xml_decl = original.lstrip().startswith(b"<?xml")
    doctype_match = re.search(
        rb"<!DOCTYPE[^>]*(?:\[[\s\S]*?\]\s*)?>",
        original[:10000],
        flags=re.IGNORECASE,
    )
    doctype = doctype_match.group(0).decode("utf-8", errors="replace") if doctype_match else None

    data = etree.tostring(
        tree,
        encoding="UTF-8",
        xml_declaration=xml_decl,
        doctype=doctype,
        pretty_print=False,
        with_tail=True,
    )

    return data


def _all_slots(root: etree._Element) -> list[TextSlot]:
    slots: list[TextSlot] = []

    for element in root.iter():
        if _has_skipped_ancestor(element):
            continue

        if element.text and _is_translatable(element.text):
            slots.append(TextSlot(element, False))

        for child in element:
            if child.tail and _is_translatable(child.tail):
                if not _has_skipped_ancestor(child):
                    slots.append(TextSlot(child, True))

    return slots


def _nearest_block(element: etree._Element) -> etree._Element:
    block_names = {
        "p", "div", "section", "article", "li", "td", "th",
        "blockquote", "h1", "h2", "h3", "h4", "h5", "h6",
        "figcaption", "caption", "title", "a", "dt", "dd",
    }
    current = element
    while current.getparent() is not None:
        if _local_name(current) in block_names:
            return current
        current = current.getparent()
    return element


def build_translation_units(
    root: etree._Element,
    max_chars: int = 3500,
) -> list[TranslationUnit]:
    """
    Group nearby text nodes so sentences crossing inline tags retain context.

    Each node gets a stable SEG_n placeholder. The model must return the
    placeholders unchanged; translated text between them is mapped back to
    the exact original text/tail slots.
    """
    slots = _all_slots(root)
    if not slots:
        return []

    units: list[TranslationUnit] = []
    current_slots: list[TextSlot] = []
    current_len = 0
    current_parent = None
    unit_index = 0

    def flush():
        nonlocal current_slots, current_len, current_parent, unit_index
        if current_slots:
            units.append(
                TranslationUnit(
                    unit_id=f"unit_{unit_index:06d}",
                    slots=current_slots,
                    source_texts=[s.text for s in current_slots],
                )
            )
            unit_index += 1
        current_slots = []
        current_len = 0
        current_parent = None

    for slot in slots:
        parent = _nearest_block(slot.element)
        text_len = len(slot.text)

        # Keep text from the same block together, unless it gets too large.
        if current_slots and (parent is not current_parent or current_len + text_len > max_chars):
            flush()

        current_slots.append(slot)
        current_len += text_len + 30
        current_parent = parent

    flush()
    return units


_SEG_RE = re.compile(
    r"⟦SEG_(\d+)⟧([\s\S]*?)⟦/SEG_\1⟧",
    flags=re.UNICODE,
)


def parse_segmented_translation(
    translated: str,
    expected_count: int,
) -> list[str] | None:
    matches = list(_SEG_RE.finditer(translated.strip()))

    if len(matches) != expected_count:
        return None

    indexes = [int(m.group(1)) for m in matches]
    if indexes != list(range(expected_count)):
        return None

    return [m.group(2) for m in matches]


def replace_unit(unit: TranslationUnit, translated: str) -> bool:
    parts = parse_segmented_translation(translated, len(unit.slots))
    if parts is None:
        return False

    for slot, value in zip(unit.slots, parts):
        slot.text = value
    return True


def extract_epub(epub_path: str | Path, work_dir: Path) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(epub_path, "r") as zf:
        zf.extractall(work_dir)


def _zipinfo_for_preservation(info):
    # We recreate files but retain useful ZIP metadata.
    return info


def rebuild_epub(work_dir: Path, output_path: str | Path) -> None:
    """
    Rebuild EPUB with mimetype first and uncompressed, as required by EPUB.
    All files not explicitly modified by the processor remain byte-for-byte
    identical inside the extracted tree.
    """
    output_path = Path(output_path)
    mimetype = work_dir / "mimetype"
    if not mimetype.exists():
        raise ValueError("EPUB does not contain a required mimetype file.")

    if mimetype.read_bytes() != b"application/epub+zip":
        raise ValueError("Invalid EPUB mimetype contents.")

    with ZipFile(output_path, "w") as zf:
        zf.write(mimetype, "mimetype", compress_type=ZIP_STORED)

        for path in sorted(work_dir.rglob("*")):
            if not path.is_file() or path == mimetype:
                continue
            arcname = path.relative_to(work_dir).as_posix()
            zf.write(path, arcname, compress_type=ZIP_DEFLATED)


def find_content_files(work_dir: Path) -> list[Path]:
    return sorted(
        p for p in work_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in HTML_SUFFIXES
    )


def find_opf_files(work_dir: Path) -> list[Path]:
    return sorted(
        p for p in work_dir.rglob("*.opf")
        if p.is_file()
    )


def process_xhtml_file(
    path: Path,
    translate_unit: Callable[[TranslationUnit], str],
    progress: Callable[[str], None] | None = None,
    max_chars: int = 3500,
    max_units: int | None = None,
) -> tuple[int, int]:
    original = path.read_bytes()
    tree = parse_xhtml(original, path)
    units = build_translation_units(tree.getroot(), max_chars=max_chars)

    if max_units is not None:
        units = units[:max_units]

    translated_count = 0
    for unit in units:
        translated = translate_unit(unit)
        if replace_unit(unit, translated):
            translated_count += 1
        else:
            raise ValueError(
                f"Model returned invalid/missing SEG placeholders for {path.name}, "
                f"{unit.unit_id}. Translation was not applied to this file."
            )

        if progress:
            progress(f"{path.name}: translated {translated_count}/{len(units)} units")

    if translated_count:
        path.write_bytes(serialize_xhtml(tree, original))

    return len(units), translated_count


def process_metadata(
    work_dir: Path,
    translate_unit: Callable[[TranslationUnit], str],
    progress: Callable[[str], None] | None = None,
    max_chars: int = 3500,
) -> tuple[int, int]:
    total_units = 0
    translated_units = 0

    for opf in find_opf_files(work_dir):
        original = opf.read_bytes()
        parser = etree.XMLParser(
            remove_blank_text=False,
            resolve_entities=False,
            no_network=True,
            recover=False,
            huge_tree=True,
        )
        tree = etree.ElementTree(etree.fromstring(original, parser))

        # Restrict metadata translation to dc:title, dc:description, dc:subject.
        metadata_tags = {
            "{http://purl.org/dc/elements/1.1/}title",
            "{http://purl.org/dc/elements/1.1/}description",
            "{http://purl.org/dc/elements/1.1/}subject",
        }

        slots = []

        for element in tree.getroot().iter():
            if not isinstance(element.tag, str):
                continue
        
            if (
                element.tag in metadata_tags
                and _is_translatable(element.text or "")
            ):
                slots.append(TextSlot(element, False))

        for idx, slot in enumerate(slots):
            unit = TranslationUnit(
                unit_id=f"metadata_{idx:06d}",
                slots=[slot],
                source_texts=[slot.text],
            )
            translated = translate_unit(unit)
            if not replace_unit(unit, translated):
                raise ValueError(f"Invalid SEG placeholder response for metadata in {opf.name}.")
            total_units += 1
            translated_units += 1

        if slots:
            opf.write_bytes(serialize_xhtml(tree, original))

        if progress and slots:
            progress(f"{opf.name}: translated {len(slots)} metadata fields")

    return total_units, translated_units


def count_words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def truncate_units_by_words(units: list[TranslationUnit], max_words: int) -> list[TranslationUnit]:
    if max_words <= 0:
        return units
    selected = []
    total = 0
    for unit in units:
        words = count_words(" ".join(unit.source_texts))
        if selected and total + words > max_words:
            break
        selected.append(unit)
        total += words
        if total >= max_words:
            break
    return selected
