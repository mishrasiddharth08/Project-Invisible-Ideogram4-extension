"""Official Ideogram 4 JSON caption verifier (vendored from ideogram-oss/ideogram4, Apache-2.0).

Verifies the structured JSON caption schema the model was trained on:
- three top-level fields, strict key order inside style_description / elements
- bbox values in [0, 1000] (y_min, x_min, y_max, x_max)
- hex colors #RRGGBB uppercase
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

PathLike = str | Path

NON_ASCII_UNICODE_ESCAPE_RE = re.compile(
  r"\\u(?:00[89a-fA-F][0-9a-fA-F]|0[1-9a-fA-F][0-9a-fA-F]{2}|[1-9a-fA-F][0-9a-fA-F]{3})"
)


class CaptionVerifier:
  """Verify Ideogram 4 JSON caption format. All methods return warning strings (empty = pass)."""

  top_level_known_keys: frozenset[str] = frozenset(
    {"high_level_description", "style_description", "compositional_deconstruction"}
  )

  style_description_key_order_photo: Sequence[str] = (
    "aesthetics", "lighting", "photo", "medium", "color_palette",
  )
  style_description_key_order_non_photo: Sequence[str] = (
    "aesthetics", "lighting", "medium", "art_style", "color_palette",
  )
  compositional_deconstruction_key_order: Sequence[str] = ("background", "elements")

  element_key_order_obj: Sequence[str] = ("type", "bbox", "desc", "color_palette")
  element_key_order_text: Sequence[str] = ("type", "bbox", "text", "desc", "color_palette")

  style_description_known_keys: frozenset[str] = frozenset(
    {"aesthetics", "lighting", "photo", "art_style", "medium", "color_palette"}
  )
  element_known_keys: frozenset[str] = frozenset({"type", "bbox", "text", "desc", "color_palette"})
  element_types: frozenset[str] = frozenset({"obj", "text"})

  bbox_min: int = 0
  bbox_max: int = 1000
  style_description_palette_max: int = 16
  element_palette_max: int = 5

  def verify(self, caption: dict) -> list[str]:
    warnings: list[str] = []
    if not isinstance(caption, dict):
      warnings.append(f"root: expected a JSON object (dict), got {type(caption).__name__}")
      return warnings
    self._check_unknown_keys(caption, self.top_level_known_keys, "root", warnings)
    if "high_level_description" in caption:
      self._verify_high_level_description(caption["high_level_description"], warnings)
    if "style_description" in caption:
      self._verify_style_description(caption["style_description"], warnings)
    if "compositional_deconstruction" in caption:
      self._verify_compositional_deconstruction(caption["compositional_deconstruction"], warnings)
    else:
      warnings.append("root: 'compositional_deconstruction' must exist")
    return warnings

  def verify_raw(self, raw_text: str) -> list[str]:
    warnings: list[str] = self.check_ensure_ascii_false(raw_text)
    try:
      caption = json.loads(raw_text)
    except json.JSONDecodeError as e:
      warnings.append(f"invalid JSON: {e}")
      return warnings
    return warnings + self.verify(caption)

  def verify_file(self, path: PathLike) -> list[str]:
    raw = Path(path).read_text(encoding="utf-8")
    return self.verify_raw(raw)

  @classmethod
  def check_ensure_ascii_false(cls, raw_text: str, max_examples: int = 3) -> list[str]:
    warnings: list[str] = []
    matches = NON_ASCII_UNICODE_ESCAPE_RE.findall(raw_text)
    if not matches:
      return warnings
    has_literal_non_ascii = any(ord(c) > 0x7F for c in raw_text)
    if has_literal_non_ascii:
      return warnings
    examples = ", ".join(sorted(set(matches))[:max_examples])
    extra = "" if len(set(matches)) <= max_examples else ", ..."
    warnings.append(
      f"raw text: found {len(matches)} non-ASCII unicode escape(s) (e.g. {examples}{extra}) "
      f"with no literal non-ASCII characters; re-save with ensure_ascii=False"
    )
    return warnings

  def _verify_high_level_description(self, hld, warnings: list[str]) -> None:
    if not isinstance(hld, str):
      warnings.append(f"high_level_description: expected a string, got {type(hld).__name__}")

  def _verify_style_description(self, sd, warnings: list[str]) -> None:
    if not isinstance(sd, dict):
      warnings.append("style_description: expected a dict")
      return
    self._check_unknown_keys(sd, self.style_description_known_keys, "style_description", warnings)
    has_photo = "photo" in sd
    has_art_style = "art_style" in sd
    if has_photo and has_art_style:
      warnings.append("style_description: contains both 'photo' and 'art_style'; expected exactly one")
      return
    if not has_photo and not has_art_style:
      warnings.append("style_description: expected one of 'photo' or 'art_style'")
      return
    try:
      self._check_key_order(sd, self._style_description_key_order(sd), "style_description", warnings)
    except ValueError:
      pass
    if "color_palette" in sd:
      self._verify_color_palette(sd["color_palette"], "style_description.color_palette", self.style_description_palette_max, warnings)

  def _verify_compositional_deconstruction(self, cd, warnings: list[str]) -> None:
    if not isinstance(cd, dict):
      warnings.append("compositional_deconstruction: expected a dict")
      return
    if "background" not in cd:
      warnings.append("compositional_deconstruction: 'background' must exist")
      return
    if not isinstance(cd["background"], str):
      warnings.append("compositional_deconstruction.background: expected a string")
      return
    if "elements" not in cd:
      warnings.append("compositional_deconstruction: 'elements' must exist")
      return
    self._check_key_order(cd, self.compositional_deconstruction_key_order, "compositional_deconstruction", warnings)
    elements = cd["elements"]
    if not isinstance(elements, list):
      warnings.append("compositional_deconstruction.elements: expected a list")
      return
    for i, elem in enumerate(elements):
      self._verify_element(i, elem, warnings)

  def _verify_element(self, i: int, elem, warnings: list[str]) -> None:
    if not isinstance(elem, dict):
      warnings.append(f"elements[{i}]: expected a dict")
      return
    self._check_unknown_keys(elem, self.element_known_keys, f"elements[{i}]", warnings)
    if "type" not in elem:
      warnings.append(f"elements[{i}]: 'type' must exist")
      return
    if elem.get("type", None) not in self.element_types:
      warnings.append(f"elements[{i}]: 'type' must be one of {sorted(self.element_types)}")
      return
    try:
      self._check_key_order(elem, self._element_key_order(elem), f"elements[{i}]", warnings)
    except ValueError:
      pass
    if "bbox" in elem:
      self._verify_bbox(i, elem["bbox"], warnings)
    if "color_palette" in elem:
      self._verify_color_palette(elem["color_palette"], f"elements[{i}].color_palette", self.element_palette_max, warnings)

  def _verify_color_palette(self, palette, path: str, max_colors: int, warnings: list[str]) -> None:
    if not isinstance(palette, list):
      warnings.append(f"{path}: expected a list")
      return
    if len(palette) > max_colors:
      warnings.append(f"{path}: too many colors ({len(palette)}), expected at most {max_colors}")
      return
    for i, color in enumerate(palette):
      if (
        not isinstance(color, str)
        or len(color) != 7
        or color[0] != "#"
        or not all(c in "0123456789ABCDEF" for c in color[1:])
      ):
        warnings.append(f"{path}[{i}]: '{color}' is not a valid #RRGGBB hex color (uppercase form required)")

  def _verify_bbox(self, i: int, bbox, warnings: list[str]) -> None:
    if not isinstance(bbox, list) or len(bbox) != 4:
      warnings.append(f"elements[{i}].bbox: expected [y_min, x_min, y_max, x_max]")
      return
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in bbox):
      warnings.append(f"elements[{i}].bbox: all values must be integers")
      return
    ymin, xmin, ymax, xmax = bbox
    if not all(self.bbox_min <= v <= self.bbox_max for v in bbox):
      warnings.append(f"elements[{i}].bbox: values must be in [{self.bbox_min}, {self.bbox_max}], got {bbox}")
    if ymin > ymax:
      warnings.append(f"elements[{i}].bbox: y_min ({ymin}) > y_max ({ymax})")
    if xmin > xmax:
      warnings.append(f"elements[{i}].bbox: x_min ({xmin}) > x_max ({xmax})")

  def _element_key_order(self, element: dict) -> Sequence[str]:
    elem_type = element.get("type", None)
    if elem_type == "text":
      element_key_order = self.element_key_order_text
    elif elem_type == "obj":
      element_key_order = self.element_key_order_obj
    else:
      raise ValueError
    element_key_order_ = []
    for key in element_key_order:
      if key in ("bbox", "color_palette"):
        if key in element:
          element_key_order_.append(key)
      else:
        element_key_order_.append(key)
    return tuple(element_key_order_)

  def _style_description_key_order(self, sd: dict) -> Sequence[str]:
    has_photo = "photo" in sd
    has_art_style = "art_style" in sd
    if has_art_style and not has_photo:
      key_order = self.style_description_key_order_non_photo
    elif not has_art_style and has_photo:
      key_order = self.style_description_key_order_photo
    else:
      raise ValueError
    key_order_ = []
    for key in key_order:
      if key == "color_palette":
        if key in sd:
          key_order_.append(key)
      else:
        key_order_.append(key)
    return tuple(key_order_)

  @staticmethod
  def _check_key_order(obj: dict, expected_order: Sequence[str], path: str, warnings: list[str]) -> None:
    """Are the keys that ARE present in the official relative order?

    ORDER only. This used to demand equality with the full expected tuple, so
    any caption that omitted an optional field was reported as misordered:

        style_description: key order is ('photo', 'medium'),
                           expected ('aesthetics', 'lighting', 'photo', 'medium')

    ('photo', 'medium') is in perfect official order - it is simply shorter.
    generate.py treats raw-JSON warnings as fatal, so that refused an entire
    generation over nothing, and it contradicted this very pipeline: the
    caption budget trims `aesthetics` and `lighting` itself, logging them as
    "optional style field". A checker cannot call optional what the trimmer
    drops and then fail the result for missing it.

    Presence is a separate question from order, and belongs to whatever
    declares a field required - not here.
    """
    expected = tuple(expected_order)
    present_keys = tuple([k for k in obj if k in expected])
    # Subsequence test: walk the expected order once and consume present keys
    # in turn. Everything matched => relative order is official.
    it = iter(expected)
    in_order = all(k in it for k in present_keys)
    if not in_order:
      warnings.append(
        f"{path}: key order is {present_keys}, which is not the official "
        f"relative order {expected}")
    extra = [k for k in obj if k not in expected]
    if extra:
      warnings.append(f"{path}: keys {extra} are not allowed in this context")

  @staticmethod
  def _check_unknown_keys(obj: dict, known: frozenset[str], path: str, warnings: list[str]) -> None:
    unknown = [k for k in obj if k not in known]
    if unknown:
      warnings.append(f"{path}: unknown keys {unknown} (not in schema)")
