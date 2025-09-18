#!/usr/bin/env python3
"""Heuristic tagger for Scryfall oracle card data."""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

COLOR_MAP = {"W": "white", "U": "blue", "B": "black", "R": "red", "G": "green"}
EM_DASH = "\u2014"

# Regex triggers that map phrases in oracle text to synergy tags. Order does not matter.
PATTERN_TAGS: Sequence[Tuple[str, str]] = [
    (r"draw (?:a|one|two|three|four|five|six|seven|eight|nine|ten|\d+) cards?", "role:card_draw"),
    (r"scry", "mechanic:scry"),
    (r"surveil", "mechanic:surveil"),
    (r"investigate", "mechanic:investigate"),
    (r"proliferate", "mechanic:proliferate"),
    (r"venture into the dungeon", "theme:dungeon"),
    (r"venture", "mechanic:venture"),
    (r"the initiative", "theme:initiative"),
    (r"energy counter", "resource:energy"),
    (r"blood token", "resource:blood"),
    (r"clue token", "resource:clue"),
    (r"food token", "resource:food"),
    (r"treasure token", "resource:treasure"),
    (r"gold token", "resource:gold"),
    (r"map token", "resource:map"),
    (r"powerstone token", "resource:powerstone"),
    (r"incubator token", "resource:incubator"),
    (r"token for each", "scaling:token"),
    (r"create a number of", "scaling:token"),
    (r"create [^\\.]* token", "role:token_generation"),
    (r"tokens you control", "theme:token_support"),
    (r"for each creature you control", "theme:go_wide"),
    (r"for each opponent", "theme:multiplayer"),
    (r"each opponent", "theme:multiplayer"),
    (r"each creature", "theme:mass_effect"),
    (r"destroy all", "role:board_wipe"),
    (r"exile all", "role:board_wipe"),
    (r"destroy target", "role:removal"),
    (r"exile target", "role:removal"),
    (r"counter target", "role:interaction"),
    (r"return target", "role:recursion"),
    (r"fight target", "role:removal"),
    (r"goad", "mechanic:goad"),
    (r"blitz", "mechanic:blitz"),
    (r"bargain", "mechanic:bargain"),
    (r"connive", "mechanic:connive"),
    (r"discover", "mechanic:discover"),
    (r"descend", "mechanic:descend"),
    (r"craft", "mechanic:craft"),
    (r"explore", "mechanic:explore"),
    (r"convoke", "mechanic:convoke"),
    (r"mutate", "mechanic:mutate"),
    (r"ninjutsu", "mechanic:ninjutsu"),
    (r"amass", "mechanic:amass"),
    (r"backup", "mechanic:backup"),
    (r"incubate", "mechanic:incubate"),
    (r"mentor", "mechanic:mentor"),
    (r"training", "mechanic:training"),
    (r"populate", "mechanic:populate"),
    (r"support", "mechanic:support"),
    (r"reinforce", "mechanic:reinforce"),
    (r"prototype", "mechanic:prototype"),
    (r"reconfigure", "mechanic:reconfigure"),
    (r"surge", "mechanic:surge"),
    (r"conspire", "mechanic:conspire"),
    (r"fabricate", "mechanic:fabricate"),
    (r"afterlife", "mechanic:afterlife"),
    (r"boast", "mechanic:boast"),
    (r"coven", "mechanic:coven"),
    (r"undergrowth", "mechanic:undergrowth"),
    (r"celebration", "mechanic:celebration"),
    (r"party", "theme:party"),
    (r"seek", "mechanic:seek"),
    (r"learn", "mechanic:learn"),
    (r"lesson", "theme:lesson"),
    (r"escape", "mechanic:escape"),
    (r"flashback", "mechanic:flashback"),
    (r"disturb", "mechanic:disturb"),
    (r"retrace", "mechanic:retrace"),
    (r"rebound", "mechanic:rebound"),
    (r"kicker", "mechanic:kicker"),
    (r"cascade", "mechanic:cascade"),
    (r"storm", "mechanic:storm"),
    (r"suspend", "mechanic:suspend"),
    (r"adventure", "mechanic:adventure"),
    (r"foretell", "mechanic:foretell"),
    (r"cycling", "mechanic:cycling"),
    (r"embalm", "mechanic:embalm"),
    (r"eternalize", "mechanic:eternalize"),
    (r"landfall", "mechanic:landfall"),
    (r"land you control", "theme:land_synergy"),
    (r"land enters the battlefield", "theme:land_synergy"),
    (r"search your library", "role:tutor"),
    (r"add \{", "role:ramp"),
    (r"mana pool", "role:ramp"),
    (r"gain life", "theme:lifegain"),
    (r"lifelink", "theme:lifegain"),
    (r"lose life", "theme:life_payment"),
    (r"sacrifice another", "theme:sacrifice"),
    (r"sacrifice a", "theme:sacrifice"),
    (r"graveyard", "theme:graveyard"),
    (r"mill", "theme:mill"),
    (r"discard", "theme:discard"),
    (r"shield counter", "resource:shield_counter"),
    (r"oil counter", "resource:oil_counter"),
    (r"experience counter", "resource:experience_counter"),
    (r"poison counter", "theme:poison"),
    (r"toxic", "mechanic:toxic"),
    (r"infect", "mechanic:infect"),
    (r"affinity", "mechanic:affinity"),
    (r"improvise", "mechanic:improvise"),
    (r"crew", "mechanic:crew"),
    (r"equip", "theme:equipment"),
    (r"aura", "theme:auras"),
    (r"vehicle", "theme:vehicles"),
    (r"planeswalker", "theme:planeswalker_support"),
    (r"double target", "theme:double_effect"),
    (r"copy target", "theme:copy"),
    (r"create a copy", "theme:copy"),
    (r"exile it, then return", "theme:blink"),
    (r"phase out", "theme:blink"),
    (r"blink", "theme:blink"),
    (r"pack tactics", "mechanic:pack_tactics"),
    (r"untap target", "role:tempo"),
    (r"prevent all damage", "role:protection"),
    (r"indestructible", "role:protection"),
    (r"hexproof", "role:protection"),
    (r"haste", "theme:aggro"),
    (r"menace", "theme:aggro"),
    (r"reach", "theme:defense"),
    (r"deathtouch", "theme:death"),
    (r"flying", "theme:evasion"),
    (r"trample", "theme:trample"),
    (r"vigilance", "theme:defense"),
    (r"exalted", "theme:single_attacker"),
    (r"heroic", "theme:spell_matter"),
    (r"prowess", "theme:spell_matter"),
    (r"magecraft", "theme:spell_matter"),
    (r"devotion", "theme:devotion"),
    (r"storm count", "theme:storm"),
    (r"draw and discard", "role:loot"),
]

TOKEN_PATTERNS: Sequence[Tuple[str, str]] = [
    (r"spirit token", "token:spirit"),
    (r"zombie token", "token:zombie"),
    (r"elf token", "token:elf"),
    (r"goblin token", "token:goblin"),
    (r"thopter token", "token:thopter"),
    (r"angel token", "token:angel"),
    (r"soldier token", "token:soldier"),
    (r"saproling token", "token:saproling"),
    (r"beast token", "token:beast"),
    (r"demon token", "token:demon"),
    (r"dragon token", "token:dragon"),
    (r"zombie army token", "token:zombie_army"),
    (r"treasure token", "token:treasure"),
]

ABILITY_WORD_TAGS: Dict[str, Sequence[str]] = {
    "threshold": ("mechanic:threshold", "theme:graveyard"),
    "delirium": ("mechanic:delirium", "theme:graveyard"),
    "domain": ("mechanic:domain", "theme:land_synergy"),
    "visit": ("mechanic:visit",),
    "heroic": ("mechanic:heroic", "theme:spell_matter"),
    "raid": ("mechanic:raid", "theme:aggro"),
    "channel": ("mechanic:channel",),
    "exhaust": ("mechanic:exhaust",),
    "metalcraft": ("mechanic:metalcraft", "theme:artifacts"),
    "constellation": ("mechanic:constellation", "theme:enchantments"),
    "morbid": ("mechanic:morbid", "theme:graveyard"),
    "magecraft": ("mechanic:magecraft", "theme:spell_matter"),
    "imprint": ("mechanic:imprint",),
    "corrupted": ("mechanic:corrupted", "theme:poison"),
    "hellbent": ("mechanic:hellbent", "theme:hand_empty"),
    "enrage": ("mechanic:enrage", "theme:damage_triggers"),
    "ferocious": ("mechanic:ferocious", "theme:power_4_plus"),
    "strive": ("mechanic:strive", "theme:spell_matter"),
    "coven": ("mechanic:coven", "theme:creature_balance"),
    "battalion": ("mechanic:battalion", "theme:go_wide"),
    "inspired": ("mechanic:inspired", "theme:untap"),
    "revolt": ("mechanic:revolt", "theme:permanent_leaves"),
    "boast": ("mechanic:boast", "theme:attack_triggers"),
    "spell mastery": ("mechanic:spell_mastery", "theme:graveyard"),
    "adamant": ("mechanic:adamant", "theme:mono_color"),
    "converge": ("mechanic:converge", "theme:multicolor"),
    "alliance": ("mechanic:alliance", "theme:token_support"),
    "rally": ("mechanic:rally", "theme:allies"),
    "bloodrush": ("mechanic:bloodrush", "theme:aggro"),
    "formidable": ("mechanic:formidable", "theme:power_4_plus"),
    "lieutenant": ("mechanic:lieutenant", "theme:commander_synergy"),
    "undergrowth": ("mechanic:undergrowth", "theme:graveyard"),
    "celebration": ("mechanic:celebration", "theme:aggro"),
    "forecast": ("mechanic:forecast",),
    "radiance": ("mechanic:radiance", "theme:multicolor"),
    "parley": ("mechanic:parley", "theme:multiplayer"),
    "pack tactics": ("mechanic:pack_tactics", "theme:aggro"),
    "tempting offer": ("mechanic:tempting_offer", "theme:multiplayer"),
    "council's dilemma": ("mechanic:councils_dilemma", "theme:multiplayer"),
    "eminence": ("mechanic:eminence", "theme:commander_synergy"),
    "fateful hour": ("mechanic:fateful_hour", "theme:life_total"),
    "cohort": ("mechanic:cohort", "theme:allies"),
    "chroma": ("mechanic:chroma", "theme:devotion"),
    "fathomless descent": ("mechanic:descend", "theme:graveyard"),
    "join forces": ("mechanic:join_forces", "theme:multiplayer"),
    "max speed": ("mechanic:max_speed", "theme:aggro"),
    "prize": ("mechanic:prize",),
    "eerie": ("mechanic:eerie", "theme:enchantments"),
    "survival": ("mechanic:survival", "theme:tap_synergy"),
    "to solve": ("mechanic:case", "theme:investigation"),
    "solved": ("mechanic:case", "theme:investigation"),
    "hero's reward": ("mechanic:heros_reward", "theme:multiplayer"),
    "void": ("mechanic:void", "theme:chaos"),
    "renew": ("mechanic:renew", "theme:graveyard"),
    "flurry": ("mechanic:flurry", "theme:spell_matter"),
    "valiant": ("mechanic:valiant", "theme:heroic"),
    "will of the council": ("mechanic:will_of_the_council", "theme:multiplayer"),
    "kinship": ("mechanic:kinship", "theme:tribal"),
    "companion": ("mechanic:companion", "theme:deckbuilding"),
    "gotcha": ("mechanic:gotcha", "theme:meta"),
    "addendum": ("mechanic:addendum", "theme:spell_matter"),
    "paradox": ("mechanic:paradox", "theme:chaos"),
}


def words_from(text: str) -> List[str]:
    """Split a string into lowercase words, stripping punctuation we do not need."""
    return [word for word in re.split(r"[\s/,-]+", text) if word]


def map_color(symbol: str) -> str:
    """Convert a mana symbol (WUBRG) into its color name."""
    return COLOR_MAP.get(symbol.upper(), symbol.lower())


def parse_stat(value: Any) -> Optional[Any]:
    """Interpret a power/toughness entry."""
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        if value.isdigit():
            return int(value)
        if "*" in value:
            return "variable"
    return None


def tag_card(card: Dict[str, Any]) -> List[str]:
    tags: Set[str] = set()

    type_line = card.get("type_line") or ""
    if type_line:
        parts = [p.strip() for p in type_line.split(EM_DASH)]
        if parts:
            for word in words_from(parts[0].lower()):
                tags.add(f"type:{word}")
        if len(parts) > 1:
            for word in words_from(parts[1].lower()):
                tags.add(f"subtype:{word}")
        lower_type = type_line.lower()
        if "saga" in lower_type:
            tags.add("type:saga")
        if "equipment" in lower_type:
            tags.add("theme:equipment")
        if "vehicle" in lower_type:
            tags.add("theme:vehicles")
        if "token" in lower_type:
            tags.add("type:token")

    for keyword in card.get("keywords") or []:
        tags.add(f"mechanic:{keyword.lower().replace(' ', '_')}")

    oracle_text = (card.get("oracle_text") or "").lower()
    for pattern, tag in PATTERN_TAGS:
        if re.search(pattern, oracle_text):
            tags.add(tag)
    for pattern, tag in TOKEN_PATTERNS:
        if re.search(pattern, oracle_text):
            tags.add(tag)
    for ability, ability_tags in ABILITY_WORD_TAGS.items():
        if re.search(rf"{re.escape(ability)} \u2014", oracle_text):
            tags.update(ability_tags)

    colors = card.get("color_identity") or []
    if colors:
        for symbol in colors:
            tags.add(f"color:{map_color(symbol)}")
        if len(colors) > 1:
            tags.add("color:multicolor")
    else:
        tags.add("color:colorless")

    for symbol in card.get("produced_mana") or []:
        tags.add(f"produces:{map_color(symbol)}")

    mana_cost = card.get("mana_cost") or ""
    if "x" in mana_cost.lower():
        tags.add("cost:variable")
    if "P" in mana_cost:
        tags.add("cost:phyrexian")

    cmc = card.get("cmc")
    if isinstance(cmc, (int, float)):
        if cmc <= 2:
            tags.add("curve:low")
        elif cmc <= 4:
            tags.add("curve:mid")
        else:
            tags.add("curve:high")

    power = parse_stat(card.get("power"))
    if power == "variable":
        tags.add("stat:power_variable")
    elif isinstance(power, int):
        if power >= 5:
            tags.add("stat:power_high")
        elif power <= 2:
            tags.add("stat:power_low")

    toughness = parse_stat(card.get("toughness"))
    if toughness == "variable":
        tags.add("stat:tough_variable")
    elif isinstance(toughness, int):
        if toughness >= 5:
            tags.add("stat:tough_high")
        elif toughness <= 2:
            tags.add("stat:tough_low")

    for fmt, status in (card.get("legalities") or {}).items():
        if status == "legal":
            tags.add(f"format:{fmt}")

    for part in card.get("all_parts") or []:
        component = part.get("component")
        if component:
            tags.add(f"related:{component}")

    layout = card.get("layout")
    if layout in {"meld", "modal_dfc", "transform", "adventure", "leveler", "class"}:
        tags.add(f"layout:{layout}")

    rarity = card.get("rarity")
    if rarity:
        tags.add(f"rarity:{rarity}")

    if oracle_text.count("token") >= 2:
        tags.add("theme:token_swarm")
    if "devotion" in oracle_text:
        tags.add("theme:devotion")
    if "artifact" in oracle_text:
        tags.add("theme:artifacts")
    if "enchantment" in oracle_text:
        tags.add("theme:enchantments")
    if "draw" in oracle_text and "discard" in oracle_text:
        tags.add("role:loot")

    return sorted(tags)


def build_record(card: Dict[str, Any], tags: List[str]) -> Dict[str, Any]:
    """Project the card data into a downstream-friendly JSON record."""
    return {
        "id": card.get("id"),
        "oracle_id": card.get("oracle_id"),
        "name": card.get("name"),
        "lang": card.get("lang"),
        "released_at": card.get("released_at"),
        "set": card.get("set"),
        "set_name": card.get("set_name"),
        "collector_number": card.get("collector_number"),
        "rarity": card.get("rarity"),
        "mana_cost": card.get("mana_cost"),
        "cmc": card.get("cmc"),
        "type_line": card.get("type_line"),
        "colors": card.get("colors"),
        "color_identity": card.get("color_identity"),
        "produced_mana": card.get("produced_mana"),
        "keywords": card.get("keywords"),
        "oracle_text": card.get("oracle_text"),
        "power": card.get("power"),
        "toughness": card.get("toughness"),
        "loyalty": card.get("loyalty"),
        "layout": card.get("layout"),
        "legalities": card.get("legalities"),
        "games": card.get("games"),
        "scryfall_uri": card.get("scryfall_uri"),
        "edhrec_rank": card.get("edhrec_rank"),
        "tags": tags,
    }


def load_cards(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(data: Iterable[Dict[str, Any]], path: str, indent: Optional[int]) -> None:
    stream = sys.stdout if path == "-" else open(path, "w", encoding="utf-8")
    try:
        json.dump(list(data), stream, ensure_ascii=False, indent=indent)
        if indent is not None or path != "-":
            stream.write("\n")
    finally:
        if stream is not sys.stdout:
            stream.close()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synergy tags for Scryfall oracle cards.")
    parser.add_argument("--input", required=True, help="Path to the Scryfall oracle JSON array.")
    parser.add_argument("--output", required=True, help="Destination file or '-' for stdout.")
    parser.add_argument("--language", default="en", help="Language code to keep (use 'any' to keep all).")
    parser.add_argument("--limit", type=int, help="Limit number of cards processed (useful for sampling).")
    parser.add_argument("--include-tokens", action="store_true", help="Keep cards with layout == 'token'.")
    parser.add_argument("--indent", type=int, default=2, help="JSON indent (<0 for compact output).")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    indent = None if args.indent is not None and args.indent < 0 else args.indent
    language = args.language.lower()
    language_filter = None if language in {"", "any", "all"} else args.language

    cards = load_cards(args.input)
    results: List[Dict[str, Any]] = []

    for card in cards:
        if language_filter and card.get("lang") != language_filter:
            continue
        if not args.include_tokens and card.get("layout") == "token":
            continue

        tags = tag_card(card)
        results.append(build_record(card, tags))

        if args.limit and len(results) >= args.limit:
            break

    dump_json(results, args.output, indent)


if __name__ == "__main__":
    main()
