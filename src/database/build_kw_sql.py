# this file generate keywords for each product in the database
import re
import sqlite3

DB_PATH = "data/canada_goose.db"

STOPWORDS |= {
  # section headers / metadata
  "features","feature","origin","disc","hood","trim","fit","length","materials","care",
  "made","canada","domestic","imported","classic","tonal","black","no","fur",

  # filler marketing
  "introducing","perfect","ideal","versatile","essential","effortless","designed","offers",
  "provides","providing","helps","help","keep","keeping","added","adds","add","allow",
  "allows","allowing","make","makes","get","today","read","learn","more","proudly",
  "collection","selection","style","silhouette","updated","update","new","most-loved",

  # repetitive action words in bullet lists
  "adjustable","closure","closures","secured","secure","double","two","way","zip","zippered",

  # common apparel noise
  "pockets","pocket","interior","exterior","front","back","upper","lower","left","right",
  "collar","cuffs","hem","panel","panels","sleeves","underarm","gussets","brim"
}

DOMAIN_KEYWORDS |= {
  # weather protection
  "waterproof","water","rain","downpour","storm","snow","wind","windproof","waterrepellent",
  "water-resistant","wind-resistant","seam-sealed","seamsealed","aquaguard","reflective",
  "venting","breathable",

  # insulation / warmth cues
  "down","downfilled","fill","thermal","warmth","insulated","extreme","arctic",

  # materials / fabric tech (very valuable)
  "cordura","merino","wool","cotton","ripstop","nylon","polartec","power","stretch",
  "arctic","tech","tri-durance","dynaluxe","ventera","aira","acclimaluxe",

  # portability / travel
  "packable","packs","pillow","travel","carabiner","backpack","straps",

  # trims / style markers people ask for
  "fur","coyote","ruff","snorkel","bomber","parka","jacket","vest","hoody","hoodie","shell",
}

PHRASES |= {
  # protection
  r"\bfully seam[- ]sealed\b": "fully_seam_sealed",
  r"\bwater[- ]repellent\b": "water_repellent",
  r"\bwater[- ]resistant\b": "water_resistant",
  r"\bwind[- ]resistant\b": "wind_resistant",
  r"\bwindproof\b": "windproof",
  r"\bhelmet[- ]compatible\b": "helmet_compatible",
  r"\bmesh venting\b": "mesh_venting",
  r"\bstorm flap\b": "storm_flap",
  r"\bwind guard\b|\bwindguard\b": "wind_guard",
  r"\baquaguard\b": "aquaguard_zippers",

  # warmth / insulation
  r"\bdown[- ]filled\b": "down_filled",
  r"\bdown[- ]filled hood\b": "down_filled_hood",
  r"\bthermal mapping\b": "thermal_mapping",
  r"\bthermal experience index\b": "tei",

  # portability / carry
  r"\bpackable into\b|\bpacks into\b": "packable",
  r"\bbackpack straps\b": "backpack_straps",

  # trims / details
  r"\bfur ruff\b": "fur_ruff",
  r"\bcoyote fur\b": "coyote_fur",
  r"\bsnorkel hood\b": "snorkel_hood",
  r"\brib[- ]knit cuffs?\b": "rib_knit_cuffs",
  r"\btricot\b|\bsueded tricot\b": "tricot",
  r"\bcordura\b": "cordura",
  r"\bmerino wool\b": "merino_wool",
}