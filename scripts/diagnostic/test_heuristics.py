import re

TAG_RULES = {
    "Dungeon Crawler": [r"dungeon crawler", r"crawling", r"dungeons"],
    "Hack and Slash": [r"hack and slash", r"hack & slash", r"slay hordes"],
    "ARPG": [r"action rpg", r"\barpg\b", r"action role-playing"],
    "Looter Shooter": [r"looter shooter", r"shoot and loot"],
    "Loot": [r"\bloot\b", r"looting"],
    "Cyberpunk": [r"cyberpunk", r"neon", r"dystopian future"],
    "Souls-like": [r"souls-like", r"soulslike", r"punishing difficulty"],
    "Open World": [r"open world", r"explore.*world", r"vast world"],
    "Choices Matter": [r"choices matter", r"multiple endings"],
    "Deckbuilder": [r"deckbuilding", r"deck-builder", r"card game"]
}

def extract_tags(text):
    text = str(text).lower()
    tags = set()
    for tag, patterns in TAG_RULES.items():
        for pat in patterns:
            if re.search(pat, text):
                tags.add(tag)
    return list(tags)

print(extract_tags("Fight your way through an exciting action-adventure game, inspired by classic dungeon crawlers and set in the Minecraft universe!"))
print(extract_tags("Torchlight II is filled to the brim with randomized levels, enemies and loot. Action RPG."))
