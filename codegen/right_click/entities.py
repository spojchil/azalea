"""Entity type -> class -> which class declares `interact` / `mobInteract`.

Reads the decompiled 26.1.2 sources (`EntityType.java` for the registered
classes, `extends` clauses for the hierarchy). Prints TSV: id, class,
interact declarer, mobInteract declarer, whether the class is a Mob.

Usage: python3 entities.py > entities.tsv  (with ROOT pointing at the
decompiled `net/minecraft/` directory)
"""
import os, re

ROOT = os.path.expanduser("~/vanilla-src-26.1.2/net/minecraft/")
src = open(ROOT + "world/entity/EntityType.java").read()
types = re.findall(r"EntityType<(\w+)> \w+ = register\(\s*\"(\w+)\"", src)

classes = {}  # 简单类名 → (父类简单名, 源码)
for dirpath, _, files in os.walk(ROOT + "world/entity"):
    for f in files:
        if f.endswith(".java"):
            text = open(os.path.join(dirpath, f), errors="replace").read()
            name = f[:-5]
            m = re.search(r"\bclass " + name + r"(?:<[^{]*?>)?\s+extends\s+(\w+)", text)
            classes[name] = (m.group(1) if m else None, text)

METHODS = {
    "interact": r"InteractionResult interact\(final Player",
    "mobInteract": r"InteractionResult mobInteract\(",
}


def declarer(cls, pattern):
    c = cls
    while c in classes:
        parent, text = classes[c]
        if re.search(pattern, text):
            return c
        c = parent
    return "-"


def is_mob(cls):
    c = cls
    while c in classes:
        if c == "Mob":
            return True
        c = classes[c][0]
    return False


for cls, name in types:
    row = [name, cls] + [declarer(cls, p) for p in METHODS.values()] + [str(is_mob(cls))]
    print("\t".join(row))
