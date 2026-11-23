"""Download the repo's bundled fonts (SIL OFL) from notofonts.github.io."""
from __future__ import annotations

import urllib.request
from pathlib import Path

B = "https://raw.githubusercontent.com/notofonts/notofonts.github.io/main/fonts/"
FONTS = [
    "NotoSansBengali/hinted/ttf/NotoSansBengali-Regular.ttf",
    "NotoSansBengali/hinted/ttf/NotoSansBengali-Bold.ttf",
    "NotoSans/hinted/ttf/NotoSans-Regular.ttf",
    "NotoSans/hinted/ttf/NotoSans-Bold.ttf",
]


def main() -> None:
    out = Path("fonts")
    out.mkdir(exist_ok=True)
    for rel in FONTS:
        name = rel.rsplit("/", 1)[1]
        urllib.request.urlretrieve(B + rel, str(out / name))
        print("fetched", name)


if __name__ == "__main__":
    main()
