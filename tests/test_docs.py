"""Every relative link in the Markdown docs points to a file or folder that exists."""
import glob
import os
import re

import pytest

from conftest import ROOT

DOCS = sorted({os.path.relpath(p, ROOT) for pattern in ("*.md", "docs/*.md", "experiments/**/*.md", "archive/*.md", "scripts/**/*.md")
               for p in glob.glob(os.path.join(ROOT, pattern), recursive=True)})
LINK = re.compile(r"\]\(([^)\s]+)\)")


@pytest.mark.parametrize("doc", DOCS)
def test_relative_links_resolve(doc):
    text = open(os.path.join(ROOT, doc)).read()
    text = re.sub(r"```.*?```", "", text, flags=re.S)                  # ignore code blocks
    broken = []
    for target in LINK.findall(text):
        if re.match(r"[a-z]+:", target) or target.startswith("#"):    # URLs, mailto, same-page anchors
            continue
        path = target.split("#")[0]
        if not os.path.exists(os.path.normpath(os.path.join(ROOT, os.path.dirname(doc), path))):
            broken.append(target)
    assert not broken, f"{doc}: broken links {broken}"
