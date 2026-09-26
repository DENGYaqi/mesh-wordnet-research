"""Install the pinned offline English-to-Chinese model in the viewer image."""

import hashlib
import tempfile
import urllib.request
import zipfile
from pathlib import Path

MODEL_URL = "https://argos-net.com/v1/translate-en_zh-1_9.argosmodel"
MODEL_SHA256 = "433e7c4f034d87fbe2353161e05f18646d7999452f801a4e1f0378522b9850ab"

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "en_zh.argosmodel"
    request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response:
        path.write_bytes(response.read())
    if hashlib.sha256(path.read_bytes()).hexdigest() != MODEL_SHA256:
        raise RuntimeError("English-Chinese translation model checksum mismatch")
    with zipfile.ZipFile(path) as model:
        model.extractall("/opt")
