"""Download the shared GEPC archive with python -m script.setup."""
import base64
import json
import os
import shutil
import stat
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen
from zipfile import ZipFile

import hydra
from omegaconf import DictConfig
from tqdm.auto import tqdm


RESOLVE_URL = (
    "https://ckdatabasews.icloud.com/database/1/"
    "com.apple.cloudkit/production/public/records/resolve"
)
DATA_FOLDERS = {"simulate", "train-ckpt", "train-log"}


def resolve_archive(link: str) -> tuple[str, int]:
    share_id = urlsplit(link).path.rstrip("/").rsplit("/", 1)[-1]
    request = Request(
        RESOLVE_URL,
        data=json.dumps({"shortGUIDs": [{"value": share_id}]}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=30) as response:
        result = json.load(response)["results"][0]
    record = result.get("rootRecord", {})
    if result.get("requireAppleLogin") or record.get("recordType") != "content":
        raise ValueError("Expected a publicly shared iCloud file")

    fields = record["fields"]
    if fields.get("extension", {}).get("value", "").lower() != "zip":
        raise ValueError("Expected a ZIP archive")
    name = base64.b64decode(fields["encryptedBasename"]["value"]).decode()
    asset = fields["fileContent"]["value"]
    url = asset["downloadURL"].replace("${f}", quote(f"{name}.zip", safe=""))
    return url, int(asset["size"])


def unpack(archive: Path, staging: Path) -> list[Path]:
    files = []
    with ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            path = PurePosixPath(entry.filename)
            if "__MACOSX" in path.parts or path.name == ".DS_Store":
                continue
            if (
                not path.parts or path.parts[0] not in DATA_FOLDERS
                or ".." in path.parts or "\\" in entry.filename or ":" in entry.filename
                or stat.S_ISLNK(entry.external_attr >> 16)
            ):
                raise ValueError(f"Unexpected archive entry: {entry.filename!r}")
            if entry.is_dir():
                continue
            if len(path.parts) < 2:
                raise ValueError(f"Expected a file inside a data folder: {entry.filename!r}")
            relative = Path(*path.parts)
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(entry) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
            files.append(relative)
    if {path.parts[0] for path in files} != DATA_FOLDERS:
        raise ValueError("Expected simulate/, train-ckpt/, and train-log/ in the archive")
    return files


@hydra.main(version_base=None, config_path="../config", config_name="pipeline/setup")
def main(cfg: DictConfig) -> None:
    data_dir = Path(hydra.utils.to_absolute_path(cfg.data_dir))
    url, size = resolve_archive(cfg.archive_link)
    data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=data_dir) as temporary:
        staging = Path(temporary)
        archive = staging / "data.zip"
        with urlopen(url, timeout=60) as source, archive.open("wb") as output:
            with tqdm(total=size, desc=archive.name, unit="B", unit_scale=True) as bar:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
                    bar.update(len(chunk))
        if archive.stat().st_size != size:
            raise OSError(f"Incomplete download (expected {size:,} bytes)")

        # Read and CRC-check every file before publishing any of them.
        files = unpack(archive, staging)
        installed = 0
        for relative in files:
            target = data_dir / relative
            if any((data_dir / part).is_symlink() for part in relative.parents):
                raise ValueError(f"Cannot install through a symbolic link: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(staging / relative, target)
            except FileExistsError:
                if target.is_symlink() or not target.is_file():
                    raise
            else:
                installed += 1
    print(f"Setup complete in {data_dir} (installed: {installed}, skipped: {len(files) - installed})")


if __name__ == "__main__":
    main()
