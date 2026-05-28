import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

DIV2K_URLS = {
    "train": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "valid": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
}



def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dst}")
    urlretrieve(url, dst)


def extract(zip_path: Path, out_dir: Path):
    print(f"Extracting {zip_path} -> {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def move_inner_folder(extract_root: Path, folder_name: str, data_root: Path):
    src = extract_root / folder_name
    if not src.exists():
        candidates = list(extract_root.rglob(folder_name))
        if not candidates:
            raise FileNotFoundError(f"Cannot find extracted folder: {folder_name}")
        src = candidates[0]

    dst = data_root / folder_name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(src), str(dst))
    return dst


def count_images(folder: Path) -> int:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)


def main(args):
    data_root = Path(args.data_root)
    cache_dir = data_root / "_downloads"
    extract_root = data_root / "_extract"

    targets = ["train", "valid"] if args.split == "all" else [args.split]

    for split in targets:
        url = DIV2K_URLS[split]
        zip_name = Path(url).name
        zip_path = cache_dir / zip_name
        if not zip_path.exists() or args.redownload:
            download(url, zip_path)
        else:
            print(f"Using cached file: {zip_path} (md5={md5sum(zip_path)})")

        extract(zip_path, extract_root)
        folder_name = "DIV2K_train_HR" if split == "train" else "DIV2K_valid_HR"
        out = move_inner_folder(extract_root, folder_name, data_root)

        n = count_images(out)
        print(f"Prepared {out} with {n} images")

        min_required = 800 if split == "train" else 100
        if n < min_required:
            raise RuntimeError(f"{folder_name} image count too low: {n} < {min_required}")

    if args.clean_temp:
        if extract_root.exists():
            shutil.rmtree(extract_root)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        print("Temporary download/extract folders removed")

    print("Done. You can now run train.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto download and prepare DIV2K dataset")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--split", type=str, choices=["train", "valid", "all"], default="all")
    parser.add_argument("--redownload", action="store_true")
    parser.add_argument("--clean_temp", action="store_true")
    args = parser.parse_args()
    main(args)
