from __future__ import annotations
from pathlib import Path
from typing import List

from loguru import logger
from tqdm import tqdm


def mkdir_if_not_exists(dir: Path, main_process: bool = True):
    dir = Path(dir)
    if not dir.exists() and main_process:
        logger.info(f"Creating directory {dir} ...")
        dir.mkdir(parents=True, exist_ok=True)


def md5_checksum(filepath: Path) -> str:
    """Compute MD5 hex digest of a file."""
    import hashlib

    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url, target_path, chunk_size=8192):
    import requests

    target_path = Path(target_path) if isinstance(target_path, str) else target_path
    existing_size = target_path.stat().st_size if target_path.exists() else 0
    headers = {"Range": f"bytes={existing_size}-"} if existing_size > 0 else {}

    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()

        # If server ignores Range header (returns 200 instead of 206),
        # we must overwrite rather than append.
        if existing_size > 0 and r.status_code != 206:
            existing_size = 0

        mode = "ab" if existing_size > 0 else "wb"
        total_size = int(r.headers.get("content-length", 0)) + existing_size
        with (
            open(target_path, mode) as f,
            tqdm(
                total=total_size,
                unit="iB",
                unit_scale=True,
                desc=f"Downloading {target_path.name}",
                leave=True,
                initial=existing_size,
            ) as pbar,
        ):
            for chunk in r.iter_content(chunk_size):
                size = f.write(chunk)
                pbar.update(size)

    pbar.set_description(f"Downloaded {target_path.name} ({total_size} bytes)")
    print()  # Add a newline after tqdm


def unzip_file(zip_file, dest_dir):
    import zipfile

    mkdir_if_not_exists(dest_dir)
    logger.info(f"Unzipping {zip_file} to {dest_dir} ...")
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(dest_dir)
    logger.info(f"Unzipping completed: {zip_file} extracted to {dest_dir}.")


def untar_file(tar_file, dest_dir):
    import tarfile

    mkdir_if_not_exists(dest_dir)
    logger.info(f"Extracting  {tar_file} to {dest_dir} ...")
    with tarfile.open(tar_file, "r|gz") as tar_ref:
        tar_ref.extractall(dest_dir)
    logger.info(f"Extracting completed: {tar_file} extracted to {dest_dir}.")


def verify_zenodo_files(zenodo_id: str, target_dir: str) -> bool:
    """Verify extracted files against Zenodo's per-file MD5 checksums.

    Fetches the file manifest from the Zenodo API, then checks each file
    on disk.  On mismatch or missing file, re-downloads individually and
    verifies again.  Returns True if all files pass.
    """
    import requests

    target_dir = Path(target_dir)
    api_url = f"https://zenodo.org/api/records/{zenodo_id}"

    try:
        resp = requests.get(api_url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"Cannot reach Zenodo API for verification: {e} — skipping.")
        return True

    file_list = resp.json().get("files", [])
    if not file_list:
        logger.warning(
            f"Zenodo record {zenodo_id} has no files listed — skipping verification."
        )
        return True

    all_ok = True
    for file_info in file_list:
        name = file_info["key"]
        filepath = target_dir / name
        expected_md5 = file_info["checksum"].removeprefix("md5:")
        download_url = file_info["links"]["self"]

        if not filepath.exists():
            logger.warning(f"Missing file {name}, downloading individually...")
            download_file(download_url, filepath)

        actual_md5 = md5_checksum(filepath)
        if actual_md5 == expected_md5:
            logger.debug(f"Verified {name}")
            continue

        # Mismatch — re-download the individual file
        logger.warning(
            f"MD5 mismatch for {name}: expected {expected_md5}, got {actual_md5}. Re-downloading..."
        )
        filepath.unlink()
        download_file(download_url, filepath)
        actual_md5 = md5_checksum(filepath)

        if actual_md5 != expected_md5:
            logger.error(f"MD5 verification failed after re-download: {name}")
            all_ok = False
        else:
            logger.info(f"Re-downloaded and verified {name}")

    if all_ok:
        logger.info(
            f"All {len(file_list)} files verified for Zenodo record {zenodo_id}."
        )
    return all_ok


def download_zenodo_record(
    zenodo_id: str,
    target_dir: str,
    force_download: bool = False,
    temp_dir: None | str = None,
):
    import shutil
    import tempfile

    target_dir = Path(target_dir)
    target_zip_path = target_dir / f"{zenodo_id}.zip"

    if not force_download and target_zip_path.exists():
        logger.info(f"{target_zip_path} already exists, skipping download.")
    else:
        temp_zip_path = Path(tempfile.gettempdir()) / f"{zenodo_id}.zip"
        zenodo_archive_url = f"https://zenodo.org/api/records/{zenodo_id}/files-archive"
        download_file(zenodo_archive_url, temp_zip_path)
        shutil.move(temp_zip_path, target_zip_path)
        logger.info(f"Downloading completed: {zenodo_id} saved to {target_zip_path}.")

    try:
        unzipped_flag = target_dir / ".unzipped"
        if not unzipped_flag.exists():
            unzip_file(target_zip_path, target_dir)
            unzipped_flag.touch()
        else:
            logger.info(f"{target_zip_path} already unzipped, skipping unzip.")
    except Exception as e:
        logger.error(f"Failed to unzip {target_zip_path} to {target_dir}: {e}.")
        logger.error(
            f"Remove {target_zip_path} and retry, or download manually using `tools/download_manually.sh`."
        )
        raise e

    # Verify extracted files against Zenodo checksums
    if not verify_zenodo_files(zenodo_id, target_dir):
        raise RuntimeError(
            f"MD5 verification failed for Zenodo record {zenodo_id}. "
            f"Try: rm -rf {target_dir} and re-run."
        )


def attr_from_py_path(path: str, endswith: str | None = None) -> type:
    from importlib import import_module

    module_name = path.replace("/", ".")
    # Strip ending
    if module_name.endswith(".py"):
        module_name = module_name[:-3]  # Remove last 3 characters (".py")

    try:
        module = import_module(module_name)
    except ModuleNotFoundError:
        raise ValueError(f"Module not found: {module_name}")

    attr_list = [m for m in dir(module) if not endswith or m.endswith(endswith)]
    if len(attr_list) != 1:
        raise ValueError(
            f"Expected 1 class with endswith={endswith}, got {len(attr_list)}"
        )

    return getattr(module, attr_list[0])


def download_hf_model_to_local(model_names: str | List[str], output_root: str = "."):
    # Download the model to the local directory to avoid issues under multi-processes
    if isinstance(model_names, str):
        model_names = [model_names]

    if "bert-base-uncased:tokenizer" in model_names:
        model_name = "google-bert/bert-base-uncased"
        output_path = Path(output_root) / model_name
        if not output_path.exists():
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.save_pretrained(output_path)
        model_names.remove("bert-base-uncased:tokenizer")

    if "qwen2" in model_names:
        model_name = "Qwen/Qwen2.5-0.5B"
        output_path = Path(output_root) / model_name
        if not output_path.exists():
            from transformers import AutoModelForCausalLM, AutoTokenizer

            for m in [
                AutoModelForCausalLM.from_pretrained(model_name),
                AutoTokenizer.from_pretrained(model_name),
            ]:
                m.save_pretrained(output_path)
        model_names.remove("qwen2")

    if len(model_names) > 0:
        logger.warning(f"Models {model_names} are not supported for download")
