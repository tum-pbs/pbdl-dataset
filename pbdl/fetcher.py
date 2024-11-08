import requests
import h5py
import os
import urllib
import io
import json
import pbdl.normalization as norm
import pkg_resources
import sys
from pbdl.colors import colors
from pbdl.logging import info, success, warn, fail


def dl_parts(dset: str, config, sims: list[int] = None, disable_progress=False):
    os.makedirs(config["global_dataset_dir"], exist_ok=True)
    dest = os.path.join(config["global_dataset_dir"], dset + config["dataset_ext"])

    # TODO dispatching
    prog_hook = None if disable_progress else print_download_progress
    modified = dl_parts_from_huggingface(dset, dest, config, sims, prog_hook=prog_hook)

    # normalization data will not incorporate all sims after download
    if modified:
        with h5py.File(dest, "r+") as dset:
            norm.clear_cache(dset)


def dl_single_file(dset: str, config, disable_progress=False):
    os.makedirs(config["global_dataset_dir"], exist_ok=True)
    dest = os.path.join(config["global_dataset_dir"], dset + config["dataset_ext"])

    if os.path.exists(dest):
        # dataset already downloaded
        return

    prog_hook = None if disable_progress else print_download_progress
    dl_single_file_from_huggingface(dset, dest, config, prog_hook=prog_hook)


def fetch_index(config):
    # TODO dispatching
    return fetch_index_from_huggingface(config)


def dl_single_file_from_huggingface(dset: str, dest: str, config, prog_hook=None):
    repo_id = config["hf_repo_id"]
    url_ds = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{dset}{config['dataset_ext']}"

    with urllib.request.urlopen(url_ds) as response:
        total_size = int(response.info().get("Content-Length").strip())
        block_size = 1024
        with open(dest, "wb") as out_file:
            for count, data in enumerate(iter(lambda: response.read(block_size), b"")):
                out_file.write(data)
                if prog_hook:
                    prog_hook(count, block_size, total_size)

    if prog_hook:
        prog_hook(1, 1, 1, message="download completed")


def dl_parts_from_huggingface(
    dataset: str, dest: str, config, sims: list[int] = None, prog_hook=None
):
    """Adds partitions to hdf5 file. If parts is not specified, alls partitions are added."""

    repo_id = config["hf_repo_id"]

    # look up partitions, if none selected
    if not sims:
        files = get_hf_repo_file_list(repo_id)

        # filter files for dataset sim files
        sim_files = [f for f in files if f.startswith(dataset + "/sim")]

        # expect numbering to be consecutive
        sims = range(len(sim_files))

    url_ds = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{dataset}"

    modified = False
    with h5py.File(dest, "a") as f:
        for i, s in enumerate(sims):
            if prog_hook:
                prog_hook(
                    i,
                    1,
                    len(sims),
                    message=f"downloading sim {s}",
                )

            if "sims/sim" + str(s) not in f:
                modified = True

                url_sim = url_ds + "/sim" + str(s) + config["dataset_ext"]

                with urllib.request.urlopen(url_sim) as response:
                    with h5py.File(io.BytesIO(response.read()), "r") as dset_sim:

                        if len(dset_sim) != 1:
                            raise ValueError(
                                f"A partition file must contain exactly one simulation."
                            )

                        sim = f.create_dataset(
                            "sims/sim" + str(s), data=dset_sim["sims/sim0"]
                        )

                        for key, value in dset_sim["sims/sim0"].attrs.items():
                            sim.attrs[key] = value

        # update meta all
        meta_all_url = url_ds + "/meta_all.json"
        with urllib.request.urlopen(meta_all_url) as response:
            meta_all = json.loads(response.read().decode())
            for key, value in meta_all.items():
                f["sims/"].attrs[key] = value

    if prog_hook:
        prog_hook(1, 1, 1, message="download completed")

    return modified


def fetch_index_from_huggingface(config):
    repo_id = config["hf_repo_id"]
    url_repo = f"https://huggingface.co/datasets/{repo_id}/resolve/main/"
    index_path = pkg_resources.resource_filename(__name__, "global_index.json")

    try:
        files = get_hf_repo_file_list(repo_id)
        first_level_dirs = {file.split("/")[0] for file in files if "/" in file}
        first_level_files = {
            file[: -len(config["dataset_ext"])]
            for file in files
            if not "/" in file and file.endswith(config["dataset_ext"])
        }

        meta_all_combined = {}
        for d in first_level_dirs:
            url_meta_all = url_repo + d + "/meta_all.json"
            meta_all = json.load(urllib.request.urlopen(url_meta_all))
            meta_all["isSingleFile"] = False
            meta_all_combined[d] = meta_all

        for r in first_level_files:
            url_meta_all = url_repo + r + ".json"

            # meta data file for single-file datasets may not exist
            try:
                meta_all = json.load(urllib.request.urlopen(url_meta_all))
            except urllib.error.URLError as e:
                meta_all = dict()

            meta_all["isSingleFile"] = True
            meta_all_combined[r] = meta_all

        # cache index for offline access
        with open(index_path, "w") as f:
            json.dump(meta_all_combined, f)

    except urllib.error.URLError:
        warn("Failed to fetch global dataset index. Check your internet connection.")

    try:
        with open(index_path) as index_file:
            return json.load(index_file)
    except (FileNotFoundError, json.JSONDecodeError):
        warn(
            "Global index is not in cache or corrupted. Global datasets will not be accessible."
        )
        return {}


def get_hf_repo_file_list(repo_id: str):
    url_api = f"https://huggingface.co/api/datasets/{repo_id}"
    response = requests.get(url_api)
    repo_info = response.json()
    siblings = repo_info.get("siblings", [])
    return [s["rfilename"] for s in siblings]


def dl_parts_from_lrz():
    pass


def fetch_index_from_lrz():
    pass


def print_download_progress(count, block_size, total_size, message=None):
    progress = count * block_size
    percent = int(progress * 100 / total_size)
    bar_length = 50
    bar = (
        "━" * int(percent / 2)
        + colors.DARKGREY
        + "━" * (bar_length - int(percent / 2))
        + colors.OKBLUE
    )

    def format_size(size):
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} PB"

    downloaded_str = format_size(progress)
    total_str = format_size(total_size)

    sys.stdout.write(
        colors.OKBLUE
        + "\r\033[K"
        + (message if message else f"{downloaded_str} / {total_str}")
        + f"\t {bar} {percent}%"
        + colors.ENDC
    )
    sys.stdout.flush()

    if progress >= total_size:
        sys.stdout.write("\n")
        sys.stdout.flush()
