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


def dl_parts(dset: str, config, sims: list[int] = None):
    os.makedirs(config["global_dataset_dir"], exist_ok=True)

    dest = os.path.join(config["global_dataset_dir"], dset + config["dataset_ext"])

    # TODO dispatching
    modified = dl_parts_from_huggingface(
        dset, dest, config, sims, prog_hook=print_download_progress
    )

    # normalization data will not incorporate all sims after download
    if modified:
        with h5py.File(dest, "r+") as dset:
            norm.clear_cache(dset)


def fetch_index(config):
    # TODO dispatching
    return fetch_index_from_huggingface(config)


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

            if "sims/sim" + str(i) not in f:
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
        prog_hook(len(sims), 1, len(sims), message="download completed")

    return modified


def fetch_index_from_huggingface(config):
    repo_id = config["hf_repo_id"]
    url_repo = f"https://huggingface.co/datasets/{repo_id}/resolve/main/"
    index_path = pkg_resources.resource_filename(__name__, "global_index.json")

    try:
        files = get_hf_repo_file_list(repo_id)
        first_level_dirs = {file.split("/")[0] for file in files if "/" in file}

        meta_all_combined = {}
        for ds in first_level_dirs:
            url_meta_all = url_repo + ds + "/meta_all.json"
            meta_all = json.load(urllib.request.urlopen(url_meta_all))

            meta_all_combined[ds] = meta_all

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

    if progress == total_size:
        sys.stdout.write("\n")
        sys.stdout.flush()
