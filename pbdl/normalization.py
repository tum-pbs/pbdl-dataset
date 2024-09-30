import numpy as np
import h5py
import sys
from abc import ABC, abstractmethod
from itertools import groupby

from pbdl.utilities import get_const_sim, get_meta_data
from pbdl.logging import info, success, warn, fail

NORM_DATA_ARR = [
    "norm_fields_sca_mean",
    "norm_fields_sca_std",
    "norm_fields_std",
    "norm_fields_sca_min",
    "norm_fields_sca_max",
    "norm_const_mean",
    "norm_const_std",
]


class NormStrategy(ABC):

    @abstractmethod
    def normalize_data(self, data):
        pass

    @abstractmethod
    def normalize_data_rev(self, data):
        pass

    def normalize_const(self, const):
        """Constant normalization always uses mean and standard deviation across all variants."""
        return (
            const - self.const_mean
        ) / self.const_std  # TODO handling zero const_std?

    def prepare(self, dset, sel_const):
        self.sel_const = sel_const
        self.meta = get_meta_data(dset)

        if not all(key in dset for key in NORM_DATA_ARR):
            self.__cache_norm_data__(dset)

        self.__load_and_validate_norm_data__(dset)

        if sel_const:
            indices = [
                i
                for i, const in enumerate(self.meta["const"])
                if const in self.sel_const
            ]
            self.const_std = self.const_std[indices]
            self.const_mea = self.const_mean[indices]

    def __cache_norm_data__(self, dset):
        info(
            "No precomputed normalization data found (or not complete). Calculating data..."
        )

        num_sca_fields = self.meta["num_sca_fields"]

        # calculate the starting indices for fields
        field_indices = [0]
        for _, f in groupby(self.meta["fields_scheme"]):
            field_indices.append(field_indices[-1] + len(list(f)))  # TODO

        # slim means that for vector (non-scalar) fields the std must first be broadcasted to the original size
        fields_std_slim = [0] * self.meta["num_fields"]

        fields_sca_std = np.full(
            (num_sca_fields,) + (1,) * self.meta["num_spatial_dim"], 0
        )
        fields_sca_mean = np.full(
            (num_sca_fields,) + (1,) * self.meta["num_spatial_dim"], 0
        )

        fields_sca_min = np.full(
            (num_sca_fields,) + (1,) * self.meta["num_spatial_dim"], float("inf")
        )
        fields_sca_max = np.full(
            (num_sca_fields,) + (1,) * self.meta["num_spatial_dim"], -float("inf")
        )

        const_stacked = []

        # sequential loading of sims, norm data will be combined in the end
        for s, sim in enumerate(dset["sims/"]):

            sim = dset["sims/" + sim]

            axis = (0,) + tuple(range(2, 2 + self.meta["num_spatial_dim"]))
            fields_sca_std = np.add(
                fields_sca_std, np.std(sim, axis=axis, keepdims=True)[0]
            )
            fields_sca_mean = np.add(
                fields_sca_mean, np.mean(sim, axis=axis, keepdims=True)[0]
            )

            fields_sca_min = np.minimum(
                fields_sca_min, np.min(sim, axis=axis, keepdims=True)[0]
            )
            fields_sca_max = np.maximum(
                fields_sca_max, np.max(sim, axis=axis, keepdims=True)[0]
            )

            for f in range(self.meta["num_fields"]):
                field = sim[:, field_indices[f] : field_indices[f + 1], ...]

                # vector norm
                field_norm = np.linalg.norm(field, axis=1, keepdims=True)

                # frame dim + spatial dims
                axis = (0,) + tuple(range(2, 2 + self.meta["num_spatial_dim"]))

                # std over frame dim and spatial dims
                fields_std_slim[f] += np.std(field_norm, axis=axis, keepdims=True)[0]

            const_stacked.append(get_const_sim(dset, s))

        fields_sca_mean = np.array(fields_sca_mean) / self.meta["num_sims"]
        fields_sca_std = np.array(fields_sca_std) / self.meta["num_sims"]

        # TODO overall std is calculated by averaging the std of all sims, efficient but mathematically not correct
        fields_std = []
        for f in range(self.meta["num_fields"]):
            field_std_avg = fields_std_slim[f] / self.meta["num_sims"]
            field_len = field_indices[f + 1] - field_indices[f]
            fields_std.append(
                np.broadcast_to(  # broadcast to original field dims
                    field_std_avg,
                    (field_len,) + (1,) * self.meta["num_spatial_dim"],
                )
            )
        fields_std = np.concatenate(fields_std, axis=0)

        # caching norm data
        dset["norm_fields_sca_mean"] = fields_sca_mean
        dset["norm_fields_sca_std"] = fields_sca_std
        dset["norm_fields_std"] = fields_std
        dset["norm_fields_sca_min"] = fields_sca_min
        dset["norm_fields_sca_max"] = fields_sca_max
        dset["norm_const_mean"] = np.mean(const_stacked, axis=0, keepdims=False)
        dset["norm_const_std"] = np.std(const_stacked, axis=0, keepdims=False)

    def __load_and_validate_norm_data__(self, dset):
        # load normalization data
        # [()] reads the entire array TODO
        self.fields_sca_mean = dset["norm_fields_sca_mean"][()]
        self.fields_sca_std = dset["norm_fields_sca_std"][()]
        self.fields_std = dset["norm_fields_std"][()]
        self.fields_sca_min = dset["norm_fields_sca_min"][()]
        self.fields_sca_max = dset["norm_fields_sca_max"][()]
        self.const_mean = dset["norm_const_mean"][()]
        self.const_std = dset["norm_const_std"][()]

        # do basic checks on shape
        if self.fields_std.shape[0] != self.meta["sim_shape"][1]:
            raise ValueError(
                "Inconsistent number of fields between normalization data and simulation data."
            )

        if self.const_mean.shape[0] != self.meta["num_const"]:
            raise ValueError(
                "Mean data of constants does not match shape of constants."
            )

        if self.const_std.shape[0] != self.meta["num_const"]:
            raise ValueError("Std data of constants does not match shape of constants.")


class StdNorm(NormStrategy):
    """Normalizes fields using only the standard deviation."""

    def normalize_data(self, data):
        return data / self.fields_std

    def normalize_data_rev(self, data):
        return data * self.fields_std

        # if (const_std < 10e-10).any():
        #     const_norm = np.zeros_like(const)
        # else:
        #     const_norm = (const - const_mean) / const_std


class MeanStdNorm(NormStrategy):
    """Normalizes fields using both mean and standard deviation. Ignores vector fields and treats them like scalar fields, thus does not use the field scheme."""

    def normalize_data(self, data):
        return (data - self.fields_sca_mean) / self.fields_sca_std

    def normalize_data_rev(self, data):
        return data * self.fields_sca_std + self.fields_sca_mean


class MinMaxNorm(NormStrategy):
    """Scales fields to a min-max range."""

    def __init__(self, min_val=0, max_val=1):
        self.min_val = min_val
        self.max_val = max_val

    def normalize_data(self, data):
        return (data - self.fields_sca_min) / (
            self.fields_sca_max - self.fields_sca_min
        ) * (self.max_val - self.min_val) + self.min_val

    def normalize_data_rev(self, data):
        return ((data - self.min_val) / (self.max_val - self.min_val)) * (
            self.fields_sca_max - self.fields_sca_min
        ) + self.fields_sca_min


def clear_cache(dset):

    for key in NORM_DATA_ARR:
        dset.pop(key, None)


STR_TO_NORM_STRAT = {
    "std": (StdNorm, {}),
    "mean-std": (MeanStdNorm, {}),
    "zero-to-one": (MinMaxNorm, {"min_val": 0, "max_val": 1}),
    "minus-one-to-one": (MinMaxNorm, {"min_val": -1, "max_val": 1}),
}

def get_norm_strat_from_str(str):
    if str not in STR_TO_NORM_STRAT.keys():
        suggestions = ", ".join(STR_TO_NORM_STRAT.keys())
        fail(
            f"Unknown normalization strategy `{str}`. Supported strategies are: {suggestions}."
        )
        sys.exit(0)

    norm_class, args = STR_TO_NORM_STRAT.get(str)
    return norm_class(**args)
