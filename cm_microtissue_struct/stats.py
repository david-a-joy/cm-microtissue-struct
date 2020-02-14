""" Stat utils for summarizing distributions

* :py:func:`bin_cm_counts`: Bin counts for cardiomyocyte cells only
* :py:func:`bin_any_counts`: Bin counts for all cells

"""

# Imports
import pandas as pd

import numpy as np

# Functions


def bin_cm_counts(cm_df: pd.DataFrame) -> pd.DataFrame:
    """ Bin the cardiomyocyte counts by invididual image

    :param DataFrame cm_df:
        The CM comparison dataframe
    :returns:
        A new dataframe binned by neighbor counts
    """
    cm_counts = {}
    all_bins = set(cm_df['Num CM Near CM']) | set(cm_df['Num CF Near CM'])
    all_bins = list(sorted(all_bins))

    image_total = 0
    cm_total = 0
    cf_total = 0
    for image_name in np.unique(cm_df['ImageName']):
        image_mask = cm_df['ImageName'] == image_name
        image_df = cm_df[image_mask]
        image_total += 1
        for bin in all_bins:
            count = np.sum(image_df['Num CM Near CM'] == bin)
            cm_counts.setdefault('Num CM Near CM', []).append(bin)
            cm_counts.setdefault('Count CM Near CM', []).append(count)
            cm_total += count
        for bin in all_bins:
            count = np.sum(image_df['Num CF Near CM'] == bin)
            cm_counts.setdefault('Num CF Near CM', []).append(bin)
            cm_counts.setdefault('Count CF Near CM', []).append(count)
            cf_total += count

    cm_total /= image_total
    cf_total /= image_total

    cm_counts = pd.DataFrame(cm_counts)
    cm_counts['Percent CM Near CM'] = cm_counts['Count CM Near CM'] / cm_total * 100
    cm_counts['Percent CF Near CM'] = cm_counts['Count CF Near CM'] / cf_total * 100
    return cm_counts


def bin_any_counts(any_df: pd.DataFrame, max_r: float = 150) -> pd.DataFrame:
    """ Bin the cardiomyocyte counts by invididual image

    :param DataFrame any_df:
        The all cells comparison dataframe
    :returns:
        A new dataframe binned by radius counts
    """
    any_counts = {}
    all_bins = np.concatenate([np.linspace(0, 1.01, 10), [np.inf]], axis=0)
    any_df['Norm All to Center'] = any_df['Dist All to Center'] / max_r

    image_total = 0
    any_total = 0
    for image_name in np.unique(any_df['ImageName']):
        image_mask = any_df['ImageName'] == image_name
        image_df = any_df[image_mask]
        image_total += 1
        for bin_st, bin_ed in zip(all_bins[:-1], all_bins[1:]):
            if not np.isfinite(bin_ed):
                continue
            mask = np.logical_and(image_df['Norm All to Center'] >= bin_st,
                                  image_df['Norm All to Center'] < bin_ed)
            count = np.sum(mask)
            any_counts.setdefault('Dist to Center', []).append((bin_st+bin_ed)/2.0 * max_r)
            any_counts.setdefault('Count Dist to Center', []).append(count)
            any_total += count

    any_total /= image_total

    any_counts = pd.DataFrame(any_counts)
    any_counts['Percent Dist to Center'] = any_counts['Count Dist to Center'] / any_total * 100
    return any_counts
