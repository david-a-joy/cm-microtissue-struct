""" Input/output functions for 3D data

Read and write cell segmentation data:

* :py:func:`load_summary_datadir`: Load all the summary stats files in a directory
* :py:func:`load_position_data`: Read the point positions from Imaris
* :py:func:`save_position_data`: Save the point positions in an Imaris-like way

"""

# Imports
import re
from typing import Tuple, Optional, Dict
import pathlib

# 3rd party
import numpy as np

import pandas as pd

# Constants
reEMPIRICAL_STUDY = re.compile(r'''^
    (?P<image_name>[fac]4_4color_sphere[0-9]+)_plots_([a-z]+)
$''', re.IGNORECASE | re.VERBOSE)
reSIMULATED_STUDY = re.compile(r'''^
    (?P<image_name>(uniform|right|left|inside|outside)[0-9]+)_plots_([a-z]+)
$''', re.IGNORECASE | re.VERBOSE)

# Functions


def save_position_data(position_file: pathlib.Path,
                       position_data: Tuple[np.ndarray]):
    """ Load the position file

    :param Path position_file:
        The Imaris position data file
    :param Tuple[ndarray] position_data:
        x, y, z coordinates
    """
    position_file.parent.mkdir(exist_ok=True, parents=True)

    num_recs = len(position_data[0])
    df = pd.DataFrame({
        'Position X': position_data[0],
        'Position Y': position_data[1],
        'Position Z': position_data[2],
        'Unit': ['um' for _ in range(num_recs)],
        'Category': ['Spot' for _ in range(num_recs)],
        'Collection': ['Position' for _ in range(num_recs)],
        'Time': [1 for _ in range(num_recs)],
        'ID': list(range(num_recs)),
    })
    with position_file.open('wt') as fp:
        fp.write('\nPosition\n====================\n')
        df.to_csv(
            fp,
            index=False,
            columns=['Position X', 'Position Y', 'Position Z', 'Unit', 'Category',
                     'Collection', 'Time', 'ID'],
        )


def load_position_data(position_file: pathlib.Path) -> np.ndarray:
    """ Load the position file

    :param Path position_file:
        The Imaris position data file
    :returns:
        x, y, z coordinates
    """
    df = pd.read_csv(str(position_file), header=2)
    if 'Cell Position X' in df.columns:
        x = df['Cell Position X']
        y = df['Cell Position Y']
        z = df['Cell Position Z']
    elif 'Position X' in df.columns:
        x = df['Position X']
        y = df['Position Y']
        z = df['Position Z']
    return x.values, y.values, z.values


def parse_empirical_study(dirname: str) -> Optional[Dict[str, str]]:
    """ Parse the data from the empricial studies

    :param str dirname:
        The directory name to parse
    :returns:
        A dictionary, or None if the directory doesn't match
    """
    match = reEMPIRICAL_STUDY.match(dirname)
    if not match:
        return None

    # Decode the image name and fibroblast source
    image_name = match.group('image_name').lower().strip()
    if image_name.startswith('f'):
        fib_source = 'fetal'
    elif image_name.startswith('a'):
        fib_source = 'adult'
    elif image_name.startswith('c'):
        fib_source = 'cm_only'
    else:
        raise ValueError(f'Unknown fibroblast source in image: {image_name}')
    return {'image_name': image_name, 'fib_source': fib_source}


def parse_simulated_study(dirname: str) -> Optional[Dict[str, str]]:
    """ Parse the data from the simulated studies

    :param str dirname:
        The directory name to parse
    :returns:
        A dictionary, or None if the directory doesn't match
    """
    match = reSIMULATED_STUDY.match(dirname)
    if not match:
        return None

    # Decode the image name and fibroblast source
    image_name = match.group('image_name').lower().strip()
    if image_name.startswith('uniform'):
        fib_source = 'uniform'
    elif image_name.startswith('outside'):
        fib_source = 'outside'
    elif image_name.startswith('inside'):
        fib_source = 'inside'
    elif image_name.startswith('left'):
        fib_source = 'left'
    elif image_name.startswith('right'):
        fib_source = 'right'
    else:
        raise ValueError(f'Unknown fibroblast source in image: {image_name}')
    return {'image_name': image_name, 'fib_source': fib_source}


def load_summary_datadir(datadir: pathlib.Path) -> Tuple[pd.DataFrame]:
    """ Load the data from different studies

    :param Path datadir:
        Directory containing multiple studies to process
    :returns:
        A CM DataFrame and a CF DataFrame with all studies concatenated together
    """
    all_cm_data = []
    all_cf_data = []
    all_any_data = []
    all_volume_data = []
    for subdir in datadir.iterdir():
        if not subdir.is_dir() or subdir.name.startswith('.'):
            continue

        print(subdir)

        # Try to load the directory from different study formats
        res = parse_empirical_study(subdir.name)
        if res is None:
            res = parse_simulated_study(subdir.name)
        if res is None:
            continue

        # Unpack the results
        image_name = res['image_name']
        fib_source = res['fib_source']

        for subfile in subdir.iterdir():
            if not subfile.is_file():
                continue
            if not subfile.name.lower().startswith(image_name):
                continue

            if subfile.name.endswith('_cm_stats.xlsx'):
                df = pd.read_excel(str(subfile))
                df['ImageName'] = image_name
                df['FibSource'] = fib_source
                df['CellType'] = 'CM'
                all_cm_data.append(df)
                continue

            if subfile.name.endswith('_cf_stats.xlsx'):
                df = pd.read_excel(str(subfile))
                df['ImageName'] = image_name
                df['FibSource'] = fib_source
                df['CellType'] = 'CF'
                all_cf_data.append(df)
                continue

            if subfile.name.endswith('_all_stats.xlsx'):
                df = pd.read_excel(str(subfile))
                df['ImageName'] = image_name
                df['FibSource'] = fib_source
                df['CellType'] = 'Any'
                all_any_data.append(df)
                continue

            if subfile.name.endswith('_volume_stats.xlsx'):
                df = pd.read_excel(str(subfile))
                df['ImageName'] = image_name
                df['FibSource'] = fib_source
                all_volume_data.append(df)
                continue

    all_cm_data = pd.concat(all_cm_data, ignore_index=True)
    del all_cm_data['Unnamed: 0']
    all_cf_data = pd.concat(all_cf_data, ignore_index=True)
    del all_cf_data['Unnamed: 0']
    all_any_data = pd.concat(all_any_data, ignore_index=True)
    del all_any_data['Unnamed: 0']

    all_volume_data = pd.concat(all_volume_data, ignore_index=True)
    del all_volume_data['Unnamed: 0']

    return all_cm_data, all_cf_data, all_any_data, all_volume_data
