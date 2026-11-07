import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_tif_files(image_path):
    """
    Get all the .tif files in the image folder.

    Parameters
    ----------
    image_path: pathlib Path
        Directory to search for tif files
    Returns:
        A list of .tif filenames
    """
    files = []
    for dir_file in image_path.iterdir():
        if str(dir_file).endswith("tif"):
            files.append(str(dir_file.name))
    return files


def load_clean_yield_data(yield_data_filepath):
    """
    Cleans the yield data by making sure any Nan values in the columns we care about
    are removed
    """
    important_columns = ["Year", "State ANSI", "County ANSI", "Value"]
    yield_data = pd.read_csv(yield_data_filepath).dropna(
        subset=important_columns, how="any"
    )
    return yield_data


def visualize_modis(data):
    """Visualize a downloaded MODIS file.

    Takes the red, green and blue bands to plot a
    'colour image' of a downloaded tif file.

    Note that this is not a true colour image, since
    this is a complex thing to represent. It is a 'basic
    true colour scheme'
    http://www.hdfeos.org/forums/showthread.php?t=736

    Parameters
    ----------
    data: a rasterio mimic Python file object
    """
    arr_red = data.read(1)
    arr_green = data.read(4)
    arr_blue = data.read(3)

    im = np.dstack((arr_red, arr_green, arr_blue))

    im_norm = im / im.max()

    plt.imshow(im_norm)

def select_counties(
    indices,
    years,
    counties_per_state=8,
    required_years=range(2003, 2017),
    random_state=42,
):
    required_years = set(required_years)

    county_data = pd.DataFrame({
        "state": indices[:, 0],
        "county": indices[:, 1],
        "year": years,
    })

    county_years = (
        county_data.groupby(["state", "county"])["year"]
        .agg(lambda x: set(x))
    )
    eligible_counties = county_years[county_years.apply(lambda x: required_years.issubset(x))].index.to_frame(index=False)

    rng = np.random.default_rng(random_state)
    selected = []

    for state, group in eligible_counties.groupby("state"):
        group = group.sort_values("county")

        if len(group) < counties_per_state:
            raise ValueError(
                f"State {state} only has {len(group)} eligible counties, but {counties_per_state} were requested."
            )

        chosen_indices = rng.choice(len(group),size=counties_per_state,replace=False,)
        selected.append(group.iloc[chosen_indices])

    selected_counties = pd.concat(selected,ignore_index=True,)
    selected_pairs = set(
        zip(
            selected_counties["state"],
            selected_counties["county"],
        )
    )

    selected_mask = np.array([
        (state, county) in selected_pairs
        for state, county in zip(
            indices[:, 0],
            indices[:, 1],
        )
    ])

    return selected_mask, selected_counties
