import datetime
import functools
from google.cloud import storage
from graphcast import autoregressive, casting, checkpoint, data_utils as du, graphcast, normalization, rollout
import haiku as hk
import isodate
import jax
import math
import numpy as np
import pandas as pd
from pysolar.radiation import get_radiation_direct
from pysolar.solar import get_altitude
import pytz
import scipy
from typing import Dict
import xarray

# ... (rest of the code remains the same)

def getSingleAndPressureValues(year):
    """Loads ERA5 data for a single year."""

    singlelevel = xarray.open_dataset(f'Dataset/single-level-{year}.nc', engine=scipy.__name__).to_dataframe()
    singlelevel = singlelevel.rename(columns={col: singlelevelfields[ind] for ind, col in enumerate(singlelevel.columns.values.tolist())})
    singlelevel = singlelevel.rename(columns={'geopotential': 'geopotential_at_surface'})

    # Calculating the sum of the last 6 hours of rainfall.
    singlelevel = singlelevel.sort_index()
    singlelevel['total_precipitation_6hr'] = singlelevel.groupby(level=[0, 1])['total_precipitation'].rolling(window=6, min_periods=1).sum().reset_index(level=[0, 1], drop=True)
    singlelevel.pop('total_precipitation')

    pressurelevel = xarray.open_dataset(f'Dataset/pressure-level-{year}.nc', engine=scipy.__name__).to_dataframe()
    pressurelevel = pressurelevel.rename(columns={col: pressurelevelfields[ind] for ind, col in enumerate(pressurelevel.columns.values.tolist())})

    return singlelevel, pressurelevel

# ... (rest of the code remains the same)

if __name__ == '__main__':

    values: Dict[str, xarray.Dataset] = {}

    # Get data for a specific year (e.g., 2024)
    year = 2024
    single, pressure = getSingleAndPressureValues(year)
    values['inputs'] = pd.merge(pressure, single, left_index=True, right_index=True, how='inner')
    values['inputs'] = integrateProgress(values['inputs'])
    values['inputs'] = formatData(values['inputs'])

    values['targets'] = getTargets(first_prediction, values['inputs'])

    values['forcings'] = getForcings(values['targets'])

    values = {value: makeXarray(values[value]) for value in values}

    predictions = Predictor.predict(values['inputs'], values['targets'], values['forcings'])
    predictions.to_dataframe().to_csv('predictions.csv', sep=',')