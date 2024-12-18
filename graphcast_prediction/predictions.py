# predictions.py
from typing import List, Dict
import datetime
import functools
import google.auth
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
import xarray as xr
import netCDF4
import os
import sys


# Define location options
print("Defining location options\n")
location_options = {
    "Foz de Iguazu": (-25.26, -54.32, "A846"),
    "Grajau": (-22.93, -43.26, "A636"),
    "Campinas": (-22.46, -47.00, "A846"), 
    "Sao Jose": (-22.75, -43.33, "A621"),
    "Tocantins": (-10.66, -48.29, "MTBA-PLUH"),
    "Paraíba do Sul": (-22.16, -43.29, "RESA-PLUH")
}

# Obtain the default credentials
print("Obtaining default credentials\n")
credentials, project_id = google.auth.default()

# Configure Google Cloud Storage
print("Configuring Google Cloud Storage\n")
gcs_bucket_name = os.environ.get('GRAPHCAST_BUCKET_NAME', 'elet-dm-graphcast')  # Use consistent variable name
client = storage.Client()
gcs_bucket = client.get_bucket(gcs_bucket_name)

# Define fields and parameters
print("Defining fields and parameters\n")
singlelevelfields = [
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    '2m_temperature',
    'geopotential',
    'land_sea_mask',
    'mean_sea_level_pressure',
    'toa_incident_solar_radiation',
    'total_precipitation'
]

pressurelevelfields = [
    'u_component_of_wind',
    'v_component_of_wind',
    'geopotential',
    'specific_humidity',
    'temperature',
    'vertical_velocity'
]

predictionFields = [
    'u_component_of_wind',
    'v_component_of_wind',
    'geopotential',
    'specific_humidity',
    'temperature',
    'vertical_velocity',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    '2m_temperature',
    'mean_sea_level_pressure',
    'total_precipitation_6hr'
]
pressure_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
pi = math.pi
gap = 6
predictions_steps = 4
watts_to_joules = 3600
first_prediction = datetime.datetime(2024, 1, 1, 18, 0)
lat_range = range(-90, 91, 1)
lon_range = range(0, 360, 1)

# AssignCoordinates class
class AssignCoordinates:
    coordinates = {
        '2m_temperature': ['batch', 'lon', 'lat', 'time'],
        'mean_sea_level_pressure': ['batch', 'lon', 'lat', 'time'],
        '10m_v_component_of_wind': ['batch', 'lon', 'lat', 'time'],
        '10m_u_component_of_wind': ['batch', 'lon', 'lat', 'time'],
        'total_precipitation_6hr': ['batch', 'lon', 'lat', 'time'],
        'temperature': ['batch', 'lon', 'lat', 'level', 'time'],
        'geopotential': ['batch', 'lon', 'lat', 'level', 'time'],
        'u_component_of_wind': ['batch', 'lon', 'lat', 'level', 'time'],
        'v_component_of_wind': ['batch', 'lon', 'lat', 'level', 'time'],
        'vertical_velocity': ['batch', 'lon', 'lat', 'level', 'time'],
        'specific_humidity': ['batch', 'lon', 'lat', 'level', 'time'],
        'toa_incident_solar_radiation': ['batch', 'lon', 'lat', 'time'],
        'year_progress_cos': ['batch', 'time'],
        'year_progress_sin': ['batch', 'time'],
        'day_progress_cos': ['batch', 'lon', 'time'],
        'day_progress_sin': ['batch', 'lon', 'time'],
        'geopotential_at_surface': ['lon', 'lat'],
        'land_sea_mask': ['lon', 'lat'],
    }


# Check if GRAPHCAST_BUCKET_NAME environment variable is set
#gcs_bucket_name = os.environ.get('GRAPHCAST_BUCKET_NAME', 'elet-dm-graphcast')
params_bucket_name = os.environ.get('GRAPHCAST_PARAMS_BUCKET', 'params')
stats_bucket_name = os.environ.get('GRAPHCAST_STATS_BUCKET', 'stats')
data_bucket_name = os.environ.get('GRAPHCAST_DATA_BUCKET', 'dataset')
model_path = os.environ.get('GRAPHCAST_MODEL_PATH',
                            '"GraphCast_operational.npz"')

# Load model parameters and configurations
print("Loading model parameters and configurations\n")
# Construct the full path to the model file
model_path = f"{params_bucket_name}/{model_path}"

# Access the model file using the Storage client and the full path
client = storage.Client()
bucket = client.get_bucket(gcs_bucket_name)
blob = bucket.blob(model_path)

with blob.open('rb') as model:
    ckpt = checkpoint.load(model, graphcast.CheckPoint)
    params = ckpt.params
    state = {}
    model_config = ckpt.model_config
    task_config = ckpt.task_config

# Load statistical data
print("Loading statistical data\n")

# Construct the full paths to the statistical data files
diffs_stddev_path = f"{stats_bucket_name}/diffs_stddev_by_level.nc"
mean_path = f"{stats_bucket_name}/mean_by_level.nc"
stddev_path = f"{stats_bucket_name}/stddev_by_level.nc"

# Load diffs_stddev_by_level.nc
blob = bucket.blob(diffs_stddev_path)
with blob.open('rb') as f:
    diffs_stddev_by_level = xr.load_dataset(f).compute()

# Load mean_by_level.nc
blob = bucket.blob(mean_path)
with blob.open('rb') as f:
    mean_by_level = xr.load_dataset(f).compute()

# Load stddev_by_level.nc
blob = bucket.blob(stddev_path)
with blob.open('rb') as f:
    stddev_by_level = xr.load_dataset(f).compute()

# Construct the GraphCast predictor
def construct_wrapped_graphcast(model_config:graphcast.ModelConfig, task_config:graphcast.TaskConfig):
    print("Constructing wrapped Graphcast\n")
    predictor = graphcast.GraphCast(model_config, task_config)
    predictor = casting.Bfloat16Cast(predictor)
    predictor = normalization.InputsAndResiduals(predictor, diffs_stddev_by_level = diffs_stddev_by_level, mean_by_level = mean_by_level, stddev_by_level = stddev_by_level)
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing = True)
    return predictor

# Define the run_forward function
@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):
    predictor = construct_wrapped_graphcast(model_config, task_config)
    return predictor(inputs, targets_template = targets_template, forcings = forcings)

# Define helper functions
def with_configs(fn):
    return functools.partial(fn, model_config = model_config, task_config = task_config)

def with_params(fn):
    return functools.partial(fn, params = params, state = state)

def drop_state(fn):
    return lambda **kw: fn(**kw)[0]

# JIT compile the run_forward function
run_forward_jitted = drop_state(with_params(jax.jit(with_configs(run_forward.apply))))

# Predictor class
class Predictor:
    @classmethod
    def predict(cls, inputs, targets, forcings) -> xr.Dataset:
        predictions = rollout.chunked_prediction(run_forward_jitted, rng = jax.random.PRNGKey(0), inputs = inputs, targets_template = targets, forcings = forcings)
        return predictions

# Helper functions
def toDatetime(dt) -> datetime.datetime:
    if isinstance(dt, datetime.date) and isinstance(dt, datetime.datetime):
        return dt
    elif isinstance(dt, datetime.date) and not isinstance(dt, datetime.datetime):
        return datetime.datetime.combine(dt, datetime.datetime.min.time())
    elif isinstance(dt, str):
        if 'T' in dt:
            return isodate.parse_datetime(dt)
        else:
            return datetime.datetime.combine(isodate.parse_date(dt), datetime.datetime.min.time())

def nans(*args) -> list:
    return np.full((args), np.nan)

def deltaTime(dt, **delta) -> datetime.datetime:
    return dt + datetime.timedelta(**delta)

def addTimezone(dt, tz = pytz.UTC) -> datetime.datetime:
    print("Adding time zone.\n")
    dt = toDatetime(dt)
    if dt.tzinfo is None:
        return pytz.UTC.localize(dt).astimezone(tz)
    else:
        return dt.astimezone(tz)

# Load ERA5 data for a single year and month
def getSingleAndPressureValues(year, month, day):
    """
    Loads single-level and pressure-level data for the specified year and month.

    Args:
        year: The year for which to load the data.
        month: The month for which to load the data.
        data_bucket_name: The name of the GCS bucket containing the data.

    Returns:
        A tuple containing two pandas DataFrames:
            - singlelevel: DataFrame with single-level data.
            - pressurelevel: DataFrame with pressure-level data.
    """

    # Construct the full paths to the data files
    single_level_path = f"{data_bucket_name}/single-level-{year}-{month:02d}.nc"
    pressure_level_path = f"{data_bucket_name}/pressure-level-{year}-{month:02d}.nc"

    # Access the data files using the Storage client
    client = storage.Client()  # Create a Storage client
    bucket = client.get_bucket(gcs_bucket_name)  # Get the bucket

    # Load single-level data
    print("loading file: gs://{}/{}\n".format(gcs_bucket_name, single_level_path))
    blob = bucket.blob(single_level_path)

    # Load single-level data using netCDF4, filtering by day
    blob = bucket.blob(single_level_path)
    with blob.open('rb') as f:
        data = f.read()
        nc = netCDF4.Dataset('in-memory.nc', 'r', memory=data)
        ds = xr.open_dataset(xr.backends.NetCDF4DataStore(nc))
        singlelevel = ds.sel(
            time=str(year)+f'-{month:02d}-{day:02d}').to_dataframe()  # Filter by day
        
    print("Renaming single level columns using values from list\n")
    # Drop the 'number' and 'expver' columns
    singlelevel = singlelevel.drop(columns=['number', 'expver'])
    # DEBUG checking column names
    print("Single-level columns:{}\n".format(singlelevel.columns))
    singlelevel = singlelevel.rename(columns={col: singlelevelfields[ind] for ind, col in enumerate(singlelevel.columns.values.tolist())})
    print("Renaming geopotential column to geopotential_at_surface\n")
    singlelevel = singlelevel.rename(columns={'geopotential': 'geopotential_at_surface'})

    # Calculating the sum of the last 6 hours of rainfall.
    print("Calculating the sum of the last 6 hours of rainfall\n")
    singlelevel = singlelevel.sort_index()
    singlelevel['total_precipitation_6hr'] = singlelevel.groupby(level=[0, 1])['total_precipitation'].rolling(
        window=6, min_periods=1).sum().reset_index(level=[0, 1], drop=True)
    singlelevel.pop('total_precipitation')

    # Load pressure-level data
    print("loading file: {}\n".format(pressure_level_path))

    # Load pressure-level data using netCDF4, filtering by day
    blob = bucket.blob(pressure_level_path)
    with blob.open('rb') as f:
        data = f.read()
        nc = netCDF4.Dataset('in-memory.nc', 'r', memory=data)
        ds = xr.open_dataset(xr.backends.NetCDF4DataStore(nc))
        pressurelevel = ds.sel(
            time=str(year)+f'-{month:02d}-{day:02d}').to_dataframe()  # Filter by day
    
    # Drop the 'number' and 'expver' columns
    pressurelevel = pressurelevel.drop(columns=['number', 'expver'])
    # DEBUG checking column names
    print("Pressure-level columns:{}\n".format(pressurelevel.columns))

    print(f"Renaming pressure level columns using values from list\n")
    pressurelevel = pressurelevel.rename(columns={col: pressurelevelfields[ind] for ind, col in enumerate(pressurelevel.columns.values.tolist())})

    print("Finished processing single level and pressure level data\n")
    return singlelevel, pressurelevel

# Add sin and cos of the year progress
def addYearProgress(secs, data):
    print("Adding yearly progress.\n")
    progress = du.get_year_progress(secs)
    data['year_progress_sin'] = np.sin(2 * pi * progress)
    data['year_progress_cos'] = np.cos(2 * pi * progress)
    return data

# Add sin and cos of the day progress
def addDayProgress(secs, lon:str, data:pd.DataFrame):
    print("Adding progress for the day\n")
    lons = data.index.get_level_values(lon).unique()
    progress:np.ndarray = du.get_day_progress(secs, np.array(lons))
    prxlon = {lon:prog for lon, prog in list(zip(list(lons), progress.tolist()))}
    data['day_progress_sin'] = data.index.get_level_values(lon).map(lambda x: math.sin(2 * pi * prxlon[x]))
    data['day_progress_cos'] = data.index.get_level_values(lon).map(lambda x: math.cos(2 * pi * prxlon[x]))
    return data

# Integrate progress data
def integrateProgress(data:pd.DataFrame):
    print("Integrating progress\n")
    for dt in data.index.get_level_values('time').unique():
        seconds_since_epoch = toDatetime(dt).timestamp()
        data = addYearProgress(seconds_since_epoch, data)
        data = addDayProgress(seconds_since_epoch, 'longitude' if 'longitude' in data.index.names else 'lon', data)
    return data

# Calculate solar radiation
def getSolarRadiation(longitude, latitude, dt):
    print("Getting solar radiation data\n")
    altitude_degrees = get_altitude(latitude, longitude, addTimezone(dt))
    solar_radiation = get_radiation_direct(dt, altitude_degrees) if altitude_degrees > 0 else 0
    return solar_radiation * watts_to_joules

# Integrate solar radiation data
def integrateSolarRadiation(data:pd.DataFrame):
    print("Integrating solar radiation\n")
    dates = list(data.index.get_level_values('time').unique())
    coords = [[lat, lon] for lat in lat_range for lon in lon_range]
    values = []
    for dt in dates:
        values.extend(list(map(lambda coord:{'time': dt, 'lon': coord[1], 'lat': coord[0], 'toa_incident_solar_radiation': getSolarRadiation(coord[1], coord[0], dt)}, coords)))
    values = pd.DataFrame(values).set_index(keys = ['lat', 'lon', 'time'])
    return pd.merge(data, values, left_index = True, right_index = True, how = 'inner')

# Modify coordinates in xarray dataset
def modifyCoordinates(data:xr.Dataset):
    print("Modifying grid coordinates.\n")
    for var in list(data.data_vars):
        varArray:xr.DataArray = data[var]
        nonIndices = list(set(list(varArray.coords)).difference(set(AssignCoordinates.coordinates[var])))
        data[var] = varArray.isel(**{coord: 0 for coord in nonIndices})
    data = data.drop_vars('batch')
    return data

# Convert pandas dataframe to xarray dataset
def makeXarray(data:pd.DataFrame) -> xr.Dataset:
    print("Creating XArray.\n")
    data = data.to_xarray()
    data = modifyCoordinates(data)
    return data

# Format data for prediction
def formatData(data:pd.DataFrame) -> pd.DataFrame:
    print("Formatting data.\n")
    data = data.rename_axis(index = {'latitude': 'lat', 'longitude': 'lon'})
    if 'batch' not in data.index.names:
        data['batch'] = 0
        data = data.set_index('batch', append = True)
    return data

# Generate target data for prediction
def getTargets(dt, data:pd.DataFrame):
    print("Getting target data.\n")
    lat, lon, levels, batch = sorted(data.index.get_level_values('lat').unique().tolist()), sorted(data.index.get_level_values('lon').unique().tolist()), sorted(data.index.get_level_values('level').unique().tolist()), data.index.get_level_values('batch').unique().tolist()
    time = [deltaTime(dt, hours = days * gap) for days in range(predictions_steps)]
    target = xr.Dataset({field: (['lat', 'lon', 'level', 'time'], nans(len(lat), len(lon), len(levels), len(time))) for field in predictionFields}, coords = {'lat': lat, 'lon': lon, 'level': levels, 'time': time, 'batch': batch})
    return target.to_dataframe()

# Generate forcing data for prediction
def getForcings(data:pd.DataFrame):
    print("Getting forcings data.\n")
    forcingdf = data.reset_index(level = 'level', drop = True).drop(labels = predictionFields, axis = 1)
    forcingdf = pd.DataFrame(index = forcingdf.index.drop_duplicates(keep = 'first'))
    forcingdf = integrateProgress(forcingdf)
    forcingdf = integrateSolarRadiation(forcingdf)
    return forcingdf

def generate_forecast_batch(init_date: datetime.datetime, forecast_steps: int) -> List[Dict]:
    """
    Generates batched weather forecasts for multiple locations and a specified number of 6-hour horizons.

    Args:
        init_date: The initial date for the forecast.
        forecast_steps: The number of 6-hour forecast steps to generate.

    Returns:
        A list of dictionaries, where each dictionary represents a forecast record
        for a specific location, time, and prediction variables.
    """
    # data_bucket_name = os.environ.get('GRAPHCAST_DATA_BUCKET', 'gs://elet-dm-graphcast/dataset')

    # Determine the month for file retrieval
    day = init_date.day
    month = init_date.month
    year = init_date.year

    forecast_results = []

    for location_name, (lat, lon, station_id) in location_options.items():
        values: Dict[str, xr.Dataset] = {}

        if year in range(2022, 2023):     # <-------- Year validation only for testing
            print("Getting single level and pressure level values\n")
            single, pressure = getSingleAndPressureValues(year, month, day)
            

            print("Merging pressure level and single level data\n")
            print("Merging pressure level and single level data\n")
            values['inputs'] = pd.merge(pressure, single, left_index=True, right_index=True, how='inner')
            print("Formatting lat lon data\n")
            values['inputs'] = values['inputs'].xs((lat, lon), level=('lat', 'lon'))

            # Roll out the forecast for the specified number of steps
            print("Calculating forecast steps\n")
            current_time = datetime.datetime(year, month, init_date.day, 6, 0)  # Start at 6:00 AM
            end_time = current_time + datetime.timedelta(hours=6 * forecast_steps)  # Calculate end time based on steps

            print("Rolling out forecast steps\n")
            while current_time < end_time:
                try:
                    print("Getting forecast predictions at step.\n")
                    values['inputs'] = values['inputs'].loc[pd.Timestamp(current_time.year, current_time.month, current_time.day, current_time.hour, current_time.minute)]
                    values['targets'] = getTargets(first_prediction, values['inputs'])
                    values['forcings'] = getForcings(values['targets'])
                    values = {value: makeXarray(values[value]) for value in values}
                    print("making predictions for step.\n")
                    predictions = Predictor.predict(values['inputs'], values['targets'], values['forcings'])

                    print("Preparing forecast step info\n")
                    forecast_step = {
                        'init_time': init_date.isoformat(),  # ISO format for init_date
                        'station_id': station_id,
                        'name': location_name,
                        'hours': int((current_time - datetime.datetime(year, month, init_date.day)).total_seconds() / 3600),  # Hours since the beginning of the day
                    }

                    for field in predictionFields:
                        forecast_step[field] = predictions[field].sel(time=pd.Timestamp(current_time), method='nearest').values.tolist()

                    print("Adding forecast step prediction to results.\n")
                    forecast_results.append(forecast_step)

                except KeyError:
                    print(f"Warning: No forecast data found for time {current_time}. Skipping.\n")

                print("Moving to next forecast step.\n")
                current_time += datetime.timedelta(hours=6)  # Increment by 6 hours

        else:
            raise ValueError("Invalid year. Please provide a year between 2022 and 2024.\n")

    return forecast_results

def write_to_bigquery(forecast_results: List[Dict]):
    """
    Writes forecast results to BigQuery table.

    Args:
        forecast_results: A list of dictionaries containing forecast data.
    """
    # Construct the BigQuery client
    client = bigquery.Client()
    table_id = 'your-project-id.elet_meteorologia_datos_bq.elet_meteorologia_datos_predictions'  # Replace with your actual table ID

    # Convert forecast_results to a list of rows compatible with BigQuery schema
    rows_to_insert = []
    for result in forecast_results:
        row = {
            'init_time': result['init_time'],
            'station_id': result['station_id'],
            'name': result['name'],
            'hours': result['hours'],
            # TODO: map other fields from forecast_results to BigQuery columns ...
        }
        rows_to_insert.append(row)

    # Insert rows into BigQuery table
    errors = client.insert_rows_json(table_id, rows_to_insert)  # API request
    if errors == []:
        print("New rows have been added.\n")
    else:
        print(f"Encountered errors while inserting rows: {errors}\n")

def main(init_date_str, forecast_steps):  # Accept parameters
    try:
        print("Starting main method.\n")
        init_date = datetime.datetime.strptime(init_date_str, '%Y-%m-%d')
        forecast_results = generate_forecast_batch(init_date, forecast_steps)

        # Write the results to BigQuery
        print("Writing forecast results to BigQuery\n")
        write_to_bigquery(forecast_results)

        print('Batch forecast generated and written to BigQuery\n')

    except Exception as e:
        print(f'Error: {str(e)}\n')

if __name__ == '__main__':
    init_date_str =  '2022-01-01' #sys.argv[2]  # Get init_date from command-line arguments
    print('Forecast Init time: {}\n'.format(init_date_str))
    forecast_steps = 10 #int(sys.argv[3])  # Get forecast_steps from command-line arguments
    print("Forecast steps: {}\n".format(forecast_steps))
    main(init_date_str, forecast_steps)  # Pass arguments to main function