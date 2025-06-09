from pathlib import Path
import pandas as pd

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def load_data(*args, **kwargs):
    """
    Load yellow taxi data from March 2023 parquet file.

    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    path_data = Path("/home/src/data")  # Path inside container (from volume mount)
    file_path = path_data / "yellow_tripdata_2023-03.parquet"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Parquet file not found at: {file_path}")

    df = pd.read_parquet(file_path)

    return df


@test
def test_output(output, *args) -> None:
    """
    Simple test to ensure the data frame is loaded and not empty.
    """
    assert output is not None, 'The output is undefined'
    assert isinstance(output, pd.DataFrame), 'Output is not a DataFrame'
    assert not output.empty, 'The DataFrame is empty'
