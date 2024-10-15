"""Querying EE STAC JSON files."""

from concurrent import futures
import datetime
import json
import logging
import pandas as pd
from typing import Iterable, Optional, Sequence
from typing_extensions import Self
import re
from contextlib import redirect_stdout
import geemap
import io

from google.cloud import storage
from google.cloud.storage import blob
import iso8601
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from stac import bboxes
from stac import stac_lib as stac


def matches_interval(
    collection_interval: tuple[datetime.datetime, datetime.datetime],
    query_interval: tuple[datetime.datetime, datetime.datetime],
):
  """Checks if the collection's datetime interval matches the query datetime.

  Args:
    collection_interval: Temporal interval of the collection.
    query_interval: a tuple with the query interval start and end

  Returns:
    True if the datetime interval matches
  """
  start_query, end_query = query_interval
  start_collection, end_collection = collection_interval
  if end_collection is None:
    # End date should always be set in STAC JSON files, but just in case...
    end_collection = datetime.datetime.now(tz=datetime.UTC)
  return end_query > start_collection and start_query <= end_collection


def matches_datetime(
    collection_interval: tuple[datetime.datetime, Optional[datetime.datetime]],
    query_datetime: datetime.datetime,
):
  """Checks if the collection's datetime interval matches the query datetime.

  Args:
    collection_interval: Temporal interval of the collection.
    query_datetime: a datetime coming from a query

  Returns:
    True if the datetime interval matches
  """
  if collection_interval[1] is None:
    # End date should always be set in STAC JSON files, but just in case...
    end_date = datetime.datetime.now(tz=datetime.UTC)
  else:
    end_date = collection_interval[1]
  return collection_interval[0] <= query_datetime <= end_date


class CollectionList(Sequence[stac.Collection]):
  """List of stac.Collections; can be filtered to return a smaller sublist."""

  _collections = Sequence[stac.Collection]

  def __init__(self, collections: Sequence[stac.Collection]):
    self._collections = tuple(collections)

  # Define immutable list interface for convenience, though one could
  # argue this should be more like a set.
  def __iter__(self):
    return iter(self._collections)

  def __getitem__(self, index):
    return self._collections[index]

  def __len__(self):
    return len(self._collections)

  def __eq__(self, other: object) -> bool:
    if isinstance(other, CollectionList):
      return self._collections == other._collections
    return False

  def __hash__(self) -> int:
    return hash(self._collections)

  def filter_by_ids(self, ids: Iterable[str]) -> Self:
    """Returns a sublist with only the collections matching the given ids."""
    return self.__class__(
        [c for c in self._collections if c.public_id() in ids]
    )

  def filter_by_datetime(
      self,
      query_datetime: datetime.datetime,
  ) -> Self:
    """Returns a sublist with the time interval matching the given time."""
    result = []
    for collection in self._collections:
      for datetime_interval in collection.datetime_interval_list():
        if matches_datetime(datetime_interval, query_datetime):
          result.append(collection)
          break
    return self.__class__(result)

  def filter_by_interval(
      self,
      query_interval: tuple[datetime.datetime, datetime.datetime],
  ) -> Self:
    """Returns a sublist with the time interval matching the given interval."""
    result = []
    for collection in self._collections:
      for datetime_interval in collection.datetime_interval_list():
        if matches_interval(datetime_interval, query_interval):
          result.append(collection)
          break
    return self.__class__(result)

  def filter_by_bounding_box(
      self, query_bbox: bboxes.BBox) -> Self:
    """Returns a sublist with the bbox matching the given bbox."""
    result = []
    for collection in self._collections:
      for collection_bbox in collection.bbox_list():
        if collection_bbox.intersects(query_bbox):
          result.append(collection)
          break
    return self.__class__(result)


  def sort_by_spatial_resolution(self, reverse=False):
        """
        Sorts the collections based on their spatial resolution.
        Collections with spatial_resolution_m() == -1 are pushed to the end.

        Args:
            reverse (bool): If True, sort in descending order (highest resolution first).
                            If False (default), sort in ascending order (lowest resolution first).

        Returns:
            CollectionList: A new CollectionList instance with sorted collections.
        """
        def sort_key(collection):
            resolution = collection.spatial_resolution_m()
            if resolution == -1:
                return float('inf') if not reverse else float('-inf')
            return resolution

        sorted_collections = sorted(
            self._collections,
            key=sort_key,
            reverse=reverse
        )
        return self.__class__(sorted_collections)


  def limit(self, n: int):
    """
    Returns a new CollectionList containing the first n entries.

    Args:
        n (int): The number of entries to include in the new list.

    Returns:
        CollectionList: A new CollectionList instance with at most n collections.
    """
    return self.__class__(self._collections[:n])


  def to_df(self):
    """Converts a collection list to a dataframe with a select set of fields."""

    rows = []
    for col in self._collections:
      # Remove text in parens in dataset name.
      short_title = re.sub(r'\([^)]*\)', '', col.get('title')).strip()

      row = {
          'id': col.public_id(),
          'name': short_title,
          'temp_res': col.temporal_resolution_str(),
          'spatial_res_m': col.spatial_resolution_m(),
          'earliest': col.start().strftime("%Y-%m-%d"),
          'latest': col.end().strftime("%Y-%m-%d"),
          'url': col.catalog_url()
      }
      rows.append(row)
    return pd.DataFrame(rows)


class Catalog:
    """Class containing all collections in the EE STAC catalog."""

    collections: CollectionList

    def __init__(self, storage_client: storage.Client):
        self.collections = CollectionList(self._load_collections(storage_client))

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _read_file(self, file_blob: blob.Blob) -> stac.Collection:
        """Reads the contents of a file from the specified bucket with retry logic."""
        file_contents = file_blob.download_as_string().decode()
        return stac.Collection(json.loads(file_contents))

    def _read_files(self, file_blobs: list[blob.Blob]) -> list[stac.Collection]:
        """Processes files in parallel with a progress bar."""
        result = []
        with futures.ThreadPoolExecutor(max_workers=10) as executor:
            file_futures = [
                executor.submit(self._read_file, file_blob)
                for file_blob in file_blobs
            ]
            for future in tqdm(futures.as_completed(file_futures), total=len(file_futures), desc="Reading files"):
                result.append(future.result())
        return result

    def _load_collections(self, storage_client: storage.Client) -> Sequence[stac.Collection]:
        """Loads all EE STAC JSON files from GCS, with datetimes as objects."""
        bucket = storage_client.get_bucket('earthengine-stac')
        files = [
            x
            for x in bucket.list_blobs(prefix='catalog/')
            if x.name.endswith('.json')
            and not x.name.endswith('/catalog.json')
            and not x.name.endswith('/units.json')
        ]
        logging.warning('Found %d files, loading...', len(files))
        
        stac_objects = self._read_files(files[:50])

        res = []
        for c in stac_objects:
            if c.is_deprecated():
                continue
            res.append(c)
        logging.warning(
            'Loaded %d collections (skipping deprecated ones)', len(res)
        )
        # Returning a tuple for immutability.
        return tuple(res)
    
class SampleCode():
  """Class containing sample code snippets for each dataset in the public datat catalog."""

  def __init__(self, storage_client: storage.Client):
      self.code_samples_dict = self._load_all_code_samples(storage_client)

  def js_code(self, collection_id: str):
     normalized_id = collection_id.replace('/', '_')
     return self.code_samples_dict.get(normalized_id)['js_code']
  
  def python_code(self, collection_id: str):
     normalized_id = collection_id.replace('/', '_')
     return self.code_samples_dict.get(normalized_id)['py_code']

      
  def _load_all_code_samples(self, storage_client: storage.Client):
    """Loads js + py example scripts from GCS into dict keyed by dataset ID."""

    # Get json file from GCS bucket
    # 'gs://earthengine-catalog/catalog/example_scripts.json'
    bucket = storage_client.get_bucket('earthengine-catalog')
    blob= bucket.blob('catalog/example_scripts.json')
    file_contents = blob.download_as_string().decode()
    data = json.loads(file_contents)

    # Flatten json to get a map from ID (using '_' rather than '/') to code
    # sample.
    all_datasets_by_provider = data[0]['contents']
    code_samples_dict = {}
    for provider in all_datasets_by_provider:
      for dataset in provider['contents']:
        js_code = dataset['code']
        py_code = self._make_python_code_sample(js_code)

        code_samples_dict[dataset['name']] = {
            'js_code': js_code, 'py_code': py_code}

    return code_samples_dict
  
  def _make_python_code_sample(self, js_code: str) -> str:
    """Converts EE JS code into python."""

    # geemap appears to have some stray print statements.
    _ = io.StringIO()
    with redirect_stdout(_):
      code_list = geemap.js_snippet_to_py(js_code,
                                      add_new_cell=False,
                                      import_ee=False,
                                      import_geemap=False,
                                      show_map=False)
    return ''.join(code_list)



def main():
  storage_client = storage.Client()
  catalog = Catalog(storage_client)
  collections = catalog.collections

  # Example usage
  bbox = bboxes.BBox.from_list([-120, 30, -100, 40])
  filtered_by_bbox = collections.filter_by_bounding_box(bbox)

  print(f'\nCollections filtered by bounding box {bbox}:')
  print(len(filtered_by_bbox))

  print(collections.limit(5).sort_by_spatial_resolution().to_df())

  sample_code = SampleCode(storage_client)
  print(sample_code.python_code('LANDSAT/LC09/C02/T1_L2'))




if __name__ == '__main__':
  main()
