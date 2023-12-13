# Copyright 2023 The Kubeflow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Feature Store grounding pipeline."""

from google_cloud_pipeline_components._implementation.model_evaluation.chunking.component import chunking as ChunkingOp
from google_cloud_pipeline_components.types.artifact_types import VertexModel
from google_cloud_pipeline_components.v1 import bigquery
from google_cloud_pipeline_components.v1.batch_predict_job import ModelBatchPredictOp
import kfp


_PIPELINE_NAME = 'feature-store-grounding-pipeline'


@kfp.dsl.component
def _get_batch_predict_input_table(
    bigquery_dataset_uri: str,
) -> str:
  """Get the batch predict input BigQuery table.

  Args:
    bigquery_dataset_uri: The URI to a bigquery dataset with expected format
      `bg://project-id.bigquery-dataset-id`.

  Returns:
    URI to the batch predict input BigQuery table.
  """
  return f'{bigquery_dataset_uri}.batch_predict_input'


@kfp.dsl.component
def _get_batch_predict_output_table(
    bigquery_dataset_uri: str,
) -> str:
  """Get the batch predict output BigQuery table.

  Args:
    bigquery_dataset_uri: The URI to a bigquery dataset with expected format
      `bg://project-id.bigquery-dataset-id`.

  Returns:
    URI to the batch predict output BigQuery table.
  """
  return f'{bigquery_dataset_uri}.batch_predict_output'


@kfp.dsl.component
def _compose_bq_query_create_table(
    bigquery_dataset_uri: str,
) -> str:
  """Compose the BQ query for table creation.

  Args:
    bigquery_dataset_uri: The URI to a bigquery dataset with expected format
      `bg://project-id.bigquery-dataset-id`.

  Returns:
    The composed query.
  """
  if bigquery_dataset_uri.startswith('bq://'):
    bigquery_dataset_uri = bigquery_dataset_uri.replace('bq://', '')

  return f"""
CREATE TABLE {bigquery_dataset_uri}.batch_predict_input (
  vertex_generated_chunk_id STRING ,
  feature_timestamp TIMESTAMP,
  source_uri STRING,
  chunk_size STRING,
  overlap_size STRING,
  content STRING,
  embedding ARRAY<FLOAT64>,
  model_source STRING,
)
"""


@kfp.dsl.component
def _compose_bq_query_format_conversion(
    bigquery_dataset_uri: str,
) -> str:
  """Compose the BQ query for format conversion.

  Args:
    bigquery_dataset_uri: The URI to a bigquery dataset with expected format
      `bg://project-id.bigquery-dataset-id`.

  Returns:
    The composed query.
  """
  if bigquery_dataset_uri.startswith('bq://'):
    bigquery_dataset_uri = bigquery_dataset_uri.replace('bq://', '')

  inseration_query = (
      f'UPDATE `{bigquery_dataset_uri}.batch_predict_input` destTable'
      ' SET embedding=ARRAY( select cast (str_element as float64) from'
      " unnest(JSON_VALUE_ARRAY(prediction, '$.embeddings.values')) as"
      ' str_element)'
  )
  fetch_data_query = (
      'FROM (SELECT vertex_generated_chunk_id, prediction FROM'
      f' `{bigquery_dataset_uri}.batch_predict_output` cross join'
      ' unnest(JSON_EXTRACT_ARRAY(predictions)) as prediction) sourceTable'
      ' WHERE'
      ' destTable.vertex_generated_chunk_id=sourceTable.vertex_generated_chunk_id'
  )
  return f'{inseration_query} {fetch_data_query};'


@kfp.dsl.pipeline(name=_PIPELINE_NAME)
def feature_store_grounding_pipeline(
    project: str,
    location: str,
    input_text_gcs_dir: str,
    bigquery_dataset_uri: str,
    output_text_gcs_dir: str,
    output_error_file_path: str,
    model_name: str,
    generation_threshold_microseconds: str = '0',
    machine_type: str = 'e2-highmem-16',
    service_account: str = '',
    encryption_spec_key_name: str = '',
):
  """The Feature Store grounding pipeline.

  Args:
    project: Required. The GCP project that runs the pipeline components.
    location: Required. The GCP region that runs the pipeline components.
    input_text_gcs_dir: the GCS directory containing the files to chunk.
    bigquery_dataset_uri: The URI to a bigquery dataset with expected format
      `bg://project-id.bigquery-dataset-id`.
    output_text_gcs_dir: The GCS folder to hold intermediate data for chunking.
    output_error_file_path: The path to the file containing chunking error.
    model_name: The path for model to generate embeddings, example,
      'publishers/google/models/textembedding-gecko@latest'
    generation_threshold_microseconds: only files created on/after this
      generation threshold will be processed, in microseconds.
    machine_type: The machine type to run chunking component in the pipeline.
    service_account: Service account to run the pipeline.
    encryption_spec_key_name: Customer-managed encryption key options for the
      CustomJob. If this is set, then all resources created by the CustomJob
      will be encrypted with the provided encryption key.
  """

  get_vertex_model_task = kfp.dsl.importer(
      artifact_uri=(
          f'https://{location}-aiplatform.googleapis.com/v1/{model_name}'
      ),
      artifact_class=VertexModel,
      metadata={'resourceName': model_name},
  )
  get_vertex_model_task.set_display_name('get-vertex-model')

  get_batch_predict_input_table_task = _get_batch_predict_input_table(
      bigquery_dataset_uri=bigquery_dataset_uri
  )
  get_batch_predict_output_table_task = _get_batch_predict_output_table(
      bigquery_dataset_uri=bigquery_dataset_uri
  )

  compose_bq_query_create_table_task = _compose_bq_query_create_table(
      bigquery_dataset_uri=bigquery_dataset_uri
  )
  bigquery_create_table_task = bigquery.BigqueryQueryJobOp(
      project=project,
      location=location,
      query=compose_bq_query_create_table_task.output,
  ).set_display_name('create-batch-table')

  chunking_task = ChunkingOp(
      project=project,
      location=location,
      input_text_gcs_dir=input_text_gcs_dir,
      output_bq_destination=get_batch_predict_input_table_task.output,
      output_text_gcs_dir=output_text_gcs_dir,
      output_error_file_path=output_error_file_path,
      generation_threshold_microseconds=generation_threshold_microseconds,
      machine_type=machine_type,
      service_account=service_account,
      encryption_spec_key_name=encryption_spec_key_name,
  )

  chunking_task.after(bigquery_create_table_task)

  batch_predict_task = ModelBatchPredictOp(
      job_display_name='feature-store-grounding-batch-predict-{{$.pipeline_job_uuid}}-{{$.pipeline_task_uuid}}',
      project=project,
      location=location,
      model=get_vertex_model_task.outputs['artifact'],
      bigquery_source_input_uri=get_batch_predict_input_table_task.output,
      bigquery_destination_output_uri=get_batch_predict_output_table_task.output,
      service_account=service_account,
      encryption_spec_key_name=encryption_spec_key_name,
  )
  batch_predict_task.after(chunking_task)

  compose_bq_query_format_conversion_task = _compose_bq_query_format_conversion(
      bigquery_dataset_uri=bigquery_dataset_uri,
  )

  bigquery_format_conversion_task = bigquery.BigqueryQueryJobOp(
      project=project,
      location=location,
      query=compose_bq_query_format_conversion_task.output,
  ).set_display_name('bigquery-format-conversion')
  bigquery_format_conversion_task.after(batch_predict_task)
