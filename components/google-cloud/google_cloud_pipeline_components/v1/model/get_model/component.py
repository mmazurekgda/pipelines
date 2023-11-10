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

from google_cloud_pipeline_components import _image
from google_cloud_pipeline_components import _placeholders
from google_cloud_pipeline_components.types.artifact_types import VertexModel
from kfp import dsl


@dsl.container_component
def model_get(
    model: dsl.Output[VertexModel],
    model_name: str,
    project: str = _placeholders.PROJECT_ID_PLACEHOLDER,
    location: str = 'us-central1',
):
  # fmt: off
  """Gets a model artifact based on the model name of an existing Vertex model.

  Args:
    project: Project from which to get the Model. Defaults to the project in which the PipelineJob is run.
    model_name: Vertex model resource name in the format of
    `projects/{project}/locations/{location}/models/{model}`
    or
    `projects/{project}/locations/{location}/models/{model}@{model_version_id or model_version_alias}`.
    If no version ID or alias is specified, the "default" version will be returned.
    location: Location from which to get the Model. Defaults to `us-central1`.

  Returns:
      model: Artifact of the Vertex Model.
  """
  # fmt: on
  return dsl.ContainerSpec(
      image=_image.GCPC_IMAGE_TAG,
      command=[
          'python3',
          '-u',
          '-m',
          'google_cloud_pipeline_components.container.v1.model.get_model.launcher',
      ],
      args=[
          '--project',
          project,
          '--location',
          location,
          '--model_name',
          model_name,
          '--executor_input',
          '{{$}}',
      ],
  )
