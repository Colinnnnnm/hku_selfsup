# coding: utf-8

"""
    Lightly API

    Lightly.ai enables you to do self-supervised learning in an easy and intuitive way. The lightly.ai OpenAPI spec defines how one can interact with our REST API to unleash the full potential of lightly.ai  # noqa: E501

    The version of the OpenAPI document: 1.0.0
    Contact: support@lightly.ai
    Generated by OpenAPI Generator (https://openapi-generator.tech)

    Do not edit the class manually.
"""


from __future__ import annotations
import pprint
import re  # noqa: F401
import json


from typing import Any, Dict, Optional
from pydantic import Extra,  BaseModel, Field
from lightly.openapi_generated.swagger_client.models.docker_worker_type import DockerWorkerType
from lightly.openapi_generated.swagger_client.models.selection_config import SelectionConfig

class DockerWorkerConfig(BaseModel):
    """
    DockerWorkerConfig
    """
    worker_type: DockerWorkerType = Field(..., alias="workerType")
    docker: Optional[Dict[str, Any]] = Field(None, description="docker run configurations, keys should match the structure of https://github.com/lightly-ai/lightly-core/blob/develop/onprem-docker/lightly_worker/src/lightly_worker/resources/docker/docker.yaml ")
    lightly: Optional[Dict[str, Any]] = Field(None, description="lightly configurations which are passed to a docker run, keys should match structure of https://github.com/lightly-ai/lightly/blob/master/lightly/cli/config/config.yaml ")
    selection: Optional[SelectionConfig] = None
    __properties = ["workerType", "docker", "lightly", "selection"]

    class Config:
        """Pydantic configuration"""
        allow_population_by_field_name = True
        validate_assignment = True
        use_enum_values = True
        extra = Extra.forbid

    def to_str(self, by_alias: bool = False) -> str:
        """Returns the string representation of the model"""
        return pprint.pformat(self.dict(by_alias=by_alias))

    def to_json(self, by_alias: bool = False) -> str:
        """Returns the JSON representation of the model"""
        return json.dumps(self.to_dict(by_alias=by_alias))

    @classmethod
    def from_json(cls, json_str: str) -> DockerWorkerConfig:
        """Create an instance of DockerWorkerConfig from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of selection
        if self.selection:
            _dict['selection' if by_alias else 'selection'] = self.selection.to_dict(by_alias=by_alias)
        # set to None if docker (nullable) is None
        # and __fields_set__ contains the field
        if self.docker is None and "docker" in self.__fields_set__:
            _dict['docker' if by_alias else 'docker'] = None

        # set to None if lightly (nullable) is None
        # and __fields_set__ contains the field
        if self.lightly is None and "lightly" in self.__fields_set__:
            _dict['lightly' if by_alias else 'lightly'] = None

        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> DockerWorkerConfig:
        """Create an instance of DockerWorkerConfig from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return DockerWorkerConfig.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in DockerWorkerConfig) in the input: " + str(obj))

        _obj = DockerWorkerConfig.parse_obj({
            "worker_type": obj.get("workerType"),
            "docker": obj.get("docker"),
            "lightly": obj.get("lightly"),
            "selection": SelectionConfig.from_dict(obj.get("selection")) if obj.get("selection") is not None else None
        })
        return _obj

