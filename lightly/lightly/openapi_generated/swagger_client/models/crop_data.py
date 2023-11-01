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


from typing import Union
from pydantic import Extra,  BaseModel, Field, confloat, conint, constr, validator

class CropData(BaseModel):
    """
    CropData
    """
    parent_id: constr(strict=True) = Field(..., alias="parentId", description="MongoDB ObjectId")
    prediction_uuid_timestamp: conint(strict=True, ge=0) = Field(..., alias="predictionUUIDTimestamp", description="unix timestamp in milliseconds")
    prediction_index: conint(strict=True, ge=0) = Field(..., alias="predictionIndex", description="the index of this crop within all found prediction singletons of a sampleId (the parentId)")
    prediction_task_name: constr(strict=True, min_length=1) = Field(..., alias="predictionTaskName", description="A name which is safe to have as a file/folder name in a file system")
    prediction_task_category_id: conint(strict=True, ge=0) = Field(..., alias="predictionTaskCategoryId", description="The id of the category. Needs to be a positive integer but can be any integer (gaps are allowed, does not need to be sequential)")
    prediction_task_score: Union[confloat(le=1, ge=0, strict=True), conint(le=1, ge=0, strict=True)] = Field(..., alias="predictionTaskScore", description="the score for the prediction task which yielded this crop")
    __properties = ["parentId", "predictionUUIDTimestamp", "predictionIndex", "predictionTaskName", "predictionTaskCategoryId", "predictionTaskScore"]

    @validator('parent_id')
    def parent_id_validate_regular_expression(cls, value):
        """Validates the regular expression"""
        if not re.match(r"^[a-f0-9]{24}$", value):
            raise ValueError(r"must validate the regular expression /^[a-f0-9]{24}$/")
        return value

    @validator('prediction_task_name')
    def prediction_task_name_validate_regular_expression(cls, value):
        """Validates the regular expression"""
        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9 ._-]+$", value):
            raise ValueError(r"must validate the regular expression /^[a-zA-Z0-9][a-zA-Z0-9 ._-]+$/")
        return value

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
    def from_json(cls, json_str: str) -> CropData:
        """Create an instance of CropData from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> CropData:
        """Create an instance of CropData from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return CropData.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in CropData) in the input: " + str(obj))

        _obj = CropData.parse_obj({
            "parent_id": obj.get("parentId"),
            "prediction_uuid_timestamp": obj.get("predictionUUIDTimestamp"),
            "prediction_index": obj.get("predictionIndex"),
            "prediction_task_name": obj.get("predictionTaskName"),
            "prediction_task_category_id": obj.get("predictionTaskCategoryId"),
            "prediction_task_score": obj.get("predictionTaskScore")
        })
        return _obj
