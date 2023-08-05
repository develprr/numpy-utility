# (C) Heikki Kupiainen 2023    

# D2FloatArray is a type safe wrapper & initializer for a two dimensional ndarray containing float values

from typing import List
from pydantic import BaseModel, StrictStr, ConfigDict, ValidationError, validate_call
from d2floatarray import D2FloatArray
import numpy as np
from numpy import ndarray
  
class D3FloatArray(BaseModel):
  model_config = ConfigDict(arbitrary_types_allowed=True)
    
  ndarray: ndarray

  @staticmethod
  @validate_call(config=dict(arbitrary_types_allowed=True))
  def new(arrays: list[D2FloatArray]):
    return D3FloatArray(**{
      'ndarray': np.array(list(map(lambda x: x.ndarray, arrays)))
    })

def test_new__pass_when_created_from_a_list_of_two_dimensional_arrays():
  d2_array_1 = D2FloatArray.new([[1,2,3], [4,5,6]])
  d2_array_2 = D2FloatArray.new([[7,8,9], [10,11,12]])
  array = D3FloatArray.new([d2_array_1, d2_array_2]).ndarray
  assert(array.dtype.name) == "float64"

def test_new__fail_when_arbitrary_lists_are_given():
  try:
    array = D3FloatArray.new([[1, 2, 3], ["neljä", 5, 6]]).ndarray
  except:
    print("pass: exception was raised as expected")
    return
  raise Exception("fail: exception was not raised as expected")