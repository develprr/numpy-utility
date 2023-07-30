# (C) Heikki Kupiainen 2023    

# MSArray is a type safe wrapper & initializer for ndarray 

from typing import List
from pydantic import BaseModel, StrictStr, ConfigDict, ValidationError, validate_call

import numpy as np
from numpy import ndarray

class MSArray(BaseModel):
  model_config = ConfigDict(arbitrary_types_allowed=True)
  
  array: ndarray
        
  @staticmethod
  def new_shaped(item_count: int, axis_count: int, elements_per_axis_count: int):
    return MSArray(**{
      'array': np.arange(item_count).reshape(axis_count, elements_per_axis_count)
    })
  
  @staticmethod
  def new_from_list(list: list):
    return MSArray(**{
      'array': np.array(list)
    })
  
  @validate_call
  def accept_int(self, number: int):
    return int

def test_new_shaped():
  array = MSArray.new_shaped(15,3,5).array
  assert(type(array)) == ndarray 
  assert(array.shape) == (3,5)
  assert(array.itemsize) == 8
  assert(array.dtype.name) == 'int64'
  assert(array.ndim) == 2

def test_new_from_list():
  array = MSArray.new_from_list([1,2,3]).array
  assert(array.dtype.name)  == 'int64'
  array = MSArray.new_from_list([1,2,3.]).array
  assert(array.dtype.name)  == 'float64'

def test_accept_int():
  msarray = MSArray.new_from_list([1,2,3])
  msarray.accept_int(3)
  try:
    msarray.accept_int(3.1)
  except:
    return
  raise Exception('Float value was incorrectly accepted')