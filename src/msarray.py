# (C) Heikki Kupiainen 2023    

# MSArray is a type safe wrapper & initializer for ndarray 

from typing import List, Optional
from pydantic import BaseModel, StrictStr, ConfigDict, ValidationError, validate_call

import numpy as np
from numpy import ndarray
  
class MSArray(BaseModel):
  model_config = ConfigDict(arbitrary_types_allowed=True)
    
  array: ndarray
        
  @staticmethod
  @validate_call
  def new_shaped(item_count: int, axis_count: int, elements_per_axis_count: int):
    return MSArray(**{
      'array': np.arange(item_count).reshape(axis_count, elements_per_axis_count)
    })
  
  @staticmethod
  @validate_call
  def new_from_int_list(list: list[int]):
    return MSArray(**{
      'array': np.array(list)
    })
  
  @staticmethod
  @validate_call(config=dict(arbitrary_types_allowed=True))
  def new_from_ndarray(array: ndarray):
    return MSArray(**{
      'array': array
    })
    
  @staticmethod
  @validate_call(config=dict(arbitrary_types_allowed=True))
  def new_zero_dimensional(value: int):
    return MSArray(**{
      'array': np.array(value)
    })  

def test_new_shaped():
  array = MSArray.new_shaped(15,3,5).array
  assert(type(array)) == ndarray 
  assert(array.shape) == (3,5)
  assert(array.itemsize) == 8
  assert(array.dtype.name) == 'int64'
  assert(array.ndim) == 2

def test_new_from_int_list__works_when_list_of_integers_is_given():
  array = MSArray.new_from_int_list([1, 2, 3]).array
  assert(array.dtype.name)  == 'int64'

def test_new_from_int_list__fails_when_not_all_values_are_integers():
  try:
    array = MSArray.new_from_int_list([1, 2, 3.1]).array
  except:
    print("new_from_int_list: an exception was raised as expected!")
    return
  raise Exception('test fail: float value was incorrectly accepted in list argument')

def test_new_from_ndarray__works_when_ndarray_is_given():
  array = MSArray.new_from_ndarray(np.array([1,2,3])).array
  assert(array.dtype.name)  == 'int64'

def test_new_from_ndarray__fails_when_list_is_given():
  try:
    array = MSArray.new_from_ndarray([1,2,3]).array
  except:
    print("new_from_ndarray: an exception was raised as expected!")
    return
  raise Exception('test fail: constructor did not raise expection on wrong input parameter type')

def test_new_zero_dimensional():
  array = MSArray.new_zero_dimensional(42).array
  assert(array == 42)
  
def test_validation():
  try:
    array = MSArray.new_shaped(15.1,3,5).array
  except:
    return
  raise Exception('float value was incorrectly accepted')