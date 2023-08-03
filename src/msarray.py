# (C) Heikki Kupiainen 2023    

# MSArray is a type safe wrapper & initializer for ndarray 

from typing import List
from pydantic import BaseModel, StrictStr, ConfigDict, ValidationError, validate_call

import numpy as np
from numpy import ndarray
  
class MSArray(BaseModel):
  model_config = ConfigDict(arbitrary_types_allowed=True)
    
  ndarray: ndarray
        
  @staticmethod
  @validate_call
  def new_shaped(item_count: int, axis_count: int, elements_per_axis_count: int):
    return MSArray(**{
      'ndarray': np.arange(item_count).reshape(axis_count, elements_per_axis_count)
    })
  
  @staticmethod
  @validate_call
  def new_int_based_from_list(list: list[int]):
    return MSArray(**{
      'ndarray': np.array(list)
    })
  
  @staticmethod
  @validate_call(config=dict(arbitrary_types_allowed=True))
  def new_from_ndarray(array: ndarray):
    return MSArray(**{
      'ndarray': array
    })
    
  @staticmethod
  @validate_call
  def new_zero_dimensional(value: int):
    return MSArray(**{
      'ndarray': np.array(value)
    })  

  @staticmethod
  @validate_call
  def new_two_dimensional(list1: list, list2: list):
    return MSArray(**{
      'ndarray': np.array([list1, list2])
    })
   
  @staticmethod
  @validate_call
  def new_float_based_two_dimensional(list1: list[float], list2: list[float]):
    return MSArray(**{
      'ndarray': np.array([list1, list2])
    })
    
def test_new_shaped():
  array = MSArray.new_shaped(15,3,5).ndarray
  assert(type(array)) == ndarray 
  assert(array.shape) == (3,5)
  assert(array.itemsize) == 8
  assert(array.dtype.name) == 'int64'
  assert(array.ndim) == 2

def test_new_int_based_from_list__pass_when_list_of_integers_is_given():
  array = MSArray.new_int_based_from_list([1, 2, 3]).ndarray
  assert(array.dtype.name)  == 'int64'

def test_new_int_based_from_list__fail_when_not_all_values_are_integers():
  try:
    array = MSArray.new_int_based_from_list([1, 2, 3.1]).ndarray
  except:
    print("new_int_based_from_list: an exception was raised as expected!")
    return
  raise Exception('test fail: float value was incorrectly accepted in list argument')

def test_new_from_ndarray__pass_when_ndarray_is_given():
  array = MSArray.new_from_ndarray(np.array([1,2,3])).ndarray
  assert(array.dtype.name)  == 'int64'

def test_new_from_ndarray__fail_when_list_is_given():
  try:
    array = MSArray.new_from_ndarray([1,2,3]).ndarray
  except:
    print("new_from_ndarray: an exception was raised as expected!")
    return
  raise Exception('test fail: constructor did not raise exception on wrong input parameter type')

def test_new_zero_dimensional():
  array = MSArray.new_zero_dimensional(42).ndarray
  assert(array == 42)
  
def test_new_two_dimensional__pass_when_two_lists_are_given():
  array = MSArray.new_two_dimensional([1,2,3], [4,5,6])
  
def test_new_two_dimensional__pass_when_any_kinds_of_lists_are_given():
  array = MSArray.new_two_dimensional([1, 2, 3.8], [4, "viis", 6.3]).ndarray
  assert(array.dtype.name) == "str1024"
  
def test_new_two_dimensional__fail_when_other_than_two_lists_are_given():
  try: 
    array = MSArray.new_two_dimensional([1, 2, 3.8], 12).ndarray
  except:
    print("new_two_dimensional: an exception was raised as expected!")
    return  
  raise Exception("test fail: constructor did not raise exception when other than two lists were given")
    
def test_new_float_based_two_dimensional__pass_when_two_float_lists_are_given():
  array = MSArray.new_float_based_two_dimensional([1.1, 2, 3], [4, 5, 6.3]).ndarray
  assert(array.dtype.name) == "float64"

# Surprisingly, this must pass because an int list is also a float list although a float list is not an int list.
# Just like every dog is an animal but not every animal is a dog :) 
def test_new_float_based_two_dimensional__pass_when_two_int_lists_are_given():
  array = MSArray.new_float_based_two_dimensional([1, 2, 3], [4, 5, 6]).ndarray
  assert(array.dtype.name) == "float64"

def test_new_float_based_two_dimensional__fail_when_arbitrary_lists_are_given():
  try:
    array = MSArray.new_float_based_two_dimensional([1, 2, 3], ["neljä", 5, 6]).ndarray
  except:
    print("pass: exception was raised as expected")
    return
  raise Exception("fail: exception was not raised as expected")
  
def test_validation():
  try:
    array = MSArray.new_shaped(15.1,3,5).ndarray 
  except:
    return
  raise Exception('float value was incorrectly accepted')