# (C) Heikki Kupiainen 2023    

# MSArray is a type safe wrapper & initializer for ndarray 

from typing import List
from pydantic import BaseModel, StrictStr
import numpy as np
from numpy import ndarray

class MSArray(BaseModel):
  array: ndarray

  class Config:
    arbitrary_types_allowed = True
        
  @staticmethod
  def new(item_count: int, axis_count: int, elements_per_axis_count: int):
    array = np.arange(item_count).reshape(axis_count, elements_per_axis_count)
    return MSArray(**{
      'array': array
    })
      
def test_msarray():
  msarray = MSArray.new(15,3,5)
  array = msarray.array
  assert(type(array)) == ndarray