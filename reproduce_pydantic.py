
try:
    from pydantic.v1 import BaseModel, Field
    from typing import Optional, Any

    class BaseFieldAttributes(BaseModel):
        sortable: bool = Field(default=False)

    class BaseVectorFieldAttributes(BaseModel):
        dims: int
        initial_cap: int = Field(default=None)

    print("BaseVectorFieldAttributes defined successfully")
    print(t)
except Exception as e:
    print(f"Error: {e}")
