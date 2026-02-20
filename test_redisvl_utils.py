
try:
    from redisvl.utils.utils import model_to_dict
    from pydantic import BaseModel
    print("Successfully imported redisvl.utils.utils")
    
    class TestModel(BaseModel):
        name: str
        age: int = None

    m = TestModel(name="test")
    d = model_to_dict(m)
    print(f"Serialized: {d}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
