
try:
    from redisvl.index.storage import BaseStorage, IndexType
    from pydantic import BaseModel
    print("Successfully imported redisvl.index.storage")
    
    class TestStorage(BaseStorage):
        type: IndexType = IndexType.JSON
        prefix: str = "test"
        key_separator: str = ":"
        
    s = TestStorage()
    print(f"Created storage: {s.type}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
