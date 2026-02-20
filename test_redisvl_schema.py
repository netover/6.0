
try:
    from redisvl.schema.schema import IndexSchema, IndexInfo, StorageType
    print("Successfully imported redisvl.schema.schema")
    
    schema_dict = {
        "index": {
            "name": "test-index",
            "prefix": "test",
            "key_separator": ":",
            "storage_type": "json"
        },
        "fields": [
            {"name": "user", "type": "tag"},
            {"name": "age", "type": "numeric"}
        ]
    }
    
    schema = IndexSchema(**schema_dict)
    print(f"Created schema: {schema.index.name}")
    print(f"Fields: {list(schema.fields.keys())}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
