
try:
    from redisvl.schema.fields import TextFieldAttributes, BaseVectorFieldAttributes, FlatVectorFieldAttributes
    print("Successfully imported redisvl.schema.fields")
    
    t = TextFieldAttributes()
    print(f"TextFieldAttributes: {t}")
    
    v = FlatVectorFieldAttributes(dims=128, algorithm="FLAT")
    print(f"FlatVectorFieldAttributes: {v}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
