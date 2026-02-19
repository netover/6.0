from resync.settings import Settings
try:
    s = Settings()
    print("Settings instantiated successfully")
except Exception:
    import traceback
    traceback.print_exc()
