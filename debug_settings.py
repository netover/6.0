import traceback
from resync.settings import Settings
try:
    s = Settings()
    print("Settings instantiated successfully")
except Exception:
    traceback.print_exc()
