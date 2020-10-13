import ssl
import requests
import os
print(ssl.get_default_verify_paths())
requests.get("https://adb-2361668257248726.6.azuredatabricks.net/")