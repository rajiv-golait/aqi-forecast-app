"""Helper functions for common utilities."""
 
# Implement helper functions here 

import math
import datetime

def clean_json_response(obj):
    """
    Recursively clean a dict/list of values so that all float NaN/inf/-inf are replaced with None,
    and datetime objects are converted to ISO strings.
    This ensures JSON serialization will not fail.
    """
    if isinstance(obj, dict):
        return {k: clean_json_response(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json_response(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    else:
        return obj 