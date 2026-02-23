import time

def apply_standard_headers(response, usage):
    response.headers["anthropic-version"] = "2023-06-01"
    response.headers["cache-control"] = "no-store"
    response.headers["x-ratelimit-limit"] = "1000"
    response.headers["x-ratelimit-remaining"] = "999"
    response.headers["x-ratelimit-reset"] = str(int(time.time()) + 60)

    if usage:
        response.headers["anthropic-usage"] = str(usage.to_dict())
