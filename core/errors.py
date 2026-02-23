from fastapi.responses import JSONResponse

def claude_error(message, type_="invalid_request_error", status=400):
    return JSONResponse(
        status_code=status,
        content={
            "type": "error",
            "error": {
                "type": type_,
                "message": message
            }
        }
    )
