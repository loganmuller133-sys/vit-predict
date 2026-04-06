    # app/api/middleware/auth.py
    import os
    from fastapi import Request, HTTPException
    from starlette.middleware.base import BaseHTTPMiddleware
    from dotenv import load_dotenv

    load_dotenv()

    API_KEY = os.getenv("API_KEY", "")
    AUTH_ENABLED = os.getenv("AUTH_ENABLED", "true").lower() == "true"


    class APIKeyMiddleware(BaseHTTPMiddleware):
        """API Key authentication middleware"""

        async def dispatch(self, request: Request, call_next):
            # Skip auth for health endpoint and docs if disabled
            if not AUTH_ENABLED:
                return await call_next(request)

            if request.url.path in ["/health", "/docs", "/openapi.json", "/redoc"]:
                return await call_next(request)

            # Get API key from headers
            api_key = request.headers.get("x-api-key")

            if not api_key:
                raise HTTPException(
                    status_code=401,
                    detail="Missing API key. Please provide x-api-key header"
                )

            if api_key != API_KEY:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid API key"
                )

            return await call_next(request)


    async def verify_api_key(request: Request):
        """Dependency for route-level API key validation"""
        if not AUTH_ENABLED:
            return True

        api_key = request.headers.get("x-api-key")

        if not api_key:
            raise HTTPException(status_code=401, detail="Missing API key")

        if api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")

        return True