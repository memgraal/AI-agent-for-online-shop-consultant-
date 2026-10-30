from fastapi.middleware.cors import CORSMiddleware

from app.main import app


def test_cors_allows_every_registered_api_method():
    cors = next(
        middleware
        for middleware in app.user_middleware
        if middleware.cls is CORSMiddleware
    )
    allowed_methods = set(cors.kwargs["allow_methods"])
    route_methods = {
        method
        for route in app.routes
        for method in getattr(route, "methods", set())
    }

    assert route_methods <= allowed_methods
