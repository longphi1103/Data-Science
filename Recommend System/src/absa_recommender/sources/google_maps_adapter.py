from typing import Any


class GoogleMapsAdapter:
    source = "google_maps"

    def __init__(self, allow_unlicensed_fetch: bool = False) -> None:
        self.allow_unlicensed_fetch = allow_unlicensed_fetch

    def fetch_reviews(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        if not self.allow_unlicensed_fetch:
            raise RuntimeError(
                "Google Maps review fetching is not enabled. Configure a licensed/API-compliant source."
            )
        return []
