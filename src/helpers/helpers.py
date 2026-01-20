import httpx
import base64

# helper function for making url -> base64 for gemini 
def load_image_base64(url: str, timeout: float = 30.0) -> str:
    resp = httpx.get(url, follow_redirects=True, timeout=timeout)
    resp.raise_for_status()
    return base64.b64encode(resp.content).decode("utf-8")