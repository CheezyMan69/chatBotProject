import httpx
import base64

# helper function for making url -> base64 for gemini 
def load_image_base64(url: str) -> str:
    resp = httpx.get(
        url,
        follow_redirects=True,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "image/*,*/*",
        }
    )

    ct = resp.headers.get("content-type", "")
    if not ct.startswith("image/"):
        raise ValueError(f"Not an image: content-type={ct}, url={url}")

    return base64.b64encode(resp.content).decode("utf-8")