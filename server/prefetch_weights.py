"""Downloads the DA3-BASE weights into the Hugging Face cache so the first request does not stall on a download."""
from huggingface_hub import snapshot_download

path = snapshot_download("depth-anything/DA3-BASE")
print(f"model cached at {path}")
