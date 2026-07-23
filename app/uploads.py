from pathlib import Path

from fastapi import HTTPException, UploadFile

MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15MB — generous headroom over any real phone photo


async def read_validated_image(photo: UploadFile) -> bytes:
    """Read an uploaded file, rejecting anything too large or that isn't
    actually a real, decodable image (regardless of what filename/content-type
    the client claims)."""
    data = await photo.read()

    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Photo is too large (max 15MB).")

    if not _is_real_image(data):
        raise HTTPException(status_code=400, detail="That file isn't a valid image.")

    return data


def _is_real_image(data: bytes) -> bool:
    from io import BytesIO

    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(BytesIO(data)) as image:
            image.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False


def save_validated_image(photo_path: Path, data: bytes) -> None:
    """Re-saves the image through Pillow (re-encoding as JPEG) so whatever's
    on disk is guaranteed to be a clean, real image — not just bytes that
    happened to pass the initial verify() check."""
    from io import BytesIO

    from PIL import Image

    with Image.open(BytesIO(data)) as image:
        image.convert("RGB").save(photo_path, "JPEG")
