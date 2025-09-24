import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw, ImageFont
import re
import os
import requests
from typing import Optional

def _extract_drive_file_id(drive_url: str) -> str:
    patterns = [
        r"drive\.google\.com\/file\/d\/([a-zA-Z0-9_-]+)",
        r"drive\.google\.com\/open\?id=([a-zA-Z0-9_-]+)",
        r"drive\.google\.com\/uc\?id=([a-zA-Z0-9_-]+)",
        r"drive\.google\.com\/uc\?export=download&id=([a-zA-Z0-9_-]+)",
    ]
    for pat in patterns:
        m = re.search(pat, drive_url)
        if m:
            return m.group(1)
    plain_id = re.fullmatch(r"[a-zA-Z0-9_-]{20,}", drive_url.strip())
    if plain_id:
        return drive_url.strip()
    raise ValueError("Could not parse a Google Drive FILE ID from the provided URL.")

def _get_confirm_token_from_html(text: str) -> Optional[str]:
    m = re.search(r'confirm=([0-9A-Za-z_-]+)', text)
    return m.group(1) if m else None

def _stream_to_file(response: requests.Response, destination_path: str, chunk_bytes: int = 1 << 20):
    os.makedirs(os.path.dirname(os.path.abspath(destination_path)), exist_ok=True)
    total_bytes = int(response.headers.get("Content-Length", "0")) or None
    written = 0
    with open(destination_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_bytes):
            if chunk:
                f.write(chunk)
                written += len(chunk)
    if os.path.getsize(destination_path) == 0:
        raise IOError("Downloaded file is empty; the link may not be public or is invalid.")

def download_public_gdrive_file(drive_url_or_id: str, destination_path: str, timeout_sec: int = 60):
    file_id = _extract_drive_file_id(drive_url_or_id)
    base = "https://drive.google.com/uc"
    session = requests.Session()

    # First attempt: try direct export=download (works for small files)
    params = {"id": file_id, "export": "download"}
    r = session.get(base, params=params, stream=True, timeout=timeout_sec)
    r.raise_for_status()

    # If Google shows the interstitial page, it returns HTML with a confirm token.
    if "text/html" in r.headers.get("Content-Type", ""):
        token = _get_confirm_token_from_html(r.text)
        if not token:
            # Sometimes the token is delivered via a cookie named 'download_warning'
            token = next((v for k, v in r.cookies.items() if k.startswith("download_warning")), None)

        if token:
            params["confirm"] = token
            r = session.get(base, params=params, stream=True, timeout=timeout_sec)
            r.raise_for_status()
        else:
            # If we cannot find a token, the file is likely not publicly accessible.
            raise PermissionError(
                "Could not obtain confirm token. Ensure the file is PUBLIC or use Drive mounting."
            )

    _stream_to_file(r, destination_path)

# Clear directories to regenerate file distribution
def clear_directory(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def save_combined_image(image, pred, label, test_file, pred_path, ship_iou=None, terrain_iou=None, divider_width=5, saliency_map=None):    
    """
    Saves a single image containing the original image, prediction, and label side by side,
    with thin white vertical dividers between them, and overlays IoU scores.

    Parameters:
    - image: PyTorch tensor (C, H, W) - The input image.
    - pred: NumPy array (H, W) - The predicted mask.
    - label: NumPy array (H, W) - The ground truth mask.
    - test_file: str - The original file path (used for naming output file).
    - pred_path: str - Directory to save the output image.
    - ship_iou: float - Ship IoU score to overlay.
    - terrain_iou: float - Terrain IoU score to overlay.
    - divider_width: int - Width of the divider lines.
    """

    # Convert the image (C, H, W) tensor to (H, W, C) NumPy array and scale to [0,255]
    img = image.squeeze().cpu().numpy()  # Convert to (H, W, C)
    img = (255 * img).astype(np.uint8)  # Scale to [0, 255]
    img_pil = Image.fromarray(img)  # Convert to PIL image

    # pred_img = Image.fromarray((255 * pred[0, ...]).astype(np.uint8))
    # label_img = Image.fromarray((255 * label.squeeze()).astype(np.uint8))
    pred_img = Image.fromarray((127.5 * pred[0, ...]+127.5).astype(np.uint8))
    label_img = Image.fromarray((127.5*label.squeeze()+127.5).astype(np.uint8))

    # Ensure all images have the same height
    height = img_pil.height
    pred_img = pred_img.resize((pred_img.width, height))
    label_img = label_img.resize((label_img.width, height))
    saliency_img = None
    
    if saliency_map is not None:
        saliency_img = Image.fromarray(255 * saliency_map[0])
        saliency_img = saliency_img.resize((img_pil.width, height))  # Match height & width

    # Create a white divider
    divider = Image.new("RGB", (divider_width, height), color=(128, 128, 128))

    # Calculate final image width: 3 images + 2 dividers
    combined_width = img_pil.width + pred_img.width + label_img.width + (2 * divider_width)
    if saliency_map is not None:
        combined_width += saliency_img.width + divider_width
    combined_img = Image.new("RGB", (combined_width, height), color=(0, 0, 0))

    # Paste images with dividers in between
    x_offset = 0
    combined_img.paste(img_pil, (x_offset, 0))
    x_offset += img_pil.width
    combined_img.paste(divider, (x_offset, 0))
    x_offset += divider_width
    combined_img.paste(label_img.convert("RGB"), (x_offset, 0))
    x_offset += pred_img.width
    combined_img.paste(divider, (x_offset, 0))
    x_offset += divider_width
    combined_img.paste(pred_img.convert("RGB"), (x_offset, 0))
    if saliency_map is not None:
        x_offset += pred_img.width
        combined_img.paste(divider, (x_offset, 0))
        x_offset += divider_width
        combined_img.paste(saliency_img.convert("RGB"), (x_offset, 0))

    # Draw IoU text at bottom center
    if ship_iou is not None or terrain_iou is not None:
        draw = ImageDraw.Draw(combined_img)
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except:
            font = ImageFont.load_default()

        text = f"Ship IoU: {ship_iou:.4f}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = (combined_img.width - text_width) // 2
        text_y = combined_img.height - text_height - 10  # Slight margin from bottom

        # Add black rectangle for better visibility
        padding = 5
        draw.rectangle(
            [text_x - padding, text_y - padding, text_x + text_width + padding, text_y + text_height + padding],
            fill=(255, 255, 255)
        )

        draw.text((text_x, text_y), text, font=font, fill=(255, 0, 0))  # Red text

    # Save the final combined image
    output_filename = os.path.basename(test_file).replace("_image.npy", "_combined.png")
    pred_img.save(os.path.join(pred_path, output_filename))
    
    return combined_img

def save_plot(save_path, data):
    # Create the figuredepth_grid
    plt.figure(figsize=(6, 6))
    plt.imshow(data, cmap='gray', interpolation='nearest')
    plt.colorbar()  # Optional: add a colorbar
    plt.axis('off')  # Optional: remove axes for a cleaner look

    # Save the figure
    plt.savefig(save_path)

    # Show the figure (optional)
    # plt.show()

def normalize_nonzero(image):
    """
    Normalizes nonzero values in an image (tensor or numpy array) to the range [0,1].
    
    Parameters:
        image (torch.Tensor or np.ndarray): Input image.

    Returns:
        torch.Tensor or np.ndarray: Normalized image with nonzero values scaled to [0,1].
    """
    is_tensor = isinstance(image, torch.Tensor)
    # print("is tensor", is_tensor)
    
    if not (is_tensor or isinstance(image, np.ndarray)):
        raise TypeError("Input must be a PyTorch tensor or a NumPy array.")
    
    nonzero_mask = image != 0  # Boolean mask for nonzero elements
    
    if is_tensor:
        if torch.any(nonzero_mask):  # Ensure there are nonzero values
            image_min = torch.min(image[nonzero_mask])
            image_max = torch.max(image[nonzero_mask])

            if not (image_min >= 0 and image_max <= 1):  # Only normalize if needed
                image[nonzero_mask] = (image[nonzero_mask] - image_min) / (image_max - image_min)
    
    else:  # NumPy array case
        if np.any(nonzero_mask):  # Ensure there are nonzero values
            image_min = np.min(image[nonzero_mask])
            image_max = np.max(image[nonzero_mask])

            if not (image_min >= 0 and image_max <= 1):  # Only normalize if needed
                image[nonzero_mask] = (image[nonzero_mask] - image_min) / (image_max - image_min)
    
    return image

def copy_files(src_folder, dst_folder):
    """
    Copies all files from src_folder to dst_folder.
    
    Args:
        src_folder (str): Source directory path.
        dst_folder (str): Destination directory path.
    """
    # Ensure the destination folder exists
    os.makedirs(dst_folder, exist_ok=True)

    # Loop through all files in the source directory
    for count, filename in enumerate(os.listdir(src_folder)):
        src_file = os.path.join(src_folder, filename)
        dst_file = os.path.join(dst_folder, filename)

        # Check if it's a file before copying
        if os.path.isfile(src_file):
            shutil.copy2(src_file, dst_file)  # copy2 preserves metadata

    print(f"Copied {count} ship images to {dst_folder}.")

    # print(f"All files copied from {src_folder} to {dst_folder}")
