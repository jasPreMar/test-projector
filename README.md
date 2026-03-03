# test-projector

A small “camera → LLM → projector” prototype:

- Shows a live camera feed on your main display.
- Shows a second window on a projector that can be:
  - a keystone-calibration UI, or
  - a black screen with the latest LLM response rendered as large text.
- Press **Space** to do a 3-second countdown, capture a frame, send it to an OpenAI vision model, then display the model’s response on the projector. :contentReference[oaicite:0]{index=0}

This repo is currently centered around a single script: `camera_test.py`. :contentReference[oaicite:1]{index=1}

---

## What it does

`camera_test.py`:
- Opens a camera (default `CAMERA_INDEX = 2`) and creates two windows: **Camera Feed** and **Projector Output**. :contentReference[oaicite:2]{index=2}
- Supports interactive keystone adjustment by warping the image via a perspective transform, saving the four corner points to `keystone_settings.json`. :contentReference[oaicite:3]{index=3}
- Captures a frame on demand, encodes it to base64 JPEG, calls `client.responses.create(...)` with model `gpt-4.1-mini`, and reads `response.output_text`. :contentReference[oaicite:4]{index=4}
- Renders the response text in large type with line wrapping onto the projector window. :contentReference[oaicite:5]{index=5}

<img width="413" height="353" alt="Screenshot 2026-03-03 at 3 57 06 PM" src="https://github.com/user-attachments/assets/46028668-f4dc-4709-9933-8424b5d332d8" />

---

## Demo

Here’s an imagine of testing it, with me holding up a post card and seeing what it would do: 

<img width="843" height="659" alt="Screenshot 2026-03-03 at 3 49 56 PM" src="https://github.com/user-attachments/assets/d29e63ba-5b2e-4771-b559-74b48f540d03" />

Here is an image of the generated output that was projected in the image above:

<img width="999" height="665" alt="Screenshot 2026-03-03 at 3 52 24 PM" src="https://github.com/user-attachments/assets/ac8c371f-be70-437d-9c07-ca852a0ecb85" />

Another example of generated output:

<img width="660" height="663" alt="Screenshot 2026-03-03 at 3 53 11 PM" src="https://github.com/user-attachments/assets/69299745-f25f-4d59-9b24-4b4669b8686d" />


---

## Requirements

- Python 3.x
- A webcam / capture device
- A second display (projector) recommended

Python packages used by the script:
- `opencv-python` (cv2)
- `numpy`
- `Pillow`
- `openai` :contentReference[oaicite:6]{index=6}

---

## Setup

1. Clone:
   ```bash
   git clone https://github.com/jasPreMar/test-projector.git
   cd test-projector
