import cv2
import numpy as np
import openai
import base64
import os
import time
import json
from io import BytesIO
from PIL import Image

# Set up OpenAI client
client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

# System prompt for concise responses
SYSTEM_PROMPT = "Reply to the images I share with you as though they are prompts. Do not describe what you see. Just respond to the image as you would if it were any other prompt. Do not start with things like 'the prompt says' or 'the image shows'. I already know what the image is, just respond to it. Be as concise as possible."

# Keystone save file path
KEYSTONE_SAVE_FILE = 'keystone_settings.json'

# Load keystone settings if they exist
def load_keystone_settings():
    try:
        with open(KEYSTONE_SAVE_FILE, 'r') as f:
            settings = json.load(f)
            return np.float32(settings['corners'])
    except (FileNotFoundError, json.JSONDecodeError):
        return np.float32([
            [-0.5, -0.5],  # Top-left - starts wider
            [1.5, -0.5],   # Top-right - starts wider
            [-0.5, 1.5],   # Bottom-left - starts wider
            [1.5, 1.5]     # Bottom-right - starts wider
        ])

# Save keystone settings
def save_keystone_settings(corners):
    with open(KEYSTONE_SAVE_FILE, 'w') as f:
        json.dump({
            'corners': corners.tolist()
        }, f)
    print("Keystone settings saved")

# Initialize keystone corners from saved settings
keystone_corners = load_keystone_settings()

def encode_image_to_base64(frame):
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_frame)
    
    # Resize if needed to meet size requirements
    max_size = (2000, 2000)
    pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    # Create buffer
    buffer = BytesIO()
    # Save image to buffer in JPEG format
    pil_image.save(buffer, format='JPEG')
    # Get base64 string
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def query_vision_model(frame):
    base64_image = encode_image_to_base64(frame)
    
    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[{
                "role": "system",
                "content": SYSTEM_PROMPT
            }, {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What do you see?"},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    },
                ],
            }],
        )
        return response.output_text
    except Exception as e:
        print(f"Error querying vision model: {e}")
        return None

def apply_keystone(frame, corners):
    h, w = frame.shape[:2]
    
    # Convert normalized coordinates to actual pixel coordinates
    src_points = np.float32([
        [corners[0][0] * w, corners[0][1] * h],  # Top-left
        [corners[1][0] * w, corners[1][1] * h],  # Top-right
        [corners[2][0] * w, corners[2][1] * h],  # Bottom-left
        [corners[3][0] * w, corners[3][1] * h]   # Bottom-right
    ])
    
    dst_points = np.float32([
        [0, 0], [w, 0],  # Top-left, Top-right
        [0, h], [w, h]   # Bottom-left, Bottom-right
    ])
    
    # Get perspective transform matrix and apply it
    matrix = cv2.getPerspectiveTransform(dst_points, src_points)
    warped = cv2.warpPerspective(frame, matrix, (w, h))
    return warped

def countdown_and_capture(cap):
    for i in range(3, 0, -1):
        ret, frame = cap.read()
        if not ret:
            return None
        
        # Add countdown text
        countdown_frame = frame.copy()
        cv2.putText(countdown_frame, str(i), 
                   (frame.shape[1]//2 - 50, frame.shape[0]//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 4.0, (255, 255, 255), 8)
        
        # Show frame with countdown
        cv2.imshow("Camera Feed", countdown_frame)
        cv2.imshow(proj_window, countdown_frame)
        
        # Wait for 1 second
        start_time = time.time()
        while time.time() - start_time < 1:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return None
    
    # Capture the final frame
    ret, frame = cap.read()
    if ret:
        # Flash effect
        flash_frame = np.ones_like(frame) * 255
        cv2.imshow("Camera Feed", flash_frame)
        cv2.imshow(proj_window, flash_frame)
        cv2.waitKey(50)  # Short flash
        
        return frame
    return None

def draw_keystone_points(frame, corners, selected_corner=None):
    h, w = frame.shape[:2]
    pts = []
    # Convert normalized coordinates to actual pixel coordinates
    for i, corner in enumerate(corners):
        x = int(corner[0] * w)
        y = int(corner[1] * h)
        pts.append((x, y))
        # Draw point
        color = (0, 0, 255) if i == selected_corner else (0, 255, 0)
        cv2.circle(frame, (x, y), 10, color, -1)
        # Draw number
        cv2.putText(frame, str(i+1), (x+15, y+15),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Draw lines connecting points
    cv2.line(frame, pts[0], pts[1], (0, 255, 0), 2)  # Top
    cv2.line(frame, pts[2], pts[3], (0, 255, 0), 2)  # Bottom
    cv2.line(frame, pts[0], pts[2], (0, 255, 0), 2)  # Left
    cv2.line(frame, pts[1], pts[3], (0, 255, 0), 2)  # Right
    
    return frame

def wrap_text(text, max_width, font_scale, thickness):
    words = text.split(' ')
    lines = []
    current_line = []
    
    for word in words:
        # Add word to current line
        test_line = ' '.join(current_line + [word])
        # Get size of text
        (w, h), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        if w <= max_width:
            current_line.append(word)
        else:
            # Line is full, start a new line
            if current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                # Single word is too long, force it on its own line
                lines.append(word)
                current_line = []
    
    # Add the last line if there's anything left
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines

# Camera setup
CAMERA_INDEX = 2
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError(f"Failed to open camera {CAMERA_INDEX}")

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Create windows
cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera Feed", 640, 360)

proj_window = "Projector Output"
cv2.namedWindow(proj_window, cv2.WINDOW_NORMAL)
cv2.resizeWindow(proj_window, 1280, 720)

print("Camera feed window created - keep this on your main display")
print("Drag the 'Projector Output' window to your projector display")
print("Controls:")
print("- Press 'f' to toggle fullscreen on projector")
print("- Press 'k' to toggle keystone mode")
print("- Press 't' to toggle text display mode")
print("- In keystone mode:")
print("  • Use number keys 1-4 to select corner (1=top-left, 2=top-right, 3=bottom-left, 4=bottom-right)")
print("  • Use arrow keys to move selected corner")
print("  • Press 'r' to reset keystone")
print("  • Press 's' to save keystone settings")
print("- Press 'space' to capture image and get new text")
print("- Press 'c' to clear current text")
print("- Press 'q' to quit")

is_fullscreen = False
keystone_mode = False
text_mode = False
selected_corner = None
current_vision_result = None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        # Create display frames
        display_frame = frame.copy()
        processed_frame = apply_keystone(frame, keystone_corners)
        
        # Prepare projector output
        if keystone_mode:
            # Show keystone UI when in keystone mode
            proj_frame = draw_keystone_points(display_frame.copy(), keystone_corners, selected_corner)
        else:
            # Show either black frame with text or camera feed
            if text_mode and current_vision_result:
                proj_frame = np.zeros_like(frame)
                
                # Text rendering settings with increased spacing
                font_scale = 1.5
                thickness = 2
                margin = 50
                line_spacing = 60  # Increased from 20 to 60
                max_width = proj_frame.shape[1] - 2 * margin
                
                # Wrap text into lines
                lines = wrap_text(current_vision_result, max_width, font_scale, thickness)
                
                # Calculate starting Y position to center text vertically
                total_text_height = len(lines) * (line_spacing + thickness)
                y_position = (proj_frame.shape[0] - total_text_height) // 2
                
                # Draw each line
                for line in lines:
                    y_position += line_spacing
                    cv2.putText(proj_frame, line,
                              (margin, y_position),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              font_scale,
                              (255, 255, 255),
                              thickness)
            else:
                proj_frame = np.zeros_like(frame)  # Black frame when not in any mode
        
        # Show frames
        cv2.imshow("Camera Feed", display_frame)
        cv2.imshow(proj_window, proj_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1)
        
        if key & 0xFF == ord('q'):
            save_keystone_settings(keystone_corners)
            break
        elif key & 0xFF == ord('f'):
            is_fullscreen = not is_fullscreen
            if is_fullscreen:
                cv2.setWindowProperty(proj_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(proj_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        elif key & 0xFF == ord('k'):
            keystone_mode = not keystone_mode
            selected_corner = 0 if keystone_mode else None
            print("Keystone mode:", "ON" if keystone_mode else "OFF")
        elif key & 0xFF == ord('t'):
            text_mode = not text_mode
            print("Text display mode:", "ON" if text_mode else "OFF")
        elif key & 0xFF == ord('r'):
            keystone_corners = np.float32([
                [-0.5, -0.5],  # Top-left
                [1.5, -0.5],   # Top-right
                [-0.5, 1.5],   # Bottom-left
                [1.5, 1.5]     # Bottom-right
            ])
            print("Keystone reset")
        elif key & 0xFF == ord('s'):
            save_keystone_settings(keystone_corners)
        elif key & 0xFF == ord('c'):
            current_vision_result = None
            print("Text cleared")
        
        # Handle corner selection and movement
        if keystone_mode and selected_corner is not None:
            # Corner selection
            if key & 0xFF in [ord('1'), ord('2'), ord('3'), ord('4')]:
                selected_corner = (key & 0xFF) - ord('1')
                print(f"Selected corner {selected_corner + 1}")
            
            # Corner movement using arrow keys
            move_amount = 0.05  # Movement sensitivity
            if key == 63232:  # Up arrow on macOS
                keystone_corners[selected_corner][1] -= move_amount
                print(f"Moving corner {selected_corner + 1} up")
            elif key == 63233:  # Down arrow on macOS
                keystone_corners[selected_corner][1] += move_amount
                print(f"Moving corner {selected_corner + 1} down")
            elif key == 63234:  # Left arrow on macOS
                keystone_corners[selected_corner][0] -= move_amount
                print(f"Moving corner {selected_corner + 1} left")
            elif key == 63235:  # Right arrow on macOS
                keystone_corners[selected_corner][0] += move_amount
                print(f"Moving corner {selected_corner + 1} right")
        
        elif key & 0xFF == ord(' '):
            print("\nStarting countdown...")
            captured_frame = countdown_and_capture(cap)
            if captured_frame is not None:
                # Apply keystone correction to the captured frame
                processed_capture = apply_keystone(captured_frame, keystone_corners)
                print("Processing image...")
                vision_result = query_vision_model(processed_capture)
                if vision_result:
                    current_vision_result = vision_result
                    text_mode = True  # Automatically enable text mode when new text is received
                    print("\nVision Model Response:")
                    print(vision_result)

finally:
    # Save settings before closing
    if keystone_corners is not None:
        save_keystone_settings(keystone_corners)
    cap.release()
    cv2.destroyAllWindows()