import cv2
import numpy as np
import openai
import base64
import os
import time
from io import BytesIO
from PIL import Image

# Set up OpenAI client
client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

# System prompt for concise responses
SYSTEM_PROMPT = "The image is my query. Reply with a succinct response to what you see, no more than several words in length."

# Initialize keystone corners with a larger capture area (normalized coordinates)
keystone_corners = np.float32([
    [-0.5, -0.5],  # Top-left - starts wider
    [1.5, -0.5],   # Top-right - starts wider
    [-0.5, 1.5],   # Bottom-left - starts wider
    [1.5, 1.5]     # Bottom-right - starts wider
])

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
print("- Press 'k' to toggle keystone correction mode")
print("- In keystone mode:")
print("  • Use number keys 1-4 to select corner (1=top-left, 2=top-right, 3=bottom-left, 4=bottom-right)")
print("  • Use WASD keys to move selected corner")
print("  • Press 'r' to reset keystone")
print("- Press 'space' to start countdown and capture")
print("- Press 'q' to quit")

is_fullscreen = False
keystone_mode = False
selected_corner = None
current_vision_result = None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        # Apply keystone correction
        processed_frame = apply_keystone(frame, keystone_corners)
        
        # Show live preview
        cv2.imshow("Camera Feed", frame)
        
        # Show keystone controls or result on projector
        display_frame = frame.copy() if keystone_mode else processed_frame.copy()
        
        if keystone_mode:
            # Draw control points
            h, w = frame.shape[:2]
            for i, corner in enumerate(keystone_corners):
                pt = (int(corner[0] * w), int(corner[1] * h))
                color = (0, 0, 255) if i == selected_corner else (0, 255, 0)
                cv2.circle(display_frame, pt, 5, color, -1)
                cv2.putText(display_frame, str(i+1), 
                          (pt[0]+10, pt[1]+10),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        if current_vision_result and not keystone_mode:
            # Add vision model result to frame
            text_bg = display_frame.copy()
            cv2.rectangle(text_bg, (0, 0), (display_frame.shape[1], 60), (0, 0, 0), -1)
            display_frame = cv2.addWeighted(display_frame, 1, text_bg, 0.5, 0)
            
            # Add text
            cv2.putText(display_frame, current_vision_result, 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow(proj_window, display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            is_fullscreen = not is_fullscreen
            if is_fullscreen:
                cv2.setWindowProperty(proj_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(proj_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        elif key == ord('k'):
            keystone_mode = not keystone_mode
            selected_corner = 0 if keystone_mode else None
            print("Keystone mode:", "ON" if keystone_mode else "OFF")
        elif key == ord('r'):
            # Reset keystone corners
            keystone_corners = np.float32([
                [-0.5, -0.5],  # Top-left
                [1.5, -0.5],   # Top-right
                [-0.5, 1.5],   # Bottom-left
                [1.5, 1.5]     # Bottom-right
            ])
            print("Keystone reset")
        elif keystone_mode and selected_corner is not None:
            # Corner selection
            if key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                selected_corner = key - ord('1')
                print(f"Selected corner {selected_corner + 1}")
            
            # Corner movement
            move_amount = 0.05  # Increased for more noticeable movement
            if key == ord('w'):  # Up
                keystone_corners[selected_corner][1] -= move_amount
                print(f"Moving corner {selected_corner + 1} up")
            elif key == ord('s'):  # Down
                keystone_corners[selected_corner][1] += move_amount
                print(f"Moving corner {selected_corner + 1} down")
            elif key == ord('a'):  # Left
                keystone_corners[selected_corner][0] -= move_amount
                print(f"Moving corner {selected_corner + 1} left")
            elif key == ord('d'):  # Right
                keystone_corners[selected_corner][0] += move_amount
                print(f"Moving corner {selected_corner + 1} right")
        elif key == ord(' '):
            print("\nStarting countdown...")
            captured_frame = countdown_and_capture(cap)
            if captured_frame is not None:
                # Apply keystone correction to the captured frame
                processed_capture = apply_keystone(captured_frame, keystone_corners)
                print("Processing image...")
                vision_result = query_vision_model(processed_capture)
                if vision_result:
                    current_vision_result = vision_result
                    print("\nVision Model Response:")
                    print(vision_result)

finally:
    cap.release()
    cv2.destroyAllWindows()