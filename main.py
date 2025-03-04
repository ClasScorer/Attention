import os
from PIL import Image, ImageDraw
import torch
import numpy as np
import matplotlib.pyplot as plt
from retinaface import RetinaFace

def load_gazelle_model():
    """Load and prepare the Gazelle model"""
    model, transform = torch.hub.load('fkryan/gazelle', 'gazelle_dinov2_vitl14_inout')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)
    return model, transform, device

def detect_faces(image):
    """Detect faces in the image using RetinaFace"""
    width, height = image.size
    # Convert PIL Image to numpy array for RetinaFace
    face_detection = RetinaFace.detect_faces(np.array(image))
    
    if not face_detection:
        return []
    
    # Extract facial areas
    bboxes = [face_detection[key]['facial_area'] for key in face_detection.keys()]
    
    # Normalize bounding boxes
    norm_bboxes = [[bbox[0]/width, bbox[1]/height, bbox[2]/width, bbox[3]/height] for bbox in bboxes]
    return norm_bboxes

def process_image(image_path, output_dir="output"):
    """Process an image with the Gazelle model and save visualizations"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model, transform, device = load_gazelle_model()
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    
    # Detect faces
    norm_bboxes = detect_faces(image)
    if not norm_bboxes:
        print(f"No faces detected in {image_path}")
        return
    
    # Prepare model input
    img_tensor = transform(image).unsqueeze(0).to(device)
    model_input = {
        "images": img_tensor,
        "bboxes": [norm_bboxes]
    }
    
    # Generate predictions
    with torch.no_grad():
        output = model(model_input)
    
    # Save original image
    image.save(os.path.join(output_dir, "original.jpg"))
    
    # Calculate gaze angles and focus
    gaze_angles = []
    focus_status = []
    for i in range(len(norm_bboxes)):
        heatmap = output['heatmap'][0][i].detach().cpu().numpy()
        
        # Find max point in heatmap
        max_index = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        gaze_target_y = max_index[0] / heatmap.shape[0] * height
        gaze_target_x = max_index[1] / heatmap.shape[1] * width
        
        # Calculate face center
        bbox = norm_bboxes[i]
        bbox_center_x = ((bbox[0] + bbox[2]) / 2) * width
        bbox_center_y = ((bbox[1] + bbox[3]) / 2) * height
        
        # Calculate angle (in degrees)
        # atan2 takes (y, x) and returns angle in radians
        dx = gaze_target_x - bbox_center_x
        dy = gaze_target_y - bbox_center_y
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)
        
        # Normalize to 0-360 range
        if angle_deg < 0:
            angle_deg += 360
            
        gaze_angles.append(angle_deg)
        
        # Determine if looking at camera
        # Calculate vector length from face to gaze point
        gaze_vector_length = np.sqrt(dx**2 + dy**2)
        
        # Calculate diagonal of face bbox
        face_width = (bbox[2] - bbox[0]) * width
        face_height = (bbox[3] - bbox[1]) * height
        face_diagonal = np.sqrt(face_width**2 + face_height**2)
        
        # If vector length is short relative to face size, person is likely looking at camera
        # We use a threshold of 2x the face diagonal
        is_focused = gaze_vector_length < face_diagonal * 2
        focus_status.append("focused" if is_focused else "unfocused")
    
    # Visualize and save individual heatmaps
    for i in range(len(norm_bboxes)):
        heatmap_img = visualize_heatmap(
            image, 
            output['heatmap'][0][i], 
            norm_bboxes[i],
            output['inout'][0][i] if output['inout'] is not None else None,
            focus_status[i]
        )
        heatmap_img.save(os.path.join(output_dir, f"heatmap_person_{i+1}.png"))
    
    # Create and save combined visualization
    combined_img = visualize_all(
        image, 
        output['heatmap'][0], 
        norm_bboxes, 
        output['inout'][0] if output['inout'] is not None else None,
        gaze_angles=gaze_angles,
        focus_status=focus_status
    )
    combined_img.save(os.path.join(output_dir, "combined_gaze.png"))
    
    # Return results
    results = {
        "heatmaps": output['heatmap'][0].cpu().numpy(),
        "inout_scores": output['inout'][0].cpu().numpy() if output['inout'] is not None else None,
        "bboxes": norm_bboxes,
        "gaze_angles": gaze_angles,
        "focus_status": focus_status
    }
    return results

def visualize_heatmap(pil_image, heatmap, bbox=None, inout_score=None, focus_status=None):
    """Visualize a single person's gaze heatmap"""
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()
    
    # Resize heatmap to match image size
    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(pil_image.size, Image.BILINEAR)
    
    # Apply colormap
    heatmap_colored = plt.cm.jet(np.array(heatmap_img) / 255.)
    heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
    heatmap_colored = Image.fromarray(heatmap_colored).convert("RGBA")
    heatmap_colored.putalpha(90)  # Semi-transparent
    
    # Overlay on original image
    overlay_image = Image.alpha_composite(pil_image.convert("RGBA"), heatmap_colored)
    
    # Draw bounding box if provided
    if bbox is not None:
        width, height = pil_image.size
        xmin, ymin, xmax, ymax = bbox
        draw = ImageDraw.Draw(overlay_image)
        draw.rectangle(
            [xmin * width, ymin * height, xmax * width, ymax * height], 
            outline="lime", 
            width=int(min(width, height) * 0.01)
        )
        
        # Add inout score if available
        text_y_offset = int(height * 0.01)
        if inout_score is not None:
            text = f"in-frame: {inout_score:.2f}"
            text_x = xmin * width
            text_y = ymax * height + text_y_offset
            draw.text((text_x, text_y), text, fill="lime")
            text_y_offset += int(height * 0.02)
            
        # Add focus status if available
        if focus_status is not None:
            focus_text = f"attention: {focus_status}"
            text_x = xmin * width
            text_y = ymax * height + text_y_offset
            text_color = "lime" if focus_status == "focused" else "red"
            draw.text((text_x, text_y), focus_text, fill=text_color)
            
    return overlay_image

def visualize_all(pil_image, heatmaps, bboxes, inout_scores, inout_thresh=0.5, gaze_angles=None, focus_status=None):
    """Visualize all people's gaze directions in one image"""
    colors = ['lime', 'tomato', 'cyan', 'fuchsia', 'yellow']
    overlay_image = pil_image.convert("RGBA")
    draw = ImageDraw.Draw(overlay_image)
    width, height = pil_image.size

    for i in range(len(bboxes)):
        bbox = bboxes[i]
        xmin, ymin, xmax, ymax = bbox
        color = colors[i % len(colors)]
        
        # Draw bounding box
        draw.rectangle(
            [xmin * width, ymin * height, xmax * width, ymax * height], 
            outline=color, 
            width=int(min(width, height) * 0.01)
        )

        # Add inout score, angle, and focus status if available
        text_y_offset = int(height * 0.01)
        if inout_scores is not None:
            inout_score = inout_scores[i]
            inout_text = f"in-frame: {inout_score:.2f}"
            text_x = xmin * width
            text_y = ymax * height + text_y_offset
            draw.text((text_x, text_y), inout_text, fill=color)
            text_y_offset += int(height * 0.02)
            
        if gaze_angles is not None:
            angle_text = f"angle: {gaze_angles[i]:.1f}°"
            text_x = xmin * width
            text_y = ymax * height + text_y_offset
            draw.text((text_x, text_y), angle_text, fill=color)
            text_y_offset += int(height * 0.02)
        
        if focus_status is not None:
            focus_text = f"attention: {focus_status[i]}"
            text_x = xmin * width
            text_y = ymax * height + text_y_offset
            text_color = color if focus_status[i] == "focused" else "red"
            draw.text((text_x, text_y), focus_text, fill=text_color)

        # Draw gaze direction for people looking inside the frame
        if inout_scores is not None and inout_score > inout_thresh:
            heatmap = heatmaps[i]
            heatmap_np = heatmap.detach().cpu().numpy()
            
            # Find max point in heatmap
            max_index = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape)
            gaze_target_x = max_index[1] / heatmap_np.shape[1] * width
            gaze_target_y = max_index[0] / heatmap_np.shape[0] * height
            
            # Calculate face center
            bbox_center_x = ((xmin + xmax) / 2) * width
            bbox_center_y = ((ymin + ymax) / 2) * height

            # Draw gaze target and line
            draw.ellipse(
                [(gaze_target_x-5, gaze_target_y-5), (gaze_target_x+5, gaze_target_y+5)], 
                fill=color, 
                width=int(0.005*min(width, height))
            )
            draw.line(
                [(bbox_center_x, bbox_center_y), (gaze_target_x, gaze_target_y)], 
                fill=color, 
                width=int(0.005*min(width, height))
            )

    return overlay_image

if __name__ == "__main__":
    try:
        # Check if in.jpeg exists
        input_path = "in.jpeg"
        if not os.path.exists(input_path):
            print(f"Error: Input image '{input_path}' not found")
        else:
            print(f"Processing image: {input_path}")
            results = process_image(input_path)
            print(f"Processing complete. Results saved to 'output' directory.")
            
            # Print some summary stats
            if results:
                num_faces = len(results["bboxes"])
                print(f"Detected {num_faces} faces in the image")
                
                for i in range(num_faces):
                    person_info = f"Person {i+1}:"
                    
                    if results["inout_scores"] is not None:
                        score = results["inout_scores"][i]
                        looking = "Looking inside frame" if score > 0.5 else "Looking outside frame"
                        person_info += f" {looking} (score: {score:.2f})"
                    
                    if "gaze_angles" in results:
                        angle = results["gaze_angles"][i]
                        person_info += f", Gaze angle: {angle:.1f}° from horizontal"
                        
                        # Add cardinal direction interpretation
                        direction = ""
                        if 22.5 <= angle < 67.5:
                            direction = "northeast"
                        elif 67.5 <= angle < 112.5:
                            direction = "upward"
                        elif 112.5 <= angle < 157.5:
                            direction = "northwest"
                        elif 157.5 <= angle < 202.5:
                            direction = "leftward"
                        elif 202.5 <= angle < 247.5:
                            direction = "southwest"
                        elif 247.5 <= angle < 292.5:
                            direction = "downward"
                        elif 292.5 <= angle < 337.5:
                            direction = "southeast"
                        else:  # 337.5 <= angle < 360 or 0 <= angle < 22.5
                            direction = "rightward"
                        
                        person_info += f" (looking {direction})"
                    
                    if "focus_status" in results:
                        focus = results["focus_status"][i]
                        person_info += f", Attention: {focus}"
                    
                    print(person_info)
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")