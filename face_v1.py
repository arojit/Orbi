#!/usr/bin/env python3
# orbi_enhanced.py
# Professional 3D-styled Orbi faces with proper alignment and modern design
import sys, time, math, random
import spidev as SPI
from PIL import Image, ImageDraw
sys.path.append("..")
from lib import LCD_2inch

# ---- Pins (adjust if needed) ----
RST = 27
DC = 25
BL = 18

# ---- Modern 3D Color Palette ----
BG = (12, 15, 25)           # Rich dark blue
WHITE = (255, 255, 255)     # Pure white
LIGHT_GRAY = (220, 225, 235) # Light highlights
MID_GRAY = (180, 190, 210)   # Mid tones
DARK_GRAY = (80, 90, 110)    # Shadows
ACCENT_BLUE = (100, 150, 255) # Modern blue accent
GLOW_BLUE = (150, 200, 255)   # Soft glow
DEPTH_SHADOW = (25, 30, 40)   # 3D shadows

# ---- Supersampling ----
SS = 2  # render scale (2x)

def aa_img(size):
    W, H = size
    return Image.new("RGB", (W*SS, H*SS), BG), W*SS, H*SS

def down(img, size):
    return img.resize(size, Image.LANCZOS)

def qbez(p0, p1, p2, steps=64):
    (x0,y0),(x1,y1),(x2,y2) = p0, p1, p2
    pts=[]
    for i in range(steps+1):
        t = i/steps
        x = (1-t)*(1-t)*x0 + 2*(1-t)*t*x1 + t*t*x2
        y = (1-t)*(1-t)*y0 + 2*(1-t)*t*y1 + t*t*y2
        pts.append((x,y))
    return pts

# ------------------ 3D Drawing Primitives ------------------

def draw_3d_circle(d, cx, cy, radius, base_color, light_angle=135):
    """Draw a 3D sphere with proper lighting and depth - error-safe version"""
    # Ensure minimum radius
    if radius < SS:
        radius = SS
    
    # Convert angle to radians
    angle_rad = math.radians(light_angle)
    light_x = math.cos(angle_rad)
    light_y = math.sin(angle_rad)
    
    # Draw multiple layers for 3D effect
    layers = 8
    for i in range(layers):
        layer_radius = radius - i * SS
        if layer_radius <= 0:
            break
            
        # Calculate lighting based on position
        intensity = 1.0 - (i / layers) * 0.6
        
        # Create gradient color
        r = min(255, max(0, int(base_color[0] * intensity)))
        g = min(255, max(0, int(base_color[1] * intensity)))
        b = min(255, max(0, int(base_color[2] * intensity)))
        layer_color = (r, g, b)
        
        # Offset for 3D effect
        offset_x = int(i * light_x * 0.5)
        offset_y = int(i * light_y * 0.5)
        
        # Ensure valid coordinates
        left = cx - layer_radius + offset_x
        right = cx + layer_radius + offset_x
        top = cy - layer_radius + offset_y  
        bottom = cy + layer_radius + offset_y
        
        if right > left and bottom > top:
            d.ellipse([left, top, right, bottom], fill=layer_color)
    
    # Add highlight - ensure it's within bounds
    highlight_r = max(SS, int(radius * 0.3))
    highlight_x = cx - int(radius * 0.3)
    highlight_y = cy - int(radius * 0.4)
    
    hl_left = highlight_x - highlight_r
    hl_right = highlight_x + highlight_r
    hl_top = highlight_y - highlight_r
    hl_bottom = highlight_y + highlight_r
    
    if hl_right > hl_left and hl_bottom > hl_top:
        d.ellipse([hl_left, hl_top, hl_right, hl_bottom], 
                 fill=WHITE, outline=LIGHT_GRAY, width=SS)

def draw_3d_eye(d, cx, cy, radius, pupil_pos=(0,0), blink_factor=1.0):
    """Professional 3D eye with proper depth and lighting"""
    # Ensure blink_factor is valid
    blink_factor = max(0.05, min(1.0, blink_factor))
    
    # Eye socket shadow (depth)
    socket_r = radius + 8*SS
    d.ellipse([cx - socket_r, cy - socket_r, cx + socket_r, cy + socket_r], 
             fill=DEPTH_SHADOW)
    
    # Blinking - compress vertically but ensure minimum size
    eye_height = max(6*SS, int(radius * 2 * blink_factor))
    eye_width = radius
    
    # Main eyeball with 3D shading
    eyeball_layers = 6
    for i in range(eyeball_layers):
        layer_r = radius - i * SS
        if layer_r <= 0:
            break
        
        intensity = 1.0 - (i / eyeball_layers) * 0.2
        gray_val = int(255 * intensity)
        layer_color = (gray_val, gray_val, gray_val)
        
        # Ensure valid coordinates
        top = cy - eye_height//2
        bottom = cy + eye_height//2
        left = cx - layer_r
        right = cx + layer_r
        
        if bottom > top and right > left:
            d.ellipse([left, top, right, bottom], fill=layer_color)
    
    # Main white eyeball - ensure valid coordinates
    top = cy - eye_height//2
    bottom = cy + eye_height//2
    left = cx - eye_width
    right = cx + eye_width
    
    if bottom > top and right > left:
        d.ellipse([left, top, right, bottom], fill=WHITE)
    
    if blink_factor > 0.3:  # Only draw details when eye is open enough
        px, py = pupil_pos
        
        # Iris with 3D depth - ensure it fits within the eye
        iris_r = min(int(radius * 0.5), eye_height//4)
        if iris_r > 2*SS:  # Only draw if large enough
            # Iris gradient layers
            iris_layers = 4
            for i in range(iris_layers):
                layer_r = iris_r - i * SS
                if layer_r <= 0:
                    break
                intensity = 0.7 - (i / iris_layers) * 0.3
                iris_color = (int(ACCENT_BLUE[0] * intensity),
                             int(ACCENT_BLUE[1] * intensity),
                             int(ACCENT_BLUE[2] * intensity))
                
                # Ensure iris stays within eye bounds
                iris_left = cx + px - layer_r
                iris_right = cx + px + layer_r
                iris_top = cy + py - layer_r
                iris_bottom = cy + py + layer_r
                
                # Clamp to eye boundaries
                eye_left = cx - radius
                eye_right = cx + radius
                eye_top = cy - eye_height//2
                eye_bottom = cy + eye_height//2
                
                if (iris_right > iris_left and iris_bottom > iris_top and
                    iris_left >= eye_left and iris_right <= eye_right and
                    iris_top >= eye_top and iris_bottom <= eye_bottom):
                    d.ellipse([iris_left, iris_top, iris_right, iris_bottom], fill=iris_color)
        
        # Pupil with depth
        pupil_r = min(int(radius * 0.25), iris_r//2, eye_height//6)
        if pupil_r > SS:  # Only draw if large enough
            pupil_left = cx + px - pupil_r
            pupil_right = cx + px + pupil_r
            pupil_top = cy + py - pupil_r
            pupil_bottom = cy + py + pupil_r
            
            if pupil_right > pupil_left and pupil_bottom > pupil_top:
                d.ellipse([pupil_left, pupil_top, pupil_right, pupil_bottom], fill=BG)
                
                # Highlights only if pupil is large enough
                if pupil_r > 3*SS:
                    # Main highlight
                    hl_r = max(SS, int(pupil_r * 0.4))
                    hl_x = cx + px - int(pupil_r * 0.4)
                    hl_y = cy + py - int(pupil_r * 0.5)
                    
                    hl_left = hl_x - hl_r
                    hl_right = hl_x + hl_r
                    hl_top = hl_y - hl_r
                    hl_bottom = hl_y + hl_r
                    
                    if hl_right > hl_left and hl_bottom > hl_top:
                        d.ellipse([hl_left, hl_top, hl_right, hl_bottom], fill=WHITE)
                    
                    # Secondary highlight
                    if pupil_r > 5*SS:
                        hl2_r = max(SS//2, int(pupil_r * 0.2))
                        hl2_x = cx + px + int(pupil_r * 0.3)
                        hl2_y = cy + py - int(pupil_r * 0.2)
                        
                        hl2_left = hl2_x - hl2_r
                        hl2_right = hl2_x + hl2_r
                        hl2_top = hl2_y - hl2_r
                        hl2_bottom = hl2_y + hl2_r
                        
                        if hl2_right > hl2_left and hl2_bottom > hl2_top:
                            d.ellipse([hl2_left, hl2_top, hl2_right, hl2_bottom], fill=LIGHT_GRAY)

def draw_3d_brow(d, cx, cy, width, height, angle=0, intensity=1.0):
    """3D eyebrow with proper thickness and shading"""
    # Calculate brow points
    hw = width // 2
    
    # Create brow shape with depth
    brow_points = []
    steps = 32
    for i in range(steps + 1):
        t = i / steps
        # Curved brow shape
        x = cx + (t - 0.5) * width
        y_curve = math.sin(math.pi * t) * height
        y = cy - y_curve + angle * (t - 0.5) * width * 0.1
        brow_points.append((x, y))
    
    # Draw brow with 3D effect
    brow_thickness = int(8 * SS * intensity)
    
    # Shadow layer
    shadow_points = [(x + SS, y + SS) for x, y in brow_points]
    d.line(shadow_points, fill=DEPTH_SHADOW, width=brow_thickness + 2*SS, joint="curve")
    
    # Main brow
    d.line(brow_points, fill=WHITE, width=brow_thickness, joint="curve")
    
    # Highlight
    highlight_points = [(x - SS//2, y - SS//2) for x, y in brow_points]
    d.line(highlight_points, fill=LIGHT_GRAY, width=max(SS, brow_thickness//2), joint="curve")

def draw_3d_mouth(d, cx, cy, expression="neutral", intensity=1.0, animation_phase=0):
    """Modern 3D mouth with sophisticated styling"""
    base_w = int(40 * SS * intensity)
    base_h = int(25 * SS * intensity)
    
    if expression == "speak":
        # Modern speaking mouth - rounded rectangle with depth
        mouth_w = base_w + int(10 * SS * math.sin(animation_phase * 8))
        mouth_h = base_h + int(8 * SS * math.sin(animation_phase * 6))
        
        # Mouth cavity depth
        cavity_layers = 6
        for i in range(cavity_layers):
            layer_w = mouth_w - i * 2 * SS
            layer_h = mouth_h - i * 2 * SS
            if layer_w <= 0 or layer_h <= 0:
                break
            
            depth_intensity = 1.0 - (i / cavity_layers)
            depth_color = (int(DEPTH_SHADOW[0] * depth_intensity),
                          int(DEPTH_SHADOW[1] * depth_intensity),
                          int(DEPTH_SHADOW[2] * depth_intensity))
            
            d.ellipse([cx - layer_w//2, cy - layer_h//2, 
                      cx + layer_w//2, cy + layer_h//2], fill=depth_color)
        
        # Lip highlight
        lip_thickness = 3 * SS
        d.ellipse([cx - mouth_w//2, cy - mouth_h//2, 
                  cx + mouth_w//2, cy + mouth_h//2], 
                 outline=LIGHT_GRAY, width=lip_thickness)
        
        # Inner lip detail
        inner_w = mouth_w - 6 * SS
        inner_h = mouth_h - 6 * SS
        d.ellipse([cx - inner_w//2, cy - inner_h//2, 
                  cx + inner_w//2, cy + inner_h//2], 
                 outline=MID_GRAY, width=SS)
    
    elif expression == "smile":
        # 3D smile curve with depth
        smile_w = int(base_w * 1.3)
        curve_height = int(base_h * 0.8)
        
        # Create smile curve points
        smile_points = []
        steps = 48
        for i in range(steps + 1):
            t = i / steps
            angle = math.pi * t
            x = cx + (smile_w//2) * math.cos(angle)
            curve_depth = math.sin(angle) * curve_height
            y = cy + int(curve_depth * (0.8 + 0.2 * math.sin(animation_phase * 3)))
            smile_points.append((x, y))
        
        # Shadow
        shadow_points = [(x + SS, y + 2*SS) for x, y in smile_points]
        d.line(shadow_points, fill=DEPTH_SHADOW, width=6*SS, joint="curve")
        
        # Main smile
        d.line(smile_points, fill=WHITE, width=4*SS, joint="curve")
        
        # Highlight
        highlight_points = [(x, y - SS) for x, y in smile_points]
        d.line(highlight_points, fill=LIGHT_GRAY, width=2*SS, joint="curve")
    
    elif expression == "surprised":
        # Perfect circle with 3D depth for surprise
        surprise_scale = 1.0 + 0.2 * math.sin(animation_phase * 4)
        mouth_r = int(base_w//2 * surprise_scale)
        
        # Depth layers
        depth_layers = 8
        for i in range(depth_layers):
            layer_r = mouth_r - i * SS
            if layer_r <= 0:
                break
            
            depth_val = 1.0 - (i / depth_layers) * 0.8
            depth_color = (int(DEPTH_SHADOW[0] * depth_val),
                          int(DEPTH_SHADOW[1] * depth_val),
                          int(DEPTH_SHADOW[2] * depth_val))
            
            d.ellipse([cx - layer_r, cy - layer_r, cx + layer_r, cy + layer_r], 
                     fill=depth_color)
        
        # Lip rim
        d.ellipse([cx - mouth_r, cy - mouth_r, cx + mouth_r, cy + mouth_r], 
                 outline=WHITE, width=3*SS)
        d.ellipse([cx - mouth_r, cy - mouth_r, cx + mouth_r, cy + mouth_r], 
                 outline=LIGHT_GRAY, width=SS)
    
    else:  # neutral
        # Small modern mouth
        mouth_w = int(base_w * 0.6)
        mouth_h = int(base_h * 0.5)
        
        # 3D small mouth
        draw_3d_circle(d, cx, cy, mouth_w//2, WHITE, 135)


def draw_realistic_listening_hand(img, cx, cy, *, scale=1.0, animation_phase=0):
    """Anatomically correct listening hand - cupped behind ear like a real person"""
    d = ImageDraw.Draw(img)
    
    base_scale = scale * SS
    
    # === STEP 1: ANATOMICAL PALM ===
    # Real hand proportions - palm is roughly 3:4 ratio
    palm_width = int(42 * base_scale)
    palm_height = int(56 * base_scale)
    
    # Palm is tilted at listening angle (like cupping your ear)
    palm_tilt = -12 + 2 * math.sin(animation_phase * 1.8)  # Gentle rocking
    
    # Draw palm with proper hand anatomy
    palm_points = []
    # Create hand-shaped palm (not just oval)
    angles = [0, 45, 90, 135, 180, 225, 270, 315]  # 8 points for hand shape
    for i, angle in enumerate(angles):
        rad = math.radians(angle + palm_tilt)
        
        # Hand-like proportions (narrower at wrist, wider at knuckles)
        if 45 <= angle <= 135:  # Top (knuckle area)
            radius_x = palm_width * 0.5
            radius_y = palm_height * 0.45
        elif 225 <= angle <= 315:  # Bottom (wrist area)  
            radius_x = palm_width * 0.35
            radius_y = palm_height * 0.5
        else:  # Sides
            radius_x = palm_width * 0.45
            radius_y = palm_height * 0.48
            
        x = cx + radius_x * math.cos(rad)
        y = cy + radius_y * math.sin(rad)
        palm_points.append((x, y))
    
    # Draw 3D palm with depth
    for depth in range(6, 0, -1):
        depth_color = (int(255 - depth * 25), int(255 - depth * 25), int(255 - depth * 25))
        offset_points = [(x + depth, y + depth) for x, y in palm_points]
        try:
            d.polygon(offset_points, fill=depth_color)
        except:
            pass
    
    # Main palm
    try:
        d.polygon(palm_points, fill=WHITE)
    except:
        # Fallback to ellipse if polygon fails
        d.ellipse([cx - palm_width//2, cy - palm_height//2, 
                  cx + palm_width//2, cy + palm_height//2], fill=WHITE)
    
    # === STEP 2: REALISTIC FINGERS ===
    # Real finger proportions and positions for "cupping ear" gesture
    finger_data = [
        # (name, base_x_offset, base_y_offset, length, width, base_angle, curl_amount)
        ("pinky", -0.35, -0.42, 32, 7, -25, 35),      # Pinky curves most
        ("ring", -0.12, -0.48, 38, 8, -15, 25),       # Ring finger
        ("middle", 0.08, -0.5, 42, 9, -8, 15),        # Middle (longest)
        ("index", 0.25, -0.45, 39, 8, 0, 20),         # Index points up
    ]
    
    furthest_extent = palm_width//2
    
    for name, rel_x, rel_y, length, width, base_angle, curl in finger_data:
        # Animate each finger independently for natural movement
        finger_hash = hash(name) % 100
        
        # Listening animation - fingers gently adjust as if focusing on sound
        angle_wobble = 3 * math.sin(animation_phase * 2.1 + finger_hash * 0.1)
        length_pulse = 0.95 + 0.08 * math.sin(animation_phase * 1.7 + finger_hash * 0.2)
        curl_adjust = curl + 5 * math.sin(animation_phase * 1.3 + finger_hash * 0.15)
        
        finger_angle = base_angle + angle_wobble
        finger_length = int(length * base_scale * length_pulse)
        finger_width = int(width * base_scale)
        
        # Finger base position on palm edge
        base_x = cx + int(rel_x * palm_width)
        base_y = cy + int(rel_y * palm_height)
        
        # === DRAW FINGER WITH JOINTS ===
        # Real fingers have 3 segments (phalanges)
        segments = 3
        segment_length = finger_length // segments
        
        current_x, current_y = base_x, base_y
        current_angle = finger_angle
        
        for segment in range(segments):
            # Each segment curves more (like real finger joints)
            segment_curve = curl_adjust * (segment + 1) / segments
            segment_angle = current_angle + segment_curve
            
            # Segment end position
            end_x = current_x + int(segment_length * math.cos(math.radians(segment_angle)))
            end_y = current_y + int(segment_length * math.sin(math.radians(segment_angle)))
            
            # Finger segment thickness decreases toward tip
            seg_width = int(finger_width * (1.0 - segment * 0.15))
            
            # Draw 3D finger segment
            # Shadow
            d.line([(current_x + SS, current_y + SS), (end_x + SS, end_y + SS)], 
                   fill=DEPTH_SHADOW, width=seg_width + SS, joint="curve")
            
            # Main segment
            d.line([(current_x, current_y), (end_x, end_y)], 
                   fill=WHITE, width=seg_width, joint="curve")
            
            # Highlight
            highlight_width = max(SS, seg_width // 3)
            d.line([(current_x - SS//2, current_y - SS//2), (end_x - SS//2, end_y - SS//2)], 
                   fill=LIGHT_GRAY, width=highlight_width, joint="curve")
            
            # Knuckle joints (small circles at joints)
            if segment < segments - 1:  # Not at fingertip
                joint_r = seg_width // 3
                if joint_r > SS//2:
                    d.ellipse([end_x - joint_r, end_y - joint_r, 
                              end_x + joint_r, end_y + joint_r], 
                             fill=LIGHT_GRAY, outline=MID_GRAY, width=SS//2)
            
            # Move to next segment
            current_x, current_y = end_x, end_y
            current_angle = segment_angle
        
        # Fingertip with nail
        tip_radius = seg_width // 2
        if tip_radius > SS:
            # Fingertip
            d.ellipse([current_x - tip_radius, current_y - tip_radius,
                      current_x + tip_radius, current_y + tip_radius], 
                     fill=WHITE)
            
            # Fingernail
            nail_r = tip_radius // 2
            nail_x = current_x - int(tip_radius * 0.3 * math.cos(math.radians(current_angle)))
            nail_y = current_y - int(tip_radius * 0.3 * math.sin(math.radians(current_angle)))
            d.ellipse([nail_x - nail_r, nail_y - nail_r, 
                      nail_x + nail_r, nail_y + nail_r], 
                     fill=LIGHT_GRAY, outline=MID_GRAY, width=SS//2)
        
        # Track hand extent
        furthest_extent = max(furthest_extent, abs(current_x - cx) + tip_radius)
    
    # === STEP 3: ANATOMICALLY CORRECT THUMB ===
    # Thumb opposes fingers and curves inward for cupping
    thumb_base_x = cx - int(palm_width * 0.38)
    thumb_base_y = cy + int(palm_height * 0.15)
    
    # Thumb animation - slight adjustment like focusing
    thumb_curl = 25 + 4 * math.sin(animation_phase * 1.9)
    thumb_length = int(34 * base_scale * (0.98 + 0.04 * math.sin(animation_phase * 2.3)))
    
    # Thumb has 2 main segments
    thumb_seg1_len = int(thumb_length * 0.6)
    thumb_seg2_len = int(thumb_length * 0.4)
    thumb_width = int(11 * base_scale)
    
    # First thumb segment (from palm)
    thumb_angle1 = 15
    thumb_mid_x = thumb_base_x + int(thumb_seg1_len * math.cos(math.radians(thumb_angle1)))
    thumb_mid_y = thumb_base_y - int(thumb_seg1_len * math.sin(math.radians(thumb_angle1)))
    
    # Second thumb segment (curves inward)
    thumb_angle2 = thumb_angle1 + thumb_curl
    thumb_tip_x = thumb_mid_x + int(thumb_seg2_len * math.cos(math.radians(thumb_angle2)))
    thumb_tip_y = thumb_mid_y - int(thumb_seg2_len * math.sin(math.radians(thumb_angle2)))
    
    # Draw thumb segments
    for (start_x, start_y), (end_x, end_y), width_mult in [
        ((thumb_base_x, thumb_base_y), (thumb_mid_x, thumb_mid_y), 1.0),
        ((thumb_mid_x, thumb_mid_y), (thumb_tip_x, thumb_tip_y), 0.8)
    ]:
        seg_width = int(thumb_width * width_mult)
        
        # 3D thumb segment
        d.line([(start_x + SS, start_y + SS), (end_x + SS, end_y + SS)], 
               fill=DEPTH_SHADOW, width=seg_width + SS, joint="curve")
        d.line([(start_x, start_y), (end_x, end_y)], 
               fill=WHITE, width=seg_width, joint="curve")
        d.line([(start_x - SS//2, start_y - SS//2), (end_x - SS//2, end_y - SS//2)], 
               fill=LIGHT_GRAY, width=max(SS, seg_width//3), joint="curve")
    
    # Thumb tip and nail
    thumb_tip_r = thumb_width // 3
    if thumb_tip_r > SS//2:
        d.ellipse([thumb_tip_x - thumb_tip_r, thumb_tip_y - thumb_tip_r,
                  thumb_tip_x + thumb_tip_r, thumb_tip_y + thumb_tip_r], fill=WHITE)
        
        # Thumb nail
        nail_r = thumb_tip_r // 2  
        d.ellipse([thumb_tip_x - nail_r, thumb_tip_y - nail_r,
                  thumb_tip_x + nail_r, thumb_tip_y + nail_r], 
                 fill=LIGHT_GRAY, outline=MID_GRAY, width=SS//2)
    
    # === STEP 4: CUPPING DEPTH INDICATION ===
    # Add visual cues that show the hand is cupped (hollow for sound)
    try:
        # Inner shadow showing the cupped hollow
        cup_points = [
            (cx - palm_width//4, cy - palm_height//4),
            (cx + palm_width//8, cy - palm_height//3),
            (cx + palm_width//6, cy + palm_height//8),
            (cx - palm_width//8, cy + palm_height//4),
        ]
        d.polygon(cup_points, fill=DEPTH_SHADOW)
        
        # Subtle reflection lines showing the curved palm surface
        for i in range(3):
            curve_y = cy - palm_height//4 + i * palm_height//8
            curve_start_x = cx - palm_width//4 + i * 2
            curve_end_x = cx + palm_width//6 - i * 2
            d.line([(curve_start_x, curve_y), (curve_end_x, curve_y)], 
                   fill=MID_GRAY, width=SS//2)
    except:
        pass  # Skip decorative elements if they cause issues
    
    return furthest_extent

# ------------------ Properly Aligned Expression Frames ------------------

def frame_thinking(size, t):
    """Thinking expression with perfect alignment"""
    img, W, H = aa_img(size)
    d = ImageDraw.Draw(img)
    cx, cy = W//2, H//2
    
    # Properly spaced eyes
    eye_spacing = int(0.35 * W)  # Increased spacing to prevent overlap
    eye_y = cy - int(0.06 * H)
    eye_r = int(0.08 * W)  # Reduced size to fit better
    
    # Ensure eyes don't go off screen
    left_eye_x = max(eye_r + 4*SS, cx - eye_spacing//2)
    right_eye_x = min(W - eye_r - 4*SS, cx + eye_spacing//2)
    
    # Breathing animation
    breathe = 1.0 + 0.02 * math.sin(t * 2.5)
    scaled_eye_r = int(eye_r * breathe)
    
    # Looking up and slightly right
    look_x = int(scaled_eye_r * 0.3)
    look_y = -int(scaled_eye_r * 0.4)
    drift_x = int(scaled_eye_r * 0.1 * math.sin(t * 0.8))
    drift_y = int(scaled_eye_r * 0.1 * math.cos(t * 1.2))
    
    pupil_pos = (look_x + drift_x, look_y + drift_y)
    
    # Draw eyes
    draw_3d_eye(d, left_eye_x, eye_y, scaled_eye_r, pupil_pos)
    draw_3d_eye(d, right_eye_x, eye_y, scaled_eye_r, pupil_pos)
    
    # Properly positioned brows (no overlap)
    brow_y = eye_y - scaled_eye_r - int(0.04 * H)
    brow_width = int(0.12 * W)
    brow_height = int(0.02 * H)
    
    # Left brow
    draw_3d_brow(d, left_eye_x, brow_y, brow_width, brow_height, 0.1)
    # Right brow  
    draw_3d_brow(d, right_eye_x, brow_y, brow_width, brow_height, -0.1)
    
    # Mouth positioned properly below
    mouth_y = cy + int(0.12 * H)
    draw_3d_mouth(d, cx, mouth_y, "neutral", 0.8, t)
    
    return img

def frame_happy(size, t):
    """Happy expression with perfect spacing"""
    img, W, H = aa_img(size)
    d = ImageDraw.Draw(img)
    cx, cy = W//2, H//2
    
    # Well-spaced happy eyes
    eye_spacing = int(0.32 * W)
    eye_y = cy - int(0.05 * H)
    eye_r = int(0.075 * W)
    
    left_eye_x = max(eye_r + 4*SS, cx - eye_spacing//2)
    right_eye_x = min(W - eye_r - 4*SS, cx + eye_spacing//2)
    
    # Happy bounce
    bounce = 1.0 + 0.08 * math.sin(t * 4)
    scaled_eye_r = int(eye_r * bounce)
    
    # Happy squint
    blink = 0.75 + 0.15 * math.sin(t * 3)
    
    # Looking at viewer with joy wiggle
    wiggle_x = int(2 * SS * math.sin(t * 5))
    wiggle_y = int(1 * SS * math.cos(t * 4))
    pupil_pos = (wiggle_x, wiggle_y)
    
    # Draw squinted eyes
    draw_3d_eye(d, left_eye_x, eye_y, scaled_eye_r, pupil_pos, blink)
    draw_3d_eye(d, right_eye_x, eye_y, scaled_eye_r, pupil_pos, blink)
    
    # Raised happy brows
    brow_y = eye_y - scaled_eye_r - int(0.05 * H)
    brow_width = int(0.10 * W)
    brow_height = int(0.025 * H)
    
    draw_3d_brow(d, left_eye_x, brow_y, brow_width, brow_height, 0.2)
    draw_3d_brow(d, right_eye_x, brow_y, brow_width, brow_height, -0.2)
    
    # Big smile
    mouth_y = cy + int(0.15 * H)
    draw_3d_mouth(d, cx, mouth_y, "smile", 1.2, t)
    
    return img

def frame_listening_enhanced(size, t):
    """Pure listening expression - focused, attentive, no hand needed"""
    img, W, H = aa_img(size)
    d = ImageDraw.Draw(img)
    cx, cy = W//2, H//2
    
    # === LISTENING EYES - Focused and Attentive ===
    eye_spacing = int(0.28 * W)
    eye_y = cy - int(0.08 * H)
    eye_r = int(0.09 * W)
    
    left_eye_x = cx - eye_spacing//2
    right_eye_x = cx + eye_spacing//2
    
    # Ensure eyes stay on screen
    left_eye_x = max(eye_r + 4*SS, left_eye_x)
    right_eye_x = min(W - eye_r - 4*SS, right_eye_x)
    
    # LISTENING BEHAVIOR: Eyes track slowly left to right as if following conversation
    listening_cycle = 8.0  # Slow tracking for attentive listening
    track_phase = math.sin(math.pi * t / listening_cycle)
    
    # Occasional focused blinks (longer, more thoughtful)
    blink_cycle = t % 6
    if 5.5 <= blink_cycle <= 5.8:
        blink_factor = 0.3  # Thoughtful partial blink
    elif blink_cycle > 5.8:
        blink_factor = max(0.1, 1.0 - (blink_cycle - 5.8) * 5)  # Slow close
    else:
        blink_factor = 1.0
    
    # Listening pupil behavior - tracks sound source
    track_range = int(eye_r * 0.4)
    pupil_x = int(track_phase * track_range)
    # Slight upward focus (looking at speaker's face level)
    pupil_y = -int(eye_r * 0.15) + int(2 * SS * math.sin(t * 0.8))
    
    pupil_pos = (pupil_x, pupil_y)
    
    # Draw attentive eyes
    try:
        draw_3d_eye(d, left_eye_x, eye_y, eye_r, pupil_pos, blink_factor)
        draw_3d_eye(d, right_eye_x, eye_y, eye_r, pupil_pos, blink_factor)
    except:
        d.ellipse([left_eye_x - eye_r, eye_y - eye_r, left_eye_x + eye_r, eye_y + eye_r], fill=WHITE)
        d.ellipse([right_eye_x - eye_r, eye_y - eye_r, right_eye_x + eye_r, eye_y + eye_r], fill=WHITE)
    
    # === LISTENING BROWS - Slightly Raised in Attention ===
    brow_y = eye_y - eye_r - int(0.05 * H)
    brow_width = int(0.11 * W)
    brow_height = int(0.025 * H)
    
    # Gentle attention lift with micro-expressions
    attention_lift = 0.02 + 0.005 * math.sin(t * 1.2)
    
    try:
        draw_3d_brow(d, left_eye_x, brow_y - int(attention_lift * H), 
                    brow_width, brow_height, 0.1)
        draw_3d_brow(d, right_eye_x, brow_y - int(attention_lift * H), 
                    brow_width, brow_height, 0.1)
    except:
        d.line([(left_eye_x - brow_width//2, brow_y), (left_eye_x + brow_width//2, brow_y)], 
               fill=WHITE, width=4*SS)
        d.line([(right_eye_x - brow_width//2, brow_y), (right_eye_x + brow_width//2, brow_y)], 
               fill=WHITE, width=4*SS)
    
    # === LISTENING MOUTH - Small, Closed, Neutral ===
    # When listening, mouth should be still and neutral
    mouth_y = cy + int(0.15 * H)
    
    # Small neutral mouth - just a gentle line or small oval
    mouth_w = int(20 * SS)
    mouth_h = int(6 * SS)
    
    # Very subtle "concentration" micro-movement
    concentration = 0.98 + 0.04 * math.sin(t * 0.7)
    
    try:
        draw_3d_mouth(d, cx, mouth_y, "neutral", 0.6 * concentration, t)
    except:
        # Fallback - simple neutral mouth
        d.ellipse([cx - mouth_w//2, mouth_y - mouth_h//2, 
                  cx + mouth_w//2, mouth_y + mouth_h//2], fill=WHITE)
    
    # === LISTENING VISUAL CUES ===
    # Add subtle visual indicators that Orbi is actively listening
    
    # 1. Subtle "sound wave" indicators near the ears (optional)
    if t % 3 < 0.5:  # Occasional subtle indication
        wave_alpha = math.sin(t * 12) * 0.3 + 0.7
        wave_color = (int(ACCENT_BLUE[0] * wave_alpha),
                     int(ACCENT_BLUE[1] * wave_alpha),
                     int(ACCENT_BLUE[2] * wave_alpha))
        
        # Small arc near ear area
        ear_x = cx + int(0.4 * W)
        ear_y = cy
        
        try:
            for i in range(2):
                arc_r = int((15 + i * 8) * SS)
                d.ellipse([ear_x - arc_r, ear_y - arc_r, 
                          ear_x + arc_r, ear_y + arc_r], 
                         outline=wave_color, width=SS//2)
        except:
            pass
    
    return img

def frame_speaking(size, t):
    """Speaking expression - animated mouth, engaging eyes"""
    img, W, H = aa_img(size)
    d = ImageDraw.Draw(img)
    cx, cy = W//2, H//2
    
    # === SPEAKING EYES - Engaging and Expressive ===
    eye_spacing = int(0.26 * W)
    eye_y = cy - int(0.06 * H)
    eye_r = int(0.08 * W)
    
    left_eye_x = cx - eye_spacing//2
    right_eye_x = cx + eye_spacing//2
    
    # Ensure eyes stay on screen
    left_eye_x = max(eye_r + 4*SS, left_eye_x)
    right_eye_x = min(W - eye_r - 4*SS, right_eye_x)
    
    # SPEAKING BEHAVIOR: More animated, engaging eye movement
    speaking_energy = 1.0 + 0.1 * math.sin(t * 3.5)
    scaled_eye_r = int(eye_r * speaking_energy)
    
    # Regular expressive blinking
    blink_cycle = t % 3.5
    if blink_cycle > 3.2:
        blink_factor = max(0.2, 1.0 - (blink_cycle - 3.2) * 3)
    else:
        blink_factor = 1.0
    
    # Speaking pupil behavior - looks at listener, more direct
    pupil_x = int(3 * SS * math.sin(t * 2.8))  # Slight horizontal movement
    pupil_y = int(2 * SS * math.sin(t * 2.1))  # Slight vertical movement
    pupil_pos = (pupil_x, pupil_y)
    
    # Draw expressive speaking eyes
    try:
        draw_3d_eye(d, left_eye_x, eye_y, scaled_eye_r, pupil_pos, blink_factor)
        draw_3d_eye(d, right_eye_x, eye_y, scaled_eye_r, pupil_pos, blink_factor)
    except:
        d.ellipse([left_eye_x - scaled_eye_r, eye_y - scaled_eye_r, 
                  left_eye_x + scaled_eye_r, eye_y + scaled_eye_r], fill=WHITE)
        d.ellipse([right_eye_x - scaled_eye_r, eye_y - scaled_eye_r, 
                  right_eye_x + scaled_eye_r, eye_y + scaled_eye_r], fill=WHITE)
    
    # === SPEAKING BROWS - More Animated ===
    brow_y = eye_y - scaled_eye_r - int(0.04 * H)
    brow_width = int(0.10 * W)
    brow_height = int(0.022 * H)
    
    # Expressive brow movement while speaking
    expression_lift = 0.01 + 0.015 * math.sin(t * 2.5)
    
    try:
        draw_3d_brow(d, left_eye_x, brow_y - int(expression_lift * H), 
                    brow_width, brow_height, 0.05)
        draw_3d_brow(d, right_eye_x, brow_y - int(expression_lift * H), 
                    brow_width, brow_height, 0.05)
    except:
        d.line([(left_eye_x - brow_width//2, brow_y), (left_eye_x + brow_width//2, brow_y)], 
               fill=WHITE, width=4*SS)
        d.line([(right_eye_x - brow_width//2, brow_y), (right_eye_x + brow_width//2, brow_y)], 
               fill=WHITE, width=4*SS)
    
    # === SPEAKING MOUTH - Actively Animated ===
    mouth_y = cy + int(0.12 * H)
    
    # Dynamic speaking mouth animation
    speech_intensity = 0.8 + 0.4 * math.sin(t * 6.5)  # Fast speech rhythm
    speech_shape = math.sin(t * 8.2)  # Shape changes
    
    # Alternate between different mouth shapes for natural speech
    mouth_phase = (t * 4) % 1.0
    
    if mouth_phase < 0.3:
        # Open vowel sound
        try:
            draw_3d_mouth(d, cx, mouth_y, "speak", speech_intensity, t)
        except:
            mouth_r = int(12 * SS * speech_intensity)
            d.ellipse([cx - mouth_r, mouth_y - mouth_r, cx + mouth_r, mouth_y + mouth_r], 
                     outline=WHITE, width=2*SS)
    elif mouth_phase < 0.6:
        # Consonant shape - smaller, more horizontal
        mouth_w = int(25 * SS * speech_intensity)
        mouth_h = int(8 * SS)
        try:
            d.ellipse([cx - mouth_w//2, mouth_y - mouth_h//2, 
                      cx + mouth_w//2, mouth_y + mouth_h//2], 
                     outline=WHITE, width=2*SS)
        except:
            pass
    else:
        # Brief pause/transition
        mouth_w = int(15 * SS)
        mouth_h = int(5 * SS)
        try:
            d.ellipse([cx - mouth_w//2, mouth_y - mouth_h//2, 
                      cx + mouth_w//2, mouth_y + mouth_h//2], fill=WHITE)
        except:
            pass
    
    return img

def frame_surprised(size, t):
    """Surprised expression with NO eye overlap"""
    img, W, H = aa_img(size)
    d = ImageDraw.Draw(img)
    cx, cy = W//2, H//2
    
    # Large surprised eyes with proper spacing
    surprise_scale = 1.15 + 0.08 * math.sin(t * 8)
    eye_r = int(0.085 * W * surprise_scale)
    
    # Calculate spacing to prevent overlap with margin
    min_spacing = (eye_r * 2) + int(0.08 * W)  # Minimum gap between eyes
    eye_spacing = max(min_spacing, int(0.4 * W))
    
    # Position eyes ensuring they stay on screen
    left_eye_x = max(eye_r + 4*SS, cx - eye_spacing//2)
    right_eye_x = min(W - eye_r - 4*SS, cx + eye_spacing//2)
    
    # Final check - if still overlapping, reduce eye size
    actual_spacing = right_eye_x - left_eye_x
    if actual_spacing < eye_r * 2:
        eye_r = int(actual_spacing * 0.4)  # Make eyes smaller to fit
    
    eye_y = cy - int(0.08 * H)
    
    # Wide-eyed surprise look (pupils slightly up and dilated)
    pupil_pos = (0, -int(eye_r * 0.15))
    
    # Draw surprised eyes
    draw_3d_eye(d, left_eye_x, eye_y, eye_r, pupil_pos, 1.0)
    draw_3d_eye(d, right_eye_x, eye_y, eye_r, pupil_pos, 1.0)
    
    # High surprised brows
    brow_y = eye_y - eye_r - int(0.06 * H)
    brow_raise = 0.04 + 0.01 * math.sin(t * 4)
    brow_width = int(0.11 * W)
    brow_height = int(0.03 * H)
    
    draw_3d_brow(d, left_eye_x, brow_y - int(brow_raise * H), brow_width, brow_height, 0.3)
    draw_3d_brow(d, right_eye_x, brow_y - int(brow_raise * H), brow_width, brow_height, -0.3)
    
    # Surprised open mouth
    mouth_y = cy + int(0.15 * H)
    draw_3d_mouth(d, cx, mouth_y, "surprised", 1.0, t)
    
    return img

# ------------------ Main animator with smooth transitions ------------------

def show(disp, pil_img):
    disp.ShowImage(pil_img.rotate(180))

def run_expression(disp, name, seconds=4.0, fps=50):
    """Ultra-smooth animation with higher frame rate"""
    size = (disp.height, disp.width)
    t0 = time.time()
    prev = t0
    
    while True:
        now = time.time()
        t = now - t0
        dt = now - prev
        prev = now
        
        if name == "thinking":
            frame = frame_thinking(size, t)
        elif name == "happy":
            frame = frame_happy(size, t)
        elif name == "listen":
            frame = frame_listening_enhanced(size, t)
        elif name == "speaking":
            frame = frame_speaking(size, t)
        elif name == "surprised":
            frame = frame_surprised(size, t)
        else:
            # Default to thinking
            frame = frame_thinking(size, t)
        
        out = down(frame, size)
        show(disp, out)
        
        # Ultra-smooth frame pacing
        time.sleep(max(0.0, 1.0/fps - (time.time()-now)))
        
        if t >= seconds:
            break

def main():
    disp = LCD_2inch.LCD_2inch(spi=SPI.SpiDev(0,0), rst=RST, dc=DC, bl=BL)
    disp.Init()
    disp.clear()
    disp.bl_Frequency(2400)
    disp.bl_DutyCycle(98)  # Maximum brightness for crisp 3D effects
    
    try:
        if len(sys.argv) > 1:
            expression = sys.argv[1].lower()
            run_expression(disp, expression, seconds=9999)
        else:
            # Professional presentation playlist
            playlist = [
                ("thinking", 6.0),
                ("happy", 4.5),
                ("listen", 7.0),
                ("speaking", 4.0),
                ("surprised", 3.5),
            ]
            
            while True:
                for name, secs in playlist:
                    # Add subtle timing variation for natural feel
                    varied_time = secs + random.uniform(-0.3, 0.8)
                    run_expression(disp, name, seconds=varied_time)
                    
    except KeyboardInterrupt:
        pass
    finally:
        disp.module_exit()

if __name__ == "__main__":
    main()
