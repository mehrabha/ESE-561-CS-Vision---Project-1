import numpy as np
import cv2
import math

# ==========================================
# CONSTANTS & SETUP (From Project Specs)
# ==========================================
F = 40.0                # Focal length in mm
P = 0.01                # Pixel size in mm
ROWS, COLS = 600, 600   # Image size in pixels

MAX_TIME = 10.0          # Max time in seconds
DT = 0.1                # Time step
T0 = np.array([0.0, 0.0, 500.0]) # Initial translation
VELOCITY = np.array([5.0, 15.0, 0.0]) 
ACC = np.array([0.0, -5.0, 0.0])
THETA0 = 0.0
W0 = 45.0               # Angular velocity in degrees/second

# Cube Vertices (WCS)
L = 10.0                # Edge length
V1 = np.array([0, 0, 0])
V2 = np.array([0, L, 0])
V3 = np.array([L, L, 0])
V4 = np.array([L, 0, 0])
V5 = np.array([L, 0, L])
V6 = np.array([0, L, L])
V7 = np.array([0, 0, L])
V8 = np.array([L, L, L])
vertices = [V1, V2, V3, V4, V5, V6, V7, V8]
edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), # Front face (v1-v2, v2-v3, v3-v4, v4-v1)
        (6, 5), (5, 7), (7, 4), (4, 6), # Back face (v7-v6, v6-v8, v8-v5, v5-v7)
        (0, 6), (1, 5), (2, 7), (3, 4)  # Connecting edges (v1-v7, v2-v6, v3-v8, v4-v5)
    ]

# Load texture image
tex_img = cv2.imread('einstein50x50v.jpg', cv2.IMREAD_GRAYSCALE)
if tex_img is None:
    tex_img = np.ones((50, 50), dtype=np.uint8) * 128
tex_rows, tex_cols = tex_img.shape

# part 2 changes
bg_img = cv2.imread('background.jpg', cv2.IMREAD_GRAYSCALE)
bg_img_bgr = cv2.cvtColor(bg_img, cv2.COLOR_GRAY2BGR)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def MapIndex(x, y, rows, cols):
    """Converts 2D physical coordinates (mm) to pixel indices."""
    col = int(round(x)) + cols // 2
    row = rows // 2 - int(round(y))
    return col, row

def Map2Da(P3D, K):
    """Projects a 3D point in CCS to 2D physical image coordinates."""
    if P3D[2] <= 0:
        return None, None
        
    P2D_homo = np.dot(K, P3D)
    x = P2D_homo[0] / P2D_homo[2]
    
    # TODO: Task 1 - Fill in the missing piece of Map2Da which concerns the y-value for the returned 2D point.
    y = P2D_homo[1] / P2D_homo[2]
    
    return x, y

def DrawLine(img, q1, q2, color):
    """Draws a line between two pixel coordinates."""
    if q1 is None or q2 is None:
        return
        
    # TODO: Task 2 - Replace the call to cv2.line() with your own code.
    q1 = np.array(q1, dtype=float)
    q2 = np.array(q2, dtype=float)

    diff = q2 - q1
    d = np.linalg.norm(diff)

    if d == 0:
        x = int(round(q1[0]))
        y = int(round(q1[1]))

        if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
            img[y, x] = color
    else:
        u = diff / d
        for c in np.arange(0.0, d + 0.5, 0.5):
            q = q1 + c * u
            x = int(round(q[0]))
            y = int(round(q[1]))
            if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                img[y, x] = color
    

# ==========================================
# PRE-COMPUTATIONS (Tasks 3 - 5)
# ==========================================

# TODO: Task 3 - Compute the axis of rotation N0 (unit vector for V8-V1) and construct matrix N.
N0 = V8 - V1
N0 = N0 / np.linalg.norm(N0)
nx, ny, nz = N0
N = [
    [0, -nz, ny],
    [nz, 0, -nx],
    [-ny, nx, 0]
]

# TODO: Task 4 - Initialize the intrinsic matrix K given focal length f.
K = [
    [F / P, 0, 0],
    [0, F / P, 0],
    [0, 0, 1]
]

# TODO: Task 5 - Compute u21 and u41, the unit vectors corresponding to (V2-V1) and (V4-V1).
u21_vec = V2 - V1
u21 = u21_vec / np.linalg.norm(u21_vec)

u41_vec = V4 - V1
u41 = u41_vec / np.linalg.norm(u41_vec)

# ==========================================
# MAIN ANIMATION LOOP
# ==========================================
out_video = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (COLS, ROWS))

for t in np.arange(0.0, MAX_TIME, DT):
    # part 2 changes
    frame = bg_img_bgr.copy()
    
    # Update translation
    T = T0 + VELOCITY * t + 0.5 * ACC * t**2
    
    # TODO: Task 7 - Compute the rotation matrix R as defined in eq 2.34 (using theta, N).
    theta = THETA0 + W0 * t
    theta_rad = math.radians(theta)
    N1 = np.array(N)
    N2 = N1 @ N1
    I = np.eye(3)
    K1 = np.array(K)

    R = I + math.sin(theta_rad) * N1 + (1 - math.cos(theta_rad)) * N2

    # part 2 changes
    eyes = [
        {'T_eye': T, 'color': (0, 0, 255), 'channel': 2}, # Camera 1 (Red)
        {'T_eye': T - np.array([10.0, 0.0, 0.0]), 'color': (255, 0, 0), 'channel': 0} # Camera 2 (Blue)
    ]

    for eye in eyes:
        T_eye = eye['T_eye']
        color = eye['color']
        channel = eye['channel']

        # TODO: Task 8 - Map all eight 3D vertices to their 2D counterpart using Map2Da.
        proj_vertices = []
        for V in vertices:
            P_ccs = R @ V + T_eye
            ix, iy = Map2Da(P_ccs, K1)

            if ix is not None and iy is not None:
                px, py = MapIndex(ix, iy, ROWS, COLS)
                proj_vertices.append((px, py))
            else:
                proj_vertices.append(None)

        
        # TODO: Task 9 - Use DrawLine to draw the 12 edges of the cube.
        for edge in edges:
            p1 = proj_vertices[edge[0]]
            p2 = proj_vertices[edge[1]]
            
            DrawLine(frame, p1, p2, color)


        # --- TEXTURE MAPPING ---
        for i in range(tex_rows):
            for j in range(tex_cols):
                intensity = tex_img[i, j]
                
                # TODO: Task 6 - When computing all 3D points on the face, complete the code which saves the Y and Z values.
                P_face_X = V1[0] + (j / tex_cols) * L * u41[0] + (i / tex_rows) * L * u21[0]
                P_face_Y = V1[1] + (j / tex_cols) * L * u41[1] + (i / tex_rows) * L * u21[1]
                P_face_Z = V1[2] + (j / tex_cols) * L * u41[2] + (i / tex_rows) * L * u21[2]
                
                P_face_WCS = np.array([P_face_X, P_face_Y, P_face_Z])
                P_face_CCS = R @ P_face_WCS + T_eye
                
                ix, iy = Map2Da(P_face_CCS, K1)

                if ix is not None and iy is not None:
                    # TODO: Task 10 - Compute ir and jr, which are the row and column indices of the output image.
                    jr, ir = MapIndex(ix, iy, ROWS, COLS)
                    
                    if 0 <= jr < COLS and 0 <= ir < ROWS:
                        # part 2 changes
                        pixel_color = [0, 0, 0]
                        pixel_color[channel] = intensity
                        frame[ir, jr] = tuple(pixel_color)

    out_video.write(frame)
    # cv2.imshow('Frame', frame)
    # cv2.waitKey(1)

out_video.release()