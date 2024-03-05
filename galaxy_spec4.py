import pyqtgraph.opengl as gl
import numpy as np
from PyQt5.QtWidgets import QApplication
from pyqtgraph.Qt import QtGui, QtCore
import os
import matplotlib.cm as cm
import librosa

class CustomGLViewWidget(gl.GLViewWidget):
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()

# Load the audio

audio_data, sr = librosa.load("audio_file.wav", sr=None)
hop_length = 512
total_frames = len(audio_data)
current_frame = 0

# Folder set-up
img_path = os.path.abspath('file_path')
os.makedirs(img_path, exist_ok=True)

# Create a 3D plot widget
app = QApplication([])
w = CustomGLViewWidget()
w.setGeometry(0, 0, 1920, 1080)
w.show()
w.setWindowTitle('Galaxy Scatterplot')

# Set the camera position and orientation
w.setCameraPosition(distance=20, elevation=25, azimuth=0)

# Define the number of points to plot for the galaxy and stars
n = 50000
n_stars = 10000

# Generate stars
x_stars = np.random.uniform(-50, 50, n_stars)
y_stars = np.random.uniform(-50, 50, n_stars)
z_stars = np.random.uniform(-10, 10, n_stars)
star_sizes = np.random.uniform(0.1, 2.5, n_stars)
star_colors = np.ones((n_stars, 4)) * [1, 1, 1, 0.75]  # Dim white color

# Add a central white line through the scatter plots
line_length = 30  # Adjust as needed for the size of your space
line_data = np.array([[-line_length/2, 0, 0], [line_length/2, 0, 0]])  # Line from -line_length/2 to line_length/2 along x-axis
central_line = gl.GLLinePlotItem(pos=line_data, color=(1, 1, 1, 0), width=0.1)  # White line, fully opaque
w.addItem(central_line)
central_line.rotate(90, 0, 1, 0)

# Parameters for the dynamic scatter plots
n_dynamic_scatter = 50000  # Number of dynamic scatter points
scatter_line_length = line_length  # Use the same length as the central line

# Function to create dynamic scatter points along the line with audio influence
def create_dynamic_scatter(audio_influence):
    # Adjust the scatter line length based on audio influence
    influenced_line_length = scatter_line_length * audio_influence

    # Generate linearly spaced points along the line
    scatter_x = np.linspace(-influenced_line_length / 2, influenced_line_length / 2, n_dynamic_scatter)
    scatter_y = np.zeros(n_dynamic_scatter)
    scatter_z = np.zeros(n_dynamic_scatter)

    # Calculate distance from the midpoint for noise adjustment
    mid_point = 0  # Middle of the stream
    distance_from_mid = np.abs(scatter_x - mid_point)
    max_distance_from_mid = influenced_line_length / 2
    noise_strength = np.interp(distance_from_mid, [0, max_distance_from_mid], [0, 0.25 * audio_influence])
    scatter_x += np.random.normal(0, noise_strength, n_dynamic_scatter)
    scatter_y += np.random.normal(0, noise_strength, n_dynamic_scatter)
    scatter_z += np.random.normal(0, noise_strength, n_dynamic_scatter)

    # Calculate distance from the center for fading effect and size adjustment
    distances = np.sqrt(scatter_x**2 + scatter_y**2 + scatter_z**2)
    max_distance = np.max(distances)

    # Apply colormap based on distance
    colormap = cm.get_cmap('bone_r')  # Choose a colormap
    normalized_distances = distances / max_distance
    scatter_colors = colormap(normalized_distances)

    # Adjust alpha values based on audio influence and distance
    alphas = (1 - distances / max_distance) * audio_influence
    sphere_radius = 0.1  # Radius of the central sphere
    alphas[distances < sphere_radius] = 0  # Set alpha to zero for points within the sphere
    scatter_colors[:, 3] = alphas

    # Size variation based on audio
    sizes = np.interp(distances, [0, max_distance], [0.1, 3])

    # Create scatter plot item
    dynamic_scatter = gl.GLScatterPlotItem(pos=np.column_stack((scatter_x, scatter_y, scatter_z)),
                                           size=sizes, color=scatter_colors)
    dynamic_scatter.rotate(90, 0, 1, 0)
    return dynamic_scatter


# Create initial dynamic scatter plot
dynamic_scatter = create_dynamic_scatter(1)
w.addItem(dynamic_scatter)

# Create a scatterplot item for stars and add it to the plot widget
stars = gl.GLScatterPlotItem(pos=np.column_stack((x_stars, y_stars, z_stars)), size=star_sizes, color=star_colors)
w.addItem(stars)

# Generate random x, y, and z coordinates for the galaxy points with some noise
noise_strength = 1
x = np.random.normal(size=n) + np.random.normal(0, noise_strength, n)
y = np.random.normal(size=n) + np.random.normal(0, noise_strength, n)
z = np.random.normal(size=n) + np.random.normal(0, noise_strength, n)

# Calculate the distance of each point from the origin
r = np.sqrt(x**2 + y**2 + z**2)

# Flatten the z-axis to give a galaxy-like appearance
z_flatten_factor = 0.125
# Increase flatness towards the center
flatness_increase_factor = 1.5  # Adjust this factor as needed
z *= z_flatten_factor * np.interp(r, [0, np.max(r)], [flatness_increase_factor, 1])

# Adjust size based on distance
sizes = 2 / (1.1 - r/np.max(r))
alphas = 1 - r/np.max(r)

# Define a colormap using matplotlib's colormaps
colormap = cm.get_cmap('twilight_r')

# Normalize the distances to range [0, 1] for colormap application
norm_r = r / r.max()

# Apply the colormap to normalized distances to get RGB values
colors_rgb = colormap(norm_r)

# Convert RGB to RGBA by incorporating the alphas
colors_rgba = np.zeros((n, 4))
colors_rgba[:, :3] = colors_rgb[:, :3]
colors_rgba[:, 3] = alphas

# Set the colors for the scatter plot item
sp = gl.GLScatterPlotItem(pos=np.column_stack((x, y, z)), size=sizes, color=colors_rgba)
w.addItem(sp)

# Create a sphere mesh item and add it to the plot widget
md = gl.MeshData.sphere(rows=20, cols=20)
sphere = gl.GLMeshItem(meshdata=md, smooth=True, color=(0, 0, 0, 1), shader='balloon')
sphere.translate(0, 0, 0)  # This ensures the sphere is at the center
sphere.scale(0.1, 0.1, 0.1)  # Adjust the size as needed
w.addItem(sphere)

# Define the number of cone plots and their range
n_cone = 50000
cone_range = 2

# Generate random theta and radii for polar coordinates
theta = np.random.uniform(0, 2 * np.pi, n_cone)
radii = np.sqrt(np.random.uniform(0, cone_range**2, n_cone))

# Convert polar coordinates to Cartesian coordinates
x_cone = radii * np.cos(theta)
y_cone = radii * np.sin(theta)

# Scaling factor for the cone's height
z_scale = 4

# Calculate z values for the cone using the cone equation
max_z_cone_up = z_scale * np.sqrt(x_cone**2 + y_cone**2)
max_z_cone_down = -max_z_cone_up

# Generate z-values inside the cone's void
z_cone_up = np.random.uniform(max_z_cone_up, cone_range * z_scale)
z_cone_down = np.random.uniform(-cone_range * z_scale, max_z_cone_down)

# Combine the up and down cone coordinates
x_cone_combined = np.concatenate([x_cone, x_cone])
y_cone_combined = np.concatenate([y_cone, y_cone])
z_cone_combined = np.concatenate([z_cone_up, z_cone_down])

# Set the color for the cone plots to red
cone_color = np.ones((2*n_cone, 4)) * [0, 0, 0, 1]

# Create a scatterplot item for the cone and add it to the plot widget
cone = gl.GLScatterPlotItem(pos=np.column_stack((x_cone_combined, y_cone_combined, z_cone_combined)), size=0.1, color=cone_color)
w.addItem(cone)

# Variable to store the current line
current_line = None

def create_line_to_random_red_point(audio_influence=1):
    global current_line
    if current_line:
        w.removeItem(current_line)  # remove the previous line from the plot widget

    # Randomly select one point from the cone plots
    idx = np.random.randint(2 * n_cone)
    xi, yi, zi = x_cone_combined[idx], y_cone_combined[idx], z_cone_combined[idx]
    
    # Modify the target point based on the audio influence
    xi *= audio_influence
    yi *= audio_influence
    zi *= audio_influence
    
    # Generate line points between center and target point
    num_intermediate_points = 15
    line_x = np.linspace(0, xi, num_intermediate_points)
    line_y = np.linspace(0, yi, num_intermediate_points)
    line_z = np.linspace(0, zi, num_intermediate_points)

    # Calculate distance of each point from the midpoint of the line for noise adjustment
    mid_point_idx = num_intermediate_points // 2
    distances_from_mid = np.abs(np.arange(num_intermediate_points) - mid_point_idx)
    max_distance_from_mid = np.max(distances_from_mid)
    noise_factors = np.interp(distances_from_mid, [0, max_distance_from_mid], [0.1, 0.25])  # Less noise at midpoint

    # Add noise to the line
    line_x += np.random.normal(0, noise_factors, num_intermediate_points)
    line_y += np.random.normal(0, noise_factors, num_intermediate_points)
    line_z += np.random.normal(0, noise_factors, num_intermediate_points)

    line_data = np.column_stack([line_x, line_y, line_z])
    
    current_line = gl.GLLinePlotItem(pos=line_data * 2, color=(1, 1, 1, 1), width=0.5, antialias=True)
    w.addItem(current_line)


# Call the function to initially create a line
create_line_to_random_red_point()

# Brightness range
min_brightness = 0.5
max_brightness = 1.0

counter = 0
c2 = 0
c3 = 0
c4 = 0

# Define a threshold for the low frequency intensity
threshold = 50  # Adjust based on your audio data

def update():
    global current_frame, x, y, z, r, colors_rgba, counter, c2, c3, dynamic_scatter, c4

    # Get audio influence

    chunk = audio_data[current_frame:current_frame + hop_length]

    # Compute STFT and extract magnitude
    D = np.abs(librosa.stft(chunk, n_fft=2048, hop_length=hop_length))

    # Extract the magnitude corresponding to 20-100 Hz
    min_idx = int(20 * 2048 / sr)
    max_idx = int(100 * 2048 / sr)
    avg_magnitude = np.mean(D[min_idx:max_idx])

    # Normalize the average magnitude
    audio_influence = avg_magnitude / np.max(D)

    audio_frame = audio_data[current_frame:current_frame + hop_length]
    fft_result = np.fft.fft(audio_frame)    
    freqs = np.fft.fftfreq(len(audio_frame), d=1/sr)
    low_freq_magnitude = np.abs(fft_result[(freqs >= 20) & (freqs <= 100)]).mean()

    # Adjust brightness based on audio influence
    brightness_factor = min_brightness + (max_brightness - min_brightness) * audio_influence
    adjusted_alphas = alphas * brightness_factor
    adjusted_alphas = np.clip(adjusted_alphas, 0.1, 1)  # Ensure values stay within [0, 1]

    colors_rgba[:, 3] = adjusted_alphas

    # Convert x, y to polar coordinates
    r_xy = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    # Adjust theta based on distance. Points closer to the origin will rotate faster.
    rotation_speed = 0.025 
    theta += rotation_speed * (1 - r_xy / r_xy.max()) * audio_influence
    
    # Convert back to Cartesian coordinates
    x = r_xy * np.cos(theta)
    y = r_xy * np.sin(theta)

    sp.setData(pos=np.column_stack((x, y, z)), size=sizes, color=colors_rgba)

    # Update the line to a new random point
    create_line_to_random_red_point(audio_influence)

    counter += 1/32
    c2 += 1/128
    c3 += 1
    c4 += 1/64

    distance = 25 + 10 * np.sin(c2)
    #'''
    # Apply noise to camera position based on low frequency intensity
    if low_freq_magnitude > threshold:
        noise_azimuth = np.random.uniform(-0.25, 0.25)
        noise_elevation = np.random.uniform(-0.25, 0.25)
        w.opts['azimuth'] += noise_azimuth
        w.opts['elevation'] += noise_elevation

    else:
        w.setCameraPosition(elevation=25, azimuth=counter)
    #'''
    
    # Remove old scatter plot and create a new one
    w.removeItem(dynamic_scatter)
    dynamic_scatter = create_dynamic_scatter(audio_influence)
    w.addItem(dynamic_scatter)

    w.setCameraPosition(distance=distance, azimuth=counter)
    stars.setData(size=star_sizes * audio_influence, color=star_colors * audio_influence)
    
    w.grabFramebuffer().save(os.path.join(img_path, f'img_{c3}.png'))
    print(c3)

    current_frame += hop_length
    if current_frame >= total_frames - hop_length:
        t.stop()

t = QtCore.QTimer()
t.timeout.connect(update)
t.start(17) #17ms = 60fps

# Start the event loop
app.exec_()
