# Audio Reactive Scatterplot Galaxy


3D scatterplot visualization of audio files (.wav only).

Creates a 3D plot widget of a dynamic scatter plot.


This is a hobby project I created to:

* visualize music I've created

* familiarize myself with PyQt Graph and OpenGL

* explore matplotlib colormaps

Examples:
----
![img_54](https://github.com/RaidenIV/Audio-Visualizer/assets/110344184/c0115376-05f5-41ce-99fd-3b3aa59fdd98)
![img_13](https://github.com/RaidenIV/Audio-Visualizer/assets/110344184/3159efe5-0893-492f-b492-dda6f00b65eb)

Visualizer options:
----
* **audio_filename**: filepath of .wav file to play (and plot)

* **cmap**: matplotlib colormap to use for plotting

* **camera**: distance, azimuth, elevation

* **n**: number of background stars

* **n_stars**: number of stars in galaxy

* **rotation_speed**: speed the galaxy rotates

* **line length**: length of simulated lightning

* **refresh_ms**: time between each plot refresh, in milliseconds (17=60fps)


Links and Documentation

* [colormaps](https://matplotlib.org/stable/users/explain/colors/colormaps.html)

* [PyQt Graph](https://pyqtgraph.readthedocs.io/en/latest/api_reference/graphicsItems/scatterplotitem.html)
