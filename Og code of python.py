
import numpy as np
import plotly.graph_objs as go

# Constants
mu0 = 4 * np.pi * 1e-7  # T·m/A
I = 1.0  # Current in Amps
R = 0.3  # Radius of circular loop
N = 200  # Number of segments in the loop

# Parameterize the circular loop (in xy-plane)
theta = np.linspace(0, 2 * np.pi, N)
x_loop = R * np.cos(theta)
y_loop = R * np.sin(theta)
z_loop = np.zeros_like(theta)

loop_points = np.vstack((x_loop, y_loop, z_loop)).T
dl = np.roll(loop_points, -1, axis=0) - loop_points
midpoints = (loop_points + np.roll(loop_points, -1, axis=0)) / 2

# Field points (3D grid)
x, y, z = np.meshgrid(np.linspace(-0.5, 0.5, 10),
                      np.linspace(-0.5, 0.5, 10),
                      np.linspace(-0.3, 0.3, 6))

Bx, By, Bz = np.zeros_like(x), np.zeros_like(y), np.zeros_like(z)

# Biot–Savart Law
for i in range(N):
    r_vec = np.stack((x - midpoints[i, 0],
                      y - midpoints[i, 1],
                      z - midpoints[i, 2]), axis=-1)
    r_mag = np.linalg.norm(r_vec, axis=-1)[..., np.newaxis]
    r_mag[r_mag == 0] = 1e-20  # Avoid division by zero

    dl_cross_r = np.cross(dl[i], r_vec)
    dB = (mu0 * I / (4 * np.pi)) * dl_cross_r / (r_mag**3)

    Bx += dB[..., 0]
    By += dB[..., 1]
    Bz += dB[..., 2]

# Normalize for display
B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
Bx_unit = Bx / (B_mag + 1e-20)
By_unit = By / (B_mag + 1e-20)
Bz_unit = Bz / (B_mag + 1e-20)

# Magnetic field vectors
field_trace = go.Cone(
    x=x.flatten(), y=y.flatten(), z=z.flatten(),
    u=Bx_unit.flatten(), v=By_unit.flatten(), w=Bz_unit.flatten(),
    colorscale='Viridis', sizemode='absolute', sizeref=0.3,
    showscale=True, name="Magnetic Field"
)

# Circular loop trace
loop_trace = go.Scatter3d(
    x=x_loop, y=y_loop, z=z_loop,
    mode='lines', line=dict(color='black', width=4),
    name="Current Loop"
)

# Plot
fig = go.Figure(data=[loop_trace, field_trace])
fig.update_layout(
    title="Magnetic Field Around a Circular Current Loop (Biot–Savart Law)",
    scene=dict(
        xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
        aspectmode='cube'
    )
)
fig.show()
 
 
