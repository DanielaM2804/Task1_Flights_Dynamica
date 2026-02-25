The program is an interactive visualization tool that displays an aircraft model and its flight-related vectors using a 3D render plus 2D cockpit-like instruments. It relies on **PyVista** for 3D rendering (aircraft STL, arrows) and **VTK** for 2D overlays (compass and attitude horizon).

A **ZYX Euler rotation matrix** (yaw–pitch–roll) is implemented in `R_zyx_long(φ, θ, ψ)`. Since the implementation uses **row vectors**, the Body→NED transformation uses the **transpose** of that matrix:

$$
\mathbf{v}_{NED}=\mathbf{v}_B\,R,
\qquad
R=\left(R_{ZYX}(\phi,\theta,\psi)\right)^T
$$

The same rotation matrix is also used to rotate the aircraft mesh in the scene.

Before visualization, the STL model is processed in `prepare_mesh()`: it is **centered**, **uniformly scaled** using its bounding box, and visually corrected (a fixed rotation if needed and a **Z-axis flip**) to match the chosen display convention (**Z down**).

During updates, the application computes the **air-relative velocity** by converting the wind from NED to Body and subtracting it from the body velocity:

$$
\mathbf{V}_{\infty,B}=\mathbf{V}_B-\mathbf{W}_B
$$

Then it transforms the air-relative velocity to NED for display:

$$
\mathbf{V}_{\infty,NED}=\mathbf{V}_{\infty,B}\,R
$$

Finally, it draws: (1) the body axes expressed in NED, and (2) the $\mathbf{V}_\infty$ vector as a purple arrow. The compass rotates with yaw ($\psi$), and the horizon line tilts with roll ($\phi$) and shifts with pitch ($\theta$). A HUD and optional terminal output report $\alpha$, $\beta$, and $\gamma$, along with velocities in Body and NED.
