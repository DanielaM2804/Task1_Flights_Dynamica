import os
import numpy as np
import pyvista as pv
import vtk

os.system("cls" if os.name == "nt" else "clear")

# ============================================================
# ROTACIÓN (ZYX) - DCM
# Convención usada aquí:
#   - vectores FILA: v_NED = v_B @ R
#   - R = (L_BV)^T  = Body -> NED
# ============================================================
def R_zyx_long(phi, theta, psi):
    """
    L_BV: NED -> BODY (yaw-psi, pitch-theta, roll-phi), secuencia ZYX.
    """
    cph, sph = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cps, sps = np.cos(psi), np.sin(psi)
    return np.array([
        [cth * cps,                   cth * sps,                   -sth],
        [sph * sth * cps - cph * sps,  sph * sth * sps + cph * cps,  sph * cth],
        [cph * sth * cps + sph * sps,  cph * sth * sps - sph * cps,  cph * cth]
    ], dtype=float)

def R_body_to_ned_row(phi, theta, psi):
    # Vectores FILA: v_NED = v_B @ R
    return R_zyx_long(phi, theta, psi).T

# ============================================================
# STL: centrar, escalar, forward, Z down
# ============================================================
def prepare_mesh(stl_path: str, target_size=2.0):
    m = pv.read(stl_path)

    # Centrar en el origen
    m = m.translate(-np.array(m.center), inplace=False)

    # Escalar a tamaño objetivo
    b = np.array(m.bounds)
    ext = np.array([b[1]-b[0], b[3]-b[2], b[5]-b[4]])
    max_extent = float(ext.max()) if float(ext.max()) > 0 else 1.0
    s = target_size / max_extent
    m = m.scale([s, s, s], inplace=False)

    # Corregir "mirando hacia atrás" (180° sobre Z)
    R_fix_forward = np.array([[-1, 0, 0],
                              [ 0,-1, 0],
                              [ 0, 0, 1]], float)
    m.points = m.points @ R_fix_forward

    # Convención: Z hacia abajo (solo visual)
    m.points[:, 2] *= -1
    return m

# ============================================================
# VTK 2D helpers
# ============================================================
def make_polyline2d_actor(points_xy_px, color=(0, 0, 0), width=2):
    pts = vtk.vtkPoints()
    for (x, y) in points_xy_px:
        pts.InsertNextPoint(float(x), float(y), 0.0)

    polyline = vtk.vtkPolyLine()
    polyline.GetPointIds().SetNumberOfIds(len(points_xy_px))
    for i in range(len(points_xy_px)):
        polyline.GetPointIds().SetId(i, i)

    cells = vtk.vtkCellArray()
    cells.InsertNextCell(polyline)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(pts)
    polydata.SetLines(cells)

    mapper = vtk.vtkPolyDataMapper2D()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor2D()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetLineWidth(width)

    return actor, polydata, pts

def make_text_actor(text, x, y, color=(0, 0, 0), font_size=18, bold=True):
    ta = vtk.vtkTextActor()
    ta.SetInput(str(text))
    tp = ta.GetTextProperty()
    tp.SetColor(color)
    tp.SetFontSize(int(font_size))
    tp.SetBold(1 if bold else 0)
    tp.SetFontFamilyToArial()
    ta.SetPosition(float(x), float(y))
    return ta

def make_rect_outline(x0, y0, w, h):
    return [(x0, y0), (x0+w, y0), (x0+w, y0+h), (x0, y0+h), (x0, y0)]

def make_circle(cx, cy, r, n=90):
    ang = np.linspace(0, 2*np.pi, n, endpoint=True)
    return [(cx + r*np.cos(a), cy + r*np.sin(a)) for a in ang]

# ============================================================
# Flechas 3D
# ============================================================
def normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v if n < eps else v / n

def make_arrow(origin, direction, length):
    direction = normalize(direction)
    return pv.Arrow(
        start=origin,
        direction=direction,
        tip_length=0.25,
        tip_radius=0.06,
        shaft_radius=0.02,
        scale=length
    )

# ============================================================
# Casos A/B/C
# ============================================================
CASES = {
    "A": dict(name="Case A – Straight & level", phi_deg=0.0,  theta_deg=2.0,  psi_deg=0.0,  Vb=[70.0, 0.0, 2.5], wind=[0.0, 0.0, 0.0]),
    "B": dict(name="Case B – Climb",            phi_deg=0.0,  theta_deg=10.0, psi_deg=15.0, Vb=[65.0, 0.5, 12.0], wind=[0.0, 0.0, 0.0]),
    "C": dict(name="Case C – Turn",             phi_deg=25.0, theta_deg=5.0,  psi_deg=60.0, Vb=[75.0, 0.2, 4.0],  wind=[0.0, 0.0, 0.0]),
}

# ============================================================
# APP
# ============================================================
def main(stl_path: str):
    pv.set_plot_theme("document")

    mesh0 = prepare_mesh(stl_path, target_size=2.0)
    mesh_live = mesh0.copy(deep=True)
    points0 = mesh0.points.copy()

    pl = pv.Plotter(window_size=(1400, 850))
    pl.enable_anti_aliasing("msaa")
    pl.enable_eye_dome_lighting()
    pl.add_axes(interactive=True)

    pl.add_mesh(mesh_live, color="#f6a04d", smooth_shading=True,
                specular=0.4, specular_power=30)

    # Estado
    state = {
        "case": "A",
        "phi": 0.0,     # [rad]
        "theta": 0.0,   # [rad]
        "psi": 0.0,     # [rad]
        "Vb": np.array([20.0, 0.0, 0.0], float),         # [u,v,w] en cuerpo (m/s)
        "Vviento_ned": np.array([0.0, 0.0, 0.0], float)  # viento NED (m/s)
    }

    # Flechas (ejes cuerpo + V∞)
    origin = np.array([0.0, 0.0, 0.0])
    L_axes = 1.2
    L_vinf = 1.6
    axis_actors = [None, None, None]  # x,y,z
    vinf_actor = None

    def dibujar_ejes_y_vinf(R, Vinf_ned):
        nonlocal vinf_actor

        # borrar anteriores
        for i in range(3):
            if axis_actors[i] is not None:
                pl.remove_actor(axis_actors[i])
                axis_actors[i] = None
        if vinf_actor is not None:
            pl.remove_actor(vinf_actor)
            vinf_actor = None

        # ejes del cuerpo en NED
        ex = np.array([1.0, 0.0, 0.0]) @ R
        ey = np.array([0.0, 1.0, 0.0]) @ R
        ez = np.array([0.0, 0.0, 1.0]) @ R

        axis_actors[0] = pl.add_mesh(make_arrow(origin, ex, L_axes), color="red")
        axis_actors[1] = pl.add_mesh(make_arrow(origin, ey, L_axes), color="green")
        axis_actors[2] = pl.add_mesh(make_arrow(origin, ez, L_axes), color="blue")

        # V∞ (en NED)
        if np.linalg.norm(Vinf_ned) > 1e-9:
            vinf_actor = pl.add_mesh(make_arrow(origin, Vinf_ned, L_vinf), color="purple")

    # Instrumentos 2D
    W, H = pl.window_size

    # Brújula (arriba-derecha)
    r = 80
    cx = W - 140
    cy = H - 190

    compass_circle_actor, _, _ = make_polyline2d_actor(make_circle(cx, cy, r, n=110), width=2)
    needle_actor, needle_poly, needle_vtkpts = make_polyline2d_actor([(cx, cy), (cx, cy + r)], color=(1, 0, 0), width=3)

    txtN = make_text_actor("N", cx - 8,  cy + r + 10, font_size=18)
    txtS = make_text_actor("S", cx - 8,  cy - r - 22, font_size=18)
    txtE = make_text_actor("E", cx + r + 10, cy - 10, font_size=18)
    txtO = make_text_actor("O", cx - r - 26, cy - 10, font_size=18)

    # Horizonte (abajo-derecha)
    box_w, box_h = 260, 180
    hx0 = W - box_w - 40
    hy0 = 60

    horizon_box_actor, _, _ = make_polyline2d_actor(make_rect_outline(hx0, hy0, box_w, box_h), width=2)
    horizon_line_actor, horizon_line_poly, horizon_line_vtkpts = make_polyline2d_actor(
        [(hx0+30, hy0+box_h/2), (hx0+box_w-30, hy0+box_h/2)],
        color=(0, 0, 1), width=3
    )
    ref_actor, _, _ = make_polyline2d_actor(
        [(hx0+box_w/2-10, hy0+box_h/2), (hx0+box_w/2+10, hy0+box_h/2)],
        width=2
    )

    pl.add_text("", name="compass_txt", position=(W - 260, H - 250), font_size=12, color="black")

    # Agregar 2D
    ren = pl.renderer
    ren.AddViewProp(compass_circle_actor)
    ren.AddViewProp(needle_actor)
    ren.AddViewProp(txtN); ren.AddViewProp(txtS); ren.AddViewProp(txtE); ren.AddViewProp(txtO)
    ren.AddViewProp(horizon_box_actor)
    ren.AddViewProp(horizon_line_actor)
    ren.AddViewProp(ref_actor)

    def actualizar_brujula_y_horizonte():
        # Brújula (rumbo = psi)
        heading_deg = (np.degrees(state["psi"]) % 360.0)
        ang = np.deg2rad(-heading_deg)
        x2 = cx + r * np.sin(ang)
        y2 = cy + r * np.cos(ang)

        needle_vtkpts.SetPoint(0, float(cx), float(cy), 0.0)
        needle_vtkpts.SetPoint(1, float(x2), float(y2), 0.0)
        needle_vtkpts.Modified()
        needle_poly.Modified()

        pl.add_text(f"RUMBO: {heading_deg:6.1f}°", name="compass_txt",
                    position=(W - 260, H - 250), font_size=12, color="black")

        # Horizonte (roll inclina, pitch desplaza)
        roll = state["phi"]
        pitch = state["theta"]

        px_per_deg = 1.5
        dy = -np.degrees(pitch) * px_per_deg

        xL = hx0 + 30
        xR = hx0 + box_w - 30
        yC = hy0 + box_h / 2 + dy

        cxh = hx0 + box_w / 2
        cyh = yC

        ca, sa = np.cos(-roll), np.sin(-roll)

        def rot2(x, y):
            dx, dy0 = x - cxh, y - cyh
            return (cxh + ca*dx - sa*dy0, cyh + sa*dx + ca*dy0)

        x1, y1 = rot2(xL, yC)
        x2h, y2h = rot2(xR, yC)

        horizon_line_vtkpts.SetPoint(0, float(x1), float(y1), 0.0)
        horizon_line_vtkpts.SetPoint(1, float(x2h), float(y2h), 0.0)
        horizon_line_vtkpts.Modified()
        horizon_line_poly.Modified()

    # HUD y terminal
    def imprimir_terminal(alpha_deg, beta_deg, gamma_deg, Vinf_b, Vinf_ned):
        print("\n========== ESTADO DE LA AERONAVE ==========")
        print(f"Ángulos aerodinámicos [deg]: alpha={alpha_deg:.2f}, beta={beta_deg:.2f}, gamma(climb)={gamma_deg:.2f}")
        print(f"Velocidad V∞ en CUERPO [m/s] (u,v,w): [{Vinf_b[0]:.2f}, {Vinf_b[1]:.2f}, {Vinf_b[2]:.2f}]")
        print(f"Velocidad V∞ en NED   [m/s] (N,E,D): [{Vinf_ned[0]:.2f}, {Vinf_ned[1]:.2f}, {Vinf_ned[2]:.2f}]")
        print(f"Actitud [deg] (phi,theta,psi): [{np.degrees(state['phi']):.2f}, {np.degrees(state['theta']):.2f}, {np.degrees(state['psi']):.2f}]")
        print("===========================================\n")

    # Guardamos últimos valores por si presionas T
    cache = {"alpha": 0.0, "beta": 0.0, "gamma": 0.0, "Vinf_b": np.zeros(3), "Vinf_ned": np.zeros(3), "case_name": ""}

    def actualizar_hud(alpha_deg, beta_deg, gamma_deg, Vinf_b, Vinf_ned):
        txt = (
            f"CASO: {cache['case_name']}\n"
            "INTERFAZ / SALIDA (Guía)\n"
            f"Ángulos Euler [deg]:  φ={np.degrees(state['phi']):7.2f}  θ={np.degrees(state['theta']):7.2f}  ψ={np.degrees(state['psi']):7.2f}\n\n"
            "Ángulos aerodinámicos [deg]\n"
            f"  α (ataque)  : {alpha_deg:7.2f}\n"
            f"  β (sideslip): {beta_deg:7.2f}\n"
            f"  γ (climb)   : {gamma_deg:7.2f}\n\n"
            "Velocidades\n"
            f"  V∞_cuerpo [u,v,w] (m/s) : [{Vinf_b[0]:7.2f} {Vinf_b[1]:7.2f} {Vinf_b[2]:7.2f}]\n"
            f"  V∞_NED   [N,E,D] (m/s)  : [{Vinf_ned[0]:7.2f} {Vinf_ned[1]:7.2f} {Vinf_ned[2]:7.2f}]\n\n"
            f"Viento NED (m/s): [{state['Vviento_ned'][0]:.2f} {state['Vviento_ned'][1]:.2f} {state['Vviento_ned'][2]:.2f}]\n\n"
            "Controles:\n"
            "  1/2/3 = Casos A/B/C\n"
            "  P = ingresar φ θ ψ (deg)\n"
            "  V = ingresar u v w (m/s)  (velocidad en CUERPO)\n"
            "  W = ingresar viento NED (m/s)\n"
            "  T = imprimir estado en terminal\n"
            "  R = reset ángulos\n"
            "  Q = salir\n"
        )
        pl.add_text(txt, name="hud", position="upper_left", font_size=13, color="black")

    # Update general
    def update_scene():
        # 1) rotación (Body -> NED) para vectores fila
        R = R_body_to_ned_row(state["phi"], state["theta"], state["psi"])

        # 2) rotar avión (puntos como vectores fila)
        mesh_live.points = points0 @ R

        # 3) calcular V(infinito) en CUERPO (aire relativo)
        Vviento_b = state["Vviento_ned"] @ R.T    # NED -> Body, fila
        Vinf_b = state["Vb"] - Vviento_b
        u, v, w = map(float, Vinf_b)

        # 4) ángulos (definiciones del trabajo)
        Vmag = float(np.linalg.norm(Vinf_b))
        if Vmag < 1e-12:
            Vmag = 1e-12

        alpha = float(np.arctan2(w, u))  # z_body positivo hacia abajo
        beta = float(np.arcsin(np.clip(v / Vmag, -1.0, 1.0)))

        # 5) V(infinito) en NED
        Vinf_ned = Vinf_b @ R

        # gamma desde NED: gamma = atan2(-VD, sqrt(VN^2 + VE^2))
        VN, VE, VD = map(float, Vinf_ned)
        Vh = float(np.hypot(VN, VE))
        gamma = float(np.arctan2(-VD, Vh))

        alpha_deg = float(np.degrees(alpha))
        beta_deg  = float(np.degrees(beta))
        gamma_deg = float(np.degrees(gamma))

        # cache para terminal
        cache["alpha"], cache["beta"], cache["gamma"] = alpha_deg, beta_deg, gamma_deg
        cache["Vinf_b"], cache["Vinf_ned"] = Vinf_b.copy(), Vinf_ned.copy()

        # dibujar ejes y V(infinito)
        dibujar_ejes_y_vinf(R, Vinf_ned)

        # instrumentos 2D
        actualizar_brujula_y_horizonte()

        # HUD
        actualizar_hud(alpha_deg, beta_deg, gamma_deg, Vinf_b, Vinf_ned)

        pl.render()

    # Entrada por terminal
    def snap_0p1(x):
        return round(float(x) * 10.0) / 10.0

    def prompt_angles():
        raw = input("Escribe φ θ ψ (deg)  ej: 10.5 -3.2 45  > ").strip()
        if not raw:
            return
        p = raw.replace(",", " ").split()
        if len(p) != 3:
            print("Formato inválido. Ej: 10.5 -3.2 45")
            return
        ph, th, ps = (snap_0p1(x) for x in p)
        state["phi"] = np.deg2rad(ph)
        state["theta"] = np.deg2rad(th)
        state["psi"] = np.deg2rad(ps)
        cache["case_name"] = "Custom"
        update_scene()

    def prompt_velocity_body():
        raw = input("Escribe u v w (m/s)  ej: 70 0 2  > ").strip()
        if not raw:
            return
        p = raw.replace(",", " ").split()
        if len(p) != 3:
            print("Formato inválido. Ej: 70 0 2")
            return
        u, v, w = (float(x) for x in p)
        state["Vb"] = np.array([u, v, w], float)
        cache["case_name"] = "Custom"
        update_scene()

    def prompt_wind_ned():
        raw = input("Escribe viento NED (m/s)  ej: -5 0 0  > ").strip()
        if not raw:
            return
        p = raw.replace(",", " ").split()
        if len(p) != 3:
            print("Formato inválido. Ej: -5 0 0")
            return
        vn, ve, vd = (float(x) for x in p)
        state["Vviento_ned"] = np.array([vn, ve, vd], float)
        cache["case_name"] = "Custom"
        update_scene()

    def print_terminal():
        imprimir_terminal(cache["alpha"], cache["beta"], cache["gamma"], cache["Vinf_b"], cache["Vinf_ned"])

    def reset_angles():
        state["phi"] = 0.0
        state["theta"] = 0.0
        state["psi"] = 0.0
        cache["case_name"] = "Custom"
        update_scene()

    def quit_app():
        pl.close()

    def load_case(case_key: str):
        c = CASES[case_key]
        cache["case_name"] = c["name"]
        state["phi"] = np.deg2rad(c["phi_deg"])
        state["theta"] = np.deg2rad(c["theta_deg"])
        state["psi"] = np.deg2rad(c["psi_deg"])
        state["Vb"] = np.array(c["Vb"], float)
        state["Vviento_ned"] = np.array(c["wind"], float)
        update_scene()
        print_terminal()

    pl.add_key_event("1", lambda: load_case("A"))
    pl.add_key_event("2", lambda: load_case("B"))
    pl.add_key_event("3", lambda: load_case("C"))
    pl.add_key_event("p", prompt_angles)
    pl.add_key_event("v", prompt_velocity_body)
    pl.add_key_event("w", prompt_wind_ned)
    pl.add_key_event("t", print_terminal)
    pl.add_key_event("r", reset_angles)
    pl.add_key_event("q", quit_app)

    # Primer render (Case A)
    load_case("A")
    pl.show()

if __name__ == "__main__":
    # El STL debe estar en la misma carpeta
    STL_PATH = os.path.join(os.path.dirname(__file__), "avion.stl")
    main(STL_PATH)
