import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go

st.set_page_config(layout='wide')
st.title("🚀 Rocket Launch & Descent Simulator with Escape Check")

# ------------------------------------
# Sidebar: Planet selection & constants
# ------------------------------------
planet = st.sidebar.selectbox("Select planet:", ["Earth", "Mars", "Moon", "Venus"])
planets = {
    "Earth": {"g": 9.80665, "rho0": 1.225, "H": 8500,  "R": 6_371_000},
    "Mars":  {"g": 3.711,   "rho0": 0.020, "H": 11_100, "R": 3_389_500},
    "Moon":  {"g": 1.62,    "rho0": 0.0,   "H": 1,      "R": 1_737_100},
    "Venus": {"g": 8.87,    "rho0": 65.0,  "H": 15_000, "R": 6_051_800},
}
props = planets[planet]

# ------------------------------
# Sidebar: Rocket configuration
# ------------------------------
st.sidebar.header("Rocket parameters")
dry_mass   = st.sidebar.number_input("Dry mass (kg)",      value=100_000.0, step=1_000.0, format="%.1f")
fuel_mass  = st.sidebar.number_input("Propellant mass (kg)", value=400_000.0, step=1_000.0, format="%.1f")
thrust     = st.sidebar.number_input("Thrust (N)",           value=7.6e6,    format="%.2e")
Isp        = st.sidebar.number_input("Specific impulse (s)", value=300.0,    format="%.1f")
Cd         = st.sidebar.number_input("Drag coefficient",     value=0.50,     format="%.2f")
A          = st.sidebar.number_input("Cross‑sectional area (m²)", value=10.0, format="%.1f")
angle      = st.sidebar.slider("Угол наклона (°)", min_value=0, max_value=90, value=15, step=1)

# --------------------
# Sidebar: Simulation
# --------------------
st.sidebar.header("Simulation controls")
max_time = st.sidebar.number_input("Max simulation time (s)", value=500.0,  step=10.0)
dt       = st.sidebar.number_input("Solver max step (s)",     value=0.5,    step=0.1)
num_pts  = st.sidebar.number_input("Number of points",        min_value=10, max_value=10_000, value=200, step=10)
run_sim  = st.sidebar.button("Run Simulation")

# -----------------
# Start simulation
# -----------------
if run_sim:
    # Planet properties
    g, rho0, H, R = props['g'], props['rho0'], props['H'], props['R']

    # Mass bookkeeping
    initial_mass = dry_mass + fuel_mass  # total at liftoff
    g0 = 9.80665                          # standard gravity for Isp
    mdot = thrust / (Isp * g0)           # propellant mass‑flow rate (kg/s)

    # "mu" – standard gravitational parameter for chosen planet
    mu = g * R**2
    
    # Преобразование угла в радианы
    angle_rad = np.radians(angle)

    # --- Helper functions ---
    def density(alt):
        """Exponential atmosphere model (returns 0 in vacuum)."""
        return rho0 * np.exp(-alt / H) if rho0 > 0 else 0.0
    def rocket_ode(t, y):
        # Расширенный вектор состояния: [x, y, vx, vy, mass]
        x, y, vx, vy, mass = y
        
        # Вычисляем модуль скорости для расчета сопротивления
        vel = np.sqrt(vx**2 + vy**2)
        
        # Thrust only while propellant remains
        F = thrust if mass > dry_mass else 0.0
        
        # Разложение вектора тяги на компоненты
        # angle_rad - угол от вертикали, поэтому:
        # Fx = F * sin(angle_rad)
        # Fy = F * cos(angle_rad)
        Fx = F * np.sin(angle_rad)
        Fy = F * np.cos(angle_rad)
        
        # Drag - сопротивление воздуха
        rho = density(y)  # y - это высота
        D_mag = 0.5 * rho * vel**2 * Cd * A     # всегда положительная
        
        # Проекции сопротивления на оси (противоположны скорости)
        Dx = -D_mag * vx / vel if vel > 0 else 0
        Dy = -D_mag * vy / vel if vel > 0 else 0
        
        # Local gravity (действует только по вертикали)
        g_curr = mu / (R + y)**2
        
        # Ускорения по осям
        ax = Fx / mass + Dx / mass
        ay = Fy / mass + Dy / mass - g_curr
        
        # Mass rate (propellant burn)
        m_dot = -mdot if mass > dry_mass else 0.0
        
        return [vx, vy, ax, ay, m_dot]

    # --- Events ---
    def hit_ground(t, y):
        return y[1]  # y-coordinate (altitude)
    hit_ground.terminal = True
    hit_ground.direction = -1

    def escape_event(t, y):
        x, y, vx, vy, _ = y
        r = R + y  # Расстояние от центра планеты
        v_squared = vx**2 + vy**2
        return v_squared / 2 - mu / r
    escape_event.terminal = True
    escape_event.direction = 1

    # --- Solve ODE ---
    # Начальные условия: [x, y, vx, vy, mass]
    y0 = [0.0, 0.0, 0.0, 0.0, initial_mass]
    t_span = (0, max_time)
    t_eval = np.linspace(0, max_time, int(num_pts))

    sol = solve_ivp(
        rocket_ode, t_span, y0,
        t_eval=t_eval,
        events=[hit_ground, escape_event],
        max_step=dt, rtol=1e-6, atol=1e-9,
    )

    # --------------------
    # Post‑processing
    # --------------------
    escaped = sol.t_events[1].size > 0
    hit     = sol.t_events[0].size > 0

    if escaped:
        t_event = sol.t_events[1][0]
        msg     = f"✅ Ракета достигла энергии ухода на t = {t_event:.1f} с и покинет планету."
        idx     = sol.t <= t_event
    elif hit:
        t_event = sol.t_events[0][0]
        msg     = f"💥 Ракета упала и ударилась о землю на t = {t_event:.1f} с."
        idx     = sol.t <= t_event
    else:
        msg = "⏱️ Симуляция завершилась до ухода или удара."
        idx = np.ones_like(sol.t, dtype=bool)

    t   = sol.t[idx]
    x   = sol.y[0][idx]  # горизонтальная позиция
    alt = sol.y[1][idx]  # высота (вертикальная позиция)
    vx  = sol.y[2][idx]
    vy  = sol.y[3][idx]
    
    # Вычисляем модуль скорости
    vel = np.sqrt(vx**2 + vy**2)

    # --------------------
    # Build animation
    # --------------------
    frames = [
        go.Frame(
            data=[go.Scatter(x=[xi], y=[h], mode='markers', 
                           marker=dict(size=30, symbol='triangle-up', 
                                    angle=90-angle if angle else 0))],
            name=str(i)
        ) for i, (xi, h) in enumerate(zip(x, alt))
    ]

    fig = go.Figure(
        data=[go.Scatter(x=[x[0]], y=[alt[0]], mode='markers', 
                        marker=dict(size=30, symbol='triangle-up', 
                                 angle=90-angle if angle else 0))],
        frames=frames,
    )

    play_button = dict(
        type="buttons",
        showactive=False,
        buttons=[{
            "label": "▶️ Play",
            "method": "animate",
            "args": [None, {"frame": {"duration": 50, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}, "fromcurrent": True}],
        }],
    )

    slider_steps = []
    for i, fr in enumerate(frames):
        slider_steps.append({
            "method": "animate",
            "label": f"{t[i]:.1f}s",
            "args": [[fr.name], {"mode": "immediate", "frame": {"duration":0, "redraw": True}, "transition": {"duration": 0}}],
        })

    slider = {
        "active": 0,
        "pad": {"t": 50},
        "len": 0.9,
        "x": 0.05,
        "y": 0,
        "steps": slider_steps,
        "currentvalue": {"visible": True, "prefix": "Время: ", "xanchor": "right", "font": {"size": 16}},
    }

    # Определение диапазона осей для графика
    max_x = max(abs(x.min()), abs(x.max()), 1) * 1.1
    max_y = max(alt.max() * 1.1, 1)
    
    fig.update_layout(
        xaxis=dict(range=[-max_x, max_x], title="Горизонтальная дистанция (м)"),
        yaxis=dict(range=[0, max_y], title="Высота (м)"),
        title="Анимация траектории ракеты",
        updatemenus=[play_button],
        sliders=[slider],
    )

    # Добавление траектории на график
    fig.add_trace(go.Scatter(
        x=x, y=alt,
        mode='lines',
        line=dict(color='rgba(0,0,255,0.3)', width=2),
        name='Траектория'
    ))

    # --------------------
    # Display
    # --------------------
    st.subheader("Анимированная траектория")
    st.plotly_chart(fig, use_container_width=True)
    st.write(msg)
    st.write(f"Максимальная достигнутая высота: {alt.max():.1f} м** на t = {t[np.argmax(alt)]:.1f} с")
    st.write(f"Максимальная горизонтальная дистанция: {x.max():.1f} м")

    if not escaped and hit:
        st.write(f"Скорость удара: {vel[-1]:.1f} м/с на t = {t[-1]:.1f} с")