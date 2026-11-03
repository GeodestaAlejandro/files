from __future__ import annotations

import math
import importlib
from dataclasses import dataclass
import re
import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import hmac
import os
import tarfile
import tempfile
import shutil
import time
import requests

try:
    GeoidKarney = importlib.import_module("pygeodesy.geoids").GeoidKarney
    HAS_GEOID_READER = True
except ImportError:
    GeoidKarney = None
    HAS_GEOID_READER = False

logger = logging.getLogger(__name__)

# =========================================================
# NUEVA CONFIGURACIÓN DE PÁGINA Y ESTILOS (TEMA ESMERALDA/PIZARRA)
# =========================================================
st.set_page_config(
    page_title="Plataforma de Geodesia",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Estilos Generales y Tipografía */
    .stApp {
        background-color: #f4f4f5;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Encabezados principales */
    h1, h2, h3 {
        color: #27272a;
        font-weight: 300 !important;
        letter-spacing: 0.5px;
    }
    
    /* Rediseño de las Tarjetas de Métricas (Borde izquierdo acentuado) */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border-left: 5px solid #10b981;
        padding: 15px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border-radius: 0px 8px 8px 0px;
    }
    div[data-testid="stMetric"] label {
        color: #71717a !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.75rem !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #27272a !important;
        font-weight: 400 !important;
        font-size: 1.8rem !important;
    }

    /* Rediseño de Botones (Estilo Outline/Plano) */
    .stButton > button {
        background-color: transparent;
        color: #10b981;
        border: 2px solid #10b981;
        border-radius: 4px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.85rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #10b981;
        color: #ffffff;
        box-shadow: none;
        transform: translateY(-1px);
    }

    /* Campos de entrada */
    .stTextInput input, .stNumberInput input {
        border-radius: 4px !important;
        border: 1px solid #d4d4d8 !important;
        background-color: #fafafa !important;
    }
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #10b981 !important;
        box-shadow: 0 0 0 1px #10b981 !important;
    }
    
    /* Barra Lateral */
    [data-testid="stSidebar"] {
        background-color: #18181b;
    }
    [data-testid="stSidebar"] * {
        color: #e4e4e7;
    }
    /* Selectores en la barra lateral */
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: #27272a !important;
        color: #ffffff !important;
        border: 1px solid #3f3f46 !important;
    }
</style>
""", unsafe_allow_html=True)


# =========================
# AUTENTICACIÓN (NUEVO DISEÑO)
# =========================
def require_login():
    if st.session_state.get("authenticated", False):
        return

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.write("<br><br><br>", unsafe_allow_html=True)
        st.markdown("## 🌍 Portal Geodésico")
        st.caption("Autenticación requerida para acceder al entorno de cálculo.")
        st.divider()
        
        with st.form("auth_form"):
            username = st.text_input("Identificador de Usuario")
            password = st.text_input("Clave de Acceso", type="password")
            submitted = st.form_submit_button("Ingresar al Sistema")

        if submitted:
            try:
                users_section = st.secrets.get("users", {})
            except Exception:
                st.error("Archivo de secretos no configurado.")
                st.stop()
                
            users = {str(k): str(v) for k, v in dict(users_section).items()}
            valid_password = users.get(username.strip())

            if isinstance(valid_password, str) and hmac.compare_digest(password, valid_password):
                st.session_state["authenticated"] = True
                st.session_state["username"] = username.strip()
                st.rerun()
            else:
                st.error("Credenciales inválidas. Verifique sus datos.")

    st.stop()

require_login()


# =========================
# BASE DE DATOS Y MODELO
# =========================
ELLIPSOIDS = {
    "WGS84": {"a": 6378137.0, "b": 6356752.314245},
    "GRS80": {"a": 6378137.0, "b": 6356752.314140},
    "WGS72": {"a": 6378135.0, "b": 6356750.520016},
    "Clarke1866": {"a": 6378206.4, "b": 6356583.8},
    "Clarke1880": {"a": 6378249.145, "b": 6356514.86955},
    "International1924": {"a": 6378388.0, "b": 6356911.946},
    "Airy1830": {"a": 6377563.396, "b": 6356256.909},
    "Bessel1841": {"a": 6377397.155, "b": 6356078.963},
    "Krassovsky1940": {"a": 6378245.0, "b": 6356863.019},
    "Everest1830": {"a": 6377276.345, "b": 6356075.413}
}

MARGIN_HEIGHT = 10000.0

@dataclass(frozen=True)
class Ellipsoid:
    name: str
    a: float
    b: float
    @property
    def f(self) -> float: return (self.a - self.b) / self.a
    @property
    def inv_f(self) -> float: return float("inf") if abs(self.f) < 1e-18 else 1.0 / self.f
    @property
    def e2(self) -> float: return (self.a**2 - self.b**2) / self.a**2
    @property
    def ep2(self) -> float: return (self.a**2 - self.b**2) / self.b**2
    @property
    def limit_xy(self) -> float: return self.a + MARGIN_HEIGHT
    @property
    def limit_z(self) -> float: return self.b + MARGIN_HEIGHT

def get_ellipsoid(name: str) -> Ellipsoid:
    data = ELLIPSOIDS[name]
    return Ellipsoid(name=name, a=float(data["a"]), b=float(data["b"]))


# =========================
# VALIDACIONES Y NÚCLEO
# =========================
EPS = 1e-15
def is_zero(value: float, eps: float = EPS) -> bool: return abs(value) <= eps

def parse_required_float(label: str, raw_value: str, errors: list[str], min_value: float | None = None, max_value: float | None = None, forbid_zero: bool = False) -> float | None:
    text = raw_value.strip()
    if text == "":
        errors.append(f"• {label}: Vacío.")
        return None
    try:
        value = float(text.replace(",", "."))
    except ValueError:
        errors.append(f"• {label}: Valor numérico inválido.")
        return None
    if not math.isfinite(value):
        errors.append(f"• {label}: Infinito o NaN no permitido.")
        return None
    if forbid_zero and is_zero(value):
        errors.append(f"• {label}: No puede ser 0.")
        return None
    if min_value is not None and value < min_value:
        errors.append(f"• {label}: Debe ser ≥ {min_value}.")
    if max_value is not None and value > max_value:
        errors.append(f"• {label}: Debe ser ≤ {max_value}.")
    return value

def parse_angle(label: str, raw_value: str, errors: list[str], angle_type: str, forbid_zero: bool = True) -> float | None:
    text = raw_value.strip()
    if text == "":
        errors.append(f"• {label}: Requerido.")
        return None
    cleaned = text.upper().replace(",", ".")
    cleaned = re.sub(r"[°º'’′\"”″]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    tokens = cleaned.split(" ")
    hemispheres = [t for t in tokens if t in {"N", "S", "E", "W"}]
    if len(hemispheres) > 1:
        errors.append(f"• {label}: Hemisferio redundante.")
        return None
    hemisphere = hemispheres[0] if hemispheres else None
    numeric_tokens = [t for t in tokens if t not in {"N", "S", "E", "W"}]
    if len(numeric_tokens) not in (1, 2, 3):
        errors.append(f"• {label}: Formato inválido.")
        return None
    try:
        first_number = float(numeric_tokens[0])
    except ValueError:
        errors.append(f"• {label}: Entrada no numérica.")
        return None
    numeric_sign = -1.0 if first_number < 0 else 1.0
    if len(numeric_tokens) == 1:
        value_abs = abs(first_number)
    else:
        try:
            degrees = float(numeric_tokens[0])
            minutes = float(numeric_tokens[1])
            seconds = float(numeric_tokens[2]) if len(numeric_tokens) == 3 else 0.0
        except ValueError:
            errors.append(f"• {label}: Grados/Min/Seg inválidos.")
            return None
        if minutes < 0 or minutes >= 60 or seconds < 0 or seconds >= 60:
            errors.append(f"• {label}: Min/Seg fuera de [0, 59.99].")
            return None
        value_abs = abs(degrees) + (minutes / 60.0) + (seconds / 3600.0)
    if hemisphere is not None:
        if angle_type == "lat" and hemisphere not in {"N", "S"}:
            errors.append(f"• {label}: Latitud requiere N/S.")
            return None
        if angle_type == "lon" and hemisphere not in {"E", "W"}:
            errors.append(f"• {label}: Longitud requiere E/W.")
            return None
        hemi_sign = -1.0 if hemisphere in {"S", "W"} else 1.0
        if numeric_sign < 0 and hemi_sign > 0:
            errors.append(f"• {label}: Conflicto de signo/hemisferio.")
            return None
        sign = hemi_sign
    else:
        sign = numeric_sign
    value = sign * value_abs
    limit = 90.0 if angle_type == "lat" else 180.0
    if abs(value) > limit:
        errors.append(f"• {label}: Fuera de rango (±{limit}°).")
        return None
    return value

def validate_ecef_values(x: float, y: float, z: float, ell: Ellipsoid, errors: list[str]) -> None:
    if abs(x) > ell.limit_xy or abs(y) > ell.limit_xy or abs(z) > ell.limit_z:
        errors.append("• ECEF: Coordenadas fuera de rango terrestre.")
    r = math.sqrt(x * x + y * y + z * z)
    if r < 6_000_000 or r > 7_000_000:
        errors.append("• ECEF: Punto demasiado alejado de la superficie.")

def show_errors(errors: list[str]) -> None:
    if errors:
        st.error("Se detectaron los siguientes problemas:\n\n" + "\n".join(errors))

# Matemáticas base
def deg_to_rad(v: float) -> float: return math.radians(v)
def rad_to_deg(v: float) -> float: return math.degrees(v)
def meridian_arc_coefficients(e2: float) -> tuple[float, float, float, float]:
    return (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256,
            3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024,
            15 * e2**2 / 256 + 45 * e2**3 / 1024,
            35 * e2**3 / 3072)
def calc_meridian_arc_distance(lat_rad: float, a: float, e2: float) -> float:
    a0, a2, a4, a6 = meridian_arc_coefficients(e2)
    return a * (a0 * lat_rad - a2 * math.sin(2 * lat_rad) + a4 * math.sin(4 * lat_rad) - a6 * math.sin(6 * lat_rad))
def calc_simulated_geoid(lat, lon):
    return (-105 * np.exp(-((lon - 75)**2 + (lat - (-10))**2) / 1000.0) +
             85 * np.exp(-((lon - 140)**2 + (lat - 0)**2) / 1000.0) +
             60 * np.exp(-((lon - (-15))**2 + (lat - 50)**2) / 1000.0) -
             60 * np.exp(-((lon - (-70))**2 + (lat - 15)**2) / 1000.0))
def prime_vertical_radius(lat_rad: float, ell: Ellipsoid) -> float:
    return ell.a / math.sqrt(1.0 - ell.e2 * math.sin(lat_rad)**2)
def meridian_radius(lat_rad: float, ell: Ellipsoid) -> float:
    return ell.a * (1.0 - ell.e2) / ((1.0 - ell.e2 * math.sin(lat_rad)**2) ** 1.5)

# Transformaciones
def geodetic_to_ecef(lat_deg: float, lon_deg: float, h_m: float, ell: Ellipsoid) -> tuple[float, float, float]:
    lat, lon = deg_to_rad(lat_deg), deg_to_rad(lon_deg)
    N = prime_vertical_radius(lat, ell)
    return (N + h_m) * math.cos(lat) * math.cos(lon), (N + h_m) * math.cos(lat) * math.sin(lon), (N * (1.0 - ell.e2) + h_m) * math.sin(lat)

def ecef_to_geodetic(x: float, y: float, z: float, ell: Ellipsoid, tol: float = 1e-12, max_iter: int = 15) -> tuple[float, float, float, int]:
    p = math.hypot(x, y)
    if p < 1e-12:
        return rad_to_deg(math.copysign(math.pi / 2.0, z)), 0.0, abs(z) - ell.b, 0
    lon = math.atan2(y, x)
    theta = math.atan2(z * ell.a, p * ell.b)
    lat = math.atan2(z + ell.ep2 * ell.b * math.sin(theta)**3, p - ell.e2 * ell.a * math.cos(theta)**3)
    iterations = 0
    for i in range(max_iter):
        iterations = i + 1
        N = prime_vertical_radius(lat, ell)
        cos_lat = math.cos(lat)
        h = abs(z) - ell.b if abs(cos_lat) < 1e-15 else p / cos_lat - N
        lat_new = math.atan2(z, p * (1.0 - ell.e2 * N / (N + h)))
        if abs(lat_new - lat) < tol:
            lat = lat_new
            break
        lat = lat_new
    N = prime_vertical_radius(lat, ell)
    h = abs(z) - ell.b if abs(math.cos(lat)) < 1e-15 else p / math.cos(lat) - N
    return rad_to_deg(lat), rad_to_deg(lon), h, iterations

def parallel_arc_length(lat_deg: float, lon1_deg: float, lon2_deg: float, ell: Ellipsoid) -> tuple[float, float]:
    lat = deg_to_rad(lat_deg)
    dlon_abs = abs((deg_to_rad(lon2_deg - lon1_deg) + math.pi) % (2 * math.pi) - math.pi)
    return prime_vertical_radius(lat, ell) * math.cos(lat) * dlon_abs, rad_to_deg(dlon_abs)

def trapezoidal_integral(x_values, y_values) -> float:
    return float(sum(0.5 * (y_values[i] + y_values[i + 1]) * (x_values[i + 1] - x_values[i]) for i in range(len(x_values) - 1)))

def geodetic_quadrilateral(lat1: float, lon1: float, lat2: float, lon2: float, ell: Ellipsoid) -> dict:
    s_rad, n_rad = deg_to_rad(min(lat1, lat2)), deg_to_rad(max(lat1, lat2))
    dlon = deg_to_rad(max(lon1, lon2) - min(lon1, lon2))
    phis = np.linspace(s_rad, n_rad, 2000)
    integrand = np.array([meridian_radius(phi, ell) * prime_vertical_radius(phi, ell) * math.cos(phi) for phi in phis])
    area_m2 = abs(dlon * trapezoidal_integral(phis, integrand))
    return {"area_m2": area_m2, "area_km2": area_m2 / 1e6}

def geodetic_to_ctm12(lat_deg: float, lon_deg: float, ell: Ellipsoid) -> tuple[float, float]:
    lat_rad, lon_rad = deg_to_rad(lat_deg), deg_to_rad(lon_deg)
    lat0_rad, lon0_rad = deg_to_rad(4.0), deg_to_rad(-73.0)
    k0, E0, N0 = 0.9992, 5000000.0, 2000000.0
    dlon = lon_rad - lon0_rad
    N_rad = prime_vertical_radius(lat_rad, ell)
    T, C, A = math.tan(lat_rad)**2, ell.ep2 * math.cos(lat_rad)**2, dlon * math.cos(lat_rad)
    M, M0 = calc_meridian_arc_distance(lat_rad, ell.a, ell.e2), calc_meridian_arc_distance(lat0_rad, ell.a, ell.e2)
    este = E0 + k0 * N_rad * (A + (1 - T + C) * A**3 / 6.0 + (5 - 18*T + T**2 + 72*C - 58*ell.ep2) * A**5 / 120.0)
    norte = N0 + k0 * (M - M0 + N_rad * math.tan(lat_rad) * (A**2 / 2.0 + (5 - T + 9*C + 4*C**2) * A**4 / 24.0 + (61 - 58*T + T**2 + 600*C - 330*ell.ep2) * A**6 / 720.0))
    return norte, este

def ctm12_to_geodetic(norte: float, este: float, ell: Ellipsoid) -> tuple[float, float]:
    lat0_rad, lon0_rad = deg_to_rad(4.0), deg_to_rad(-73.0)
    k0, E0, N0 = 0.9992, 5000000.0, 2000000.0
    x, y = este - E0, norte - N0
    M = calc_meridian_arc_distance(lat0_rad, ell.a, ell.e2) + y / k0
    a0, _, _, _ = meridian_arc_coefficients(ell.e2)
    mu = M / (ell.a * a0)
    e1 = (1 - math.sqrt(1 - ell.e2)) / (1 + math.sqrt(1 - ell.e2))
    phi1 = mu + (3*e1/2 - 27*e1**3/32)*math.sin(2*mu) + (21*e1**2/16 - 55*e1**4/32)*math.sin(4*mu) + (151*e1**3/96)*math.sin(6*mu) + (1097*e1**4/512)*math.sin(8*mu)
    N1, R1 = prime_vertical_radius(phi1, ell), meridian_radius(phi1, ell)
    T1, C1, D = math.tan(phi1)**2, ell.ep2 * math.cos(phi1)**2, x / (N1 * k0)
    lat_rad = phi1 - (N1 * math.tan(phi1) / R1) * (D**2 / 2.0 - (5 + 3*T1 + 10*C1 - 4*C1**2 - 9*ell.ep2) * D**4 / 24.0 + (61 + 90*T1 + 298*C1 + 45*T1**2 - 252*ell.ep2 - 3*C1**2) * D**6 / 720.0)
    lon_rad = lon0_rad + (D - (1 + 2*T1 + C1) * D**3 / 6.0 + (5 - 2*C1 + 28*T1 - 3*C1**2 + 8*ell.ep2 + 24*T1**2) * D**5 / 120.0) / math.cos(phi1)
    return rad_to_deg(lat_rad), rad_to_deg(lon_rad)

def resolver_triseccion_tienstra(ea, na, eb, nb, ec, nc, alpha, beta, gamma):
    def ang(e1, n1, e2, n2, e3, n3):
        a2, b2, c2 = (e3-e2)**2+(n3-n2)**2, (e1-e3)**2+(n1-n3)**2, (e2-e1)**2+(n2-n1)**2
        if 0 in (a2, b2, c2): return 0
        return math.acos(max(-1.0, min(1.0, (b2 + c2 - a2) / (2 * math.sqrt(b2) * math.sqrt(c2)))))
    A, B, C = ang(ea, na, eb, nb, ec, nc), ang(eb, nb, ec, nc, ea, na), ang(ec, nc, ea, na, eb, nb)
    try:
        wA, wB, wC = 1/(1/math.tan(A) - 1/math.tan(math.radians(alpha))), 1/(1/math.tan(B) - 1/math.tan(math.radians(beta))), 1/(1/math.tan(C) - 1/math.tan(math.radians(gamma)))
        sum_w = wA + wB + wC
        return (wA * ea + wB * eb + wC * ec) / sum_w, (wA * na + wB * nb + wC * nc) / sum_w
    except Exception:
        return None, None

def resolver_biseccion_azimut(ea, na, eb, nb, az_a_deg, az_b_deg):
    rad_a, rad_b = math.radians(az_a_deg), math.radians(az_b_deg)
    dx_a, dy_a, dx_b, dy_b = math.sin(rad_a), math.cos(rad_a), math.sin(rad_b), math.cos(rad_b)
    det = dx_a * dy_b - dy_a * dx_b
    if abs(det) < 1e-9: return None, None
    t_a = ((eb - ea) * dy_b - (nb - na) * dx_b) / det
    return ea + t_a * dx_a, na + t_a * dy_a


# =========================
# GRÁFICOS PLOTLY (REDESIGN: TEMA ESMERALDA)
# =========================
SURFACE_LIGHTING = {"ambient": 0.45, "diffuse": 0.9, "specular": 0.55, "roughness": 0.35, "fresnel": 0.15}
SURFACE_LIGHT_POSITION = {"x": 10000, "y": 10000, "z": 20000}
COLOR_PRIMARIO = "#10b981"
COLOR_SECUNDARIO = "#3f3f46"
COLOR_ALERTA = "#f43f5e"

def ellipsoid_surface_mesh(ell: Ellipsoid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, v = np.linspace(0, 2 * math.pi, 160), np.linspace(0, math.pi, 80)
    return ell.a * np.outer(np.cos(u), np.sin(v)), ell.a * np.outer(np.sin(u), np.sin(v)), ell.b * np.outer(np.ones_like(u), np.cos(v))

def fig_meridian_ellipse(ell: Ellipsoid) -> go.Figure:
    t = np.linspace(0, 2 * math.pi, 360)
    fig = go.Figure(go.Scatter(x=ell.a * np.cos(t), y=ell.b * np.sin(t), mode="lines", line=dict(color=COLOR_PRIMARIO, width=3), fill='toself', fillcolor='rgba(16, 185, 129, 0.1)'))
    fig.update_layout(title="Perfil Meridiano", xaxis_title="Ecuador (m)", yaxis_title="Polos (m)", template="plotly_white", height=450, margin=dict(l=0, r=0, t=40, b=0))
    return fig

def fig_ecef_3d(x: float, y: float, z: float, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=[0, x], y=[0, y], z=[0, z], mode="lines+markers", line=dict(color=COLOR_PRIMARIO, width=5), marker=dict(size=[4, 8], color=[COLOR_SECUNDARIO, COLOR_ALERTA])))
    fig.update_layout(title=title, template="plotly_white", scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"), height=450, margin=dict(l=0, r=0, t=30, b=0))
    return fig

def fig_ecef_surface_point(x: float, y: float, z: float, ell: Ellipsoid) -> go.Figure:
    fig = go.Figure()
    xs, ys, zs = ellipsoid_surface_mesh(ell)
    fig.add_trace(go.Surface(x=xs, y=ys, z=zs, colorscale='Greens', opacity=0.2, showscale=False, lighting=dict(SURFACE_LIGHTING), lightposition=dict(SURFACE_LIGHT_POSITION)))
    fig.add_trace(go.Scatter3d(x=[0, x], y=[0, y], z=[0, z], mode='lines+markers', line=dict(color=COLOR_ALERTA, width=4), marker=dict(size=[3, 6], color=[COLOR_SECUNDARIO, COLOR_ALERTA])))
    fig.update_layout(scene=dict(aspectmode='data'), height=450, margin=dict(l=0, r=0, t=0, b=0), template="plotly_white")
    return fig

def fig_parallel_arc(lat_deg: float, lon1_deg: float, lon2_deg: float, ell: Ellipsoid) -> go.Figure:
    lat_rad = math.radians(lat_deg)
    N = prime_vertical_radius(lat_rad, ell)
    lons_rad = np.radians(np.linspace(lon1_deg, lon2_deg, 100))
    X, Y, Z = N * math.cos(lat_rad) * np.cos(lons_rad), N * math.cos(lat_rad) * np.sin(lons_rad), (N * (1.0 - ell.e2)) * math.sin(lat_rad) * np.ones_like(lons_rad)
    fig = go.Figure()
    xs, ys, zs = ellipsoid_surface_mesh(ell)
    fig.add_trace(go.Surface(x=xs, y=ys, z=zs, colorscale='Greys', opacity=0.15, showscale=False))
    fig.add_trace(go.Scatter3d(x=X, y=Y, z=Z, mode='lines', line=dict(color=COLOR_PRIMARIO, width=6)))
    fig.update_layout(scene=dict(aspectmode='data'), height=450, margin=dict(l=0, r=0, t=0, b=0), template="plotly_white")
    return fig

def fig_mapa_2d(pts_dict, title="Plano Cartesiano"):
    fig = go.Figure()
    for name, coords in pts_dict.items():
        if coords[0] is not None:
            c = COLOR_ALERTA if "P" in name or "CTM" in name else COLOR_PRIMARIO
            fig.add_trace(go.Scatter(x=[coords[0]], y=[coords[1]], mode="markers+text", marker=dict(size=12, color=c), text=[name], textposition="top center", name=name))
    fig.update_layout(title=title, template="plotly_white", xaxis_title="Este (E)", yaxis_title="Norte (N)", height=450)
    return fig

def fig_mapa_base_mercator(lat: float, lon: float):
    try:
        trace = go.Scattermap(lat=[lat], lon=[lon], mode='markers', marker=dict(size=14, color=COLOR_ALERTA))
        layout_opts = dict(map_style="light", map=dict(center=dict(lat=lat, lon=lon), zoom=12))
    except AttributeError:
        trace = go.Scattermapbox(lat=[lat], lon=[lon], mode='markers', marker=dict(size=14, color=COLOR_ALERTA))
        layout_opts = dict(mapbox_style="light", mapbox=dict(center=dict(lat=lat, lon=lon), zoom=12))
    fig = go.Figure(trace)
    fig.update_layout(height=450, margin={"r": 0, "t": 0, "l": 0, "b": 0}, **layout_opts)
    return fig

def fig_perfil_elevacion(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Distancia_Acumulada'], y=df['Cota_Ajustada'], mode='lines+markers', line=dict(color=COLOR_PRIMARIO, width=3), fill='tozeroy', fillcolor='rgba(16, 185, 129, 0.1)'))
    fig.update_layout(title="Perfil Longitudinal", xaxis_title="Abscisado (m)", yaxis_title="Elevación (msnm)", template="plotly_white", height=400)
    return fig

@st.cache_resource(show_spinner="Descargando modelo EGM2008 (esto puede tardar unos minutos)...")
def load_geoid():
    if not HAS_GEOID_READER:
        logger.warning(
            "Falta PyGeodesy; instala la dependencia con: python -m pip install pygeodesy"
        )
        return None
    base_dir = os.path.dirname(os.path.abspath(__file__))
    geoids_dir = os.path.join(base_dir, "geoids")
    model_name = "egm2008-5"
    model_filename = f"{model_name}.pgm"
    model_file = os.path.join(geoids_dir, model_filename)
    os.makedirs(geoids_dir, exist_ok=True)

    if not os.path.exists(model_file):
        url = f"https://sourceforge.net/projects/geographiclib/files/geoids-distrib/{model_name}.tar.bz2/download"
        headers = {"User-Agent": "GeodesiaLab/1.0"}
        file_descriptor, archive_path = tempfile.mkstemp(
            prefix=f"{model_name}-", suffix=".tar.bz2", dir=geoids_dir
        )
        os.close(file_descriptor)
        model_temp_path = None
        try:
            for attempt in range(3):
                try:
                    with requests.get(
                        url, headers=headers, stream=True, timeout=(10, 60)
                    ) as response:
                        response.raise_for_status()
                        with open(archive_path, "wb") as out_file:
                            for chunk in response.iter_content(chunk_size=64 * 1024):
                                if chunk:
                                    out_file.write(chunk)
                    break
                except requests.exceptions.RequestException as error:
                    if attempt == 2:
                        logger.warning(
                            "No se pudo descargar el modelo EGM2008 tras 3 intentos: %s",
                            error,
                        )
                        return None
                    time.sleep(2**attempt)

            file_descriptor, model_temp_path = tempfile.mkstemp(
                prefix=f".{model_name}-", suffix=".pgm.part", dir=geoids_dir
            )
            os.close(file_descriptor)
            with tarfile.open(archive_path, "r:bz2") as archive:
                model_members = [
                    member for member in archive.getmembers()
                    if os.path.basename(member.name) == model_filename and member.isfile()
                ]
                if len(model_members) != 1:
                    raise ValueError(f"El archivo no contiene un único {model_filename}.")
                model_stream = archive.extractfile(model_members[0])
                if model_stream is None:
                    raise tarfile.ExtractError(f"No se pudo leer {model_filename}.")
                with model_stream, open(model_temp_path, "wb") as out_file:
                    shutil.copyfileobj(model_stream, out_file)
            os.replace(model_temp_path, model_file)
        except (OSError, tarfile.TarError, ValueError) as error:
            logger.warning("No se pudo instalar el modelo EGM2008: %s", error)
            return None
        finally:
            for temporary_path in (archive_path, model_temp_path):
                if temporary_path is None:
                    continue
                try:
                    os.remove(temporary_path)
                except FileNotFoundError:
                    pass

    try:
        return GeoidKarney(model_file)
    except Exception as error:
        logger.warning("No se pudo abrir el modelo EGM2008: %s", error)
        return None

def fig_geoide_3d_real(lat_pt=None, lon_pt=None, ell=None):
    geoid = load_geoid()
    lons, lats = np.linspace(-180, 180, 181), np.linspace(-90, 90, 91)
    Lon, Lat = np.meshgrid(lons, lats)
    N_real = calc_simulated_geoid(Lat, Lon)
    using_egm_real = False
    
    if geoid is not None:
        try:
            geoid_tmp = GeoidKarney(os.path.join(os.path.dirname(os.path.abspath(__file__)), "geoids", "egm2008-5.pgm"), kind=2)
            for i, latitude in enumerate(lats):
                for j, longitude in enumerate(lons):
                    N_real[i, j] = geoid_tmp.height(float(latitude), float(longitude))
            geoid_tmp.close()
            using_egm_real = True
        except Exception: pass

    n_pt = 0.0
    if lat_pt is not None and lon_pt is not None:
        n_pt = geoid.height(lat_pt, lon_pt) if using_egm_real else calc_simulated_geoid(lat_pt, lon_pt)
            
    R_base = ell.a if ell else 6378137.0
    R_def = R_base + N_real * 15000 
    phi, theta = np.radians(Lat), np.radians(Lon)
    
    fig = go.Figure(go.Surface(
        x=R_def * np.cos(phi) * np.cos(theta), y=R_def * np.cos(phi) * np.sin(theta), z=R_def * np.sin(phi),
        surfacecolor=N_real, colorscale='Teal', colorbar_title="N (m)"
    ))
    
    if lat_pt is not None and lon_pt is not None:
        phi_pt, theta_pt, r_pt = math.radians(lat_pt), math.radians(lon_pt), R_base + n_pt * 15000
        fig.add_trace(go.Scatter3d(
            x=[r_pt * math.cos(phi_pt) * math.cos(theta_pt)], y=[r_pt * math.cos(phi_pt) * math.sin(theta_pt)], z=[r_pt * math.sin(phi_pt)],
            mode='markers', marker=dict(size=8, color=COLOR_ALERTA)
        ))
        
    fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data'), margin=dict(l=0, r=0, t=0, b=0), template="plotly_white")
    return fig, n_pt, using_egm_real


# =========================================================
# DISEÑO DE ESTRUCTURA Y BARRA LATERAL (NUEVO)
# =========================================================

# Menú jerárquico dict
MENU_OPTIONS = {
    "Definiciones Geométricas": ["Propiedades del Elipsoide", "Análisis Meridiano"],
    "Transformaciones de Coordenadas": ["Paso a Sistema Cartesiano (ECEF)", "Paso a Sistema Geodésico"],
    "Superficies e Intersecciones": ["Medición de Arco Paralelo", "Geometría de Cuadriláteros"],
    "Instrumentación y Topografía": ["Problemas de Resección/Bisección", "Cartera de Nivelación"],
    "Proyecciones y Cartografía": ["Origen CTM12 (Colombia)", "Visor de Cartografía Base"],
    "Modelación Física": ["Superficie Geoidal EGM2008", "Evaluación Puntual 3D"]
}

with st.sidebar:
    st.markdown(f"### Sesión Activa\n👤 **{st.session_state.get('username', 'Admin')}**")
    if st.button("Finalizar Sesión"):
        st.session_state.clear()
        st.rerun()
        
    st.divider()
    
    st.markdown("### 🌐 Configuración Global")
    ellipsoid_name = st.selectbox("Sistema de Referencia (Elipsoide):", list(ELLIPSOIDS.keys()), index=1)
    ell = get_ellipsoid(ellipsoid_name)
    
    st.divider()
    
    st.markdown("### 🛠️ Módulos de Operación")
    categoria = st.selectbox("Categoría de Análisis:", list(MENU_OPTIONS.keys()))
    operacion = st.radio("Herramienta Específica:", MENU_OPTIONS[categoria])

st.markdown("## " + operacion)
st.caption(f"Categoría: {categoria} | Elipsoide Actual: {ell.name}")
st.markdown("<hr style='margin-top: 5px; margin-bottom: 30px; border-top: 2px solid #e4e4e7;'>", unsafe_allow_html=True)


# =========================
# RUTADORES DE MÓDULOS
# =========================

if operacion == "Propiedades del Elipsoide":
    c1, c2, c3 = st.columns(3)
    c1.metric("Semieje Mayor (a)", f"{ell.a:,.3f} m")
    c2.metric("Semieje Menor (b)", f"{ell.b:,.3f} m")
    c3.metric("Achatamiento (f)", f"{ell.f:.8f}")
    c4, c5, c6 = st.columns(3)
    c4.metric("Inv. Achatamiento (1/f)", f"{ell.inv_f:,.4f}")
    c5.metric("Excentricidad (e²)", f"{ell.e2:.8f}")
    c6.metric("2da Excentricidad (e'²)", f"{ell.ep2:.8f}")

elif operacion == "Análisis Meridiano":
    c_in, c_out = st.columns([1, 2])
    with c_in:
        with st.form("form_meridian"):
            lat_raw = st.text_input("Latitud de Interés (Decimal o GMS):", placeholder="Ej: 4.609")
            sub = st.form_submit_button("Evaluar Radios")
        if sub:
            err = []
            lat = parse_angle("Lat", lat_raw, err, "lat", False)
            if not err and lat is not None:
                phi = deg_to_rad(lat)
                st.metric("Curvatura Meridiana (M)", f"{meridian_radius(phi, ell):,.3f} m")
                st.metric("Curvatura Primer Vertical (N)", f"{prime_vertical_radius(phi, ell):,.3f} m")
            else:
                show_errors(err)
    with c_out:
        st.plotly_chart(fig_meridian_ellipse(ell), use_container_width=True)

elif operacion == "Paso a Sistema Cartesiano (ECEF)":
    with st.form("form_fwd"):
        col1, col2, col3 = st.columns(3)
        lat_raw = col1.text_input("Latitud Geodésica", "4.6")
        lon_raw = col2.text_input("Longitud Geodésica", "-74.0")
        h_raw = col3.text_input("Altura Elipsoidal (m)", "2600")
        sub = st.form_submit_button("Ejecutar Transformación")
    if sub:
        err = []
        lat, lon, h = parse_angle("Lat", lat_raw, err, "lat", False), parse_angle("Lon", lon_raw, err, "lon", False), parse_required_float("Altura", h_raw, err)
        if err: show_errors(err)
        else:
            X, Y, Z = geodetic_to_ecef(lat, lon, h, ell)
            c1, c2, c3 = st.columns(3)
            c1.metric("Eje X", f"{X:,.3f} m")
            c2.metric("Eje Y", f"{Y:,.3f} m")
            c3.metric("Eje Z", f"{Z:,.3f} m")
            st.plotly_chart(fig_ecef_3d(X, Y, Z, "Vector Geocéntrico"), use_container_width=True)

elif operacion == "Paso a Sistema Geodésico":
    with st.form("form_inv"):
        col1, col2, col3 = st.columns(3)
        x_raw = col1.text_input("Coordenada X (m)", "1749871.0")
        y_raw = col2.text_input("Coordenada Y (m)", "-6111234.0")
        z_raw = col3.text_input("Coordenada Z (m)", "508123.0")
        sub = st.form_submit_button("Recuperar Coordenadas")
    if sub:
        err = []
        x, y, z = parse_required_float("X", x_raw, err), parse_required_float("Y", y_raw, err), parse_required_float("Z", z_raw, err)
        if x and y and z: validate_ecef_values(x, y, z, ell, err)
        if err: show_errors(err)
        else:
            lat, lon, h, iters = ecef_to_geodetic(x, y, z, ell)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Latitud (°)", f"{lat:.7f}")
            c2.metric("Longitud (°)", f"{lon:.7f}")
            c3.metric("Altura (m)", f"{h:,.3f}")
            c4.metric("Iteraciones", str(iters))
            st.plotly_chart(fig_ecef_surface_point(x, y, z, ell), use_container_width=True)

elif operacion == "Medición de Arco Paralelo":
    with st.form("form_arc"):
        col1, col2, col3 = st.columns(3)
        lat_raw = col1.text_input("Paralelo de Trabajo (Lat)", "4.6")
        lon1_raw = col2.text_input("Longitud Inicio", "-74.2")
        lon2_raw = col3.text_input("Longitud Fin", "-73.8")
        sub = st.form_submit_button("Evaluar Extensión")
    if sub:
        err = []
        lat = parse_angle("Lat", lat_raw, err, "lat", False)
        lon1 = parse_angle("Lon1", lon1_raw, err, "lon", False)
        lon2 = parse_angle("Lon2", lon2_raw, err, "lon", False)
        if err: show_errors(err)
        else:
            arc_len, dlon = parallel_arc_length(lat, lon1, lon2, ell)
            c1, c2 = st.columns(2)
            c1.metric("Desplazamiento Angular", f"{dlon:.5f}°")
            c2.metric("Longitud Física", f"{arc_len:,.3f} m")
            st.plotly_chart(fig_parallel_arc(lat, lon1, lon2, ell), use_container_width=True)

elif operacion == "Geometría de Cuadriláteros":
    with st.form("form_quad"):
        st.write("Definición de Vértices Diagonales")
        r1, r2, r3, r4 = st.columns(4)
        lat1_raw = r1.text_input("Lat. Vértice 1", "4.4")
        lon1_raw = r2.text_input("Lon. Vértice 1", "-74.2")
        lat2_raw = r3.text_input("Lat. Vértice 2", "4.8")
        lon2_raw = r4.text_input("Lon. Vértice 2", "-73.8")
        sub = st.form_submit_button("Determinar Superficie")
    if sub:
        err = []
        lat1, lon1 = parse_angle("Lat1", lat1_raw, err, "lat", False), parse_angle("Lon1", lon1_raw, err, "lon", False)
        lat2, lon2 = parse_angle("Lat2", lat2_raw, err, "lat", False), parse_angle("Lon2", lon2_raw, err, "lon", False)
        if err: show_errors(err)
        else:
            res = geodetic_quadrilateral(lat1, lon1, lat2, lon2, ell)
            c1, c2 = st.columns(2)
            c1.metric("Superficie Total (m²)", f"{res['area_m2']:,.2f}")
            c2.metric("Superficie Total (km²)", f"{res['area_km2']:,.4f}")

elif operacion == "Problemas de Resección/Bisección":
    tipo = st.selectbox("Método a resolver:", ["Resección (Pothenot/Tienstra)", "Intersección Directa (Bisección)"])
    if "Resección" in tipo:
        with st.form("t_form"):
            r1, r2, r3 = st.columns(3)
            ea = r1.number_input("Este A", value=1000.0)
            na = r1.number_input("Norte A", value=1000.0)
            alpha = r1.number_input("Ángulo α", value=45.0)
            eb = r2.number_input("Este B", value=2000.0)
            nb = r2.number_input("Norte B", value=1500.0)
            beta = r2.number_input("Ángulo β", value=60.0)
            ec = r3.number_input("Este C", value=2500.0)
            nc = r3.number_input("Norte C", value=800.0)
            gamma = r3.number_input("Ángulo γ", value=75.0)
            sub = st.form_submit_button("Procesar Trisección")
        if sub:
            ep, np_c = resolver_triseccion_tienstra(ea, na, eb, nb, ec, nc, alpha, beta, gamma)
            if ep:
                st.success("Cálculo Convergente")
                st.plotly_chart(fig_mapa_2d({"A":(ea,na), "B":(eb,nb), "C":(ec,nc), "Punto (Trisección)":(ep, np_c)}), use_container_width=True)
            else: st.error("Fallo de convergencia geométrica.")
    else:
        with st.form("b_form"):
            r1, r2 = st.columns(2)
            ea = r1.number_input("Este Estación 1", value=1000.0)
            na = r1.number_input("Norte Estación 1", value=1000.0)
            aza = r1.number_input("Azimut 1-P (°)", value=45.0)
            eb = r2.number_input("Este Estación 2", value=2000.0)
            nb = r2.number_input("Norte Estación 2", value=1000.0)
            azb = r2.number_input("Azimut 2-P (°)", value=315.0)
            sub = st.form_submit_button("Procesar Intersección")
        if sub:
            ep, np_c = resolver_biseccion_azimut(ea, na, eb, nb, aza, azb)
            if ep:
                st.plotly_chart(fig_mapa_2d({"Estación 1":(ea,na), "Estación 2":(eb,nb), "Punto Intersecado":(ep, np_c)}), use_container_width=True)
            else: st.error("No hay intersección válida (Líneas paralelas).")

elif operacion == "Cartera de Nivelación":
    if "df_niv" not in st.session_state:
        st.session_state.df_niv = pd.DataFrame({
            "Punto": ["BM-1", "1", "2", "BM-2"],
            "Dist_Armado": [0.0, 20.0, 25.0, 30.0],
            "Lectura_Atras": [1.250, 1.420, 0.0, 0.0],
            "Lectura_Inter": [0.0, 0.0, 1.350, 0.0],
            "Lectura_Adelante": [0.0, 1.150, 0.0, 1.200],
            "Elev_Base": [2600.0, 0.0, 0.0, 0.0]
        })
    df_edit = st.data_editor(st.session_state.df_niv, num_rows="dynamic", use_container_width=True)
    if st.button("Ejecutar Compensación"):
        df = df_edit.copy()
        cota_actual = df.loc[0, "Elev_Base"]
        hi_actual = cota_actual + df.loc[0, "Lectura_Atras"]
        cotas = [cota_actual]
        for i in range(1, len(df)):
            v_at, v_int, v_ad = df.loc[i, "Lectura_Atras"], df.loc[i, "Lectura_Inter"], df.loc[i, "Lectura_Adelante"]
            evaluar = v_ad if v_ad > 0 else v_int
            cota_por_hi = hi_actual - evaluar
            cotas.append(cota_por_hi)
            if v_at > 0: hi_actual = cota_por_hi + v_at
        df["Cota_Ajustada"] = cotas
        df["Distancia_Acumulada"] = df["Dist_Armado"].cumsum()
        st.dataframe(df[["Punto", "Distancia_Acumulada", "Cota_Ajustada"]], use_container_width=True)
        st.plotly_chart(fig_perfil_elevacion(df), use_container_width=True)

elif operacion == "Origen CTM12 (Colombia)":
    tab1, tab2 = st.tabs(["Proyección Directa (Geo → Plana)", "Proyección Inversa (Plana → Geo)"])
    with tab1:
        with st.form("c_fwd"):
            c1, c2 = st.columns(2)
            lat_raw = c1.text_input("Lat. Geodésica", "4.60971")
            lon_raw = c2.text_input("Lon. Geodésica", "-74.08175")
            sub1 = st.form_submit_button("Obtener Cartesianas CTM12")
        if sub1:
            err = []
            lat, lon = parse_angle("Lat", lat_raw, err, "lat", False), parse_angle("Lon", lon_raw, err, "lon", False)
            if err: show_errors(err)
            else:
                n, e = geodetic_to_ctm12(lat, lon, ell)
                st.metric("Norte (N)", f"{n:,.3f}")
                st.metric("Este (E)", f"{e:,.3f}")
    with tab2:
        with st.form("c_inv"):
            c1, c2 = st.columns(2)
            n_raw = c1.text_input("Norte CTM12 (m)", "2067382.15")
            e_raw = c2.text_input("Este CTM12 (m)", "4879791.80")
            sub2 = st.form_submit_button("Obtener Geodésicas")
        if sub2:
            err = []
            n, e = parse_required_float("N", n_raw, err), parse_required_float("E", e_raw, err)
            if err: show_errors(err)
            else:
                lat, lon = ctm12_to_geodetic(n, e, ell)
                st.metric("Latitud Resultante", f"{lat:.8f}°")
                st.metric("Longitud Resultante", f"{lon:.8f}°")

elif operacion == "Visor de Cartografía Base":
    with st.form("map_f"):
        c1, c2 = st.columns(2)
        lat_raw = c1.text_input("Latitud Central", "4.60971")
        lon_raw = c2.text_input("Longitud Central", "-74.08175")
        sub = st.form_submit_button("Consultar Servidor de Mapas")
    if sub:
        err = []
        lat, lon = parse_angle("Lat", lat_raw, err, "lat", False), parse_angle("Lon", lon_raw, err, "lon", False)
        if not err: st.plotly_chart(fig_mapa_base_mercator(lat, lon), use_container_width=True)

elif operacion == "Superficie Geoidal EGM2008":
    fig, _, flag = fig_geoide_3d_real(ell=ell)
    if not flag: st.caption("Nota: Operando con topología simulada (EGM2008 inactivo).")
    st.plotly_chart(fig, use_container_width=True)

elif operacion == "Evaluación Puntual 3D":
    with st.form("3d_f"):
        c1, c2 = st.columns(2)
        lat_raw = c1.text_input("Latitud de Sondeo", "4.6097")
        lon_raw = c2.text_input("Longitud de Sondeo", "-74.0817")
        sub = st.form_submit_button("Extraer Ondulación Geoidal (N)")
    if sub:
        err = []
        lat, lon = parse_angle("Lat", lat_raw, err, "lat", False), parse_angle("Lon", lon_raw, err, "lon", False)
        if not err:
            fig, n_ext, flag = fig_geoide_3d_real(lat, lon, ell)
            st.metric(f"Valor N en la coordenada indicada", f"{n_ext:.4f} metros")
            st.plotly_chart(fig, use_container_width=True)