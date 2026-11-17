# -*- coding: utf-8 -*-
"""
SIMULADOR DE TRANSICIÓN DE PANDEO ELÁSTICO A INELÁSTICO (NSR-10)
Grupo 10: Influencia de la esbeltez y geometría en el pandeo de columnas
Universidad Santiago de Cali

Ejecución:
    pip install streamlit plotly numpy pandas
    streamlit run app_pandeo.py

Unidades internas: mm, N, MPa (N/mm²). Resultados de carga en kN.
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Pandeo de columnas - NSR-10", layout="wide")

# ----------------------------------------------------------------------
# 1. DATOS BASE
# ----------------------------------------------------------------------
# Factor de longitud efectiva K (valores teóricos y valores recomendados de diseño)
K_TEORICO = {"Empotrado - Libre": 2.0, "Articulado - Articulado": 1.0,
             "Empotrado - Articulado": 0.7, "Empotrado - Empotrado": 0.5}
K_DISENO = {"Empotrado - Libre": 2.1, "Articulado - Articulado": 1.0,
            "Empotrado - Articulado": 0.8, "Empotrado - Empotrado": 0.65}

# Materiales: (E [MPa], Fy [MPa])
MATERIALES = {"Acero A36 (Fy=250 MPa)": (200000.0, 250.0),
              "Acero A572 Gr.50 (Fy=345 MPa)": (200000.0, 345.0),
              "Acero personalizado": (200000.0, 250.0),
              "Concreto reforzado (simplificado)": (None, None)}

SECCIONES = ["Rectangular maciza", "Circular maciza",
             "Tubular circular (CHS)", "Tubular rectangular (HSS)"]
MODELOS = ["NSR-10 F.2.5 (AISC 360)", "Euler-Johnson (parábola)", "Euler truncado en Fy"]


# ----------------------------------------------------------------------
# 2. FUNCIONES DE CÁLCULO
# ----------------------------------------------------------------------
def propiedades(tipo, d):
    """Devuelve A [mm²], I mínima [mm⁴] y r mínimo [mm] de la sección."""
    if tipo == "Rectangular maciza":
        b, h = d["b"], d["h"]
        A = b * h
        Ix, Iy = b * h**3 / 12, h * b**3 / 12
    elif tipo == "Circular maciza":
        D = d["D"]
        A = np.pi * D**2 / 4
        Ix = Iy = np.pi * D**4 / 64
    elif tipo == "Tubular circular (CHS)":
        D, t = d["D"], d["t"]
        Di = D - 2 * t
        A = np.pi / 4 * (D**2 - Di**2)
        Ix = Iy = np.pi / 64 * (D**4 - Di**4)
    else:  # Tubular rectangular (HSS)
        b, h, t = d["b"], d["h"], d["t"]
        bi, hi = b - 2 * t, h - 2 * t
        A = b * h - bi * hi
        Ix = (b * h**3 - bi * hi**3) / 12
        Iy = (h * b**3 - hi * bi**3) / 12
    I = min(Ix, Iy)                     # el pandeo ocurre por el eje débil
    return {"A": A, "Ix": Ix, "Iy": Iy, "I": I, "r": np.sqrt(I / A)}


def esfuerzo_critico(lam, E, Fy, modelo):
    """Esfuerzo crítico Fcr [MPa] para esbeltez lam = KL/r (admite arreglos)."""
    lam = np.maximum(np.asarray(lam, dtype=float), 1e-6)
    Fe = np.pi**2 * E / lam**2                      # Euler
    if modelo == MODELOS[2]:                        # Euler truncado
        return np.minimum(Fe, Fy)
    if modelo == MODELOS[1]:                        # Johnson
        Cc = np.sqrt(2 * np.pi**2 * E / Fy)
        return np.where(lam <= Cc, Fy * (1 - lam**2 / (2 * Cc**2)), Fe)
    lim = 4.71 * np.sqrt(E / Fy)                    # NSR-10 F.2.5 / AISC 360
    return np.where(lam <= lim, Fy * 0.658**(Fy / Fe), 0.877 * Fe)


def esbeltez_limite(E, Fy, modelo):
    """Esbeltez de transición entre rango inelástico y elástico."""
    if modelo == MODELOS[2]:
        return np.pi * np.sqrt(E / Fy)              # Fe = Fy
    if modelo == MODELOS[1]:
        return np.sqrt(2 * np.pi**2 * E / Fy)       # Cc
    return 4.71 * np.sqrt(E / Fy)


def resolver(tipo, dims, L_mm, K, E, Fy, modelo):
    """Cálculo completo para una columna; devuelve diccionario de resultados."""
    p = propiedades(tipo, dims)
    lam = K * L_mm / p["r"]
    lam_lim = esbeltez_limite(E, Fy, modelo)
    Fcr = float(esfuerzo_critico(lam, E, Fy, modelo))
    return {**p, "lam": lam, "lam_lim": lam_lim, "Fcr": Fcr,
            "Fe": np.pi**2 * E / lam**2, "Pcr": Fcr * p["A"] / 1000.0,
            "rango": "Elástico (Euler)" if lam > lam_lim else "Inelástico / fluencia"}


def escalar_dims(tipo, d, pct):
    """Simula pérdida de sección: reduce dimensiones (o el espesor en tubos)."""
    f = 1 - pct / 100
    if tipo.startswith("Tubular"):
        return {**d, "t": d["t"] * f}
    return {k: v * f for k, v in d.items()}


# ----------------------------------------------------------------------
# 3. ENTRADAS (BARRA LATERAL)
# ----------------------------------------------------------------------
st.title("Simulador de transición de pandeo elástico a inelástico (NSR-10)")
st.caption("Grupo 10 · Influencia de la esbeltez y geometría en el pandeo de columnas")

with st.sidebar:
    st.header("1. Longitud y apoyos")
    L = st.number_input("Longitud de la columna L (m)", 0.2, 30.0, 3.0, 0.1)
    tipoK = st.radio("Valores de K", ["Teórico", "Recomendado de diseño"], horizontal=True)
    TABLA_K = K_TEORICO if tipoK == "Teórico" else K_DISENO
    apoyo = st.selectbox("Condición de apoyo", list(TABLA_K))
    K = TABLA_K[apoyo]
    st.write(f"**K = {K:.2f}**")

    st.header("2. Material")
    mat = st.selectbox("Material", list(MATERIALES))
    if mat.startswith("Concreto"):
        fc = st.number_input("f'c (MPa)", 14.0, 70.0, 21.0, 1.0)
        bd = st.slider("βdns (carga sostenida / carga total)", 0.0, 1.0, 0.6, 0.05)
        Ec = 3900 * np.sqrt(fc)                    # NSR-10 C.8.5.4.2
        E = 0.4 * Ec / (1 + bd)                    # EI efectivo, NSR-10 C.10.10.6.1
        Fy = 0.85 * fc                             # límite de aplastamiento simplificado
        st.caption(f"Ec = {Ec:,.0f} MPa · E efectivo = {E:,.0f} MPa · 0.85f'c = {Fy:.2f} MPa")
        modelos_disp = [MODELOS[2]]
    else:
        E0, F0 = MATERIALES[mat]
        E = st.number_input("E (MPa)", 10000.0, 250000.0, E0, 1000.0, disabled=mat != "Acero personalizado")
        Fy = st.number_input("Fy (MPa)", 100.0, 700.0, F0, 5.0, disabled=mat != "Acero personalizado")
        modelos_disp = MODELOS
    modelo = st.selectbox("Modelo de esfuerzo crítico", modelos_disp)

    st.header("3. Sección transversal")
    tipo = st.selectbox("Tipo de sección", SECCIONES)
    dims = {}
    if tipo == "Rectangular maciza":
        dims["b"] = st.number_input("b (mm)", 10.0, 1000.0, 50.0, 5.0)
        dims["h"] = st.number_input("h (mm)", 10.0, 1000.0, 100.0, 5.0)
    elif tipo == "Circular maciza":
        dims["D"] = st.number_input("Diámetro D (mm)", 10.0, 1000.0, 100.0, 5.0)
    elif tipo == "Tubular circular (CHS)":
        dims["D"] = st.number_input("Diámetro exterior D (mm)", 20.0, 1000.0, 100.0, 5.0)
        dims["t"] = st.number_input("Espesor t (mm)", 1.0, 50.0, 5.0, 0.5)
    else:
        dims["b"] = st.number_input("b exterior (mm)", 20.0, 1000.0, 100.0, 5.0)
        dims["h"] = st.number_input("h exterior (mm)", 20.0, 1000.0, 150.0, 5.0)
        dims["t"] = st.number_input("Espesor t (mm)", 1.0, 50.0, 5.0, 0.5)
    st.header("4. Demanda (opcional)")
    Pu = st.number_input("Carga axial de demanda Pu (kN)", 0.0, 100000.0, 0.0, 10.0)

# Validación geométrica de tubos
if "t" in dims and dims["t"] >= min(v for k, v in dims.items() if k != "t") / 2:
    st.error("El espesor t debe ser menor que la mitad de la menor dimensión exterior.")
    st.stop()

R = resolver(tipo, dims, L * 1000, K, E, Fy, modelo)

# ----------------------------------------------------------------------
# 4. RESULTADOS PRINCIPALES
# ----------------------------------------------------------------------
c = st.columns(4)
c[0].metric("Área A", f"{R['A']:,.1f} mm²")
c[1].metric("Inercia mínima I", f"{R['I']:,.0f} mm⁴")
c[2].metric("Radio de giro r", f"{R['r']:.2f} mm")
c[3].metric("Esbeltez KL/r", f"{R['lam']:.1f}")
c = st.columns(4)
c[0].metric("Esbeltez límite (transición)", f"{R['lam_lim']:.1f}")
c[1].metric("Esfuerzo crítico Fcr", f"{R['Fcr']:.2f} MPa")
c[2].metric("Carga crítica Pcr", f"{R['Pcr']:,.1f} kN")
c[3].metric("Rango de comportamiento", R["rango"])

if Pu > 0:
    phi = 0.90 if not mat.startswith("Concreto") else 1.0
    st.info(f"Demanda/capacidad = Pu / (φ·Pcr) = {Pu:.1f} / ({phi:.2f}×{R['Pcr']:.1f}) = "
            f"**{Pu / (phi * R['Pcr']):.2f}**  (φ=0.90 para compresión en acero; "
            "para concreto se muestra solo la carga crítica simplificada).")
if R["lam"] > 200:
    st.warning("KL/r > 200: supera el límite de esbeltez recomendado para miembros en compresión (NSR-10 F.2.5).")
if R["Pcr"] and tipo.startswith("Tubular"):
    st.caption("Nota: no se verifica pandeo local de las paredes del tubo (relación ancho/espesor).")

tabs = st.tabs(["Curva Fcr vs KL/r", "Apoyos (K)", "Geometría (igual área)",
                "Pcr vs longitud", "Sección dañada (sismo)"])

# ----------------------------------------------------------------------
# 5. TAB 1: CURVA FCR vs ESBELTEZ
# ----------------------------------------------------------------------
with tabs[0]:
    lam_max = max(250.0, 1.2 * R["lam"])
    x = np.linspace(5, lam_max, 600)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=np.pi**2 * E / x**2, name="Hipérbola de Euler (σe)",
                             line=dict(dash="dash", color="royalblue")))
    fig.add_trace(go.Scatter(x=[0, lam_max], y=[Fy, Fy], name="Fluencia / límite del material",
                             line=dict(dash="dot", color="crimson")))
    fig.add_trace(go.Scatter(x=x, y=esfuerzo_critico(x, E, Fy, modelo), name=f"Fcr: {modelo}",
                             line=dict(width=4, color="black")))
    fig.add_vline(x=R["lam_lim"], line_dash="dashdot", line_color="green",
                  annotation_text=f"Transición = {R['lam_lim']:.1f}")
    fig.add_trace(go.Scatter(x=[R["lam"]], y=[R["Fcr"]], mode="markers+text",
                             marker=dict(size=14, color="orange", line=dict(width=2)),
                             text=[f"Columna: {R['Fcr']:.1f} MPa"], textposition="top right",
                             name="Columna analizada"))
    fig.update_layout(xaxis_title="Relación de esbeltez KL/r", yaxis_title="Esfuerzo crítico (MPa)",
                      yaxis_range=[0, 1.3 * Fy], height=520, hovermode="x unified")
    fig.add_vrect(x0=0, x1=R["lam_lim"], fillcolor="orange", opacity=0.08, annotation_text="Inelástico")
    fig.add_vrect(x0=R["lam_lim"], x1=lam_max, fillcolor="blue", opacity=0.06, annotation_text="Elástico")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
# 6. TAB 2: COMPARACIÓN DE CONDICIONES DE APOYO
# ----------------------------------------------------------------------
with tabs[1]:
    filas = []
    for nombre, k in TABLA_K.items():
        r_ = resolver(tipo, dims, L * 1000, k, E, Fy, modelo)
        filas.append({"Apoyo": nombre, "K": k, "KL/r": round(r_["lam"], 1),
                      "Fcr (MPa)": round(r_["Fcr"], 2), "Pcr (kN)": round(r_["Pcr"], 1),
                      "Rango": r_["rango"]})
    df = pd.DataFrame(filas)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.plotly_chart(go.Figure(go.Bar(x=df["Apoyo"], y=df["Pcr (kN)"], text=df["Pcr (kN)"],
                                     marker_color=["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]))
                    .update_layout(yaxis_title="Pcr (kN)", title="Carga crítica según condición de apoyo"),
                    use_container_width=True)

# ----------------------------------------------------------------------
# 7. TAB 3: COMPARACIÓN DE GEOMETRÍAS CON IGUAL ÁREA
# ----------------------------------------------------------------------
with tabs[2]:
    st.write("Mismo material, longitud, K y **misma área**: solo cambia la distribución del material.")
    razon = st.slider("Relación D/t (o b/t) de los tubos", 8, 60, 25)
    A0 = R["A"]
    a = np.sqrt(A0)
    t_c = np.sqrt(A0 / (np.pi * (razon - 1)))
    t_h = np.sqrt(A0 / (4 * (razon - 1)))
    opciones = {"Cuadrada maciza": ("Rectangular maciza", {"b": a, "h": a}),
                "Rectangular 1:2 maciza": ("Rectangular maciza", {"b": np.sqrt(A0 / 2), "h": 2 * np.sqrt(A0 / 2)}),
                "Circular maciza": ("Circular maciza", {"D": np.sqrt(4 * A0 / np.pi)}),
                "Tubo circular": ("Tubular circular (CHS)", {"D": razon * t_c, "t": t_c}),
                "Tubo cuadrado": ("Tubular rectangular (HSS)", {"b": razon * t_h, "h": razon * t_h, "t": t_h})}
    filas = []
    for nombre, (tp, dm) in opciones.items():
        r_ = resolver(tp, dm, L * 1000, K, E, Fy, modelo)
        filas.append({"Geometría": nombre, "A (mm²)": round(r_["A"], 1), "r (mm)": round(r_["r"], 2),
                      "KL/r": round(r_["lam"], 1), "Fcr (MPa)": round(r_["Fcr"], 2),
                      "Pcr (kN)": round(r_["Pcr"], 1)})
    dg = pd.DataFrame(filas)
    st.dataframe(dg, use_container_width=True, hide_index=True)
    st.plotly_chart(go.Figure(go.Bar(x=dg["Geometría"], y=dg["Pcr (kN)"], text=dg["Pcr (kN)"]))
                    .update_layout(yaxis_title="Pcr (kN)", title=f"Pcr con A = {A0:,.0f} mm² constante"),
                    use_container_width=True)

# ----------------------------------------------------------------------
# 8. TAB 4: Pcr vs LONGITUD PARA LOS CUATRO K
# ----------------------------------------------------------------------
with tabs[3]:
    Ls = np.linspace(0.3, max(8.0, 1.5 * L), 150)
    fig = go.Figure()
    for nombre, k in TABLA_K.items():
        lam_ = k * Ls * 1000 / R["r"]
        fig.add_trace(go.Scatter(x=Ls, y=esfuerzo_critico(lam_, E, Fy, modelo) * R["A"] / 1000,
                                 name=f"{nombre} (K={k})"))
    fig.add_trace(go.Scatter(x=[L], y=[R["Pcr"]], mode="markers", name="Caso actual",
                             marker=dict(size=13, color="black", symbol="x")))
    fig.update_layout(xaxis_title="Longitud L (m)", yaxis_title="Pcr (kN)", height=500)
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
# 9. TAB 5: SECCIÓN DAÑADA (CONTEXTO SÍSMICO)
# ----------------------------------------------------------------------
with tabs[4]:
    st.write("Compara la columna íntegra con una columna con **pérdida de sección** y/o "
             "**pérdida de arriostramiento** (cambio de K), como tras un sismo.")
    pct = st.slider("Pérdida de sección (%) [dimensiones; espesor en tubos]", 0, 40, 15)
    apoyo_d = st.selectbox("Condición de apoyo tras el daño", list(TABLA_K), index=0, key="apd")
    dims_d = escalar_dims(tipo, dims, pct)
    Rd = resolver(tipo, dims_d, L * 1000, TABLA_K[apoyo_d], E, Fy, modelo)
    dc = pd.DataFrame({"Parámetro": ["K", "A (mm²)", "r (mm)", "KL/r", "Fcr (MPa)", "Pcr (kN)", "Rango"],
                       "Íntegra": [K, round(R["A"], 1), round(R["r"], 2), round(R["lam"], 1),
                                   round(R["Fcr"], 2), round(R["Pcr"], 1), R["rango"]],
                       "Dañada": [TABLA_K[apoyo_d], round(Rd["A"], 1), round(Rd["r"], 2),
                                  round(Rd["lam"], 1), round(Rd["Fcr"], 2), round(Rd["Pcr"], 1), Rd["rango"]]})
    st.dataframe(dc.astype(str), use_container_width=True, hide_index=True)
    st.metric("Pérdida de capacidad crítica", f"{100 * (1 - Rd['Pcr'] / R['Pcr']):.1f} %")
    xs = np.linspace(5, max(250.0, 1.2 * max(R["lam"], Rd["lam"])), 500)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=esfuerzo_critico(xs, E, Fy, modelo), name="Curva Fcr", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=[R["lam"]], y=[R["Fcr"]], mode="markers", name="Íntegra",
                             marker=dict(size=14, color="green")))
    fig.add_trace(go.Scatter(x=[Rd["lam"]], y=[Rd["Fcr"]], mode="markers", name="Dañada",
                             marker=dict(size=14, color="red", symbol="diamond")))
    fig.update_layout(xaxis_title="KL/r", yaxis_title="Fcr (MPa)", yaxis_range=[0, 1.3 * Fy], height=450)
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("Modelo con fines académicos: columna ideal, carga axial centrada, sin imperfecciones iniciales, "
           "sin pandeo local ni torsional. Verificar siempre contra el texto vigente del Reglamento NSR-10.")