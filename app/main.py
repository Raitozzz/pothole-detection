"""Pothole Detection System — Streamlit App."""
import sys
import io
import hashlib
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from model_utils import predict
from geo_utils import extract_exif_coords, assign_cluster, compute_priority_table
from db import save_reports, get_all_reports, clear_reports

# ── Design tokens ─────────────────────────────────────────────
NAVY    = "#0F2A43"
INK     = "#1E293B"
MUTED   = "#64748B"
LINE    = "#E2E8F0"
CARD    = "#FFFFFF"
POTHOLE = "#DC2626"
NORMAL  = "#16A34A"
VERIFIED = "#2563EB"
REVIEW  = "#D97706"
ACCENT  = "#0EA5E9"
NEUTRAL = "#64748B"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#CBD5E1",
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
})

st.set_page_config(
    page_title="Pothole Detection System",
    page_icon="🛣️",
    layout="wide",
)

if "reports" not in st.session_state:
    # Muat riwayat dari database agar data bertahan antar sesi.
    try:
        st.session_state["reports"] = get_all_reports()
    except Exception:
        st.session_state["reports"] = []


# ── Global styling ────────────────────────────────────────────
def _inject_css():
    st.markdown(
        """
        <style>
        html, body, [class*="css"] {
            font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        .stApp { background: #F8FAFC; }
        .block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1320px; }
        header[data-testid="stHeader"] { background: transparent; }
        #MainMenu, footer { visibility: hidden; }

        /* Tabs as a segmented control */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px; background: #EEF2F7; padding: 5px;
            border-radius: 12px; border: 1px solid #E2E8F0;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px; padding: 8px 20px;
            font-weight: 600; color: #475569;
        }
        .stTabs [aria-selected="true"] {
            background: #FFFFFF; color: #0F2A43;
            box-shadow: 0 1px 3px rgba(15,42,67,.10);
        }

        /* Buttons */
        .stButton > button, .stDownloadButton > button {
            border-radius: 9px; font-weight: 600;
            border: 1px solid #E2E8F0; padding: 8px 16px;
        }
        .stButton > button[kind="primary"] {
            background: #0F2A43; border: none;
        }
        .stButton > button[kind="primary"]:hover { background: #16385C; }

        /* Section subheaders */
        h3, .stMarkdown h4 { color: #0F2A43; }

        /* Dataframe */
        [data-testid="stDataFrame"] { border-radius: 10px; border: 1px solid #E2E8F0; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Components ─────────────────────────────────────────────────
def _pill(text: str, color: str) -> str:
    return (
        f'<span style="display:inline-block;padding:4px 12px;border-radius:999px;'
        f'background:{color}14;color:{color};font-weight:700;font-size:12.5px;'
        f'border:1px solid {color}33;letter-spacing:.2px">{text}</span>'
    )


def _section(title: str, subtitle: str = ""):
    sub = (f'<div style="font-size:13px;color:{MUTED};margin-top:2px">{subtitle}</div>'
           if subtitle else "")
    st.markdown(
        f"""
        <div style="margin:6px 0 14px">
          <div style="font-size:18px;font-weight:800;color:{NAVY};letter-spacing:-.2px">{title}</div>
          {sub}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _kpi_card(label: str, value, sub: str = "", color: str = NAVY):
    st.markdown(
        f"""
        <div style="background:{CARD};border:1px solid {LINE};border-radius:14px;
                    padding:18px 20px;box-shadow:0 1px 2px rgba(15,42,67,.04);height:100%">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
            <span style="width:8px;height:8px;border-radius:50%;background:{color};
                         display:inline-block"></span>
            <span style="font-size:11px;color:{MUTED};font-weight:700;
                         letter-spacing:.6px;text-transform:uppercase">{label}</span>
          </div>
          <div style="font-size:30px;font-weight:800;color:{INK};line-height:1.05">{value}</div>
          <div style="font-size:12px;color:{MUTED};margin-top:5px">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _chart_title(text: str):
    st.markdown(
        f"<div style='font-size:13px;font-weight:700;color:{INK};"
        f"margin-bottom:4px'>{text}</div>",
        unsafe_allow_html=True,
    )


def _confidence_bar(confidence: float, status: str):
    color = VERIFIED if status == "Terverifikasi" else REVIEW
    pct = int(round(confidence * 100))
    st.markdown(
        f"""
        <div style="margin:10px 0 2px">
          <div style="display:flex;justify-content:space-between;font-size:12px;
                      color:{MUTED};margin-bottom:5px">
            <span>Keyakinan model</span>
            <span style="font-weight:700;color:{INK}">{pct}%</span>
          </div>
          <div style="background:{LINE};border-radius:999px;height:8px;overflow:hidden">
            <div style="width:{pct}%;background:{color};height:8px;border-radius:999px"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Charts ─────────────────────────────────────────────────────
def _donut_chart(n_pothole: int, n_normal: int) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    sizes = [n_pothole, n_normal] if (n_pothole + n_normal) > 0 else [1, 1]
    ax.pie(
        sizes, colors=[POTHOLE, NORMAL], startangle=90,
        wedgeprops=dict(width=0.5, edgecolor="white", linewidth=2.5),
    )
    total = n_pothole + n_normal
    pct_p = (n_pothole / total * 100) if total else 0
    ax.text(0, 0.10, f"{int(round(pct_p))}%", ha="center", va="center",
            fontsize=19, fontweight="bold", color=INK)
    ax.text(0, -0.20, "pothole", ha="center", va="center",
            fontsize=9, color=MUTED)
    ax.legend(
        handles=[
            mpatches.Patch(color=POTHOLE, label=f"Pothole  ({n_pothole})"),
            mpatches.Patch(color=NORMAL, label=f"Normal  ({n_normal})"),
        ],
        loc="lower center", ncol=1, fontsize=8.5,
        frameon=False, bbox_to_anchor=(0.5, -0.16),
    )
    fig.tight_layout(pad=0.4)
    return _fig_to_buf(fig)


def _status_bar_chart(n_verified: int, n_review: int) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    labels = ["Terverifikasi", "Perlu\nTinjauan"]
    values = [n_verified, n_review]
    bars = ax.bar(labels, values, color=[VERIFIED, REVIEW], width=0.55,
                  edgecolor="white", linewidth=1.5, zorder=3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                str(val), ha="center", va="bottom", fontsize=13,
                fontweight="bold", color=INK)
    ax.set_ylim(0, max(values) * 1.35 + 1)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.yaxis.set_visible(False)
    ax.tick_params(axis="x", labelsize=10, length=0)
    ax.grid(axis="y", color="#EEF2F7", zorder=0)
    fig.tight_layout(pad=0.4)
    return _fig_to_buf(fig)


def _confidence_hist(confidences: list) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    ax.hist(confidences, bins=10, range=(0.5, 1.0),
            color=NAVY, edgecolor="white", linewidth=0.8, zorder=3)
    ax.axvline(0.70, color=POTHOLE, linewidth=1.6, linestyle="--",
               label="Ambang 70%", zorder=4)
    ax.set_xlabel("Keyakinan model", fontsize=9)
    ax.set_ylabel("Jumlah foto", fontsize=9)
    ax.legend(fontsize=8.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#EEF2F7", zorder=0)
    fig.tight_layout(pad=0.4)
    return _fig_to_buf(fig)


def _fig_to_buf(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                facecolor="white", transparent=False)
    plt.close(fig)
    buf.seek(0)
    return buf


# ── Utils ──────────────────────────────────────────────────────
def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.drop(columns=["hash"], errors="ignore").to_csv(index=False).encode("utf-8")


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _summary_kpi_and_charts(df: pd.DataFrame, title: str, subtitle: str = ""):
    """Render KPI cards + charts for a given DataFrame of results."""
    total       = len(df)
    n_pothole   = int((df["prediction"] == "Pothole").sum())
    n_normal    = int((df["prediction"] == "Normal").sum())
    n_verified  = int((df["status"] == "Terverifikasi").sum())
    n_review    = int((df["status"] == "Perlu Tinjauan").sum())

    def pct(x):
        return (x / total * 100) if total else 0

    _section(title, subtitle)

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        _kpi_card("Total Foto", total, "diproses", NAVY)
    with k2:
        _kpi_card("Pothole", n_pothole, f"{pct(n_pothole):.1f}% dari total", POTHOLE)
    with k3:
        _kpi_card("Normal", n_normal, f"{pct(n_normal):.1f}% dari total", NORMAL)
    with k4:
        _kpi_card("Terverifikasi", n_verified, f"{pct(n_verified):.1f}% · conf ≥ 70%", VERIFIED)
    with k5:
        _kpi_card("Perlu Tinjauan", n_review, f"{pct(n_review):.1f}% · conf < 70%", REVIEW)

    st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.1, 1.1, 1.7])
    with c1:
        _chart_title("Distribusi Prediksi")
        st.image(_donut_chart(n_pothole, n_normal), use_container_width=True)
    with c2:
        _chart_title("Status Verifikasi")
        st.image(_status_bar_chart(n_verified, n_review), use_container_width=True)
    with c3:
        _chart_title("Sebaran Keyakinan Model")
        st.image(_confidence_hist(df["confidence"].tolist()), use_container_width=True)


# ══════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════
_inject_css()

st.markdown(
    """
    <div style="background:linear-gradient(135deg,#0F2A43 0%,#13395E 100%);
                border-radius:16px;padding:26px 30px;margin-bottom:22px;
                box-shadow:0 10px 30px rgba(15,42,67,.18)">
      <div style="display:flex;align-items:center;gap:15px">
        <div style="width:44px;height:44px;border-radius:12px;background:rgba(125,211,252,.15);
                    display:flex;align-items:center;justify-content:center;
                    font-size:20px;color:#7DD3FC;font-weight:800;
                    border:1px solid rgba(125,211,252,.3)">P</div>
        <div>
          <div style="color:#FFFFFF;font-size:24px;font-weight:800;letter-spacing:-.4px">
            Pothole Detection System</div>
          <div style="color:#9EC1DE;font-size:13px;margin-top:3px">
            Deteksi kerusakan jalan otomatis &middot; pemetaan prioritas perbaikan</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["Upload & Deteksi", "Dashboard Prioritas", "Riwayat Sesi"])


# ══════════════════════════════════════════════════════════════
# Tab 1 — Upload & Deteksi
# ══════════════════════════════════════════════════════════════
def render_upload_tab():
    _section("Upload Foto Jalan",
             "Unggah satu atau beberapa foto sekaligus. Format didukung: JPG, JPEG, PNG.")

    uploaded = st.file_uploader(
        "Pilih foto",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if not uploaded:
        st.info("Belum ada foto yang diunggah. Pilih satu atau beberapa foto di atas untuk memulai.")
        return

    st.caption(f"{len(uploaded)} foto dipilih — memproses…")

    batch_results = []
    # Foto identik dikenali dari sidik jari (hash) isi filenya, bukan dari namanya.
    # Kumpulkan hash yang sudah ada di sesi maupun di batch ini agar tidak dihitung ganda.
    session_hashes = {r["hash"] for r in st.session_state["reports"] if "hash" in r}
    batch_hashes = set()

    for i, file in enumerate(uploaded):
        photo_hash = hashlib.md5(file.getvalue()).hexdigest()
        pil_img = Image.open(file)

        with st.container(border=True):
            cols = st.columns([1, 2, 1.6])

            with cols[0]:
                st.image(pil_img, use_container_width=True)
                st.caption(file.name)

            # Foto yang sama dengan yang sudah diproses cukup dihitung sekali.
            if photo_hash in session_hashes or photo_hash in batch_hashes:
                with cols[1]:
                    st.markdown(_pill("Duplikat", NEUTRAL), unsafe_allow_html=True)
                    st.caption("Foto ini sudah pernah diproses — hanya laporan pertama yang dihitung.")
                continue

            batch_hashes.add(photo_hash)
            result = predict(pil_img)
            coords = extract_exif_coords(pil_img)

            with cols[1]:
                pred   = result["prediction"]
                status = result["status"]

                color = POTHOLE if pred == "Pothole" else NORMAL
                label = "Pothole Terdeteksi" if pred == "Pothole" else "Jalan Normal"
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:10px'>"
                    f"<span style='width:12px;height:12px;border-radius:50%;"
                    f"background:{color}'></span>"
                    f"<span style='font-size:19px;font-weight:800;color:{color}'>{label}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                _confidence_bar(result["confidence"], status)
                st_color = VERIFIED if status == "Terverifikasi" else REVIEW
                st.markdown(_pill(status, st_color), unsafe_allow_html=True)
                st.markdown(
                    f"<div style='margin-top:10px;font-size:13px;color:{MUTED}'>"
                    f"Normal <b style='color:{INK}'>{result['prob_normal']*100:.1f}%</b>"
                    f" &nbsp;·&nbsp; "
                    f"Pothole <b style='color:{INK}'>{result['prob_pothole']*100:.1f}%</b></div>",
                    unsafe_allow_html=True,
                )

            with cols[2]:
                if coords:
                    lat, lon = coords
                    st.markdown(
                        f"<div style='font-size:11px;font-weight:700;color:{MUTED};"
                        f"letter-spacing:.5px'>KOORDINAT · EXIF</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"`{lat:.6f}, {lon:.6f}`")
                else:
                    st.markdown(
                        f"<div style='font-size:11px;font-weight:700;color:{MUTED};"
                        f"letter-spacing:.5px'>KOORDINAT · MANUAL</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption("Tidak ada EXIF — isi manual")
                    lat = st.number_input(
                        "Latitude", value=0.0, format="%.6f",
                        key=f"lat_{i}_{file.name}", label_visibility="collapsed",
                        placeholder="Latitude",
                    )
                    lon = st.number_input(
                        "Longitude", value=0.0, format="%.6f",
                        key=f"lon_{i}_{file.name}", label_visibility="collapsed",
                        placeholder="Longitude",
                    )

            batch_results.append({
                "filename":     file.name,
                "prediction":   result["prediction"],
                "confidence":   round(result["confidence"], 4),
                "status":       result["status"],
                "prob_normal":  round(result["prob_normal"], 4),
                "prob_pothole": round(result["prob_pothole"], 4),
                "lat":          lat if lat != 0.0 else None,
                "lon":          lon if lon != 0.0 else None,
                "hash":         photo_hash,
            })

    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

    if not batch_results:
        st.warning("Semua foto yang diunggah adalah duplikat. Tidak ada data baru untuk diproses.")
        return

    df_batch = pd.DataFrame(batch_results)
    _summary_kpi_and_charts(df_batch, "Ringkasan Batch Ini")

    st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
    _section("Detail Hasil")
    df_display = df_batch[["filename", "prediction", "confidence", "status", "lat", "lon"]].copy()
    df_display["confidence"] = (df_display["confidence"] * 100).round(1).astype(str) + "%"
    df_display.columns = ["Nama File", "Prediksi", "Keyakinan", "Status", "Latitude", "Longitude"]
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Tambahkan ke Sesi", type="primary", use_container_width=True):
            for row in batch_results:
                if row["lat"] is not None and row["lon"] is not None:
                    row["cluster_id"] = assign_cluster(
                        row["lat"], row["lon"],
                        st.session_state["reports"], radius=25
                    )
                else:
                    row["cluster_id"] = None
                row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state["reports"].append(row)
            try:
                save_reports(batch_results)  # simpan permanen ke database
            except Exception as e:
                st.warning(f"Gagal menyimpan ke database: {e}")
            st.success(f"{len(batch_results)} foto ditambahkan ke sesi.")

    with col_b:
        st.download_button(
            "Download CSV Deteksi (batch ini)",
            data=_csv_bytes(df_batch),
            file_name=f"deteksi_{_timestamp()}.csv",
            mime="text/csv",
            use_container_width=True,
        )


with tab1:
    render_upload_tab()


# ══════════════════════════════════════════════════════════════
# Tab 2 — Dashboard & Peta
# ══════════════════════════════════════════════════════════════
with tab2:
    reports = st.session_state["reports"]

    if not reports:
        st.info("Belum ada data. Unggah foto di tab pertama lalu klik 'Tambahkan ke Sesi'.")
    else:
        df_all = pd.DataFrame(reports)
        n_clusters = df_all["cluster_id"].nunique()

        _summary_kpi_and_charts(
            df_all, "Ringkasan Seluruh Sesi",
            f"Dari {n_clusters} lokasi unik (radius klaster 25 meter)",
        )

        st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)

        priority_df = compute_priority_table(reports)

        if priority_df.empty:
            st.warning("Belum ada foto dengan koordinat valid. Peta tidak dapat ditampilkan.")
        else:
            try:
                import folium
                from streamlit_folium import st_folium

                center_lat = priority_df["center_lat"].mean()
                center_lon = priority_df["center_lon"].mean()
                m = folium.Map(location=[center_lat, center_lon], zoom_start=15,
                               tiles="CartoDB positron")

                color_map = {"TINGGI": "#DC2626", "SEDANG": "#D97706", "RENDAH": "#CA8A04"}
                for _, row in priority_df.iterrows():
                    color  = color_map.get(row["priority_level"], "#2563EB")
                    radius = max(10, min(30, row["count_pothole"] * 5))
                    popup_html = (
                        f"<div style='font-family:sans-serif;font-size:13px'>"
                        f"<b>Peringkat #{int(row['rank'])}</b><br>"
                        f"Level: <b style='color:{color}'>{row['priority_level']}</b><br>"
                        f"Pothole: {int(row['count_pothole'])} laporan<br>"
                        f"Total foto: {int(row['total_laporan'])}<br>"
                        f"Rata-rata keyakinan: {row['avg_confidence']}%<br>"
                        f"Koordinat: {row['center_lat']:.5f}, {row['center_lon']:.5f}</div>"
                    )
                    folium.CircleMarker(
                        location=[row["center_lat"], row["center_lon"]],
                        radius=radius, color=color, weight=2,
                        fill=True, fill_color=color, fill_opacity=0.55,
                        popup=folium.Popup(popup_html, max_width=240),
                        tooltip=f"#{int(row['rank'])} — {row['priority_level']} "
                                f"({int(row['count_pothole'])} pothole)",
                    ).add_to(m)

                lats = priority_df["center_lat"].tolist()
                lons = priority_df["center_lon"].tolist()
                if len(lats) > 1:
                    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

                _section("Peta Prioritas Perbaikan")
                st.markdown(
                    f"""
                    <div style="display:flex;gap:18px;flex-wrap:wrap;margin:-6px 0 10px;
                                font-size:12.5px;color:{MUTED}">
                      <span><span style="display:inline-block;width:10px;height:10px;
                        border-radius:50%;background:#DC2626;margin-right:6px"></span>
                        TINGGI (&ge;5 pothole)</span>
                      <span><span style="display:inline-block;width:10px;height:10px;
                        border-radius:50%;background:#D97706;margin-right:6px"></span>
                        SEDANG (2–4)</span>
                      <span><span style="display:inline-block;width:10px;height:10px;
                        border-radius:50%;background:#CA8A04;margin-right:6px"></span>
                        RENDAH (1)</span>
                      <span style="color:#94A3B8">Klik marker untuk detail</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st_folium(m, width="100%", height=460)

            except ImportError:
                st.warning("Install `folium` dan `streamlit-folium` untuk menampilkan peta.")

            _section("Tabel Prioritas Lokasi")

            def _highlight(row):
                if row["Level"] == "TINGGI":
                    return ["background-color:#FEF2F2"] * len(row)
                if row["Level"] == "SEDANG":
                    return ["background-color:#FFFBEB"] * len(row)
                return [""] * len(row)

            display_cols = {
                "rank": "Peringkat", "center_lat": "Latitude", "center_lon": "Longitude",
                "total_laporan": "Total Foto", "count_pothole": "Pothole",
                "count_normal": "Normal", "avg_confidence": "Rata² Keyakinan (%)",
                "priority_level": "Level",
            }
            df_show = priority_df[list(display_cols.keys())].rename(columns=display_cols)
            st.dataframe(
                df_show.style.apply(_highlight, axis=1),
                use_container_width=True, hide_index=True,
            )

            st.download_button(
                "Download CSV Prioritas",
                data=_csv_bytes(priority_df),
                file_name=f"prioritas_{_timestamp()}.csv",
                mime="text/csv",
            )


# ══════════════════════════════════════════════════════════════
# Tab 3 — Riwayat Sesi
# ══════════════════════════════════════════════════════════════
with tab3:
    reports = st.session_state["reports"]

    if not reports:
        st.info("Belum ada data di sesi ini.")
    else:
        _section("Riwayat Sesi", f"{len(reports)} foto tersimpan pada sesi ini")

        df_hist = pd.DataFrame(reports)
        df_show = df_hist.drop(columns=["hash"], errors="ignore").copy()
        df_show["confidence"] = (df_show["confidence"] * 100).round(1).astype(str) + "%"
        st.dataframe(df_show, use_container_width=True, hide_index=True)

        col_x, col_y = st.columns(2)
        with col_x:
            st.download_button(
                "Download CSV Semua Data Sesi",
                data=_csv_bytes(df_hist),
                file_name=f"sesi_lengkap_{_timestamp()}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_y:
            if st.button("Hapus Semua Data Sesi", type="secondary", use_container_width=True):
                st.session_state["reports"] = []
                try:
                    clear_reports()  # hapus juga dari database
                except Exception as e:
                    st.warning(f"Gagal menghapus data database: {e}")
                st.rerun()
