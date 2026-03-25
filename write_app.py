app = open('app.py', 'w')
app.write("""import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from signals import ALL_SCENARIOS, get_scenario, SIGNALS, SIGNAL_UNITS
from detector import detect_all_signals
from investigator import run_full_investigation
from reporter import generate_incident_report, parse_report_sections

st.set_page_config(page_title="AnomalyAI", page_icon="🔍", layout="wide", initial_sidebar_state="collapsed")

st.markdown(\"\"\"<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[data-testid="stAppViewContainer"]{background:#060810!important;}
[data-testid="stHeader"],[data-testid="stDecoration"]{display:none!important;}
section[data-testid="stSidebar"]{display:none!important;}
[data-testid="stMainBlockContainer"]{padding:0!important;max-width:100%!important;}
.block-container{padding:0!important;max-width:100%!important;}
*{font-family:'Inter',sans-serif!important;}
button[kind="primary"]{background:linear-gradient(135deg,#e63946,#c1121f)!important;border:none!important;border-radius:6px!important;color:#fff!important;font-weight:600!important;font-size:13px!important;}
button[kind="secondary"]{background:transparent!important;border:1px solid #1e2535!important;color:#8892a4!important;border-radius:6px!important;font-size:12px!important;}
[data-testid="stSelectbox"]>div>div{background:#0d1117!important;border:1px solid #1e2535!important;border-radius:6px!important;color:#e2e8f0!important;}
</style>\"\"\", unsafe_allow_html=True)

SEVERITY_COLOR = {"critical":"#e63946","high":"#ef9f27","medium":"#ef9f27","low":"#1d9e75"}
SIGNAL_COLOR = {"temperature":"#e63946","pressure":"#ef9f27","vibration":"#7f77dd","error_rate":"#d85a30","throughput":"#1d9e75"}

def build_timeline_html(timeline):
    parts = ['<div style="background:#0a0e18;border:1px solid #1e2535;border-radius:8px;padding:14px 16px;">']
    for i, ev in enumerate(timeline):
        sc = SIGNAL_COLOR.get(ev["signal"], "#888")
        dot = "#e63946" if i == 0 else sc + "44"
        root = '<span style="font-size:9px;background:#e6394622;color:#e63946;border:1px solid #e6394644;border-radius:3px;padding:1px 5px;margin-left:4px;">ROOT</span>' if i == 0 else ""
        line = '<div style="width:1px;height:12px;background:#1e2535;margin-left:7px;margin-top:2px;"></div>' if i < len(timeline)-1 else ""
        idx = ev["index"]
        sig = ev["signal"].upper()
        evt = ev["event"]
        val = ev["value"]
        parts.append(
            '<div style="display:flex;gap:12px;align-items:flex-start;">'
            '<div style="display:flex;flex-direction:column;align-items:center;">'
            f'<div style="width:15px;height:15px;border-radius:50%;background:{dot};border:2px solid {sc};flex-shrink:0;"></div>'
            f'{line}'
            '</div>'
            '<div style="padding-bottom:8px;">'
            '<div style="display:flex;align-items:center;gap:8px;margin-bottom:2px;">'
            f'<span style="font-size:9px;color:#3d4a5c;font-family:monospace;">T+{idx}</span>'
            f'<span style="font-size:10px;font-weight:600;color:{sc};">{sig}</span>'
            f'{root}'
            '</div>'
            f'<div style="font-size:11px;color:#8892a4;">{evt} <span style="color:#556070;font-family:monospace;font-size:10px;">val={val}</span></div>'
            '</div></div>'
        )
    parts.append('</div>')
    return ''.join(parts)

def signal_chart(df, det, signals):
    fig = make_subplots(rows=len(signals), cols=1, shared_xaxes=True, vertical_spacing=0.02)
    for i, sig in enumerate(signals, 1):
        res = det[sig]
        color = SIGNAL_COLOR.get(sig, "#888")
        x = list(range(len(df)))
        y = df[sig].values
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, width=1.2), name=sig, showlegend=False), row=i, col=1)
        mask = res["anomaly_mask"]
        ax = np.array(x)[mask]
        ay = y[mask]
        if len(ax):
            fig.add_trace(go.Scatter(x=ax, y=ay, mode="markers", marker=dict(color="#e63946", size=5), showlegend=False), row=i, col=1)
        fig.update_yaxes(title_text=sig.upper(), row=i, col=1, gridcolor="rgba(255,255,255,0.03)", tickfont=dict(size=9, color="#556070"), title_font=dict(size=8, color=color), zeroline=False)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.03)", tickfont=dict(size=9, color="#556070"), zeroline=False)
    fig.update_layout(height=420, margin=dict(l=80,r=20,t=10,b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified")
    return fig

def corr_heatmap(corr_matrix):
    signals = corr_matrix.columns.tolist()
    z = corr_matrix.values.astype(float)
    fig = go.Figure(go.Heatmap(z=z, x=signals, y=signals, colorscale=[[0,"#1a0a0a"],[0.5,"#1e2535"],[1,"#e63946"]], zmid=0, zmin=-1, zmax=1, text=np.round(z,2), texttemplate="%{text}", textfont=dict(size=10, color="#e2e8f0"), hoverongaps=False))
    fig.update_layout(height=220, margin=dict(l=10,r=10,t=10,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(tickfont=dict(size=9,color="#556070")), yaxis=dict(tickfont=dict(size=9,color="#556070")))
    return fig

def rc_bars(root_causes):
    sigs = [rc["signal"] for rc in root_causes]
    scores = [rc["composite_score"] for rc in root_causes]
    colors = [SIGNAL_COLOR.get(s,"#888") for s in sigs]
    fig = go.Figure(go.Bar(x=scores, y=sigs, orientation="h", marker=dict(color=colors, opacity=0.85), text=[f"{s:.3f}" for s in scores], textposition="outside", textfont=dict(size=10, color="#8892a4")))
    fig.update_layout(height=180, margin=dict(l=10,r=60,t=10,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(range=[0,1.15], gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=9,color="#556070"), zeroline=False), yaxis=dict(tickfont=dict(size=10,color="#e2e8f0"), categoryorder="total ascending"), bargap=0.35)
    return fig

st.markdown('<div style="background:#0a0e18;border-bottom:1px solid #1e2535;padding:0 40px;height:52px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:999;"><div style="display:flex;align-items:center;gap:12px;"><div style="width:8px;height:8px;background:#e63946;border-radius:50%;box-shadow:0 0 8px #e63946;"></div><span style="font-size:13px;font-weight:700;color:#e2e8f0;letter-spacing:.05em;font-family:monospace;">ANOMALY·AI</span><span style="font-size:10px;color:#3d4a5c;font-family:monospace;">v1.0 · incident investigation console</span></div><div style="display:flex;gap:20px;align-items:center;"><span style="font-size:10px;color:#3d4a5c;">BSTS · Z-SCORE · IQR · CUSUM · GRANGER</span><div style="background:#0d1117;border:1px solid #1e2535;border-radius:4px;padding:3px 10px;font-size:10px;color:#1d9e75;font-family:monospace;">● SYSTEM ONLINE</div></div></div>', unsafe_allow_html=True)

main_left, main_right = st.columns([1, 2.6], gap="large")

with main_left:
    st.markdown('<div style="padding:24px 16px 0 32px;"><div style="font-size:9px;font-weight:600;color:#3d4a5c;text-transform:uppercase;letter-spacing:.12em;margin-bottom:16px;font-family:monospace;">· INCIDENT SELECTOR</div></div>', unsafe_allow_html=True)
    st.markdown('<div style="padding:0 16px 0 32px;">', unsafe_allow_html=True)
    scenario_name = st.selectbox("Select incident", list(ALL_SCENARIOS.keys()), label_visibility="collapsed")
    scenario = get_scenario(scenario_name)
    sev = scenario["severity"]
    sev_color = SEVERITY_COLOR.get(sev, "#888")
    st.markdown(f'<div style="margin-top:12px;background:#0a0e18;border:1px solid #1e2535;border-radius:8px;padding:14px 16px;"><div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;"><span style="font-size:9px;font-weight:600;color:#3d4a5c;text-transform:uppercase;letter-spacing:.1em;font-family:monospace;">INCIDENT TYPE</span><span style="font-size:9px;font-weight:700;color:{sev_color};background:{sev_color}22;border:1px solid {sev_color}44;border-radius:3px;padding:2px 8px;letter-spacing:.06em;font-family:monospace;">{sev.upper()}</span></div><div style="font-size:12px;color:#c9d1e0;line-height:1.6;margin-bottom:10px;">{scenario["description"]}</div><div style="display:flex;gap:8px;flex-wrap:wrap;"><span style="font-size:9px;background:#0d1117;border:1px solid #1e2535;border-radius:3px;padding:2px 7px;color:#556070;font-family:monospace;">TYPE: {scenario["incident_type"].upper()}</span><span style="font-size:9px;background:#0d1117;border:1px solid #1e2535;border-radius:3px;padding:2px 7px;color:#556070;font-family:monospace;">SIGNALS: {len(SIGNALS)}</span></div></div>', unsafe_allow_html=True)
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    investigate_btn = st.button("⚡ Investigate incident", type="primary", use_container_width=True)
    for method, desc in [("Z-SCORE","Spike detection · σ>3"),("IQR","Outlier detection · 2.5×IQR"),("CUSUM","Drift detection · cumulative sum"),("GRANGER","Causal direction · p<0.05")]:
        st.markdown(f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;"><div style="width:3px;height:24px;background:#e63946;border-radius:2px;flex-shrink:0;opacity:.6;"></div><div><div style="font-size:9px;font-weight:600;color:#e2e8f0;font-family:monospace;">{method}</div><div style="font-size:10px;color:#556070;">{desc}</div></div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with main_right:
    st.markdown('<div style="padding:24px 32px 0 16px;">', unsafe_allow_html=True)
    if not investigate_btn:
        st.markdown('<div style="height:520px;background:#0a0e18;border:1px dashed #1e2535;border-radius:12px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:20px;"><div style="font-family:monospace;font-size:32px;color:#1e2535;letter-spacing:4px;">◉ ◉ ◉ ◉ ◉</div><div style="font-size:13px;color:#3d4a5c;font-family:monospace;">SELECT INCIDENT · CLICK INVESTIGATE</div></div>', unsafe_allow_html=True)
    else:
        with st.spinner("Running detection and investigation..."):
            det = detect_all_signals(scenario["data"], SIGNALS)
            inv = run_full_investigation(scenario, det, SIGNALS)
        total_anom = sum(det[s]["anomaly_count"] for s in SIGNALS)
        affected = sum(1 for s in SIGNALS if det[s]["anomaly_count"] > 0)
        top_sig = inv["top_cause"]
        top_color = SIGNAL_COLOR.get(top_sig, "#888")
        st.markdown(f'<div style="display:grid;grid-template-columns:2fr 1fr 1fr 1fr;gap:10px;margin-bottom:18px;"><div style="background:#0a0e18;border:1px solid {sev_color}44;border-left:3px solid {sev_color};border-radius:8px;padding:14px 18px;"><div style="font-size:9px;font-weight:600;color:{sev_color};text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px;font-family:monospace;">● INCIDENT STATUS</div><div style="font-size:17px;font-weight:700;color:{sev_color};">{sev.upper()} SEVERITY</div><div style="font-size:10px;color:#556070;margin-top:3px;">{scenario["incident_type"].replace("_"," ").title()} · {len(scenario["data"])} samples</div></div><div style="background:#0a0e18;border:1px solid #1e2535;border-radius:8px;padding:14px 18px;"><div style="font-size:9px;font-weight:600;color:#3d4a5c;text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px;font-family:monospace;">ROOT CAUSE</div><div style="font-size:15px;font-weight:700;color:{top_color};">{top_sig.replace("_"," ").upper()}</div><div style="font-size:10px;color:#556070;margin-top:3px;">score: {inv["root_causes"][0]["composite_score"]:.3f}</div></div><div style="background:#0a0e18;border:1px solid #1e2535;border-radius:8px;padding:14px 18px;"><div style="font-size:9px;font-weight:600;color:#3d4a5c;text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px;font-family:monospace;">ANOMALIES</div><div style="font-size:22px;font-weight:700;color:#e63946;">{total_anom}</div><div style="font-size:10px;color:#556070;margin-top:3px;">across {affected} signals</div></div><div style="background:#0a0e18;border:1px solid #1e2535;border-radius:8px;padding:14px 18px;"><div style="font-size:9px;font-weight:600;color:#3d4a5c;text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px;font-family:monospace;">FIRST ALERT</div><div style="font-size:22px;font-weight:700;color:#ef9f27;">T+{inv["root_causes"][0]["first_anomaly"]}</div><div style="font-size:10px;color:#556070;margin-top:3px;">sample index</div></div></div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:9px;font-weight:600;color:#3d4a5c;text-transform:uppercase;letter-spacing:.12em;margin-bottom:8px;font-family:monospace;">· SIGNAL MONITOR</div>', unsafe_allow_html=True)
        st.markdown('<div style="background:#0a0e18;border:1px solid #1e2535;border-radius:8px;padding:8px;">', unsafe_allow_html=True)
        st.plotly_chart(signal_chart(scenario["data"], det, SIGNALS), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns([1.2, 1])
        with col_a:
            st.markdown('<div style="margin-top:14px;font-size:9px;font-weight:600;color:#3d4a5c;text-transform:uppercase;letter-spacing:.12em;margin-bottom:8px;font-family:monospace;">· EVIDENCE TIMELINE</div>', unsafe_allow_html=True)
            st.markdown(build_timeline_html(inv["timeline"]), unsafe_allow_html=True)
        with col_b:
            st.markdown('<div style="margin-top:14px;font-size:9px;font-weight:600;color:#3d4a5c;text-transform:uppercase;letter-spacing:.12em;margin-bottom:8px;font-family:monospace;">· ROOT CAUSE RANKING</div>', unsafe_allow_html=True)
            st.markdown('<div style="background:#0a0e18;border:1px solid #1e2535;border-radius:8px;padding:12px;">', unsafe_allow_html=True)
            st.plotly_chart(rc_bars(inv["root_causes"]), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div style="margin-top:12px;font-size:9px;font-weight:600;color:#3d4a5c;text-transform:uppercase;letter-spacing:.12em;margin-bottom:8px;font-family:monospace;">· SIGNAL CORRELATION</div>', unsafe_allow_html=True)
            st.markdown('<div style="background:#0a0e18;border:1px solid #1e2535;border-radius:8px;padding:8px;">', unsafe_allow_html=True)
            st.plotly_chart(corr_heatmap(inv["corr_matrix"]), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div style="margin-top:14px;font-size:9px;font-weight:600;color:#3d4a5c;text-transform:uppercase;letter-spacing:.12em;margin-bottom:8px;font-family:monospace;">· AI INCIDENT REPORT</div>', unsafe_allow_html=True)
        with st.spinner("Generating incident report..."):
            raw_report = generate_incident_report(scenario, inv)
            sections = parse_report_sections(raw_report)
        sbc = SEVERITY_COLOR.get(sections["severity"].lower(), sev_color)
        st.markdown(f'<div style="background:#0a0e18;border:1px solid #1e2535;border-radius:8px;padding:20px 24px;font-family:monospace;"><div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid #1e2535;"><div style="display:flex;align-items:center;gap:10px;"><div style="width:6px;height:6px;background:#e63946;border-radius:50%;box-shadow:0 0 6px #e63946;"></div><span style="font-size:11px;font-weight:600;color:#e2e8f0;letter-spacing:.08em;">INCIDENT REPORT · {scenario["name"].upper()}</span></div><span style="font-size:9px;font-weight:700;color:{sbc};background:{sbc}22;border:1px solid {sbc}55;border-radius:3px;padding:2px 8px;">{(sections["severity"] or sev).upper()}</span></div><div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px;"><div><div style="font-size:9px;color:#556070;letter-spacing:.08em;margin-bottom:6px;">PROBABLE CAUSE</div><div style="font-size:12px;color:#c9d1e0;line-height:1.6;">{sections["probable_cause"] or "See raw report"}</div></div><div><div style="font-size:9px;color:#556070;letter-spacing:.08em;margin-bottom:6px;">TIMELINE SUMMARY</div><div style="font-size:12px;color:#c9d1e0;line-height:1.6;">{sections["timeline_summary"] or "See raw report"}</div></div></div><div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;"><div><div style="font-size:9px;color:#556070;letter-spacing:.08em;margin-bottom:6px;">CONTRIBUTING FACTORS</div><div style="font-size:12px;color:#c9d1e0;line-height:1.8;white-space:pre-line;">{sections["contributing_factors"] or "See raw report"}</div></div><div><div style="font-size:9px;color:#556070;letter-spacing:.08em;margin-bottom:6px;">RECOMMENDED ACTIONS</div><div style="font-size:12px;color:#c9d1e0;line-height:1.8;white-space:pre-line;">{sections["recommended_actions"] or "See raw report"}</div></div></div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
""")
app.close()
print("Done")
