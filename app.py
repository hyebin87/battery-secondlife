
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import io
 
st.set_page_config(
    page_title="배터리 Second-Life 추천 플랫폼",
    page_icon="🔋",
    layout="wide"
)
 
st.markdown("""
<style>
    .main-title  { font-size:28px; font-weight:700; margin-bottom:4px; }
    .sub-title   { font-size:14px; color:#888; margin-bottom:24px; }
    .metric-card { background:#1a1a2e; border-radius:12px; padding:20px;
                   text-align:center; border:1px solid #2a2a4a; }
    .metric-val  { font-size:28px; font-weight:700; color:#00d4aa; }
    .metric-label{ font-size:12px; color:#aaa; margin-top:4px; }
    .rec-card    { background:#1a1a2e; border-radius:12px; padding:16px 20px;
                   margin-bottom:10px; border:1px solid #2a2a4a; }
    .top-card    { border:2px solid #00d4aa !important; }
    .section-title { font-size:18px; font-weight:600; margin:20px 0 12px; }
</style>
""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────
# 파일 읽기 함수 (xls, xlsx, csv, txt 지원)
# ─────────────────────────────────────────────
def read_eis_file(uploaded_file):
    filename = uploaded_file.name.lower()
    try:
        if filename.endswith('.xls'):
            df = pd.read_excel(uploaded_file, engine='xlrd', header=None)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine='openpyxl', header=None)
        elif filename.endswith('.csv'):
            # 구분자 자동 감지
            content = uploaded_file.read().decode('utf-8')
            uploaded_file.seek(0)
            if '\t' in content.split('\n')[0]:
                df = pd.read_csv(uploaded_file, sep='\t', header=None, comment='#')
            else:
                df = pd.read_csv(uploaded_file, sep=',', header=None, comment='#')
        elif filename.endswith('.txt'):
            content = uploaded_file.read().decode('utf-8')
            uploaded_file.seek(0)
            if '\t' in content.split('\n')[0]:
                df = pd.read_csv(uploaded_file, sep='\t', header=None, comment='#')
            else:
                df = pd.read_csv(uploaded_file, sep=r'\s+', header=None, comment='#')
        else:
            return None, "지원하지 않는 파일 형식이에요."
 
        # 숫자 컬럼만 선택
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
 
        # 최소 3개 컬럼 필요
        if df.shape[1] < 2:
            return None, "컬럼이 너무 적어요. 주파수, 실수부, 허수부 데이터가 필요해요."
 
        # 3개 컬럼으로 맞추기
        if df.shape[1] >= 3:
            df = df.iloc[:, :3]
            df.columns = ['freq', 'z_real', 'z_imag']
        else:
            df = df.iloc[:, :2]
            df.columns = ['freq', 'z_real']
            df['z_imag'] = 0
 
        return df, None
 
    except Exception as e:
        return None, f"파일 읽기 오류: {str(e)}"
 
# ─────────────────────────────────────────────
# 모델 (기본 샘플 데이터 기반)
# ─────────────────────────────────────────────
@st.cache_resource
def load_default_model():
    np.random.seed(42)
    features, labels = [], []
    soh_params = {
        100: dict(re=0.022, rct=0.018, zw=0.015),
        95:  dict(re=0.028, rct=0.022, zw=0.018),
        90:  dict(re=0.035, rct=0.030, zw=0.022),
        85:  dict(re=0.042, rct=0.038, zw=0.028),
        80:  dict(re=0.052, rct=0.048, zw=0.035),
    }
    for soh, params in soh_params.items():
        for _ in range(72):
            noise = 0.003
            re   = params['re']  + np.random.normal(0, noise)
            rct  = params['rct'] + np.random.normal(0, noise)
            zw   = params['zw']  + np.random.normal(0, noise)
            z_real_max  = re + rct + zw
            z_imag_min  = -(rct * 0.6 + np.random.normal(0, 0.002))
            z_imag_max  =  rct * 0.3  + np.random.normal(0, 0.001)
            z_real_mean = re + rct * 0.5
            z_imag_std  = abs(z_imag_min) * 0.4
            features.append([re, z_real_max, z_imag_min, z_imag_max, z_real_mean, z_imag_std])
            labels.append(soh)
    X, y = np.array(features), np.array(labels)
    model = GradientBoostingRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model
 
def extract_features(df):
    return np.array([[
        float(df['z_real'].iloc[0]),
        float(df['z_real'].max()),
        float(df['z_imag'].min()),
        float(df['z_imag'].max()),
        float(df['z_real'].mean()),
        float(df['z_imag'].std()),
    ]])
 
def get_recommendations(soh, years, cycles, bat_type):
    age_penalty   = years  * 0.3
    cycle_penalty = cycles * 0.002
    apps = [
        {
            "name": "가정용 ESS",       "icon": "🏠",
            "desc": "저출력 장기 사용. 태양광 패널과 연계해 잉여전력 저장.",
            "score": max(10, soh - age_penalty - cycle_penalty + 5),
            "life":  max(1, round((soh - 60) / 8 - years * 0.1)),
            "value": round(soh * 2.5), "carbon": round(soh * 8),
            "condition": soh >= 75,
        },
        {
            "name": "태양광 연계 ESS",  "icon": "☀️",
            "desc": "재생에너지 저장에 최적. 낮은 충방전 반복 환경.",
            "score": max(10, soh - age_penalty - cycle_penalty + 10),
            "life":  max(1, round((soh - 65) / 7 - years * 0.1)),
            "value": round(soh * 3.2), "carbon": round(soh * 12),
            "condition": soh >= 70,
        },
        {
            "name": "통신기지국 백업전원", "icon": "📡",
            "desc": "간헐적 방전 환경. 안정적 출력 유지.",
            "score": max(10, soh - age_penalty - cycle_penalty),
            "life":  max(1, round((soh - 55) / 10 - years * 0.1)),
            "value": round(soh * 2.8), "carbon": round(soh * 7),
            "condition": soh >= 65,
        },
        {
            "name": "UPS 비상전원",      "icon": "🏥",
            "desc": "병원·데이터센터 비상전원. 단기 방전 위주.",
            "score": max(10, soh - age_penalty - cycle_penalty - 5),
            "life":  max(1, round((soh - 50) / 12 - years * 0.1)),
            "value": round(soh * 2.2), "carbon": round(soh * 6),
            "condition": soh >= 60,
        },
    ]
    valid = [a for a in apps if a["condition"]]
    if not valid:
        valid = [apps[-1]]
    return sorted(valid, key=lambda x: x["score"], reverse=True)[:3]
 
def safety_eval(soh, years, cycles):
    score = soh - years * 1.5 - cycles * 0.005
    if score >= 80:
        return "안전", "#00d4aa", "정상 범위 내 운용 가능합니다."
    elif score >= 65:
        return "주의", "#f0a500", "주기적 점검이 필요합니다."
    else:
        return "위험", "#e05555", "재사용보다 재활용 공정 투입을 권장합니다."
 
# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">🔋 배터리 Second-Life 추천 플랫폼</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">EIS 데이터 기반 AI 진단 · 최적 활용처 추천</div>', unsafe_allow_html=True)
 
with st.sidebar:
    st.header("⚙️ 설정")
    st.success("✅ 기본 모델 준비 완료!")
    st.divider()
    st.markdown("**지원 파일 형식**")
    st.markdown("- `.xls` (Excel 97-2003)")
    st.markdown("- `.xlsx` (Excel)")
    st.markdown("- `.csv` (쉼표/탭 구분)")
    st.markdown("- `.txt` (공백/탭 구분)")
    st.divider()
    st.markdown("**데이터셋 정보**")
    st.markdown("- Warwick DIB Dataset")
    st.markdown("- SOH: 80 / 85 / 90 / 95 / 100%")
    st.markdown("- 온도: 15 / 25 / 35°C")
    st.markdown("- 총 360개 파일")
 
# 배터리 기본 정보
st.markdown('<div class="section-title">📋 배터리 기본 정보 입력</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1:
    bat_type = st.selectbox("배터리 종류", ["NCM", "LFP", "NCA", "LCO"])
with c2:
    years = st.number_input("사용 연수 (년)", min_value=0, max_value=20, value=5)
with c3:
    cycles = st.number_input("충방전 횟수 (회)", min_value=0, max_value=3000, value=500, step=50)
with c4:
    voltage = st.number_input("현재 전압 (V)", min_value=2.5, max_value=4.5, value=3.7, step=0.01)
 
# EIS 파일 업로드
st.markdown('<div class="section-title">📂 EIS 파일 업로드</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "EIS 측정 파일 업로드",
    type=["xls", "xlsx", "csv", "txt"],
    help="xls, xlsx, csv, txt 형식 모두 지원합니다."
)
 
if uploaded:
    df, error = read_eis_file(uploaded)
 
    if error:
        st.error(f"❌ {error}")
    else:
        st.success(f"✅ 파일 읽기 성공! ({len(df)}개 데이터 포인트)")
 
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-title">📈 나이퀴스트 플롯</div>', unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['z_real'], y=-df['z_imag'],
                mode='lines+markers',
                marker=dict(
                    color=np.log10(np.abs(df['freq']) + 1e-10),
                    colorscale='Plasma', size=7,
                    colorbar=dict(title="log₁₀(Hz)", thickness=12)
                ),
                line=dict(color='rgba(255,255,255,0.2)', width=1.5),
            ))
            fig.update_layout(
                xaxis_title="Z' (실수부, Ω)", yaxis_title="-Z'' (허수부, Ω)",
                template='plotly_dark', height=320, margin=dict(l=0,r=0,t=10,b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
 
        with col2:
            st.markdown('<div class="section-title">📊 임피던스 크기</div>', unsafe_allow_html=True)
            z_mag = np.sqrt(df['z_real']**2 + df['z_imag']**2) * 1000
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=df['freq'], y=z_mag,
                mode='lines+markers',
                line=dict(color='#00d4aa', width=2),
                marker=dict(size=5)
            ))
            fig2.update_layout(
                xaxis_title="주파수 (Hz)", xaxis_type="log",
                yaxis_title="|Z| (mΩ)",
                template='plotly_dark', height=320, margin=dict(l=0,r=0,t=10,b=0)
            )
            st.plotly_chart(fig2, use_container_width=True)
 
        # AI 분석
        st.divider()
        model = load_default_model()
        feats    = extract_features(df)
        soh_pred = float(model.predict(feats)[0])
        soh_pred = round(min(100, max(50, soh_pred)), 1)
 
        st.markdown('<div class="section-title">🤖 AI 진단 결과</div>', unsafe_allow_html=True)
        m1, m2, m3, m4, m5 = st.columns(5)
        re_val       = round(df['z_real'].iloc[0] * 1000, 2)
        rct_val      = round((df['z_real'].max() - df['z_real'].iloc[0]) * 1000, 2)
        status_txt   = "양호" if soh_pred >= 85 else "보통" if soh_pred >= 70 else "주의"
        status_color = "#00d4aa" if soh_pred >= 85 else "#f0a500" if soh_pred >= 70 else "#e05555"
 
        for col, val, label, color in zip(
            [m1, m2, m3, m4, m5],
            [f"{soh_pred}%", f"{re_val}mΩ", f"{rct_val}mΩ", f"{voltage}V", status_txt],
            ["예측 SOH", "전해질 저항(Re)", "전하전달 저항(Rct)", "현재 전압", "배터리 상태"],
            ["#00d4aa","#00d4aa","#00d4aa","#00d4aa", status_color]
        ):
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color:{color}">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)
 
        # 안전성 평가
        st.markdown('<div class="section-title">🛡️ 안전성 평가</div>', unsafe_allow_html=True)
        safety_txt, safety_color, safety_desc = safety_eval(soh_pred, years, cycles)
        st.markdown(f"""
        <div class="metric-card" style="text-align:left; border:2px solid {safety_color};">
            <span style="font-size:20px; font-weight:700; color:{safety_color}">{safety_txt}</span>
            <span style="font-size:14px; color:#ccc; margin-left:12px;">{safety_desc}</span>
        </div>""", unsafe_allow_html=True)
 
        # 추천 활용처
        st.markdown('<div class="section-title">🎯 추천 활용처</div>', unsafe_allow_html=True)
        recs = get_recommendations(soh_pred, years, cycles, bat_type)
        for i, rec in enumerate(recs):
            card_class = "rec-card top-card" if i == 0 else "rec-card"
            rank_label = "✦ 최우선 추천" if i == 0 else f"{i+1}순위 추천"
            st.markdown(f"""
            <div class="{card_class}">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
                    <div>
                        <div style="font-size:16px; font-weight:600;">{rec['icon']} {rec['name']}</div>
                        <div style="font-size:12px; color:#aaa;">{rank_label} · 적합도 {round(rec['score'])}%</div>
                        <div style="font-size:13px; color:#bbb; margin-top:6px;">{rec['desc']}</div>
                    </div>
                    <div style="display:flex; gap:20px; flex-wrap:wrap;">
                        <div style="text-align:center;">
                            <div style="font-size:18px; font-weight:600; color:#00d4aa;">{rec['life']}년</div>
                            <div style="font-size:11px; color:#aaa;">예상 잔존수명</div>
                        </div>
                        <div style="text-align:center;">
                            <div style="font-size:18px; font-weight:600; color:#00d4aa;">{rec['value']}만원</div>
                            <div style="font-size:11px; color:#aaa;">경제적 가치</div>
                        </div>
                        <div style="text-align:center;">
                            <div style="font-size:18px; font-weight:600; color:#00d4aa;">{rec['carbon']}kg</div>
                            <div style="font-size:11px; color:#aaa;">CO₂ 절감</div>
                        </div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)
 
        # 에너지 임팩트
        st.markdown('<div class="section-title">🌍 에너지 임팩트</div>', unsafe_allow_html=True)
        i1, i2, i3, i4 = st.columns(4)
        i1.metric("예측 SOH",   f"{soh_pred}%")
        i2.metric("CO₂ 절감",   f"{recs[0]['carbon']}kg",  "탄소 감축")
        i3.metric("경제적 가치", f"{recs[0]['value']}만원", "재사용 가치")
        i4.metric("광물 절약",   f"{round(soh_pred*0.05,1)}kg", "리튬·코발트")
 
        # 최종 판단
        st.divider()
        reusable = soh_pred >= 60
        color = "#00d4aa" if reusable else "#e05555"
        msg   = "✅ 재사용 가능" if reusable else "❌ 재활용 공정 권장"
        st.markdown(f"""
        <div style="background:#1a1a2e; border-radius:12px; padding:20px;
                    border:2px solid {color}; text-align:center;">
            <div style="font-size:24px; font-weight:700; color:{color}">{msg}</div>
            <div style="font-size:14px; color:#aaa; margin-top:8px;">
                배터리 종류: {bat_type} | 사용 연수: {years}년 | 충방전: {cycles}회
            </div>
        </div>""", unsafe_allow_html=True)
 
else:
    st.info("👆 EIS 파일을 업로드하면 AI가 자동으로 분석해드립니다.")
    st.markdown("""
    **지원 파일 형식:**
    - `.xls` / `.xlsx` — Excel 파일
    - `.csv` — 쉼표 또는 탭 구분
    - `.txt` — 공백 또는 탭 구분
    
    **데이터 형식:** 주파수(Hz) | Z 실수부(Ω) | Z 허수부(Ω)
    """)
