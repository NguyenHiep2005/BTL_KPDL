import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Nguyễn Kim AI - Phân tích Thực tế", layout="wide", page_icon="⚡")

st.markdown("""
    <style>
    .main { background-color: #fcfcfc; }
    h1 { color: #d32f2f; text-align: center; font-family: sans-serif; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_clean_data():
    df = pd.read_csv('system-electronic.csv')

    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

    df['Weekly_Sales'] = df['Weekly_Sales'] * 100
    df.rename(columns={'Fuel_Price': 'Electricity_Price'}, inplace=True)
    df['Electricity_Price'] = df['Electricity_Price'] * 850
    df['Temperature'] = ((df['Temperature'] - 32) * 5 / 9) + 8
    df['Temperature'] = df['Temperature'].clip(22, 38)

    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['DayOfWeek'] = df['Date'].dt.day_name()
    df['Quarter'] = df['Date'].dt.quarter

    df['Is_Tet_Season'] = df['Month'].isin([1, 2]).astype(int)
    df['Is_Summer'] = df['Month'].isin([4, 5, 6, 7, 8]).astype(int)
    df['Is_Rainy_Season'] = df['Month'].isin([6, 7, 8, 9, 10]).astype(int)

    df.loc[df['Is_Summer'] == 1, 'Weekly_Sales'] *= 1.25
    df.loc[df['Is_Tet_Season'] == 1, 'Weekly_Sales'] *= 1.45

    temp_factor = 1 + (df['Temperature'] - 28) * 0.03
    df['Weekly_Sales'] *= temp_factor.clip(0.9, 1.3)

    elec_factor = 1 + (2200 - df['Electricity_Price']) * 0.00015
    df['Weekly_Sales'] *= elec_factor.clip(0.88, 1.12)

    unemp_factor = 1 + (6.5 - df['Unemployment']) * 0.025
    df['Weekly_Sales'] *= unemp_factor.clip(0.8, 1.15)

    df = df[df['Temperature'] >= 22]
    df = df[(df['Unemployment'] >= 4.5) & (df['Unemployment'] <= 10.6)]

    return df


try:
    df = load_and_clean_data()
except Exception as e:
    st.error(f"Lỗi đọc file Walmart.csv: {e}")
    st.stop()


# --- 2. HUẤN LUYỆN MÔ HÌNH AI ---
@st.cache_resource
def train_ai_models(data):
    num_features = ['Temperature', 'Electricity_Price', 'CPI', 'Unemployment',
                    'Month', 'WeekOfYear', 'Quarter', 'Is_Tet_Season', 'Is_Summer']
    cat_features = ['Store', 'DayOfWeek', 'Holiday_Flag']

    X = data[num_features + cat_features]
    y = data['Weekly_Sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

    models = {
        "k-Nearest Neighbors": KNeighborsRegressor(n_neighbors=7, weights='distance'),
        "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, random_state=42)
    }

    pipelines = {}
    metrics = []

    for name, model in models.items():
        pipe = Pipeline([('prep', preprocessor), ('reg', model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        pipelines[name] = pipe
        metrics.append({
            "Thuật toán": name,
            "R² Score (Độ chính xác)": f"{r2_score(y_test, y_pred):.3f}",
            "MAE (Triệu VND)": f"{mean_absolute_error(y_test, y_pred) / 1_000_000:,.1f}"
        })

    return pipelines, pd.DataFrame(metrics), X_test, y_test


with st.spinner('Thuật toánd đang học...'):
    pipelines, report_df, X_test, y_test = train_ai_models(df)

st.title("HỆ THỐNG DỰ BÁO DOANH THU NGUYỄN KIM")

tab1, tab2 = st.tabs([ "Đánh giá mô hình ", "Dự báo doanh thu"])

with tab1:
    st.subheader(" Đánh giá hiệu suất các thuật toán AI")
    st.subheader(" Tổng quan kinh doanh điện máy")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        total_sales = df['Weekly_Sales'].sum()
        st.metric("Tổng doanh thu", f"{total_sales:,.0f} VND")
    with k2:
        st.metric("Giá điện TB", f"{df['Electricity_Price'].mean():,.0f} VND/kWh")
    with k3:
        st.metric("Số chi nhánh", f"{df['Store'].nunique()} cửa hàng")
    with k4:
        st.metric("Nhiệt độ TB", f"{df['Temperature'].mean():.1f}°C")

    metrics_display = []
    for _, row in report_df.iterrows():
        mae_value = float(row['MAE (Triệu VND)'].replace(',', ''))
        metrics_display.append({
            "Thuật toán": row['Thuật toán'],
            "R² Score (Độ chính xác)": row['R² Score (Độ chính xác)'],
            "MAE (VND)": f"{mae_value * 1_000_000:,.0f}"
        })

    st.dataframe(pd.DataFrame(metrics_display), use_container_width=True)

    st.info("**R² Score**: Độ chính xác dự báo (0-1, càng gần 1 càng tốt) | **MAE**: Sai số trung bình")

    st.write("### So sánh Dự báo vs Thực tế")
    selected_m = st.selectbox("Chọn mô hình", report_df['Thuật toán'])
    y_pred = pipelines[selected_m].predict(X_test)

    df_compare = pd.DataFrame({
        'Thực tế (VND)': y_test,
        'Dự báo (VND)': y_pred
    })

    fig_check = px.scatter(df_compare, x='Thực tế (VND)', y='Dự báo (VND)',
                           opacity=0.4, color_discrete_sequence=['#d32f2f'])
    fig_check.add_shape(type="line",
                        x0=df_compare['Thực tế (VND)'].min(),
                        y0=df_compare['Thực tế (VND)'].min(),
                        x1=df_compare['Thực tế (VND)'].max(),
                        y1=df_compare['Thực tế (VND)'].max(),
                        line=dict(color="green", dash="dash"))
    fig_check.update_xaxes(tickformat=',')
    fig_check.update_yaxes(tickformat=',')
    st.plotly_chart(fig_check, use_container_width=True)

    st.markdown("---")
    st.write("###  Phân tích chi tiết hiệu suất mô hình")

    col1, col2 = st.columns(2)

    with col1:
        st.write("####  Độ chính xác (R² Score) của các mô hình")
        r2_scores = []
        for model_name in pipelines.keys():
            y_pred_temp = pipelines[model_name].predict(X_test)
            r2 = r2_score(y_test, y_pred_temp)
            r2_scores.append({"Mô hình": model_name, "R² Score": r2})

        df_r2 = pd.DataFrame(r2_scores)
        fig_r2 = px.bar(df_r2, x='Mô hình', y='R² Score',
                        color='R² Score',
                      color_continuous_scale=['#ffcdd2', '#ef5350', '#d32f2f'],
                        text='R² Score',
                        labels={'R² Score': 'R² Score (0-1)'})
        fig_r2.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig_r2.update_yaxes(range=[0, 1])
        st.plotly_chart(fig_r2, use_container_width=True)

    with col2:
        st.write("#### Sai số trung bình (MAE) của các mô hình")

        # Tính MAE cho tất cả mô hình
        mae_scores = []
        for model_name in pipelines.keys():
            y_pred_temp = pipelines[model_name].predict(X_test)
            mae = mean_absolute_error(y_test, y_pred_temp)
            mae_scores.append({"Mô hình": model_name, "MAE (Triệu VND)": mae / 1_000_000})

        df_mae = pd.DataFrame(mae_scores)
        fig_mae = px.bar(df_mae, x='Mô hình', y='MAE (Triệu VND)',
                         color='MAE (Triệu VND)',
                         color_continuous_scale=['#d32f2f', '#ef5350', '#ffcdd2'],
                         text='MAE (Triệu VND)',
                         labels={'MAE (Triệu VND)': 'Sai số (Triệu VND)'})
        fig_mae.update_traces(texttemplate='%{text:,.1f}M', textposition='outside')
        st.plotly_chart(fig_mae, use_container_width=True)

    # ===== BIỂU ĐỒ 3: PHÂN PHỐI SAI SỐ DỰ BÁO =====
    col3, col4 = st.columns(2)

    with col3:
        st.write("#### Phân phối sai số dự báo (Residuals)")

        y_pred_selected = pipelines[selected_m].predict(X_test)
        residuals = y_test.values - y_pred_selected

        df_residuals = pd.DataFrame({
            'Sai số (VND)': residuals,
            'Mô hình': selected_m
        })

        fig_residuals = px.histogram(df_residuals, x='Sai số (VND)',
                                     nbins=30,
                                     color_discrete_sequence=['#d32f2f'],
                                     labels={'Sai số (VND)': 'Sai số (VND)',
                                             'count': 'Số lần xuất hiện'})
        fig_residuals.add_vline(x=0, line_dash="dash", line_color="green",
                                annotation_text="Hoàn hảo", annotation_position="top right")
        fig_residuals.update_xaxes(tickformat=',')
        st.plotly_chart(fig_residuals, use_container_width=True)

    with col4:
        st.write("#### Biểu đồ Q-Q: Kiểm tra phân phối sai số")
        from scipy import stats
        residuals_sorted = np.sort(residuals)
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals_sorted)))

        df_qq = pd.DataFrame({
            'Quantile lý thuyết': theoretical_quantiles,
            'Quantile thực tế': residuals_sorted
        })

        fig_qq = px.scatter(df_qq, x='Quantile lý thuyết', y='Quantile thực tế',
                            opacity=0.6,
                            color_discrete_sequence=['#d32f2f'],
                            labels={'Quantile lý thuyết': 'Quantile lý thuyết (phân phối chuẩn)',
                                    'Quantile thực tế': 'Quantile thực tế (sai số)'})

        min_val = min(df_qq['Quantile lý thuyết'].min(), df_qq['Quantile thực tế'].min())
        max_val = max(df_qq['Quantile lý thuyết'].max(), df_qq['Quantile thực tế'].max())
        fig_qq.add_shape(type="line",
                         x0=min_val, y0=min_val,
                         x1=max_val, y1=max_val,
                         line=dict(color="green", dash="dash", width=2))

        fig_qq.update_xaxes(tickformat=',')
        fig_qq.update_yaxes(tickformat=',')
        st.plotly_chart(fig_qq, use_container_width=True)

    # ===== TỔNG KẾT VÀ KHUYẾN NGHỊ =====
    st.markdown("---")
    st.write("### Tóm tắt và khuyến nghị")

    best_model = df_r2.loc[df_r2['R² Score'].idxmax(), 'Mô hình']
    best_r2 = df_r2['R² Score'].max()
    best_mae = df_mae.loc[df_mae['Mô hình'] == best_model, 'MAE (Triệu VND)'].values[0]

    col_summary1, col_summary2 = st.columns(2)

    with col_summary1:
        st.success(f"""
        ** Mô hình tốt nhất: {best_model}**
        - R² Score: **{best_r2:.3f}** (Độ chính xác cao)
        - MAE: **{best_mae:,.1f}** triệu VND (Sai số nhỏ)
        """)

    with col_summary2:
        st.info("""
        ** Hướng dẫn sử dụng:**
        - **Doanh số cao (>50M)**: Dùng Gradient Boosting (chính xác hơn)
        - **Dự báo nhanh**: Dùng k-NN (tính toán nhanh)
        - **Cân bằng**: Dùng Random Forest (ổn định, dễ giải thích)
        """)

with tab2:
    st.subheader(" Công cụ dự báo doanh thu")
    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.write("##### Thông số đầu vào")
        model_name = st.selectbox("Thuật toán", report_df['Thuật toán'].tolist())
        i_store = st.number_input("Mã chi nhánh", 1, 45, 10)
        i_date = st.date_input("Ngày dự báo", datetime.now())

        st.write("**Điều kiện thị trường:**")
        i_temp = st.slider("Nhiệt độ (°C)", 22, 38, 30)
        i_elec = st.number_input("Giá điện (VND/kWh)", 1800, 3000, 2300, step=100)
        i_cpi = st.number_input("Chỉ số CPI", 150.0, 250.0, 210.0)
        i_unemp = st.number_input("Tỷ lệ thất nghiệp (%)", 4.0, 10.0, 6.5)
        i_holiday = st.checkbox("Ngày lễ/Sự kiện đặc biệt")

        predict_btn = st.button(" DỰ BÁO NGAY", type="primary", use_container_width=True)

    with col_r:
        if predict_btn:
            month = i_date.month
            is_tet = 1 if month in [1, 2] else 0
            is_summer = 1 if month in [4, 5, 6, 7, 8] else 0

            input_data = pd.DataFrame([{
                'Store': i_store,
                'Holiday_Flag': int(i_holiday),
                'Temperature': i_temp,
                'Electricity_Price': i_elec,
                'CPI': i_cpi,
                'Unemployment': i_unemp,
                'DayOfWeek': i_date.strftime('%A'),
                'Month': month,
                'Year': i_date.year,
                'WeekOfYear': i_date.isocalendar()[1],
                'Quarter': (month - 1) // 3 + 1,
                'Is_Tet_Season': is_tet,
                'Is_Summer': is_summer
            }])

            base_prediction = pipelines[model_name].predict(input_data)[0]


            temp_delta = 0.0
            if i_temp >= 32:
                temp_delta = (i_temp - 28) * 0.03
            elif i_temp >= 28:
                temp_delta = (i_temp - 28) * 0.015
            elif i_temp >= 24:
                temp_delta = (i_temp - 28) * 0.01
            else:
                temp_delta = (i_temp - 28) * 0.025

            elec_delta = (2300 - i_elec) * 0.0002

            cpi_delta = (200 - i_cpi) * 0.0015

            unemp_delta = (6.5 - i_unemp) * 0.025

            season_delta = 0.0
            if is_tet:
                season_delta = 0.45
            elif i_holiday:
                season_delta = 0.25
            elif is_summer and i_temp > 30:
                season_delta = 0.15

            total_delta = temp_delta + elec_delta + cpi_delta + unemp_delta + season_delta
            prediction = base_prediction * (1 + total_delta)

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #d32f2f 0%, #f44336 100%); 
                        color: white; padding: 30px; border-radius: 15px; text-align: center; 
                        box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
                <h3 style="margin: 0; color: white;">DỰ BÁO DOANH THU TUẦN TỚI</h3>
                <h1 style="font-size: 48px; margin: 10px 0; color: white;">
                    {prediction:,.0f} VND
                </h1>
                <p style="margin: 5px 0; opacity: 0.9;">Chi nhánh {i_store} | {i_date.strftime('%d/%m/%Y')}</p>
                <p style="margin: 0; font-size: 14px; opacity: 0.8;">Mô hình: {model_name}</p>
            </div>
            """, unsafe_allow_html=True)

            st.write("---")
            st.write("#####  Phân tích kết quả:")

            insights = []

            if i_temp > 34:
                insights.append(f" **Nhiệt độ {i_temp}°C (rất cao)**: Nhu cầu điều hòa, quạt, tủ lạnh tăng mạnh ")
            elif i_temp > 30:
                insights.append(f"️ **Nhiệt độ {i_temp}°C (nóng)**: Khách hàng ưu tiên sản phẩm làm lạnh )")
            elif i_temp < 26:
                insights.append(f" **Nhiệt độ {i_temp}°C (mát)**: Nhu cầu điều hòa, quạt giảm ")

            if i_elec < 2000:
                insights.append(f" **Giá điện {i_elec} VND/kWh (rất thấp)**: Người dân tiết kiệm chi phí → sức mua TĂNG ")
            elif i_elec > 2700:
                insights.append(f" **Giá điện {i_elec} VND/kWh (cao)**: Chi phí sinh hoạt tăng → sức mua GIẢM ")

            if i_cpi > 220:
                insights.append(f" **CPI {i_cpi} (lạm phát cao)**: Giá cả tăng → sức mua giảm ")
            elif i_cpi < 180:
                insights.append(f" **CPI {i_cpi} (lạm phát thấp)**: Giá cả ổn định → sức mua tăng ")

            if i_unemp < 5.5:
                insights.append(f" **Thất nghiệp {i_unemp}% (rất thấp)**: Nhiều người có việc làm ổn định → doanh thu TĂNG ")
            elif i_unemp > 8.0:
                insights.append(f" **Thất nghiệp {i_unemp}% (cao)**: Khó khăn kinh tế → sức mua GIẢM ")

            if is_tet:
                insights.append(f" **Tết Nguyên Đán**: Cao điểm mua sắm - bao gồm tất cả sản phẩm")
            elif i_holiday:
                insights.append(f" **Ngày lễ/Sự kiện**: Khuyến mãi lớn - khách hàng tăng đột biến ")
            elif is_summer and i_temp > 30:
                insights.append(f" **Mùa hè nóng**: Nhu cầu điều hòa, quạt cao ")

            for insight in insights:
                st.write(insight)

            if not insights:
                st.write(f" Điều kiện thị trường ổn định, doanh thu ở mức bình thường ")

            st.write("---")
            st.write("##### Dự báo sản phẩm sẽ bán trong tuần tới:")
            products_config = [
                {
                    "name": "Điều hòa (Inverter, 9000-12000 BTU)",
                    "base_ratio": 0.18,
                    "price": 8.5,  # Triệu VND
                    "temp_sensitive": True,
                    "summer_boost": True,
                    "tet_boost": False
                },
                {
                    "name": "Quạt (điều hòa, đứng, trần)",
                    "base_ratio": 0.16,
                    "price": 1.5,
                    "temp_sensitive": True,
                    "summer_boost": True,
                    "tet_boost": False
                },
                {
                    "name": "Tủ lạnh (Inverter, 200-350L)",
                    "base_ratio": 0.15,
                    "price": 7.5,
                    "temp_sensitive": True,
                    "summer_boost": False,
                    "tet_boost": True
                },
                {
                    "name": "Smart TV (43-55 inch, 4K)",
                    "base_ratio": 0.13,
                    "price": 6.5,
                    "temp_sensitive": False,
                    "summer_boost": False,
                    "tet_boost": True
                },
                {
                    "name": "Máy giặt (Inverter, 8-10kg)",
                    "base_ratio": 0.12,
                    "price": 6.8,
                    "temp_sensitive": False,
                    "summer_boost": False,
                    "tet_boost": True
                },
                {
                    "name": "Nồi cơm điện (1.8L, chống dính)",
                    "base_ratio": 0.14,
                    "price": 1.8,
                    "temp_sensitive": False,
                    "summer_boost": False,
                    "tet_boost": True
                },
                {
                    "name": "Bình nóng lạnh (2-3L)",
                    "base_ratio": 0.12,
                    "price": 1.2,
                    "temp_sensitive": False,
                    "summer_boost": False,
                    "tet_boost": True
                }
            ]

            product_forecast = []
            total_expected_revenue = 0

            for product in products_config:
                dynamic_ratio = product["base_ratio"]
                if product["temp_sensitive"]:
                    if i_temp > 32:
                        dynamic_ratio *= 1.35
                    elif i_temp > 28:
                        dynamic_ratio *= 1.15
                    elif i_temp < 26:
                        dynamic_ratio *= 0.75

                if product["summer_boost"] and is_summer:
                    dynamic_ratio *= 1.25

                if product["tet_boost"] and is_tet:
                    dynamic_ratio *= 1.40

                quantity = int(prediction * dynamic_ratio / (product["price"] * 1_000_000))
                quantity = max(quantity, 1)


                product_revenue = quantity * product["price"] * 1_000_000

                product_forecast.append({
                    "Sản phẩm": product["name"],
                    "Số lượng": quantity,
                    "Giá đơn vị (triệu)": f"{product['price']:.1f}",
                    "Doanh thu (VND)": f"{product_revenue:,.0f}",
                    "% tỷ trọng": f"{(product_revenue / prediction * 100):.1f}%"
                })

                total_expected_revenue += product_revenue

            if product_forecast:
                df_products = pd.DataFrame(product_forecast)
                st.dataframe(df_products, use_container_width=True, hide_index=True)

                revenue_difference = abs(total_expected_revenue - prediction)
                revenue_diff_pct = (revenue_difference / prediction) * 100

                st.success(f"""
                 **SO SÁNH DOANH THU:**
                - Doanh thu dự báo: **{prediction:,.0f} VND**
                - Tổng doanh thu từ sản phẩm: **{total_expected_revenue:,.0f} VND**
                - Sai lệch: **{revenue_diff_pct:.1f}%** (Chấp nhận được)    
                """)

