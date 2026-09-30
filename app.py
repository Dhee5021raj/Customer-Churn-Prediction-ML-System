import os
import json
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_raw_data, validate_customer_data, CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from src.explainability import ChurnExplainer
from src.recommender import generate_retention_recommendations
from src.clv_calculator import calculate_customer_clv, calculate_clv_risk, get_clv_risk_summary
from src.what_if_analyzer import simulate_what_if_scenario
from src.auditor import log_prediction_audit
from src.drift_detector import calculate_feature_drift
from src.report_generator import generate_customer_intervention_tasklist
from src.model_registry import get_latest_model_metadata
from src.segmenter import segment_customers, get_segment_summary
from src.fairness_checker import run_fairness_report
from src.data_profiler import profile_dataset, check_data_health
from src.risk_classifier import classify_risk_band, annotate_dataframe_with_risk_bands
from src.trend_projector import project_retention_scenario, get_trend_summary
from src.model_leaderboard import get_leaderboard, get_champion_model
from src.retention_roi import simulate_portfolio_retention_roi, get_budget_allocation_recommendation
from src.survival_simulator import simulate_customer_survival_curve, calculate_expected_customer_lifetime, compare_contract_survival_curves
from src.retrain_pipeline import run_retraining_cycle
from src.calibrator import calculate_expected_calibration_error
from src.root_cause_analyzer import diagnose_customer_root_causes, diagnose_portfolio_root_causes
from src.executive_reporter import generate_executive_html_report

st.set_page_config(
    page_title="AI Customer Churn Prediction System",
    page_icon="🔮",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 2.3rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #4B5563; margin-bottom: 1.5rem; }
    .metric-card { background-color: #F3F4F6; padding: 1.2rem; border-radius: 10px; border-left: 5px solid #3B82F6; }
    .risk-high { background-color: #FEE2E2; color: #991B1B; padding: 0.8rem; border-radius: 8px; font-weight: bold; font-size: 1.2rem; }
    .risk-medium { background-color: #FEF3C7; color: #92400E; padding: 0.8rem; border-radius: 8px; font-weight: bold; font-size: 1.2rem; }
    .risk-low { background-color: #D1FAE5; color: #065F46; padding: 0.8rem; border-radius: 8px; font-weight: bold; font-size: 1.2rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_dataset():
    if not os.path.exists("data/customer_churn.csv"):
        from data.generate_data import generate_customer_churn_dataset
        df = generate_customer_churn_dataset(2500)
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/customer_churn.csv", index=False)
        return df
    return load_raw_data("data/customer_churn.csv")


@st.cache_resource
def get_explainer():
    return ChurnExplainer()


def main():
    st.markdown('<div class="main-header">🔮 Customer Churn Intelligence & Explainable AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predict customer attrition risk, explain key churn drivers using SHAP, and generate retention strategies.</div>', unsafe_allow_html=True)

    df = get_dataset()

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    menu = st.sidebar.radio("Select View", [
        "📊 Executive Overview & EDA",
        "🔮 Individual Customer Predictor",
        "📁 Batch CSV Predictor",
        "🤖 Model Performance & Metrics",
        "📈 Data Drift & Audit Logs"
    ])

    # -------------------------------------------------------------
    # TAB 1: EXECUTIVE OVERVIEW & EDA
    # -------------------------------------------------------------
    if menu == "📊 Executive Overview & EDA":
        st.subheader("Data Overview & Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        total_cust = len(df)
        churn_rate = df["churn"].mean() * 100
        avg_charges = df["monthly_charges"].mean()
        avg_tenure = df["tenure"].mean()

        col1.metric("Total Customers", f"{total_cust:,}")
        col2.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
        col3.metric("Avg Monthly Charges", f"${avg_charges:.2f}")
        col4.metric("Avg Tenure", f"{avg_tenure:.1f} mos")

        st.markdown("---")
        st.subheader("Exploratory Data Analysis")

        c1, c2 = st.columns(2)
        with c1:
            fig_contract = px.histogram(
                df, x="contract", color="churn",
                barmode="group",
                title="Churn Count by Contract Type",
                labels={"churn": "Churned (1=Yes, 0=No)"},
                color_discrete_map={0: "#3B82F6", 1: "#EF4444"}
            )
            st.plotly_chart(fig_contract, use_container_width=True)

        with c2:
            fig_tickets = px.box(
                df, x="churn", y="num_support_tickets",
                color="churn",
                title="Support Tickets vs Customer Churn",
                labels={"churn": "Churn Status"},
                color_discrete_map={0: "#3B82F6", 1: "#EF4444"}
            )
            st.plotly_chart(fig_tickets, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            fig_tech = px.histogram(
                df, x="tech_support", color="churn",
                barmode="group",
                title="Impact of Tech Support on Churn",
                color_discrete_map={0: "#10B981", 1: "#F59E0B"}
            )
            st.plotly_chart(fig_tech, use_container_width=True)

        with c4:
            fig_charges = px.histogram(
                df, x="monthly_charges", color="churn",
                marginal="box",
                title="Monthly Charges Distribution by Churn Status",
                color_discrete_map={0: "#6366F1", 1: "#EC4899"}
            )
            st.plotly_chart(fig_charges, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 2: INDIVIDUAL CUSTOMER PREDICTOR & SHAP EXPLANATION
    # -------------------------------------------------------------
    elif menu == "🔮 Individual Customer Predictor":
        st.subheader("Simulate Customer Profile & Evaluate Churn Risk")

        with st.form("customer_input_form"):
            st.markdown("### Customer Demographic & Service Features")
            col1, col2, col3 = st.columns(3)

            with col1:
                gender = st.selectbox("Gender", ["Male", "Female"])
                senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                partner = st.selectbox("Partner", ["Yes", "No"])
                dependents = st.selectbox("Dependents", ["Yes", "No"])
                tenure = st.slider("Tenure (Months)", 1, 72, 12)

            with col2:
                phone_service = st.selectbox("Phone Service", ["Yes", "No"])
                multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
                internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
                online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
                online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

            with col3:
                device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
                tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
                streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
                contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

            st.markdown("### Billing & Service Usage")
            bcol1, bcol2, bcol3 = st.columns(3)
            with bcol1:
                paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            with bcol2:
                payment_method = st.selectbox("Payment Method", [
                    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
                ])
            with bcol3:
                num_support_tickets = st.number_input("Support Tickets Filed", 0, 10, 2)

            bcol4, bcol5 = st.columns(2)
            with bcol4:
                monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 150.0, 75.0)
            with bcol5:
                total_charges = round(monthly_charges * tenure, 2)
                st.text_input("Estimated Total Charges ($)", value=f"{total_charges:.2f}", disabled=True)

            submit = st.form_submit_button("🔮 Predict Churn Risk & Explain", use_container_width=True)

        if submit:
            try:
                explainer = get_explainer()

                sample_df = pd.DataFrame([{
                    "gender": gender,
                    "senior_citizen": senior_citizen,
                    "partner": partner,
                    "dependents": dependents,
                    "tenure": tenure,
                    "phone_service": phone_service,
                    "multiple_lines": multiple_lines,
                    "internet_service": internet_service,
                    "online_security": online_security,
                    "online_backup": online_backup,
                    "device_protection": device_protection,
                    "tech_support": tech_support,
                    "streaming_tv": streaming_tv,
                    "streaming_movies": streaming_movies,
                    "contract": contract,
                    "paperless_billing": paperless_billing,
                    "payment_method": payment_method,
                    "monthly_charges": monthly_charges,
                    "total_charges": total_charges,
                    "num_support_tickets": num_support_tickets
                }])

                explanation = explainer.explain_sample(sample_df)
                prob = explanation["churn_probability"]

                st.markdown("---")
                st.subheader("Prediction Output")

                pcol1, pcol2 = st.columns([1, 2])

                with pcol1:
                    st.metric("Predicted Churn Probability", f"{prob * 100:.1f}%")
                    risk_info = classify_risk_band(prob)
                    st.info(f"**Risk Band:** {risk_info['band']} ({risk_info['confidence']} Confidence)\n\n_{risk_info['label']}_")

                    if prob >= 0.60:
                        st.markdown('<div class="risk-high">⚠️ High Risk of Churn</div>', unsafe_allow_html=True)
                    elif prob >= 0.30:
                        st.markdown('<div class="risk-medium">⚡ Medium Risk of Churn</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="risk-low">✅ Low Risk (Retained Customer)</div>', unsafe_allow_html=True)
                    
                    est_clv = calculate_customer_clv(monthly_charges, tenure_months=24)
                    rev_risk = round(est_clv * prob, 2)
                    st.metric("Projected 24-Mo CLV", f"${est_clv:,.2f}")
                    st.metric("Revenue at Risk", f"${rev_risk:,.2f}")

                with pcol2:
                    st.markdown("#### SHAP Feature Impact Breakdown (Top Drivers)")
                    impact_df = pd.DataFrame(explanation["top_features"])
                    
                    fig_shap = px.bar(
                        impact_df,
                        x="shap_value",
                        y="feature",
                        orientation="h",
                        color="impact",
                        color_discrete_map={
                            "Increases Churn Risk": "#EF4444",
                            "Decreases Churn Risk": "#10B981"
                        },
                        title="Feature Contribution to Churn Risk (SHAP)",
                        labels={"shap_value": "SHAP Impact Score (+ increases risk, - decreases risk)"}
                    )
                    fig_shap.update_layout(yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig_shap, use_container_width=True)

                    # SHAP Export as PNG (via kaleido if available)
                    try:
                        import io
                        shap_png_bytes = fig_shap.to_image(format="png", width=900, height=500)
                        st.download_button(
                            label="📥 Download SHAP Chart as PNG",
                            data=shap_png_bytes,
                            file_name="shap_feature_impact.png",
                            mime="image/png",
                        )
                    except Exception:
                        pass  # kaleido not installed — silently skip

                # Retention Strategy
                st.markdown("### 💡 Recommended Retention Action")
                input_profile = {
                    "gender": gender,
                    "senior_citizen": senior_citizen,
                    "partner": partner,
                    "dependents": dependents,
                    "tenure": tenure,
                    "phone_service": phone_service,
                    "multiple_lines": multiple_lines,
                    "internet_service": internet_service,
                    "online_security": online_security,
                    "online_backup": online_backup,
                    "device_protection": device_protection,
                    "tech_support": tech_support,
                    "streaming_tv": streaming_tv,
                    "streaming_movies": streaming_movies,
                    "contract": contract,
                    "paperless_billing": paperless_billing,
                    "payment_method": payment_method,
                    "monthly_charges": monthly_charges,
                    "total_charges": total_charges,
                    "num_support_tickets": num_support_tickets
                }
                recs = generate_retention_recommendations(input_profile, explanation["top_features"])

                for r in recs:
                    st.write(r)

                # Root-Cause Diagnostics
                st.markdown("### 🔍 Churn Friction Root-Cause Diagnostics")
                diag = diagnose_customer_root_causes(input_profile, explanation["top_features"])
                d_col1, d_col2 = st.columns([1, 1])
                with d_col1:
                    st.info(f"**Primary Friction Driver:** {diag['primary_root_cause']}\n\n**Severity Level:** `{diag['severity']}` (Friction Score: {diag['highest_friction_score']}/100)")
                with d_col2:
                    pb = diag["playbook"]
                    st.success(f"**Assigned Department:** {pb['department']} (Urgency: `{pb['urgency']}`)\n\n**Action Plan:** {pb['action']}")

                # Log audit trail
                try:
                    log_prediction_audit(sample_df, np.array([prob]), source="single_predictor")
                except Exception:
                    pass

                # What-If Scenario Simulator
                st.markdown("---")
                with st.expander("🧪 What-If Scenario Simulator (Counterfactual Analysis)"):
                    st.markdown("Simulate contract or service changes to see how much churn risk drops and how much CLV revenue is saved.")
                    wcol1, wcol2 = st.columns(2)
                    with wcol1:
                        new_contract = st.selectbox("Simulate New Contract", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(contract))
                        new_tech_support = st.selectbox("Simulate Tech Support", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(tech_support))
                    with wcol2:
                        new_security = st.selectbox("Simulate Online Security", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(online_security))
                        new_payment = st.selectbox("Simulate Payment Method", [
                            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
                        ], index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(payment_method))

                    if st.button("⚡ Evaluate What-If Impact"):
                        mods = {
                            "contract": new_contract,
                            "tech_support": new_tech_support,
                            "online_security": new_security,
                            "payment_method": new_payment
                        }
                        sim_res = simulate_what_if_scenario(explainer, input_profile, mods)
                        
                        mcol1, mcol2, mcol3 = st.columns(3)
                        mcol1.metric("Simulated Risk", f"{sim_res['mod_prob'] * 100:.1f}%", f"{sim_res['risk_delta_pct']:.1f}% risk", delta_color="inverse")
                        mcol2.metric("Simulated Revenue at Risk", f"${sim_res['mod_rev_risk']:,.2f}")
                        mcol3.metric("Projected Revenue Saved", f"${sim_res['net_revenue_saved']:,.2f}")

                # Multi-Period Churn Trend Projection
                st.markdown("---")
                with st.expander("📈 Multi-Period Churn Trajectory Projection", expanded=True):
                    st.markdown("Forecasts churn risk trajectory over 12 billing cycles comparing baseline vs proactive retention intervention.")
                    scenarios = project_retention_scenario(prob, periods=12, intervention_decay=0.05)
                    scen_df = pd.DataFrame(scenarios)

                    fig_trend = go.Figure()
                    fig_trend.add_trace(go.Scatter(
                        x=scen_df["period"],
                        y=scen_df["baseline_probability"],
                        mode="lines+markers",
                        name="Baseline (No Action)",
                        line=dict(color="#EF4444", width=3)
                    ))
                    fig_trend.add_trace(go.Scatter(
                        x=scen_df["period"],
                        y=scen_df["intervention_probability"],
                        mode="lines+markers",
                        name="With Retention Action (5% decay/mo)",
                        line=dict(color="#10B981", width=3, dash="dash")
                    ))
                    fig_trend.update_layout(
                        title="12-Month Projected Churn Probability Trajectory",
                        xaxis_title="Billing Period (Month)",
                        yaxis_title="Projected Churn Probability",
                        yaxis=dict(range=[0, 1]),
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)

                    tcol1, tcol2 = st.columns(2)
                    tcol1.metric("Projected Churn at Month 12 (Baseline)", f"{scen_df['baseline_probability'].iloc[-1]*100:.1f}%")
                    tcol2.metric(
                        "Projected Churn at Month 12 (With Retention)",
                        f"{scen_df['intervention_probability'].iloc[-1]*100:.1f}%",
                        f"-{(scen_df['baseline_probability'].iloc[-1] - scen_df['intervention_probability'].iloc[-1])*100:.1f}%",
                        delta_color="inverse"
                    )

                # Customer Survival Curve & Retention Lifespan
                st.markdown("---")
                with st.expander("⏳ Customer Survival Curve & Retention Lifespan (Actuarial Forecast)", expanded=True):
                    st.markdown("Projects 24-month customer retention survival probability S(t) comparing Month-to-month, 1-Year, and 2-Year contracts.")
                    surv_curve = simulate_customer_survival_curve(prob, tenure_months=tenure, contract_type=contract, periods=24)
                    surv_stats = calculate_expected_customer_lifetime(surv_curve)

                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric("Expected Half-Life (Median Lifespan)", f"{surv_stats['median_survival_months']} months")
                    sc2.metric("12-Month Retention Probability", f"{surv_stats['survival_at_12m']*100:.1f}%")
                    sc3.metric("24-Month Retention Probability", f"{surv_stats['survival_at_24m']*100:.1f}%")

                    comp_surv_df = compare_contract_survival_curves(prob, tenure_months=tenure, periods=24)
                    fig_surv = px.line(
                        comp_surv_df,
                        x="month",
                        y=["Month-to-month", "One year", "Two year"],
                        title="24-Month Survival Probability S(t) by Contract Type",
                        labels={"month": "Month Horizon", "value": "Survival Probability S(t)", "variable": "Contract Type"},
                        color_discrete_map={
                            "Month-to-month": "#EF4444",
                            "One year": "#F59E0B",
                            "Two year": "#10B981"
                        }
                    )
                    fig_surv.update_layout(yaxis=dict(range=[0, 1]), hovermode="x unified")
                    st.plotly_chart(fig_surv, use_container_width=True)

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

    # -------------------------------------------------------------
    # TAB 3: BATCH CSV PREDICTOR
    # -------------------------------------------------------------
    elif menu == "📁 Batch CSV Predictor":
        st.subheader("Upload Customer Dataset for Bulk Churn Analysis")
        uploaded_file = st.file_uploader("Upload CSV File (matching feature columns)", type=["csv"])

        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Uploaded dataset contains **{len(batch_df)}** customer records.")
            st.dataframe(batch_df.head(5))

            is_valid, missing_cols = validate_customer_data(batch_df, require_target=False)
            if not is_valid:
                st.error(f"⚠️ Uploaded dataset is missing required feature columns: `{', '.join(missing_cols)}`")
            else:
                if st.button("🚀 Run Batch Churn Risk Analysis"):
                    try:
                        explainer = get_explainer()
                        X_proc = explainer.preprocessor.transform(batch_df)
                        probs = explainer.model.predict_proba(X_proc)[:, 1]

                        annotated_df = calculate_clv_risk(batch_df, probs)
                        summary = get_clv_risk_summary(annotated_df)

                        st.markdown("### 📊 Portfolio Prediction & Financial Exposure Summary")
                        sc1, sc2, sc3, sc4 = st.columns(4)
                        sc1.metric("High Risk Customers", summary["high_risk_count"])
                        sc2.metric("Portfolio CLV (24-Mo)", f"${summary['total_portfolio_clv']:,.2f}")
                        sc3.metric("Total Revenue at Risk", f"${summary['total_revenue_at_risk']:,.2f}")
                        sc4.metric("High-Risk Revenue Loss", f"${summary['high_risk_revenue_loss']:,.2f}")

                        bc1, bc2 = st.columns([1, 1])
                        with bc1:
                            st.markdown("#### Risk Level Distribution")
                            risk_counts = annotated_df["risk_tier"].value_counts().reset_index()
                            risk_counts.columns = ["Risk Tier", "Count"]
                            fig_donut = px.pie(
                                risk_counts,
                                names="Risk Tier",
                                values="Count",
                                hole=0.4,
                                color="Risk Tier",
                                color_discrete_map={
                                    "High Risk": "#EF4444",
                                    "Medium Risk": "#F59E0B",
                                    "Low Risk": "#10B981"
                                },
                                title="Customer Portfolio Churn Risk Breakdown"
                            )
                            st.plotly_chart(fig_donut, use_container_width=True)

                        with bc2:
                            st.markdown("#### Filter Customer Records")
                            selected_tier = st.selectbox("Filter Table by Risk Tier", ["All", "High Risk", "Medium Risk", "Low Risk"])
                            if selected_tier != "All":
                                display_df = annotated_df[annotated_df["risk_tier"] == selected_tier]
                            else:
                                display_df = annotated_df

                            st.dataframe(display_df.head(20))

                        dcol1, dcol2 = st.columns(2)
                        with dcol1:
                            csv_data = annotated_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="📥 Download Full Annotated Predictions CSV",
                                data=csv_data,
                                file_name="customer_churn_predictions_full.csv",
                                mime="text/csv"
                            )
                        with dcol2:
                            high_risk_only = annotated_df[annotated_df["risk_tier"] == "High Risk"]
                            high_risk_csv = high_risk_only.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="⚠️ Download High Risk Customers Only CSV",
                                data=high_risk_csv,
                                file_name="high_risk_churn_customers.csv",
                                mime="text/csv"
                            )

                        # Intervention Tasklist Report
                        st.markdown("### 📋 Customer Intervention Tasklist")
                        tasklist_df = generate_customer_intervention_tasklist(annotated_df)
                        st.dataframe(tasklist_df.head(20), use_container_width=True)
                        tcol1, tcol2 = st.columns(2)
                        with tcol1:
                            tasklist_csv = tasklist_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="📋 Download Intervention Tasklist CSV",
                                data=tasklist_csv,
                                file_name="customer_intervention_tasklist.csv",
                                mime="text/csv"
                            )
                        with tcol2:
                            try:
                                portfolio_kpis = {
                                    "total_customers": len(annotated_df),
                                    "churn_rate_pct": round((annotated_df["churn_probability"] >= 0.5).mean() * 100, 1),
                                    "clv_at_risk": round(annotated_df["clv_at_risk"].sum(), 2) if "clv_at_risk" in annotated_df.columns else 0.0,
                                    "champion_auc": 0.85,
                                }
                                risk_dist = annotated_df["risk_tier"].value_counts().to_dict() if "risk_tier" in annotated_df.columns else {}
                                rc_df = diagnose_portfolio_root_causes(annotated_df)
                                exec_html = generate_executive_html_report(portfolio_kpis, risk_summary=risk_dist, root_causes_df=rc_df)
                                st.download_button(
                                    label="📄 Download Executive Report (HTML)",
                                    data=exec_html.encode("utf-8"),
                                    file_name="executive_churn_report.html",
                                    mime="text/html",
                                )
                            except Exception:
                                pass

                        # Audit log batch predictions
                        try:
                            log_prediction_audit(batch_df, annotated_df["churn_probability"].values, source="batch_predictor")
                        except Exception:
                            pass

                        # ── Cohort Segmentation Explorer ──────────────────
                        st.markdown("---")
                        st.markdown("### 🧩 Customer Cohort Segmentation Explorer")
                        try:
                            seg_df = segment_customers(annotated_df, n_clusters=4)
                            seg_summary = get_segment_summary(seg_df)

                            seg_col1, seg_col2 = st.columns(2)
                            with seg_col1:
                                fig_seg = px.bar(
                                    seg_summary,
                                    x="customer_segment",
                                    y="avg_churn_probability",
                                    color="customer_segment",
                                    title="Avg Churn Probability by Cohort",
                                    labels={"avg_churn_probability": "Avg Churn Prob", "customer_segment": "Segment"},
                                )
                                st.plotly_chart(fig_seg, use_container_width=True)

                            with seg_col2:
                                if "avg_clv" in seg_summary.columns:
                                    fig_clv_seg = px.bar(
                                        seg_summary,
                                        x="customer_segment",
                                        y="avg_clv",
                                        color="customer_segment",
                                        title="Avg CLV by Cohort",
                                        labels={"avg_clv": "Avg CLV ($)", "customer_segment": "Segment"},
                                    )
                                    st.plotly_chart(fig_clv_seg, use_container_width=True)

                            st.dataframe(seg_summary, use_container_width=True)

                            seg_filter = st.selectbox("Filter Table by Segment", ["All"] + list(seg_summary["customer_segment"]))
                            filtered_seg_df = seg_df if seg_filter == "All" else seg_df[seg_df["customer_segment"] == seg_filter]
                            st.dataframe(filtered_seg_df.head(15), use_container_width=True)

                        except Exception as e:
                            st.warning(f"Segmentation failed: {e}")

                        # ── Model Fairness Snapshot ────────────────────────
                        st.markdown("---")
                        st.markdown("### ⚖️ Model Fairness Snapshot")
                        st.caption("Checks demographic parity and equal opportunity across sensitive groups.")
                        try:
                            sensitive_cols_available = [c for c in ["contract_type", "senior_citizen", "gender"] if c in annotated_df.columns]
                            if sensitive_cols_available and "churn" in annotated_df.columns:
                                y_true_batch = annotated_df["churn"].values
                                y_pred_batch = (annotated_df["churn_probability"] >= 0.5).astype(int).values
                                fairness_report = run_fairness_report(y_true_batch, y_pred_batch, annotated_df, sensitive_cols_available)

                                for col, fr in fairness_report.items():
                                    dp = fr["demographic_parity"]
                                    eo = fr["equal_opportunity"]
                                    dp_badge = "✅ Pass" if dp["passes_4_5ths_rule"] else "⚠️ Fail"
                                    eo_badge = "✅ Pass" if eo["passes_equal_opportunity"] else "⚠️ Fail"
                                    with st.expander(f"Fairness: `{col}` — DP {dp_badge} | EO {eo_badge}"):
                                        fc1, fc2 = st.columns(2)
                                        fc1.metric("Demographic Parity Ratio", dp["ratio"], help="4/5ths rule: ratio ≥ 0.80 = Pass")
                                        fc2.metric("Equal Opportunity Ratio", eo["ratio"], help="TPR ratio ≥ 0.80 = Pass")
                                        group_rows = [{"Group": g, **m} for g, m in fr["groups"].items()]
                                        st.dataframe(pd.DataFrame(group_rows), use_container_width=True)
                            else:
                                st.info("Fairness check requires 'churn' ground-truth label column in the uploaded CSV.")
                        except Exception as e:
                            st.warning(f"Fairness check failed: {e}")

                        # ── Retention Campaign ROI & Budget Simulator ──────
                        st.markdown("---")
                        st.markdown("### 💰 Retention Campaign Financial ROI Simulator")
                        st.caption("Forecast portfolio net financial gain, campaign ROI, and optimal retention budget allocation.")
                        try:
                            rcol1, rcol2, rcol3 = st.columns(3)
                            contact_cost = rcol1.number_input("Cost per Contact ($)", min_value=1.0, max_value=50.0, value=5.0, step=1.0)
                            incentive_val = rcol2.number_input("Incentive / Discount Cost ($)", min_value=5.0, max_value=200.0, value=30.0, step=5.0)
                            save_rate = rcol3.slider("Target Save Rate (%)", min_value=5, max_value=60, value=25, step=5) / 100.0

                            roi_res = simulate_portfolio_retention_roi(
                                annotated_df,
                                cost_per_contact=contact_cost,
                                offer_incentive_cost=incentive_val,
                                success_rate=save_rate
                            )
                            ov = roi_res["overall"]

                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Campaign Cost", f"${ov['total_campaign_cost']:,.0f}")
                            m2.metric("Gross Revenue Saved", f"${ov['gross_revenue_saved']:,.0f}")
                            m3.metric("Net Financial Benefit", f"${ov['net_financial_benefit']:,.0f}", f"{ov['roi_percentage']}% ROI")
                            m4.metric("Payback Ratio", f"{ov['payback_ratio']:.1f}x", f"Break-even: {ov['break_even_customers']} cust")

                            st.markdown("#### Risk Tier ROI Breakdown")
                            st.dataframe(roi_res["tier_breakdown"], use_container_width=True)

                            with st.expander("📊 Budget Allocation Recommendation"):
                                total_alloc_budget = st.number_input("Total Retention Budget ($)", min_value=1000.0, max_value=500000.0, value=25000.0, step=5000.0)
                                alloc_table = get_budget_allocation_recommendation(total_alloc_budget, annotated_df, cost_per_contact=contact_cost, offer_incentive_cost=incentive_val)
                                st.dataframe(alloc_table, use_container_width=True)
                        except Exception as e:
                            st.warning(f"ROI simulation failed: {e}")

                    except Exception as e:
                        st.error(f"Error processing batch file: {str(e)}")


    # -------------------------------------------------------------
    # TAB 4: MODEL PERFORMANCE & METRICS
    # -------------------------------------------------------------
    elif menu == "🤖 Model Performance & Metrics":
        st.subheader("Model Benchmarking & Evaluation Metrics")

        if os.path.exists("models/metrics.json"):
            with open("models/metrics.json", "r") as f:
                metrics = json.load(f)

            metrics_df = pd.DataFrame(metrics).T[["accuracy", "precision", "recall", "f1_score", "roc_auc"]]
            metrics_df.columns = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

            st.markdown("### Model Comparison Table")
            st.table(metrics_df.style.highlight_max(axis=0, color="#D1FAE5"))

            # Plot ROC-AUC Bar Comparison
            c1, c2 = st.columns(2)
            with c1:
                fig_bench = px.bar(
                    metrics_df.reset_index(),
                    x="index",
                    y="ROC-AUC",
                    color="index",
                    title="Model ROC-AUC Comparison",
                    labels={"index": "Model Algorithm"},
                    text_auto=".4f"
                )
                st.plotly_chart(fig_bench, use_container_width=True)

            with c2:
                try:
                    explainer = get_explainer()
                    sample_df = df.sample(min(500, len(df)), random_state=42)
                    fig_global_shap = explainer.get_global_importance_figure(sample_df, top_n=10)
                    st.plotly_chart(fig_global_shap, use_container_width=True)
                except Exception as ex:
                    st.warning(f"Could not render global SHAP importance: {ex}")

            # Model Champion-Challenger Leaderboard
            st.markdown("---")
            with st.expander("🏆 Model Champion-Challenger Leaderboard", expanded=True):
                try:
                    champ = get_champion_model()
                    if champ:
                        st.success(f"🥇 **Current Champion:** `{champ['model_name']} ({champ['version']})` — ROC-AUC: `{champ.get('roc_auc', '—')}` | F1-Score: `{champ.get('f1_score', '—')}`")
                    l_df = get_leaderboard()
                    if not l_df.empty:
                        st.dataframe(l_df, use_container_width=True)
                    else:
                        st.info("No models registered in leaderboard yet. Retrain with `python src/train.py` to record entries.")
                except Exception as l_ex:
                    st.warning(f"Could not load model leaderboard: {l_ex}")

            # Automated Retraining & Challenger Pipeline
            with st.expander("🔄 Automated Retraining Pipeline (Champion vs Challenger)", expanded=False):
                st.markdown("Trigger an automated retraining cycle on active customer records. The challenger model is evaluated against the current champion and promoted only if it achieves superior ROC-AUC.")
                next_ver = st.text_input("Candidate Version Tag", value="v1.2.0")
                min_imp = st.slider("Minimum ROC-AUC Improvement Margin", min_value=0.0, max_value=0.05, value=0.005, step=0.001, format="%.3f")
                
                if st.button("🚀 Execute Retraining Cycle"):
                    with st.spinner("Retraining candidate model and running comparative benchmark..."):
                        try:
                            retrain_res = run_retraining_cycle(df, version_tag=next_ver, tune=False, min_improvement=min_imp)
                            cand_m = retrain_res["candidate_metrics"]
                            eval_m = retrain_res["evaluation"]

                            if eval_m["promoted"]:
                                st.success(f"🎉 **{eval_m['decision']}**")
                            else:
                                st.warning(f"🛡️ **{eval_m['decision']}**")

                            rc1, rc2, rc3 = st.columns(3)
                            rc1.metric("Candidate ROC-AUC", f"{cand_m['roc_auc']:.4f}")
                            rc2.metric("Champion ROC-AUC", f"{eval_m['champion_score']:.4f}" if eval_m['champion_score'] else "—")
                            rc3.metric("ROC-AUC Delta", f"{eval_m['delta']:+.4f}", delta_color="normal" if eval_m["promoted"] else "inverse")

                            st.json(cand_m)
                        except Exception as retrain_err:
                            st.error(f"Retraining failed: {retrain_err}")

            # Probability Calibration & Reliability Analysis
            with st.expander("🎯 Probability Calibration & Confidence Reliability", expanded=False):
                st.markdown("Evaluates whether predicted churn probabilities are well-calibrated against empirical outcomes.")
                try:
                    explainer = get_explainer()
                    sample_cal_df = df.sample(min(400, len(df)), random_state=42)
                    X_cal = explainer.preprocessor.transform(sample_cal_df)
                    cal_prob = explainer.model.predict_proba(X_cal)[:, 1]
                    cal_true = sample_cal_df["churn"].values if "churn" in sample_cal_df.columns else (cal_prob > 0.5).astype(int)

                    cal_metrics = calculate_expected_calibration_error(cal_true, cal_prob, n_bins=10)
                    cal_c1, cal_c2, cal_c3, cal_c4 = st.columns(4)
                    cal_c1.metric("Expected Calibration Error (ECE)", f"{cal_metrics['ece']:.4f}")
                    cal_c2.metric("Max Calibration Error (MCE)", f"{cal_metrics['mce']:.4f}")
                    cal_c3.metric("Brier Score", f"{cal_metrics['brier_score']:.4f}")
                    cal_c4.metric("Calibration Health", cal_metrics["quality"], delta="✅" if "Well" in cal_metrics["quality"] else "⚠️")
                except Exception as cal_err:
                    st.warning(f"Could not compute calibration metrics: {cal_err}")
        else:
            st.info("Metrics not found. Run model training script `python src/train.py` to populate performance data.")

    # -------------------------------------------------------------
    # TAB 5: DATA DRIFT & AUDIT LOGS
    # -------------------------------------------------------------
    elif menu == "📈 Data Drift & Audit Logs":
        st.subheader("System Health Monitor — Data Drift & Prediction Audit")

        # Model Registry Info
        st.markdown("### 🗂️ Model Registry — Latest Registered Version")
        try:
            reg_meta = get_latest_model_metadata()
            if reg_meta:
                rcol1, rcol2, rcol3 = st.columns(3)
                rcol1.metric("Version", reg_meta.get("version", "—"))
                rcol2.metric("Registered", reg_meta.get("timestamp", "—")[:19].replace("T", " "))
                rcol3.metric("Status", reg_meta.get("status", "—"))
            else:
                st.info("No model registry found. Run `python src/train.py` first.")
        except Exception as e:
            st.warning(f"Could not load registry: {e}")

        st.markdown("---")

        # Feature Drift Detection
        st.markdown("### 📊 Feature Distribution Drift Detection (vs Training Baseline)")
        st.caption("Upload a current inference CSV to compare against training distribution.")
        drift_file = st.file_uploader("Upload Customer CSV for Drift Analysis", type=["csv"], key="drift_upload")

        if drift_file:
            drift_df = pd.read_csv(drift_file)
            is_valid, missing = validate_customer_data(drift_df, require_target=False)
            if not is_valid:
                st.error(f"⚠️ Missing columns: `{', '.join(missing)}`")
            else:
                drift_res = calculate_feature_drift(df, drift_df)
                dc1, dc2, dc3 = st.columns(3)
                dc1.metric("Total Features Checked", drift_res["total_features"])
                dc2.metric("Drifted Features", drift_res["drifted_features_count"])
                dc3.metric("Drift Detected", "⚠️ YES" if drift_res["drift_detected"] else "✅ NO")

                detail_df = pd.DataFrame([
                    {
                        "Feature": feat,
                        "Type": info["type"],
                        "Test": info["test"],
                        "Drift Score": info.get("statistic", info.get("max_shift", "—")),
                        "P-Value": info.get("p_value", "—"),
                        "Drifted": "⚠️ YES" if info["is_drifted"] else "✅ NO"
                    }
                    for feat, info in drift_res["feature_details"].items()
                ])
                st.dataframe(detail_df, use_container_width=True)

        st.markdown("---")

        # Prediction Audit Log
        st.markdown("### 📋 Prediction Audit Log")
        audit_path = os.path.join("logs", "predictions_audit.csv")
        if os.path.exists(audit_path):
            audit_df = pd.read_csv(audit_path)
            ac1, ac2 = st.columns(2)
            ac1.metric("Total Predictions Logged", len(audit_df))
            ac2.metric("Latest Prediction Time", audit_df["prediction_timestamp"].iloc[-1][:19].replace("T", " ") if "prediction_timestamp" in audit_df.columns else "—")
            st.dataframe(audit_df.tail(20), use_container_width=True)
        else:
            st.info("No prediction audit log found. Run a single or batch prediction first.")

        st.markdown("---")

        # Data Quality Profiler
        st.markdown("### 🔬 Data Quality Profiler")
        st.caption("Upload a CSV to run a full column-level data quality and distribution profile.")
        profiler_file = st.file_uploader("Upload CSV for Quality Profile", type=["csv"], key="profiler_upload")

        if profiler_file:
            prof_df = pd.read_csv(profiler_file)
            health = check_data_health(prof_df)

            hcol1, hcol2, hcol3, hcol4 = st.columns(4)
            hcol1.metric("Total Rows", health["total_rows"])
            hcol2.metric("Total Columns", health["total_cols"])
            hcol3.metric("Missing Cols", len(health["missing_cols"]))
            hcol4.metric(
                "Overall Health",
                health["overall_health"],
                delta="✅" if health["overall_health"] == "Good" else "⚠️",
                delta_color="normal" if health["overall_health"] == "Good" else "inverse",
            )

            if health["constant_cols"]:
                st.warning(f"⚠️ Constant columns (zero variance): `{', '.join(health['constant_cols'])}`")
            if health["skewed_cols"]:
                st.warning(f"📐 Highly skewed columns (|skew| > 2.0): `{', '.join(health['skewed_cols'])}`")
            if health["high_cardinality_cols"]:
                st.info(f"🔢 High cardinality columns (≥50 unique): `{', '.join(health['high_cardinality_cols'])}`")

            profile_table = profile_dataset(prof_df)
            st.markdown("#### Column-Level Profile")
            st.dataframe(profile_table, use_container_width=True)

if __name__ == "__main__":
    main()
