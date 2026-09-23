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
        "🤖 Model Performance & Metrics"
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
        else:
            st.info("Metrics not found. Run model training script `python src/train.py` to populate performance data.")

if __name__ == "__main__":
    main()
