"""
src/executive_reporter.py
-------------------------
Automated executive churn intelligence HTML report generator.

Compiles high-level KPIs, risk distribution, root-cause diagnostics, and
retention campaign financial ROI into a standalone, styled C-suite HTML report.

Functions:
    generate_executive_html_report(...) -> str
    save_executive_report(html_content, filepath) -> str
"""

import os
from datetime import datetime, timezone
import pandas as pd
from typing import Dict, Any, Optional

from src.config import BASE_DIR, setup_logger

logger = setup_logger("executive_reporter")


def generate_executive_html_report(
    kpis: Dict[str, Any],
    risk_summary: Optional[Dict[str, Any]] = None,
    roi_summary: Optional[Dict[str, Any]] = None,
    root_causes_df: Optional[pd.DataFrame] = None,
) -> str:
    """
    Compiles an executive summary report into a self-contained, styled HTML document.

    Parameters
    ----------
    kpis : dict
        Key performance indicators (e.g. total_customers, churn_rate, clv_at_risk, model_auc).
    risk_summary : dict, optional
        Counts or percentages across risk bands.
    roi_summary : dict, optional
        Summary metrics from retention ROI simulator.
    root_causes_df : pd.DataFrame, optional
        Portfolio root-cause diagnostic breakdown.

    Returns
    -------
    str
        Complete, valid HTML document string.
    """
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Safe defaults
    total_cust = kpis.get("total_customers", 0)
    churn_rate = kpis.get("churn_rate_pct", 0.0)
    clv_at_risk = kpis.get("clv_at_risk", 0.0)
    model_auc = kpis.get("champion_auc", 0.85)

    risk_html = ""
    if risk_summary:
        risk_items = "".join(
            f'<div class="badge-item"><strong>{k}:</strong> {v}</div>'
            for k, v in risk_summary.items()
        )
        risk_html = f'<div class="section-card"><h3>⚠️ Portfolio Risk Distribution</h3><div class="badge-row">{risk_items}</div></div>'

    roi_html = ""
    if roi_summary:
        roi_html = f"""
        <div class="section-card">
            <h3>💰 Retention Campaign Financial Forecast</h3>
            <div class="kpi-grid">
                <div class="kpi-card">
                    <span class="kpi-label">Total Campaign Budget</span>
                    <span class="kpi-value">${roi_summary.get('total_campaign_cost', 0):,.0f}</span>
                </div>
                <div class="kpi-card">
                    <span class="kpi-label">Gross Revenue Saved</span>
                    <span class="kpi-value">${roi_summary.get('gross_revenue_saved', 0):,.0f}</span>
                </div>
                <div class="kpi-card">
                    <span class="kpi-label">Net Financial Benefit</span>
                    <span class="kpi-value" style="color: #10B981;">${roi_summary.get('net_financial_benefit', 0):,.0f}</span>
                </div>
                <div class="kpi-card">
                    <span class="kpi-label">Projected Campaign ROI</span>
                    <span class="kpi-value" style="color: #3B82F6;">{roi_summary.get('roi_percentage', 0):.1f}%</span>
                </div>
            </div>
        </div>
        """

    root_cause_html = ""
    if root_causes_df is not None and not root_causes_df.empty:
        rows_html = "".join(
            f"""<tr>
                <td><strong>{row.get('root_cause', '')}</strong></td>
                <td>{row.get('affected_customers', '')}</td>
                <td>{row.get('share_percentage', '')}%</td>
                <td>${row.get('avg_customer_clv', 0):,.2f}</td>
                <td>{row.get('department', '')}</td>
                <td><span class="urgency-badge {str(row.get('urgency', '')).lower()}">{row.get('urgency', '')}</span></td>
            </tr>"""
            for _, row in root_causes_df.iterrows()
        )
        root_cause_html = f"""
        <div class="section-card">
            <h3>🔍 Primary Churn Friction & Department Action Plan</h3>
            <table class="report-table">
                <thead>
                    <tr>
                        <th>Root-Cause Theme</th>
                        <th>Customers</th>
                        <th>Share %</th>
                        <th>Avg CLV</th>
                        <th>Target Department</th>
                        <th>Urgency</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Customer Churn Intelligence Executive Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: #0F172A;
            color: #E2E8F0;
            margin: 0;
            padding: 30px;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
        }}
        .header {{
            border-bottom: 2px solid #334155;
            padding-bottom: 20px;
            margin-bottom: 30px;
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
        }}
        .title {{
            font-size: 26px;
            font-weight: 700;
            color: #F8FAFC;
            margin: 0;
        }}
        .subtitle {{
            font-size: 14px;
            color: #94A3B8;
            margin-top: 5px;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 30px;
        }}
        .kpi-card {{
            background: #1E293B;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 18px;
            display: flex;
            flex-direction: column;
        }}
        .kpi-label {{
            font-size: 12px;
            text-transform: uppercase;
            color: #94A3B8;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }}
        .kpi-value {{
            font-size: 24px;
            font-weight: 700;
            color: #F8FAFC;
        }}
        .section-card {{
            background: #1E293B;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 25px;
        }}
        .section-card h3 {{
            margin-top: 0;
            font-size: 18px;
            color: #F1F5F9;
            border-bottom: 1px solid #334155;
            padding-bottom: 10px;
        }}
        .badge-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .badge-item {{
            background: #334155;
            padding: 8px 14px;
            border-radius: 6px;
            font-size: 14px;
        }}
        .report-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        .report-table th, .report-table td {{
            text-align: left;
            padding: 12px 14px;
            border-bottom: 1px solid #334155;
            font-size: 14px;
        }}
        .report-table th {{
            color: #94A3B8;
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.5px;
        }}
        .urgency-badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }}
        .urgency-badge.immediate {{ background: #EF4444; color: #FFF; }}
        .urgency-badge.high {{ background: #F59E0B; color: #000; }}
        .urgency-badge.medium {{ background: #3B82F6; color: #FFF; }}
        .footer {{
            text-align: center;
            font-size: 12px;
            color: #64748B;
            margin-top: 40px;
            border-top: 1px solid #334155;
            padding-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1 class="title">Customer Churn Intelligence Report</h1>
                <div class="subtitle">AI & Predictive Analytics Leadership Briefing</div>
            </div>
            <div class="subtitle">Generated: {now_str}</div>
        </div>

        <div class="kpi-grid">
            <div class="kpi-card">
                <span class="kpi-label">Portfolio Customers</span>
                <span class="kpi-value">{total_cust:,}</span>
            </div>
            <div class="kpi-card">
                <span class="kpi-label">Projected Churn Rate</span>
                <span class="kpi-value" style="color: #EF4444;">{churn_rate:.1f}%</span>
            </div>
            <div class="kpi-card">
                <span class="kpi-label">Total CLV Exposure</span>
                <span class="kpi-value" style="color: #F59E0B;">${clv_at_risk:,.0f}</span>
            </div>
            <div class="kpi-card">
                <span class="kpi-label">Model Benchmark ROC-AUC</span>
                <span class="kpi-value" style="color: #10B981;">{model_auc:.4f}</span>
            </div>
        </div>

        {risk_html}
        {roi_html}
        {root_cause_html}

        <div class="footer">
            CONFIDENTIAL — Generated by AI Customer Churn Intelligence & Explainable AI Engine
        </div>
    </div>
</body>
</html>
"""
    return html


def save_executive_report(
    html_content: str,
    filepath: Optional[str] = None,
) -> str:
    """
    Saves executive HTML report to disk.

    Parameters
    ----------
    html_content : str
        HTML text string.
    filepath : str, optional
        Destination filepath (default: reports/executive_churn_report.html).

    Returns
    -------
    str
        Saved absolute or relative file path.
    """
    if filepath is None:
        filepath = os.path.join(BASE_DIR, "reports", "executive_churn_report.html")

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"Executive HTML report written to {filepath}")
    return filepath
