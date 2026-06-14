import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="DSO & Payment Delay", layout="wide")

############################################################
# ANALYSIS FUNCTIONS
############################################################

def clean_numeric(series):
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")

def clean_dates(series):
    return pd.to_datetime(
        series,
        errors="coerce"
    )

def normalize_columns(df):
    df.columns = df.columns.str.replace(r"[-.]", "_", regex=True)
    return df

def load_and_clean_sales(file):
    df = pd.read_csv(file, dtype=str)
    df = normalize_columns(df)
    df["Date"]      = clean_dates(df["Date"])
    df["Amount"]    = clean_numeric(df["Amount"])
    df["Type"]      = pd.to_numeric(df["Type"], errors="coerce").astype("Int64")
    df["NetAmount"] = df.apply(lambda r: -r["Amount"] if r["Type"] == 12 else r["Amount"], axis=1)
    df = df.rename(columns={
        "Oid_2":    "CustomerID",
        "Code_2":   "CustomerCode",
        "Label1_2": "CustomerName",
        "Label1_3": "CustomerCategory",
        "Label1_4": "Region",
        "Label1_5": "Wilaya",
    })
    df["CustomerID"] = df["CustomerID"].astype(str).str.strip()
    df = df[df["CustomerID"].notna() & (df["CustomerID"] != "") & (df["CustomerID"] != "nan")]
    return df[["CustomerID","CustomerCode","CustomerName","CustomerCategory","Region","Wilaya","Date","NetAmount"]]

def load_and_clean_payments(file):
    df = pd.read_csv(file, dtype=str)
    df = normalize_columns(df)
    df["TransactionDate"] = clean_dates(df["TransactionDate"])
    df["Credit"]     = clean_numeric(df["Credit"]).fillna(0)
    df["Debit"]      = clean_numeric(df["Debit"]).fillna(0)
    df["NetPayment"] = df["Credit"] - df["Debit"]
    df = df.rename(columns={
        "Oid":    "CustomerID",
        "Code":   "CustomerCode",
        "Label1": "CustomerName",
        "TransactionDate": "PaymentDate",
    })
    df["CustomerID"] = df["CustomerID"].astype(str).str.strip()
    df = df[df["CustomerID"].notna() & (df["CustomerID"] != "") & (df["CustomerID"] != "nan")]
    return df[["CustomerID","CustomerCode","CustomerName","PaymentDate","NetPayment"]]

def compute_summary(sales, payments):
    cust_sales    = sales.groupby("CustomerID")["NetAmount"].sum().reset_index().rename(columns={"NetAmount":"TotalSales"})
    cust_payments = payments.groupby("CustomerID")["NetPayment"].sum().reset_index().rename(columns={"NetPayment":"TotalPayments"})
    summary = pd.merge(cust_sales, cust_payments, on="CustomerID", how="outer").fillna(0)
    summary["OutstandingBalance"] = (summary["TotalSales"] - summary["TotalPayments"]).clip(lower=0)

    meta = sales.drop_duplicates("CustomerID")[["CustomerID","CustomerName","CustomerCode","CustomerCategory","Region","Wilaya"]]
    summary = summary.merge(meta, on="CustomerID", how="left")

    date_range = sales.groupby("CustomerID")["Date"].agg(MinDate="min", MaxDate="max").reset_index()
    date_range["PeriodDays"] = (date_range["MaxDate"] - date_range["MinDate"]).dt.days.clip(lower=1)
    summary = summary.merge(date_range[["CustomerID","PeriodDays"]], on="CustomerID", how="left")

    summary["AverageDailySales"] = np.where(
        (summary["TotalSales"] > 0) & (summary["PeriodDays"] > 0),
        summary["TotalSales"] / summary["PeriodDays"], np.nan
    )
    summary["DSO"] = np.where(
        summary["AverageDailySales"] > 0,
        (summary["OutstandingBalance"] / summary["AverageDailySales"]).clip(lower=0),
        np.nan
    )
    return summary

def fifo_match_customer(cs, cp):
    cs = cs.sort_values("Date").reset_index(drop=True)   # sort the values from the oldest to the newest 
    cp = cp.sort_values("PaymentDate").reset_index(drop=True)

    queue = []                      # list of non payed bills
    for _, row in cs.iterrows():
        amt = row["NetAmount"]
        if pd.isna(amt):
            continue
        if amt > 0:
            queue.append({"SaleDate": row["Date"], "Remaining": float(amt)})
        else:
            ret = abs(amt)
            for q in queue:
                if ret <= 0:
                    break
                reduce = min(q["Remaining"], ret)
                q["Remaining"] -= reduce
                ret -= reduce
            queue = [q for q in queue if q["Remaining"] > 1e-6]

    if not queue:
        return pd.DataFrame()

    matches = []
    queue_idx = 0

    for _, row in cp.iterrows():
        pmt = row["NetPayment"]
        if pd.isna(pmt) or pmt <= 0:
            continue
        pdate = row["PaymentDate"]

        while pmt > 0 and queue_idx < len(queue):
            cur     = queue[queue_idx]
            matched = min(pmt, cur["Remaining"])
            delay   = (pdate - cur["SaleDate"]).days if pd.notna(pdate) and pd.notna(cur["SaleDate"]) else np.nan

            matches.append({
                "CustomerID":    cs["CustomerID"].iloc[0],
                "SaleDate":      cur["SaleDate"],
                "PaymentDate":   pdate,
                "MatchedAmount": matched,
                "DelayDays":     delay,
            })

            pmt                  -= matched
            cur["Remaining"]     -= matched
            if cur["Remaining"] <= 1e-6:
                queue_idx += 1

    return pd.DataFrame(matches)

def run_fifo(sales, payments, progress_bar=None):
    customer_ids = sales["CustomerID"].unique()
    all_matches  = []

    for i, cust in enumerate(customer_ids):
        cs = sales[sales["CustomerID"] == cust]
        cp = payments[payments["CustomerID"] == cust]
        result = fifo_match_customer(cs, cp)
        if not result.empty:
            all_matches.append(result)
        if progress_bar is not None and i % 200 == 0:
            progress_bar.progress(min(int(100 * i / len(customer_ids)), 99))

    if not all_matches:
        return pd.DataFrame(columns=["CustomerID","SaleDate","PaymentDate","MatchedAmount","DelayDays"])
    return pd.concat(all_matches, ignore_index=True)

def compute_delay_metrics(matched_df):
    if matched_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    def wavg(g):
        w = g["MatchedAmount"]
        return np.average(g["DelayDays"].fillna(0), weights=w) if w.sum() > 0 else np.nan

    delay = matched_df.groupby("CustomerID").apply(lambda g: pd.Series({
        "AvgDelay":    wavg(g),
        "MedianDelay": g["DelayDays"].median(),
        "MaxDelay":    g["DelayDays"].max(),
        "P90Delay":    g["DelayDays"].quantile(0.90),
        "TotalMatched": g["MatchedAmount"].sum(),
    })).reset_index()

    speed = matched_df.groupby("CustomerID").apply(lambda g: pd.Series({
        "PctPaid30": g.loc[g["DelayDays"] <= 30, "MatchedAmount"].sum() / g["MatchedAmount"].sum(),
        "PctPaid60": g.loc[g["DelayDays"] <= 60, "MatchedAmount"].sum() / g["MatchedAmount"].sum(),
        "PctPaid90": g.loc[g["DelayDays"] <= 90, "MatchedAmount"].sum() / g["MatchedAmount"].sum(),
    })).reset_index()

    return delay, speed

def assign_segment(avg_delay):
    if pd.isna(avg_delay): return "No Payment Data"
    if avg_delay <  7:     return "Cash"
    if avg_delay < 30:     return "Fast"
    if avg_delay < 60:     return "Normal"
    if avg_delay < 90:     return "Slow"
    return "Chronic Late"

SEG_COLORS = {
    "Cash":            "#2ecc71",
    "Fast":            "#27ae60",
    "Normal":          "#f39c12",
    "Slow":            "#e67e22",
    "Chronic Late":    "#e74c3c",
    "No Payment Data": "#95a5a6",
}

############################################################
# UI
############################################################

st.title("📊 DSO & Customer Payment Delay Analysis")

tab_upload, tab_overview, tab_table, tab_fifo, tab_risk = st.tabs([
    "📁 Upload & Run", "📈 Overview", "📋 Customer Table", "🔗 FIFO Matches", "⚠️ Risk Map"
])

# ── SESSION STATE ──
if "results" not in st.session_state:
    st.session_state.results = None

# ── UPLOAD TAB ──
with tab_upload:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sales File (.csv)")
        sales_file = st.file_uploader("Upload sales CSV", type="csv", key="sales")
        st.caption("Columns needed: Date, Amount, Type, Oid-2, Code-2, Label1-2/3/4/5")
    with col2:
        st.subheader("Payments File (.csv)")
        pay_file = st.file_uploader("Upload payments CSV", type="csv", key="payments")
        st.caption("Columns needed: TransactionDate, Credit, Debit, Oid, Code, Label1")

    if st.button("▶ Run Analysis", type="primary"):
        if sales_file is None or pay_file is None:
            st.error("Please upload both files first.")
        else:
            try:
                with st.spinner("Loading and cleaning data..."):
                    sales    = load_and_clean_sales(sales_file)
                    payments = load_and_clean_payments(pay_file)

                with st.spinner("Computing customer aggregates and DSO..."):
                    summary = compute_summary(sales, payments)

                st.info(f"Running FIFO matching for {sales['CustomerID'].nunique()} customers...")
                pb = st.progress(0)
                matched_df = run_fifo(sales, payments, progress_bar=pb)
                pb.progress(100)

                with st.spinner("Computing delay metrics..."):
                    delay_metrics, payment_speed = compute_delay_metrics(matched_df)

                if not delay_metrics.empty:
                    delay_metrics["Segment"] = delay_metrics["AvgDelay"].apply(assign_segment)
                    summary = summary.merge(delay_metrics, on="CustomerID", how="left")
                    summary = summary.merge(payment_speed, on="CustomerID", how="left")
                else:
                    summary["Segment"] = "No Payment Data"

                summary["Segment"]   = summary["Segment"].fillna("No Payment Data")
                summary["LatePayer"] = ((summary["AvgDelay"].notna()) & (summary["AvgDelay"] > 60)).astype(int)

                st.session_state.results    = summary
                st.session_state.matched_df = matched_df
                st.success("✅ Analysis complete! Go to the other tabs to explore results.")

            except Exception as e:
                st.error(f"❌ Error: {e}")
                import traceback
                st.code(traceback.format_exc())

# ── OVERVIEW TAB ──
with tab_overview:
    if st.session_state.results is None:
        st.info("Run the analysis first.")
    else:
        s = st.session_state.results

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Customers",          f"{len(s):,}")
        c2.metric("Total Sales",        f"{s['TotalSales'].sum()/1e6:.1f} M")
        c3.metric("Outstanding Balance",f"{s['OutstandingBalance'].sum()/1e6:.1f} M")
        wdso = np.average(s["DSO"].dropna(), weights=s.loc[s["DSO"].notna(), "TotalSales"])
        c4.metric("Weighted Avg DSO",   f"{wdso:.1f} days")

        col1, col2 = st.columns(2)

        with col1:
            seg_counts = s["Segment"].value_counts().reset_index()
            seg_counts.columns = ["Segment", "Count"]
            fig = px.pie(seg_counts, names="Segment", values="Count",
                         color="Segment", color_discrete_map=SEG_COLORS,
                         title="Customer Segments")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            mdf = st.session_state.matched_df
            if not mdf.empty:
                d = mdf[(mdf["DelayDays"] >= 0) & (mdf["DelayDays"] <= 365)]
                fig = px.histogram(d, x="DelayDays", nbins=60,
                                   title="Payment Delay Distribution (FIFO)",
                                   labels={"DelayDays": "Delay (days)"})
                fig.update_traces(marker_color="#3498db")
                st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            avgs = {
                "≤30 days": s["PctPaid30"].mean() * 100 if "PctPaid30" in s else 0,
                "≤60 days": s["PctPaid60"].mean() * 100 if "PctPaid60" in s else 0,
                "≤90 days": s["PctPaid90"].mean() * 100 if "PctPaid90" in s else 0,
            }
            fig = go.Figure(go.Bar(
                x=list(avgs.keys()), y=list(avgs.values()),
                marker_color=["#2ecc71","#f39c12","#e74c3c"],
                text=[f"{v:.1f}%" for v in avgs.values()],
                textposition="outside"
            ))
            fig.update_layout(title="% Paid within 30/60/90 days",
                              yaxis=dict(range=[0, 110]))
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            top20 = s.nlargest(20, "OutstandingBalance").copy()
            top20["ShortName"] = top20["CustomerName"].str[:25]
            fig = px.bar(top20, x="ShortName", y="OutstandingBalance",
                         title="Top 20 by Outstanding Balance")
            fig.update_traces(marker_color="#e74c3c")
            fig.update_layout(xaxis_tickangle=-40)
            st.plotly_chart(fig, use_container_width=True)

# ── CUSTOMER TABLE TAB ──
with tab_table:
    if st.session_state.results is None:
        st.info("Run the analysis first.")
    else:
        s = st.session_state.results.copy()
        cols = ["CustomerCode","CustomerName","CustomerCategory","Region","Wilaya",
                "TotalSales","TotalPayments","OutstandingBalance","DSO",
                "AvgDelay","MedianDelay","P90Delay","PctPaid30","PctPaid60","Segment","LatePayer"]
        cols = [c for c in cols if c in s.columns]
        display = s[cols].copy()
        for c in ["TotalSales","TotalPayments","OutstandingBalance"]:
            if c in display: display[c] = display[c].round(0)
        for c in ["DSO","AvgDelay","MedianDelay","P90Delay"]:
            if c in display: display[c] = display[c].round(1)
        for c in ["PctPaid30","PctPaid60"]:
            if c in display: display[c] = (display[c] * 100).round(1)

        st.download_button("⬇ Download CSV", display.to_csv(index=False),
                           file_name=f"customer_credit_analysis_{datetime.today().date()}.csv")
        st.dataframe(display, use_container_width=True, height=600)

# ── FIFO MATCHES TAB ──
with tab_fifo:
    if st.session_state.get("matched_df") is None:
        st.info("Run the analysis first.")
    else:
        mdf = st.session_state.matched_df.copy()
        mdf["SaleDate"]    = mdf["SaleDate"].astype(str)
        mdf["PaymentDate"] = mdf["PaymentDate"].astype(str)
        mdf["MatchedAmount"] = mdf["MatchedAmount"].round(2)
        mdf["DelayDays"]     = mdf["DelayDays"].round(1)
        st.download_button("⬇ Download CSV", mdf.to_csv(index=False),
                           file_name=f"fifo_matches_{datetime.today().date()}.csv")
        st.dataframe(mdf, use_container_width=True, height=600)

# ── RISK MAP TAB ──
with tab_risk:
    if st.session_state.results is None:
        st.info("Run the analysis first.")
    else:
        s = st.session_state.results

        st.subheader("Top 50 Risk Customers")
        risk = s.dropna(subset=["AvgDelay"]).nlargest(50, "AvgDelay")[
            ["CustomerCode","CustomerName","Region","Wilaya",
             "OutstandingBalance","AvgDelay","DSO","Segment"]
        ].copy()
        risk["OutstandingBalance"] = risk["OutstandingBalance"].round(0)
        risk["AvgDelay"]           = risk["AvgDelay"].round(1)
        risk["DSO"]                = risk["DSO"].round(1)
        st.dataframe(risk, use_container_width=True)

        st.subheader("Avg Delay vs Outstanding Balance")
        d = s.dropna(subset=["AvgDelay"])
        d = d[d["TotalSales"] > 0]
        fig = px.scatter(d, x="AvgDelay", y="OutstandingBalance",
                         color="Segment", color_discrete_map=SEG_COLORS,
                         hover_data=["CustomerName","AvgDelay","OutstandingBalance"],
                         title="Avg Delay vs Outstanding Balance",
                         labels={"AvgDelay": "Avg Payment Delay (days)",
                                 "OutstandingBalance": "Outstanding Balance"})
        fig.update_traces(marker=dict(size=8, opacity=0.7))
        st.plotly_chart(fig, use_container_width=True)
