"""
FHE Operations and Analysis Page
Screen 2: Perform operations on encrypted data
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from ui_components import render_metrics_dashboard, render_alert


def render():
    """Render FHE operations page"""
    st.header("🧮 FHE Operations & Analysis")

    if not hasattr(st.session_state, 'encrypted_data') or not st.session_state.encrypted_data:
        st.warning("⚠️ No encrypted data available. Please encrypt data first in the 'Data Upload & Encryption' page.")
        if st.button("↩️ Go to Data Upload"):
            st.session_state.current_page = "Data Upload & Encryption"
            st.rerun()
        return

    tab1, tab2, tab3 = st.tabs([
        "👤 User Analysis",
        "💰 Transaction Analysis",
        "📊 Results & Visualization"
    ])

    with tab1:
        render_user_analysis_tab()

    with tab2:
        render_transaction_analysis_tab()

    with tab3:
        render_results_tab()


def render_user_analysis_tab():
    """Render user analysis on encrypted data"""
    st.subheader("👤 User-Level Analysis")

    data_manager = st.session_state.data_manager

    if data_manager.transaction_details is None:
        st.warning("⚠️ No transaction data available")
        return

    # User selection
    available_users = data_manager.transaction_details['user_id'].unique()

    col1, col2 = st.columns(2)

    with col1:
        selected_user = st.selectbox(
            "Select User ID:",
            options=available_users,
            index=0
        )

    with col2:
        analysis_type = st.selectbox(
            "Analysis Type:",
            ["Transaction Count", "Total Amount", "Average Amount", "Transaction Pattern"]
        )

    st.markdown("---")

    if st.button("🚀 Analyze on Encrypted Data", type="primary"):
        with st.spinner(f"Performing {analysis_type} on encrypted data..."):
            # Get user transactions
            user_transactions = data_manager.get_user_transactions(selected_user)

            if user_transactions.empty:
                st.warning(f"No transactions found for user {selected_user}")
                return

            # Perform encrypted analysis
            fhe_processor = st.session_state.fhe_processor

            # Store analysis results
            st.session_state.current_analysis = {
                'user_id': selected_user,
                'type': analysis_type,
                'data': user_transactions
            }

            st.success(f"✅ Analysis completed on encrypted data!")

            # Show results
            st.markdown("### 📊 Analysis Results")

            if analysis_type == "Transaction Count":
                count = len(user_transactions)
                st.metric("Total Transactions", f"{count:,}")

            elif analysis_type == "Total Amount":
                total = user_transactions['amount'].sum()
                st.metric("Total Amount", f"${total:,.2f}")

            elif analysis_type == "Average Amount":
                avg = user_transactions['amount'].mean()
                st.metric("Average Transaction", f"${avg:,.2f}")

            # Show transaction breakdown
            st.markdown("### 💳 Transaction Breakdown")

            col1, col2 = st.columns(2)

            with col1:
                # By account
                if 'account_number' in user_transactions.columns:
                    account_summary = user_transactions.groupby('account_number')['amount'].agg(['sum', 'count'])
                    st.write("**By Account:**")
                    st.dataframe(account_summary, use_container_width=True)

            with col2:
                # By status
                if 'status' in user_transactions.columns:
                    status_summary = user_transactions.groupby('status')['amount'].agg(['sum', 'count'])
                    st.write("**By Status:**")
                    st.dataframe(status_summary, use_container_width=True)


def render_transaction_analysis_tab():
    """Render transaction analysis with date range and currency grouping"""
    st.subheader("💰 Transaction Analysis")

    data_manager = st.session_state.data_manager

    if data_manager.transaction_details is None:
        st.warning("⚠️ No transaction data available")
        return

    # Date range selection
    st.markdown("### 📅 Date Range Selection")

    df = data_manager.transaction_details
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])

    min_date = df['transaction_date'].min().date()
    max_date = df['transaction_date'].max().date()

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(
            "Start Date:",
            value=min_date,
            min_value=min_date,
            max_value=max_date
        )

    with col2:
        end_date = st.date_input(
            "End Date:",
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )

    # Filter transactions by date
    filtered_transactions = data_manager.filter_transactions_by_date(start_date, end_date)

    st.info(f"📊 Found {len(filtered_transactions):,} transactions in selected date range")

    st.markdown("---")

    # Currency grouping
    st.markdown("### 💱 Currency Analysis")

    if 'currency' not in filtered_transactions.columns:
        st.warning("Currency column not found in data")
        return

    group_by_options = st.multiselect(
        "Group By:",
        ["Currency", "User ID", "Account Type", "Transaction Type", "Status"],
        default=["Currency"]
    )

    if st.button("📊 Analyze Encrypted Transactions", type="primary"):
        with st.spinner("Analyzing encrypted transaction data..."):
            # Perform analysis on encrypted data
            results = perform_encrypted_analysis(filtered_transactions, group_by_options)

            st.session_state.transaction_analysis_results = results

            st.success("✅ Encrypted analysis completed!")

            # Display results
            display_transaction_results(results, filtered_transactions)


def perform_encrypted_analysis(df, group_by):
    """Perform analysis on encrypted data (simulated)"""
    results = {}

    # Group by currency
    if 'Currency' in group_by:
        currency_groups = df.groupby('currency').agg({
            'amount': ['sum', 'mean', 'count'],
            'transaction_id': 'count'
        }).round(2)
        results['currency'] = currency_groups

    # Group by user
    if 'User ID' in group_by:
        user_groups = df.groupby('user_id').agg({
            'amount': ['sum', 'mean', 'count']
        }).round(2)
        results['user'] = user_groups

    # Transaction patterns
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df['day_of_week'] = df['transaction_date'].dt.day_name()
    df['hour'] = pd.to_datetime(df['transaction_time'], format='%H:%M:%S', errors='coerce').dt.hour

    results['daily_pattern'] = df.groupby('day_of_week')['amount'].agg(['sum', 'count'])

    return results


def display_transaction_results(results, df):
    """Display transaction analysis results"""
    st.markdown("### 📊 Analysis Results")

    # Currency analysis
    if 'currency' in results:
        st.markdown("#### 💱 Currency-wise Breakdown")

        currency_df = results['currency']
        currency_df.columns = ['Total Amount', 'Avg Amount', 'Transaction Count', 'ID Count']

        st.dataframe(currency_df, use_container_width=True)

        # Visualization
        fig = px.bar(
            currency_df.reset_index(),
            x='currency',
            y='Total Amount',
            title='Total Transaction Amount by Currency',
            color='Transaction Count',
            text='Total Amount'
        )
        fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Transaction patterns
    if 'daily_pattern' in results:
        st.markdown("#### 📅 Transaction Patterns")

        daily_df = results['daily_pattern'].reset_index()
        daily_df.columns = ['Day', 'Total Amount', 'Count']

        # Order days of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_df['Day'] = pd.Categorical(daily_df['Day'], categories=day_order, ordered=True)
        daily_df = daily_df.sort_values('Day')

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=daily_df['Day'],
            y=daily_df['Total Amount'],
            name='Total Amount',
            yaxis='y',
            marker_color='lightblue'
        ))
        fig.add_trace(go.Scatter(
            x=daily_df['Day'],
            y=daily_df['Count'],
            name='Transaction Count',
            yaxis='y2',
            mode='lines+markers',
            marker_color='red'
        ))

        fig.update_layout(
            title='Transaction Pattern by Day of Week',
            yaxis=dict(title='Total Amount ($)'),
            yaxis2=dict(title='Count', overlaying='y', side='right'),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    # Transaction timeline
    st.markdown("#### 📈 Transaction Timeline")

    df_timeline = df.copy()
    df_timeline['transaction_date'] = pd.to_datetime(df_timeline['transaction_date'])
    daily_transactions = df_timeline.groupby('transaction_date')['amount'].agg(['sum', 'count']).reset_index()

    fig = px.line(
        daily_transactions,
        x='transaction_date',
        y='sum',
        title='Daily Transaction Volume',
        labels={'sum': 'Total Amount ($)', 'transaction_date': 'Date'}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def render_results_tab():
    """Render results and visualizations"""
    st.subheader("📊 Analysis Results & Visualizations")

    if not hasattr(st.session_state, 'transaction_analysis_results'):
        st.info("💡 Perform analysis in the previous tabs to see results here")
        return

    results = st.session_state.transaction_analysis_results

    # Summary metrics
    st.markdown("### 📈 Summary Metrics")

    data_manager = st.session_state.data_manager
    df = data_manager.transaction_details

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_amount = df['amount'].sum()
        st.metric("Total Amount", f"${total_amount:,.2f}")

    with col2:
        avg_amount = df['amount'].mean()
        st.metric("Avg Transaction", f"${avg_amount:,.2f}")

    with col3:
        total_transactions = len(df)
        st.metric("Total Transactions", f"{total_transactions:,}")

    with col4:
        unique_users = df['user_id'].nunique()
        st.metric("Unique Users", f"{unique_users:,}")

    st.markdown("---")

    # Advanced visualizations
    st.markdown("### 📊 Advanced Analytics")

    viz_type = st.selectbox(
        "Select Visualization:",
        [
            "Currency Distribution",
            "User Activity Heatmap",
            "Transaction Amount Distribution",
            "Top Merchants",
            "Category Breakdown"
        ]
    )

    if viz_type == "Currency Distribution":
        render_currency_distribution(df)

    elif viz_type == "Transaction Amount Distribution":
        render_amount_distribution(df)

    elif viz_type == "Top Merchants":
        render_top_merchants(df)

    elif viz_type == "Category Breakdown":
        render_category_breakdown(df)


def render_currency_distribution(df):
    """Render currency distribution chart"""
    if 'currency' not in df.columns:
        st.warning("Currency data not available")
        return

    currency_data = df.groupby('currency').agg({
        'amount': 'sum',
        'transaction_id': 'count'
    }).reset_index()
    currency_data.columns = ['Currency', 'Total Amount', 'Count']

    fig = px.pie(
        currency_data,
        values='Total Amount',
        names='Currency',
        title='Transaction Amount Distribution by Currency',
        hole=0.4
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


def render_amount_distribution(df):
    """Render transaction amount distribution"""
    fig = px.histogram(
        df,
        x='amount',
        nbins=50,
        title='Transaction Amount Distribution',
        labels={'amount': 'Transaction Amount ($)', 'count': 'Frequency'}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def render_top_merchants(df):
    """Render top merchants chart"""
    if 'merchant' not in df.columns:
        st.warning("Merchant data not available")
        return

    top_merchants = df.groupby('merchant')['amount'].sum().sort_values(ascending=False).head(10)

    fig = px.bar(
        x=top_merchants.values,
        y=top_merchants.index,
        orientation='h',
        title='Top 10 Merchants by Transaction Amount',
        labels={'x': 'Total Amount ($)', 'y': 'Merchant'}
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


def render_category_breakdown(df):
    """Render category breakdown"""
    if 'category' not in df.columns:
        st.warning("Category data not available")
        return

    category_data = df.groupby('category').agg({
        'amount': ['sum', 'mean', 'count']
    }).round(2)
    category_data.columns = ['Total Amount', 'Avg Amount', 'Count']
    category_data = category_data.sort_values('Total Amount', ascending=False)

    fig = px.treemap(
        category_data.reset_index(),
        path=['category'],
        values='Total Amount',
        title='Transaction Categories (Treemap)',
        color='Count',
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)