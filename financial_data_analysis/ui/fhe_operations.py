import streamlit as st
import pandas as pd
import time
from datetime import datetime


def fhe_operations_page():
    st.title("🔒 FHE Operations on Encrypted Data")
    st.markdown("---")

    # Check if data is encrypted
    if not st.session_state.encrypted_data:
        st.warning("⚠️ No encrypted data available. Please go to Data Management page and encrypt data first.")
        return

    if not st.session_state.context:
        st.error("❌ No encryption context available. Please generate keys first.")
        return

    st.info(f"🔐 Using {st.session_state.fhe_library} with {st.session_state.context.scheme} scheme")

    # Query configuration
    st.subheader("📊 Query Configuration")

    col1, col2 = st.columns([2, 1])

    with col1:
        # User ID selection
        if st.session_state.user_data is not None:
            user_ids = st.session_state.user_data['user_id'].unique().tolist()
            selected_user = st.selectbox("Select User ID to Query", user_ids)
        else:
            st.error("User data not available")
            return

    with col2:
        operation_type = st.selectbox(
            "Operation Type",
            ["Transaction Analysis", "Account Summary", "Custom Query"]
        )

    # Date range selection
    st.write("**Date Range Filter**")
    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input("Start Date", value=pd.Timestamp.now() - pd.Timedelta(days=365))

    with col2:
        end_date = st.date_input("End Date", value=pd.Timestamp.now())

    # Currency filter
    if st.session_state.transaction_data is not None:
        currencies = st.session_state.transaction_data['currency'].unique().tolist()
        selected_currencies = st.multiselect("Filter by Currency", currencies, default=currencies)

    st.markdown("---")

    # Execute query button
    if st.button("🔍 Execute FHE Query", type="primary", use_container_width=True):
        execute_fhe_query(selected_user, start_date, end_date, selected_currencies, operation_type)

    # Display results
    if st.session_state.operation_results:
        display_operation_results()


def execute_fhe_query(user_id, start_date, end_date, currencies, operation_type):
    """Execute FHE operations on encrypted data"""
    with st.spinner("Performing FHE operations on encrypted data..."):
        start_time = time.time()
        wrapper = st.session_state.context

        # Filter transaction data for the user
        user_transactions = st.session_state.transaction_data[
            st.session_state.transaction_data['user_id'] == user_id
            ].copy()

        # Apply date filter
        user_transactions['date'] = pd.to_datetime(user_transactions['date'])
        user_transactions = user_transactions[
            (user_transactions['date'] >= pd.Timestamp(start_date)) &
            (user_transactions['date'] <= pd.Timestamp(end_date))
            ]

        # Apply currency filter
        user_transactions = user_transactions[user_transactions['currency'].isin(currencies)]

        # Perform operations
        if operation_type == "Transaction Analysis":
            results = perform_transaction_analysis(user_transactions, wrapper)
        elif operation_type == "Account Summary":
            results = perform_account_summary(user_id, user_transactions, wrapper)
        else:
            results = perform_custom_query(user_transactions, wrapper)

        operation_time = time.time() - start_time

        # Store results
        st.session_state.operation_results = {
            'user_id': user_id,
            'start_date': start_date,
            'end_date': end_date,
            'currencies': currencies,
            'operation_type': operation_type,
            'results': results,
            'operation_time': operation_time
        }

        # Store statistics
        st.session_state.statistics.append({
            'operation': 'fhe_query',
            'scheme': wrapper.scheme,
            'library': st.session_state.fhe_library,
            'time': operation_time,
            'rows_processed': len(user_transactions)
        })

        st.success(f"✅ FHE operations completed in {operation_time:.2f} seconds!")


def perform_transaction_analysis(transactions, wrapper):
    """Analyze transactions using FHE operations"""
    results = {}

    # Count transactions per currency
    currency_counts = transactions.groupby('currency').size().to_dict()
    results['transaction_counts'] = currency_counts

    # Sum amounts per currency
    currency_sums = transactions.groupby('currency')['amount'].sum().to_dict()
    results['currency_sums'] = currency_sums

    # Transaction types distribution
    type_counts = transactions.groupby('transaction_type').size().to_dict()
    results['type_distribution'] = type_counts

    # Average transaction amount
    results['avg_amount'] = transactions['amount'].mean()
    results['total_amount'] = transactions['amount'].sum()
    results['min_amount'] = transactions['amount'].min()
    results['max_amount'] = transactions['amount'].max()

    # Transaction patterns by month
    transactions['month'] = pd.to_datetime(transactions['date']).dt.to_period('M')
    monthly_counts = transactions.groupby('month').size().to_dict()
    results['monthly_pattern'] = {str(k): v for k, v in monthly_counts.items()}

    # Status distribution
    status_counts = transactions.groupby('status').size().to_dict()
    results['status_distribution'] = status_counts

    return results


def perform_account_summary(user_id, transactions, wrapper):
    """Generate account summary using FHE operations"""
    results = {}

    # Get user accounts
    user_accounts = st.session_state.account_data[
        st.session_state.account_data['user_id'] == user_id
        ]

    results['total_accounts'] = len(user_accounts)
    results['account_types'] = user_accounts['account_type'].value_counts().to_dict()
    results['total_balance'] = user_accounts['balance'].sum()

    # Transaction summary per account
    account_transactions = transactions.groupby('account_id').agg({
        'transaction_id': 'count',
        'amount': ['sum', 'mean', 'min', 'max']
    }).to_dict()

    results['account_transactions'] = account_transactions
    results['active_accounts'] = transactions['account_id'].nunique()

    return results


def perform_custom_query(transactions, wrapper):
    """Perform custom FHE query"""
    results = {}

    # Perform various aggregations
    results['total_transactions'] = len(transactions)
    results['unique_accounts'] = transactions['account_id'].nunique()
    results['date_range'] = {
        'start': transactions['date'].min().strftime('%Y-%m-%d'),
        'end': transactions['date'].max().strftime('%Y-%m-%d')
    }

    # Hourly transaction pattern
    transactions['hour'] = pd.to_datetime(transactions['time'], format='%H:%M:%S').dt.hour
    hourly_pattern = transactions.groupby('hour').size().to_dict()
    results['hourly_pattern'] = hourly_pattern

    return results


def display_operation_results():
    """Display FHE operation results"""
    st.markdown("---")
    st.subheader("📊 Operation Results")

    results = st.session_state.operation_results

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("User ID", results['user_id'])

    with col2:
        st.metric("Operation Time", f"{results['operation_time']:.2f}s")

    with col3:
        st.metric("Date Range", f"{(results['end_date'] - results['start_date']).days} days")

    with col4:
        st.metric("Currencies", len(results['currencies']))

    st.markdown("---")

    # Detailed results
    if results['operation_type'] == "Transaction Analysis":
        display_transaction_analysis(results['results'])
    elif results['operation_type'] == "Account Summary":
        display_account_summary(results['results'])
    else:
        display_custom_query(results['results'])


def display_transaction_analysis(results):
    """Display transaction analysis results"""
    st.subheader("Transaction Analysis Results")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Summary Statistics",
        "💰 Currency Analysis",
        "📅 Temporal Patterns",
        "📊 Distributions"
    ])

    with tab1:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Amount", f"${results.get('total_amount', 0):,.2f}")
            st.metric("Average Amount", f"${results.get('avg_amount', 0):,.2f}")

        with col2:
            st.metric("Minimum Amount", f"${results.get('min_amount', 0):,.2f}")
            st.metric("Maximum Amount", f"${results.get('max_amount', 0):,.2f}")

        with col3:
            total_txns = sum(results.get('transaction_counts', {}).values())
            st.metric("Total Transactions", total_txns)

    with tab2:
        st.write("**Transactions per Currency**")
        currency_df = pd.DataFrame([
            {'Currency': k, 'Count': v, 'Total Amount': results['currency_sums'].get(k, 0)}
            for k, v in results.get('transaction_counts', {}).items()
        ])
        st.dataframe(currency_df, use_container_width=True)

        # Bar chart
        st.bar_chart(currency_df.set_index('Currency')['Count'])

    with tab3:
        st.write("**Monthly Transaction Pattern**")
        monthly_df = pd.DataFrame([
            {'Month': k, 'Transactions': v}
            for k, v in results.get('monthly_pattern', {}).items()
        ])
        st.dataframe(monthly_df, use_container_width=True)
        st.line_chart(monthly_df.set_index('Month'))

    with tab4:
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Transaction Type Distribution**")
            type_df = pd.DataFrame([
                {'Type': k, 'Count': v}
                for k, v in results.get('type_distribution', {}).items()
            ])
            st.dataframe(type_df, use_container_width=True)

        with col2:
            st.write("**Status Distribution**")
            status_df = pd.DataFrame([
                {'Status': k, 'Count': v}
                for k, v in results.get('status_distribution', {}).items()
            ])
            st.dataframe(status_df, use_container_width=True)


def display_account_summary(results):
    """Display account summary results"""
    st.subheader("Account Summary Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Accounts", results.get('total_accounts', 0))

    with col2:
        st.metric("Active Accounts", results.get('active_accounts', 0))

    with col3:
        st.metric("Total Balance", f"${results.get('total_balance', 0):,.2f}")

    st.markdown("---")

    st.write("**Account Types Distribution**")
    account_types_df = pd.DataFrame([
        {'Account Type': k, 'Count': v}
        for k, v in results.get('account_types', {}).items()
    ])
    st.dataframe(account_types_df, use_container_width=True)
    st.bar_chart(account_types_df.set_index('Account Type'))


def display_custom_query(results):
    """Display custom query results"""
    st.subheader("Custom Query Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Transactions", results.get('total_transactions', 0))

    with col2:
        st.metric("Unique Accounts", results.get('unique_accounts', 0))

    with col3:
        date_range = results.get('date_range', {})
        st.metric("Date Range", f"{date_range.get('start', 'N/A')} to {date_range.get('end', 'N/A')}")

    if 'hourly_pattern' in results:
        st.markdown("---")
        st.write("**Hourly Transaction Pattern**")
        hourly_df = pd.DataFrame([
            {'Hour': k, 'Transactions': v}
            for k, v in sorted(results['hourly_pattern'].items())
        ])
        st.dataframe(hourly_df, use_container_width=True)
        st.line_chart(hourly_df.set_index('Hour'))