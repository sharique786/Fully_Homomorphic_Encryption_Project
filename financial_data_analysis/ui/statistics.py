import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def statistics_page():
    st.title("📈 Statistics & Scheme Comparison")
    st.markdown("---")

    # Check if statistics are available
    if not st.session_state.statistics:
        st.warning("⚠️ No statistics available yet. Please perform some operations first.")
        st.info("💡 Go to Data Management to encrypt data and FHE Operations to run queries.")
        return

    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Performance Overview",
        "⚖️ Scheme Comparison",
        "📉 Detailed Analytics",
        "🔍 Operation Breakdown"
    ])

    with tab1:
        display_performance_overview()

    with tab2:
        display_scheme_comparison()

    with tab3:
        display_detailed_analytics()

    with tab4:
        display_operation_breakdown()


def display_performance_overview():
    """Display overall performance metrics"""
    st.subheader("Performance Overview")

    stats_df = pd.DataFrame(st.session_state.statistics)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_ops = len(stats_df)
        st.metric("Total Operations", total_ops)

    with col2:
        avg_time = stats_df['time'].mean()
        st.metric("Avg Time (s)", f"{avg_time:.3f}")

    with col3:
        total_time = stats_df['time'].sum()
        st.metric("Total Time (s)", f"{total_time:.2f}")

    with col4:
        if 'rows_processed' in stats_df.columns:
            total_rows = stats_df['rows_processed'].sum()
            st.metric("Rows Processed", f"{total_rows:,}")

    st.markdown("---")

    # Time series of operations
    st.subheader("Operation Timeline")
    stats_df['operation_id'] = range(1, len(stats_df) + 1)

    fig = px.line(stats_df, x='operation_id', y='time',
                  color='operation', markers=True,
                  title='Execution Time per Operation',
                  labels={'operation_id': 'Operation Number', 'time': 'Time (seconds)'})
    st.plotly_chart(fig, use_container_width=True)

    # Operation type distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Operations by Type")
        op_counts = stats_df['operation'].value_counts()
        fig = px.pie(values=op_counts.values, names=op_counts.index,
                     title='Operation Distribution')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Time by Operation Type")
        time_by_op = stats_df.groupby('operation')['time'].sum()
        fig = px.bar(x=time_by_op.index, y=time_by_op.values,
                     labels={'x': 'Operation', 'y': 'Total Time (s)'},
                     title='Total Time by Operation Type')
        st.plotly_chart(fig, use_container_width=True)


def display_scheme_comparison():
    """Compare different FHE schemes"""
    st.subheader("Scheme Comparison")

    stats_df = pd.DataFrame(st.session_state.statistics)

    # Check if we have multiple schemes
    schemes = stats_df['scheme'].unique()

    if len(schemes) < 2:
        st.info(
            f"Currently using only {schemes[0]} scheme. To compare schemes, perform operations with different schemes.")
        display_single_scheme_info(schemes[0])
        return

    # Comparison metrics
    st.write("**Performance Comparison by Scheme**")

    scheme_stats = stats_df.groupby('scheme').agg({
        'time': ['mean', 'min', 'max', 'sum', 'count']
    }).round(4)

    scheme_stats.columns = ['Avg Time', 'Min Time', 'Max Time', 'Total Time', 'Operations']
    st.dataframe(scheme_stats, use_container_width=True)

    # Visualization
    col1, col2 = st.columns(2)

    with col1:
        # Average time comparison
        avg_times = stats_df.groupby('scheme')['time'].mean()
        fig = px.bar(x=avg_times.index, y=avg_times.values,
                     labels={'x': 'Scheme', 'y': 'Average Time (s)'},
                     title='Average Execution Time by Scheme',
                     color=avg_times.index)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Box plot for time distribution
        fig = px.box(stats_df, x='scheme', y='time',
                     title='Time Distribution by Scheme',
                     labels={'scheme': 'Scheme', 'time': 'Time (s)'})
        st.plotly_chart(fig, use_container_width=True)

    # Library comparison
    st.markdown("---")
    st.subheader("Library Performance")

    if 'library' in stats_df.columns:
        library_scheme = stats_df.groupby(['library', 'scheme'])['time'].mean().reset_index()
        fig = px.bar(library_scheme, x='scheme', y='time', color='library',
                     barmode='group',
                     title='Average Time by Library and Scheme',
                     labels={'time': 'Average Time (s)'})
        st.plotly_chart(fig, use_container_width=True)

    # Detailed comparison table
    st.markdown("---")
    st.subheader("Detailed Scheme Characteristics")

    comparison_data = []
    for scheme in schemes:
        scheme_data = stats_df[stats_df['scheme'] == scheme]
        comparison_data.append({
            'Scheme': scheme,
            'Operations': len(scheme_data),
            'Avg Time (s)': scheme_data['time'].mean(),
            'Std Dev (s)': scheme_data['time'].std(),
            'Total Time (s)': scheme_data['time'].sum(),
            'Library': scheme_data['library'].mode()[0] if 'library' in scheme_data.columns else 'N/A'
        })

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)

    # Scheme recommendations
    st.markdown("---")
    st.subheader("📋 Scheme Recommendations")

    best_scheme = comparison_df.loc[comparison_df['Avg Time (s)'].idxmin(), 'Scheme']
    st.success(f"**Fastest Scheme:** {best_scheme}")

    display_scheme_characteristics()


def display_single_scheme_info(scheme):
    """Display information about a single scheme"""
    st.markdown("---")
    st.subheader(f"Scheme Information: {scheme}")

    characteristics = {
        'CKKS': {
            'Type': 'Approximate',
            'Data Support': 'Real/Complex numbers',
            'Best For': 'Machine learning, signal processing',
            'Precision': 'Approximate (configurable)',
            'Operations': 'Addition, Multiplication, Rotation',
            'Key Features': 'SIMD operations, bootstrapping support'
        },
        'BFV': {
            'Type': 'Exact',
            'Data Support': 'Integers',
            'Best For': 'Exact computations, database operations',
            'Precision': 'Exact',
            'Operations': 'Addition, Multiplication',
            'Key Features': 'Noise growth management, batching'
        },
        'BGV': {
            'Type': 'Exact',
            'Data Support': 'Integers',
            'Best For': 'General purpose, leveled computations',
            'Precision': 'Exact',
            'Operations': 'Addition, Multiplication, Rotation',
            'Key Features': 'Modulus switching, key switching'
        }
    }

    if scheme in characteristics:
        info = characteristics[scheme]
        col1, col2 = st.columns(2)

        with col1:
            for key in ['Type', 'Data Support', 'Best For']:
                st.write(f"**{key}:** {info[key]}")

        with col2:
            for key in ['Precision', 'Operations', 'Key Features']:
                st.write(f"**{key}:** {info[key]}")


def display_scheme_characteristics():
    """Display characteristics of all schemes"""
    characteristics_data = [
        {
            'Scheme': 'CKKS',
            'Type': 'Approximate',
            'Data': 'Real/Complex',
            'Precision': '~40-60 bits',
            'Best Use': 'ML, Analytics',
            'Speed': 'Fast'
        },
        {
            'Scheme': 'BFV',
            'Type': 'Exact',
            'Data': 'Integers',
            'Precision': 'Exact',
            'Best Use': 'Database ops',
            'Speed': 'Medium'
        },
        {
            'Scheme': 'BGV',
            'Type': 'Exact',
            'Data': 'Integers',
            'Precision': 'Exact',
            'Best Use': 'General purpose',
            'Speed': 'Medium-Fast'
        }
    ]

    char_df = pd.DataFrame(characteristics_data)
    st.dataframe(char_df, use_container_width=True)


def display_detailed_analytics():
    """Display detailed analytics"""
    st.subheader("Detailed Analytics")

    stats_df = pd.DataFrame(st.session_state.statistics)

    # Time analysis
    st.write("**Time Analysis**")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Min Time", f"{stats_df['time'].min():.4f}s")

    with col2:
        st.metric("Max Time", f"{stats_df['time'].max():.4f}s")

    with col3:
        st.metric("Std Deviation", f"{stats_df['time'].std():.4f}s")

    # Histogram of execution times
    fig = px.histogram(stats_df, x='time', nbins=20,
                       title='Distribution of Execution Times',
                       labels={'time': 'Time (seconds)', 'count': 'Frequency'})
    st.plotly_chart(fig, use_container_width=True)

    # Correlation analysis
    if len(stats_df.columns) > 3:
        st.markdown("---")
        st.subheader("Correlation Analysis")

        numeric_cols = stats_df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 1:
            corr_matrix = stats_df[numeric_cols].corr()

            fig = px.imshow(corr_matrix,
                            labels=dict(color="Correlation"),
                            title="Feature Correlation Matrix",
                            color_continuous_scale='RdBu_r',
                            aspect='auto')
            st.plotly_chart(fig, use_container_width=True)

    # Efficiency metrics
    st.markdown("---")
    st.subheader("Efficiency Metrics")

    if 'rows_processed' in stats_df.columns and 'columns_encrypted' in stats_df.columns:
        stats_df['rows_per_second'] = stats_df['rows_processed'] / stats_df['time']
        stats_df['efficiency_score'] = (stats_df['rows_processed'] * stats_df['columns_encrypted']) / stats_df['time']

        col1, col2 = st.columns(2)

        with col1:
            avg_rps = stats_df['rows_per_second'].mean()
            st.metric("Avg Rows/Second", f"{avg_rps:.2f}")

        with col2:
            avg_efficiency = stats_df['efficiency_score'].mean()
            st.metric("Avg Efficiency Score", f"{avg_efficiency:.2f}")

        # Efficiency over time
        fig = px.line(stats_df, x=stats_df.index, y='efficiency_score',
                      title='Efficiency Score Over Operations',
                      labels={'index': 'Operation', 'efficiency_score': 'Efficiency Score'})
        st.plotly_chart(fig, use_container_width=True)

    # Raw data table
    st.markdown("---")
    st.subheader("Raw Statistics Data")
    st.dataframe(stats_df, use_container_width=True)

    # Download option
    csv = stats_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Statistics CSV",
        data=csv,
        file_name="fhe_statistics.csv",
        mime="text/csv"
    )


def display_operation_breakdown():
    """Display detailed operation breakdown"""
    st.subheader("Operation Breakdown")

    stats_df = pd.DataFrame(st.session_state.statistics)

    # Group by operation type
    operation_groups = stats_df.groupby('operation')

    for operation, group in operation_groups:
        with st.expander(f"📊 {operation.upper()} Operations ({len(group)} total)"):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Count", len(group))

            with col2:
                st.metric("Avg Time", f"{group['time'].mean():.3f}s")

            with col3:
                st.metric("Total Time", f"{group['time'].sum():.2f}s")

            with col4:
                st.metric("Std Dev", f"{group['time'].std():.3f}s")

            # Timeline for this operation
            group_with_index = group.copy()
            group_with_index['op_number'] = range(1, len(group) + 1)

            fig = px.scatter(group_with_index, x='op_number', y='time',
                             title=f'{operation.title()} Operation Times',
                             labels={'op_number': 'Operation Number', 'time': 'Time (s)'},
                             trendline="lowess")
            st.plotly_chart(fig, use_container_width=True)

            # Detailed table
            st.dataframe(group, use_container_width=True)

    # Performance trends
    st.markdown("---")
    st.subheader("Performance Trends")

    # Create multi-line chart for different operations
    fig = go.Figure()

    for operation in stats_df['operation'].unique():
        op_data = stats_df[stats_df['operation'] == operation].reset_index(drop=True)
        fig.add_trace(go.Scatter(
            x=op_data.index,
            y=op_data['time'],
            mode='lines+markers',
            name=operation.title()
        ))

    fig.update_layout(
        title='Performance Comparison Across Operations',
        xaxis_title='Operation Sequence',
        yaxis_title='Time (seconds)',
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Statistical summary
    st.markdown("---")
    st.subheader("Statistical Summary by Operation")

    summary = stats_df.groupby('operation')['time'].describe().round(4)
    st.dataframe(summary, use_container_width=True)

    # Key insights
    st.markdown("---")
    st.subheader("🔍 Key Insights")

    fastest_op = stats_df.loc[stats_df['time'].idxmin(), 'operation']
    slowest_op = stats_df.loc[stats_df['time'].idxmax(), 'operation']

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"**Fastest Operation:** {fastest_op}")
        st.write(f"Time: {stats_df['time'].min():.4f}s")

    with col2:
        st.warning(f"**Slowest Operation:** {slowest_op}")
        st.write(f"Time: {stats_df['time'].max():.4f}s")

    # Performance improvement suggestions
    st.info("""
    **💡 Performance Tips:**
    - CKKS is generally faster for approximate computations
    - BFV/BGV provide exact results but may be slower
    - Larger polynomial modulus degrees increase security but reduce performance
    - Batch operations when possible to amortize overhead
    - Consider the trade-off between security level and performance
    """)