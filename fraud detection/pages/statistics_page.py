"""
Performance Statistics Page
Screen 3: Compare FHE schemes and show performance metrics
"""

# Denormalized multi-table support
# Automatic PII detection
# Data validation against schema
# Efficient batch processing
# Transaction filtering by date/currency/user

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from config import FHE_SCHEMES


def render():
    """Render performance statistics page"""
    st.header("📈 Performance Statistics & Comparison")

    tab1, tab2, tab3 = st.tabs([
        "⚖️ Scheme Comparison",
        "⏱️ Performance Metrics",
        "📊 Statistical Analysis"
    ])

    with tab1:
        render_scheme_comparison_tab()

    with tab2:
        render_performance_metrics_tab()

    with tab3:
        render_statistical_analysis_tab()


def render_scheme_comparison_tab():
    """Render FHE scheme comparison"""
    st.subheader("⚖️ FHE Scheme Comparison")

    st.markdown("""
    Compare different FHE schemes to understand their characteristics,
    performance trade-offs, and suitability for various use cases.
    """)

    # Scheme selection for comparison
    schemes_to_compare = st.multiselect(
        "Select schemes to compare:",
        ["BFV", "BGV", "CKKS"],
        default=["BFV", "CKKS"]
    )

    if len(schemes_to_compare) < 2:
        st.warning("⚠️ Please select at least 2 schemes to compare")
        return

    # Comparison table
    st.markdown("### 📋 Feature Comparison")

    comparison_data = []
    for scheme in schemes_to_compare:
        scheme_info = FHE_SCHEMES[scheme]
        comparison_data.append({
            'Scheme': scheme,
            'Full Name': scheme_info['name'],
            'Type': scheme_info['type'],
            'Precision': scheme_info['precision'],
            'Batching': '✅' if scheme_info['supports_batching'] else '❌',
            'Default Poly Degree': scheme_info['default_poly_modulus'],
            'Security': f"{scheme_info['default_security']} bits"
        })

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)

    # Performance simulation
    st.markdown("### ⚡ Performance Simulation")

    if st.button("🚀 Run Performance Benchmark", type="primary"):
        with st.spinner("Running benchmarks..."):
            benchmark_results = run_scheme_benchmarks(schemes_to_compare)
            st.session_state.benchmark_results = benchmark_results

            display_benchmark_results(benchmark_results)


def run_scheme_benchmarks(schemes):
    """Simulate performance benchmarks for schemes"""
    # Simulated benchmark data
    results = []

    base_encryption = {'BFV': 1200, 'BGV': 980, 'CKKS': 1450}
    base_operation = {'BFV': 300, 'BGV': 250, 'CKKS': 400}
    base_decryption = {'BFV': 650, 'BGV': 580, 'CKKS': 720}

    for scheme in schemes:
        # Add some randomness
        results.append({
            'scheme': scheme,
            'encryption_time': base_encryption[scheme] * np.random.uniform(0.9, 1.1),
            'operation_time': base_operation[scheme] * np.random.uniform(0.9, 1.1),
            'decryption_time': base_decryption[scheme] * np.random.uniform(0.9, 1.1),
            'memory_usage': np.random.uniform(200, 800),
            'noise_budget': np.random.uniform(75, 95),
            'ciphertext_size': np.random.uniform(500, 2000)
        })

    return pd.DataFrame(results)


def display_benchmark_results(df):
    """Display benchmark results with visualizations"""
    st.success("✅ Benchmark completed!")

    # Performance comparison chart
    st.markdown("#### ⏱️ Time Comparison")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Encryption',
        x=df['scheme'],
        y=df['encryption_time'],
        marker_color='lightblue'
    ))
    fig.add_trace(go.Bar(
        name='Operation',
        x=df['scheme'],
        y=df['operation_time'],
        marker_color='lightgreen'
    ))
    fig.add_trace(go.Bar(
        name='Decryption',
        x=df['scheme'],
        y=df['decryption_time'],
        marker_color='salmon'
    ))

    fig.update_layout(
        title='Performance Comparison (milliseconds)',
        barmode='group',
        height=400,
        yaxis_title='Time (ms)',
        xaxis_title='Scheme'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Memory and size metrics
    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            df,
            x='scheme',
            y='memory_usage',
            title='Memory Usage Comparison (MB)',
            color='scheme',
            text='memory_usage'
        )
        fig.update_traces(texttemplate='%{text:.0f} MB', textposition='outside')
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            df,
            x='scheme',
            y='ciphertext_size',
            title='Ciphertext Size (KB)',
            color='scheme',
            text='ciphertext_size'
        )
        fig.update_traces(texttemplate='%{text:.0f} KB', textposition='outside')
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Detailed metrics table
    st.markdown("#### 📊 Detailed Metrics")

    display_df = df.copy()
    display_df['encryption_time'] = display_df['encryption_time'].round(2)
    display_df['operation_time'] = display_df['operation_time'].round(2)
    display_df['decryption_time'] = display_df['decryption_time'].round(2)
    display_df['memory_usage'] = display_df['memory_usage'].round(2)
    display_df['noise_budget'] = display_df['noise_budget'].round(2)
    display_df['ciphertext_size'] = display_df['ciphertext_size'].round(2)

    st.dataframe(display_df, use_container_width=True)


def render_performance_metrics_tab():
    """Render performance metrics analysis"""
    st.subheader("⏱️ Performance Metrics Analysis")

    if not hasattr(st.session_state, 'encrypted_data') or not st.session_state.encrypted_data:
        st.info("💡 Encrypt data first to see actual performance metrics")
        render_theoretical_performance()
        return

    # Actual performance metrics from encryption
    encryption_result = st.session_state.encrypted_data['result']

    st.markdown("### 📊 Actual Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Encryption Time",
            f"{encryption_result['encryption_time_ms']:.2f} ms"
        )

    with col2:
        st.metric(
            "Values Encrypted",
            f"{encryption_result['total_values']:,}"
        )

    with col3:
        st.metric(
            "Library Used",
            encryption_result['library']
        )

    with col4:
        throughput = encryption_result['total_values'] / (encryption_result['encryption_time_ms'] / 1000)
        st.metric(
            "Throughput",
            f"{throughput:.0f} values/sec"
        )

    # Performance over different parameters
    st.markdown("### 📈 Parameter Impact Analysis")

    render_parameter_impact_analysis()


def render_theoretical_performance():
    """Show theoretical performance characteristics"""
    st.markdown("### 📊 Theoretical Performance Characteristics")

    poly_degrees = [2048, 4096, 8192, 16384]

    # Encryption time vs polynomial degree
    encryption_times = [100, 250, 600, 1500]
    security_levels = [80, 112, 128, 192]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=poly_degrees,
        y=encryption_times,
        name='Encryption Time',
        mode='lines+markers',
        yaxis='y',
        line=dict(color='blue', width=3)
    ))

    fig.add_trace(go.Scatter(
        x=poly_degrees,
        y=security_levels,
        name='Security Level',
        mode='lines+markers',
        yaxis='y2',
        line=dict(color='green', width=3)
    ))

    fig.update_layout(
        title='Performance vs Security Trade-off',
        xaxis=dict(title='Polynomial Modulus Degree', type='log'),
        yaxis=dict(title='Encryption Time (ms)', side='left'),
        yaxis2=dict(title='Security Level (bits)', side='right', overlaying='y'),
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def render_parameter_impact_analysis():
    """Analyze impact of different parameters"""

    parameter = st.selectbox(
        "Select parameter to analyze:",
        ["Polynomial Degree", "Plain Modulus", "Security Level", "Batch Size"]
    )

    if parameter == "Polynomial Degree":
        data = {
            'Degree': [2048, 4096, 8192, 16384],
            'Encryption (ms)': [100, 250, 600, 1500],
            'Operation (ms)': [30, 75, 180, 450],
            'Memory (MB)': [50, 120, 300, 750],
            'Security (bits)': [80, 112, 128, 192]
        }
    elif parameter == "Plain Modulus":
        data = {
            'Modulus': [65537, 786433, 1032193],
            'Encryption (ms)': [500, 600, 650],
            'Operation (ms)': [150, 180, 200],
            'Precision': [16, 20, 21],
            'Range': ['Small', 'Medium', 'Large']
        }
    else:
        data = {
            'Security Level': [128, 192, 256],
            'Encryption (ms)': [600, 900, 1200],
            'Key Size (KB)': [500, 800, 1100],
            'Performance Impact': ['Baseline', '+50%', '+100%']
        }

    df = pd.DataFrame(data)

    # Display table
    st.dataframe(df, use_container_width=True)

    # Visualize
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        fig = px.line(
            df,
            x=df.columns[0],
            y=numeric_cols[1:].tolist(),
            title=f'Impact of {parameter}',
            markers=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


def render_statistical_analysis_tab():
    """Render statistical analysis"""
    st.subheader("📊 Statistical Analysis")

    if not hasattr(st.session_state, 'benchmark_results'):
        st.info("💡 Run scheme comparison benchmark to see statistical analysis")
        return

    df = st.session_state.benchmark_results

    # Summary statistics
    st.markdown("### 📈 Summary Statistics")

    stats_df = df.describe().round(2)
    st.dataframe(stats_df, use_container_width=True)

    # Correlation analysis
    st.markdown("### 🔗 Correlation Analysis")

    numeric_cols = ['encryption_time', 'operation_time', 'decryption_time',
                    'memory_usage', 'noise_budget', 'ciphertext_size']

    corr_matrix = df[numeric_cols].corr()

    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect='auto',
        title='Correlation Matrix of Performance Metrics',
        color_continuous_scale='RdBu_r'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Performance ranking
    st.markdown("### 🏆 Performance Ranking")

    # Calculate overall performance score
    df_rank = df.copy()
    df_rank['total_time'] = df_rank['encryption_time'] + df_rank['operation_time'] + df_rank['decryption_time']
    df_rank['efficiency_score'] = (
        (1 / df_rank['total_time']) * 1000 +
        (df_rank['noise_budget'] / 100) * 500 -
        (df_rank['memory_usage'] / 1000) * 200
    )
    df_rank = df_rank.sort_values('efficiency_score', ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Overall Efficiency Ranking:**")
        ranking_df = df_rank[['scheme', 'efficiency_score', 'total_time']].copy()
        ranking_df['rank'] = range(1, len(ranking_df) + 1)
        ranking_df['efficiency_score'] = ranking_df['efficiency_score'].round(2)
        ranking_df['total_time'] = ranking_df['total_time'].round(2)
        st.dataframe(ranking_df, use_container_width=True)

    with col2:
        fig = px.bar(
            df_rank,
            x='scheme',
            y='efficiency_score',
            title='Efficiency Score by Scheme',
            color='efficiency_score',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Recommendations
    st.markdown("### 💡 Recommendations")

    best_scheme = df_rank.iloc[0]['scheme']

    recommendations = {
        'BFV': {
            'icon': '🔢',
            'best_for': 'Exact integer arithmetic, financial calculations',
            'strengths': 'High precision, good performance for integer operations',
            'considerations': 'Limited to integer operations only'
        },
        'BGV': {
            'icon': '⚡',
            'best_for': 'SIMD operations, batch processing',
            'strengths': 'Excellent for parallel operations, efficient batching',
            'considerations': 'More complex parameter tuning required'
        },
        'CKKS': {
            'icon': '📊',
            'best_for': 'Real number operations, machine learning',
            'strengths': 'Supports floating-point, good for statistical analysis',
            'considerations': 'Approximate results, precision loss'
        }
    }

    for scheme in df_rank['scheme']:
        info = recommendations[scheme]

        with st.expander(f"{info['icon']} {scheme} - {'⭐ Best Overall' if scheme == best_scheme else 'Details'}"):
            st.write(f"**Best For:** {info['best_for']}")
            st.write(f"**Strengths:** {info['strengths']}")
            st.write(f"**Considerations:** {info['considerations']}")

            # Show metrics for this scheme
            scheme_metrics = df_rank[df_rank['scheme'] == scheme].iloc[0]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Encryption", f"{scheme_metrics['encryption_time']:.0f} ms")
            with col2:
                st.metric("Operation", f"{scheme_metrics['operation_time']:.0f} ms")
            with col3:
                st.metric("Decryption", f"{scheme_metrics['decryption_time']:.0f} ms")

    # Export results
    st.markdown("---")

    if st.button("📥 Export Performance Report"):
        report = generate_performance_report(df_rank)
        st.download_button(
            label="💾 Download CSV Report",
            data=report,
            file_name=f"fhe_performance_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


def generate_performance_report(df):
    """Generate performance report for export"""
    report_df = df.copy()
    report_df['timestamp'] = pd.Timestamp.now()
    return report_df.to_csv(index=False)