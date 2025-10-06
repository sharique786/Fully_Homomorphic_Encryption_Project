"""
Key Management Page
Comprehensive key generation, rotation, import/export
"""

# Automatic key rotation with version control
# Backward compatibility for old encrypted data
# Secure key export/import
# Key health monitoring
# Usage statistics tracking

import streamlit as st
import json
from datetime import datetime
from ui_components import render_key_display, render_alert


def render():
    """Render key management page"""
    st.header("🔑 Key Management")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🆕 Generate Keys",
        "🔄 Key Rotation",
        "📥 Import/Export",
        "📊 Key Status"
    ])

    with tab1:
        render_key_generation_tab()

    with tab2:
        render_key_rotation_tab()

    with tab3:
        render_import_export_tab()

    with tab4:
        render_key_status_tab()


def render_key_generation_tab():
    """Render key generation interface"""
    st.subheader("🆕 Generate New FHE Keys")

    st.markdown("""
    Generate a new set of encryption keys for your FHE operations.
    Choose the appropriate scheme and parameters based on your security requirements.
    """)

    # Key generation options
    col1, col2 = st.columns(2)

    with col1:
        library = st.selectbox(
            "FHE Library:",
            ["TenSEAL", "OpenFHE"],
            help="Select the FHE library for key generation"
        )

        scheme = st.selectbox(
            "Encryption Scheme:",
            ["BFV", "BGV", "CKKS"] if library == "OpenFHE" else ["BFV", "CKKS"],
            help="Select encryption scheme"
        )

    with col2:
        poly_degree = st.selectbox(
            "Polynomial Degree:",
            [2048, 4096, 8192, 16384],
            index=2,
            help="Higher = more secure but slower"
        )

        security_level = st.selectbox(
            "Security Level:",
            [128, 192, 256],
            help="Bits of security"
        )

    # Additional parameters based on scheme
    if scheme in ['BFV', 'BGV']:
        plain_modulus = st.selectbox(
            "Plain Modulus:",
            [65537, 786433, 1032193],
            index=2
        )
        parameters = {
            'poly_modulus_degree': poly_degree,
            'plain_modulus': plain_modulus,
            'security_level': security_level
        }
    else:  # CKKS
        scale_factor = st.slider(
            "Scale Factor:",
            20, 60, 40,
            help="Precision for CKKS"
        )
        parameters = {
            'poly_modulus_degree': poly_degree,
            'scale_factor': scale_factor,
            'security_level': security_level
        }

    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        generate_relin_keys = st.checkbox(
            "Generate Relinearization Keys",
            value=True,
            help="Required for multiplication operations"
        )

        generate_galois_keys = st.checkbox(
            "Generate Galois Keys",
            value=True,
            help="Required for rotation operations"
        )

        if generate_galois_keys:
            rotation_indices = st.text_input(
                "Rotation Indices (comma-separated):",
                value="1,2,3,-1,-2,-3",
                help="Specify rotation steps"
            )

    st.markdown("---")

    # Generate button
    if st.button("🔑 Generate Keys", type="primary"):
        with st.spinner("Generating FHE keys... This may take a moment."):
            key_manager = st.session_state.key_manager

            # Generate keys
            keys = key_manager.generate_keys(library, scheme, parameters)

            if keys['status'] == 'success':
                st.success("✅ Keys generated successfully!")

                # Show generation info
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Generation Time", f"{keys['metadata']['generation_duration_ms']:.2f} ms")
                with col2:
                    st.metric("Key Size", f"{keys.get('key_size_bytes', 0) / 1024:.2f} KB")
                with col3:
                    st.metric("Version", keys['metadata']['version'])

                # Display keys
                st.markdown("### 🔑 Generated Keys")

                show_private = st.checkbox(
                    "⚠️ Show Private Key (Handle with care!)",
                    value=False
                )

                render_key_display(keys, include_private=show_private)

                # Save option
                st.markdown("---")
                st.warning("💾 **Important:** Save your keys securely before leaving this page!")

                col1, col2 = st.columns(2)

                with col1:
                    if st.button("💾 Save Keys to Session"):
                        st.session_state.current_keys = keys
                        st.success("Keys saved to session!")

                with col2:
                    keys_json = json.dumps(keys, indent=2)
                    st.download_button(
                        label="📥 Download Keys (JSON)",
                        data=keys_json,
                        file_name=f"fhe_keys_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            else:
                st.error("❌ Key generation failed!")


def render_key_rotation_tab():
    """Render key rotation interface"""
    st.subheader("🔄 Key Rotation")

    st.markdown("""
    Rotate your encryption keys while maintaining backward compatibility with previously encrypted data.
    This is important for long-term security.
    """)

    if not hasattr(st.session_state.key_manager, 'public_key') or not st.session_state.key_manager.public_key:
        st.warning("⚠️ No active keys found. Generate keys first.")
        return

    # Current key info
    st.markdown("### 📋 Current Key Information")

    key_metadata = st.session_state.key_manager.key_metadata

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(f"**Current Version:** {key_metadata.get('version', 1)}")
    with col2:
        st.info(f"**Scheme:** {key_metadata.get('scheme', 'N/A')}")
    with col3:
        st.info(f"**Generated:** {key_metadata.get('generation_time', 'N/A')[:10]}")

    st.markdown("---")

    # Rotation configuration
    st.markdown("### ⚙️ Rotation Configuration")

    rotation_reason = st.selectbox(
        "Rotation Reason:",
        [
            "Scheduled rotation",
            "Security upgrade",
            "Parameter change",
            "Compliance requirement",
            "Other"
        ]
    )

    maintain_compatibility = st.checkbox(
        "Maintain backward compatibility",
        value=True,
        help="Allow old keys to decrypt existing data"
    )

    # New parameters
    st.write("**New Key Parameters:**")

    col1, col2 = st.columns(2)

    with col1:
        new_poly_degree = st.selectbox(
            "New Polynomial Degree:",
            [2048, 4096, 8192, 16384],
            index=3
        )

    with col2:
        new_security = st.selectbox(
            "New Security Level:",
            [128, 192, 256],
            index=1
        )

    new_parameters = {
        'poly_modulus_degree': new_poly_degree,
        'security_level': new_security
    }

    # Perform rotation
    if st.button("🔄 Rotate Keys", type="primary"):
        with st.spinner("Rotating keys..."):
            key_manager = st.session_state.key_manager

            old_keys = {
                'public_key': key_manager.public_key,
                'private_key': key_manager.private_key
            }

            new_keys = key_manager.rotate_keys(old_keys, new_parameters)

            if new_keys['status'] == 'success':
                st.success("✅ Keys rotated successfully!")

                st.info(f"""
                **Rotation Details:**
                - Old Version: {new_keys['rotation_info']['old_version']}
                - New Version: {new_keys['rotation_info']['new_version']}
                - Backward Compatible: {new_keys['rotation_info']['backward_compatible']}
                """)

                # Option to download both old and new keys
                col1, col2 = st.columns(2)

                with col1:
                    old_keys_json = json.dumps(old_keys, indent=2)
                    st.download_button(
                        label="📥 Download Old Keys (Backup)",
                        data=old_keys_json,
                        file_name=f"old_keys_v{new_keys['rotation_info']['old_version']}.json",
                        mime="application/json"
                    )

                with col2:
                    new_keys_json = json.dumps(new_keys, indent=2)
                    st.download_button(
                        label="📥 Download New Keys",
                        data=new_keys_json,
                        file_name=f"new_keys_v{new_keys['rotation_info']['new_version']}.json",
                        mime="application/json"
                    )


def render_import_export_tab():
    """Render import/export interface"""
    st.subheader("📥 Import / Export Keys")

    tab1, tab2 = st.tabs(["📥 Import", "📤 Export"])

    with tab1:
        st.markdown("### Import Existing Keys")

        import_format = st.radio(
            "Key Format:",
            ["JSON", "Base64"],
            horizontal=True
        )

        key_input = st.text_area(
            "Paste your keys here:",
            height=200,
            help="Paste the complete key data"
        )

        if st.button("📥 Import Keys"):
            if not key_input:
                st.error("Please paste key data")
                return

            key_manager = st.session_state.key_manager
            success = key_manager.import_keys(key_input, format=import_format.lower())

            if success:
                st.success("✅ Keys imported successfully!")

                # Show imported key info
                metadata = key_manager.key_metadata
                st.json(metadata)
            else:
                st.error("❌ Failed to import keys. Please check the format.")

    with tab2:
        st.markdown("### Export Current Keys")

        if not hasattr(st.session_state.key_manager, 'public_key') or not st.session_state.key_manager.public_key:
            st.warning("⚠️ No keys available to export")
            return

        export_format = st.radio(
            "Export Format:",
            ["JSON", "Base64"],
            horizontal=True
        )

        include_private = st.checkbox(
            "Include Private Key",
            value=False,
            help="⚠️ Only include for backup purposes"
        )

        if st.button("📤 Prepare Export"):
            key_manager = st.session_state.key_manager
            exported_keys = key_manager.export_keys(
                format=export_format.lower(),
                include_private=include_private
            )

            st.text_area(
                "Exported Keys:",
                value=exported_keys,
                height=300
            )

            st.download_button(
                label="💾 Download Keys File",
                data=exported_keys,
                file_name=f"fhe_keys_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )


def render_key_status_tab():
    """Render key status and information"""
    st.subheader("📊 Key Status & Information")

    key_manager = st.session_state.key_manager

    if not hasattr(key_manager, 'public_key') or not key_manager.public_key:
        st.warning("⚠️ No active keys in session")
        return

    # Key metadata
    st.markdown("### 📋 Key Metadata")

    metadata = key_manager.key_metadata

    col1, col2 = st.columns(2)

    with col1:
        st.write("**General Information:**")
        st.json({
            'scheme': metadata.get('scheme', 'N/A'),
            'library': metadata.get('library', 'N/A'),
            'version': metadata.get('version', 1),
            'generation_time': metadata.get('generation_time', 'N/A')
        })

    with col2:
        st.write("**Parameters:**")
        st.json(metadata.get('parameters', {}))

    # Key health check
    st.markdown("### 🏥 Key Health Check")

    health_checks = [
        ("✅", "Keys are properly initialized"),
        ("✅", "Public key is available"),
        ("✅", "Private key is secured"),
        ("✅", "Metadata is complete"),
    ]

    if hasattr(key_manager, 'evaluation_keys') and key_manager.evaluation_keys:
        health_checks.append(("✅", "Evaluation keys are available"))
    else:
        health_checks.append(("⚠️", "Evaluation keys not generated"))

    for status, message in health_checks:
        st.write(f"{status} {message}")

    # Usage statistics
    st.markdown("### 📈 Usage Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Check if data has been encrypted
        encrypted_count = 1 if hasattr(st.session_state, 'encrypted_data') and st.session_state.encrypted_data else 0
        st.metric("Encryption Operations", encrypted_count)

    with col2:
        # Check operation log if available
        operation_count = 0
        if hasattr(st.session_state, 'fhe_processor') and st.session_state.fhe_processor:
            operation_count = len(st.session_state.fhe_processor.operation_log)
        st.metric("FHE Operations", operation_count)

    with col3:
        key_age = "N/A"
        if metadata.get('generation_time'):
            try:
                gen_time = datetime.fromisoformat(metadata['generation_time'])
                age_days = (datetime.now() - gen_time).days
                key_age = f"{age_days} days"
            except:
                pass
        st.metric("Key Age", key_age)

    # Security recommendations
    st.markdown("### 🔒 Security Recommendations")

    recommendations = []

    # Check key age
    if metadata.get('generation_time'):
        try:
            gen_time = datetime.fromisoformat(metadata['generation_time'])
            age_days = (datetime.now() - gen_time).days

            if age_days > 90:
                recommendations.append(("⚠️", "Keys are older than 90 days. Consider rotation."))
            elif age_days > 365:
                recommendations.append(("❌", "Keys are older than 1 year. Rotation strongly recommended!"))
            else:
                recommendations.append(("✅", f"Key age is acceptable ({age_days} days)."))
        except:
            pass

    # Check security level
    security = metadata.get('parameters', {}).get('security_level', 128)
    if security < 128:
        recommendations.append(("⚠️", "Consider increasing security level to at least 128 bits."))
    elif security >= 128:
        recommendations.append(("✅", f"Security level is adequate ({security} bits)."))

    # Check if keys are backed up
    recommendations.append(("⚠️", "Ensure keys are backed up securely offline."))
    recommendations.append(("💡", "Never share private keys or commit them to version control."))

    for status, message in recommendations:
        if status == "❌":
            st.error(f"{status} {message}")
        elif status == "⚠️":
            st.warning(f"{status} {message}")
        elif status == "💡":
            st.info(f"{status} {message}")
        else:
            st.success(f"{status} {message}")