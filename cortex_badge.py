import streamlit as st


def show_cortex_badge():
    st.sidebar.divider()
    st.sidebar.markdown(
        """
        <div style="text-align: center; padding: 0.5rem 0;">
            <span style="font-size: 0.75rem; color: #888;">Built with</span><br/>
            <a href="https://docs.snowflake.com/en/user-guide/cortex-code/cortex-code" target="_blank"
               style="text-decoration: none; font-weight: 600; font-size: 0.95rem; color: #29B5E8;">
                ❄️ Cortex Code
            </a><br/>
            <span style="font-size: 0.7rem; color: #666;">AI-Assisted ML Development</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
