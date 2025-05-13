import streamlit as st


def custom_css_sidebar():
    st.markdown("""
    <style>
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #9a44eb;
        border-right: 1px solid #eaeaea;
    }
    
    /* Optional: Add logo area styling */
    .sidebar-logo {
        padding: 1rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Custom styling for the option menu */
    .css-16eck9z {
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

def general_css():
    st.markdown("""
<style>
    /* General Headers */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #8E24AA;
        margin-bottom: 1rem;
    }

    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #5E35B1;
        text-align: center;
        margin-top: 40px;
        margin-bottom: 1rem;
    }

    /* Hero Section */
    .home-hero {
        background: linear-gradient(135deg, #9C27B0 0%, #4A148C 100%);
        padding: 3rem;
        border-radius: 0.7rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }

    .home-hero h1 {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    .home-hero p {
        font-size: 1.2rem;
        opacity: 0.9;
    }

    /* Feature Cards */
    .feature-card {
        padding: 1.5rem;
        border-radius: 0.7rem;
        background: white;
        box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.1);
        margin-bottom: 1.5rem;
        height: 100%;
        transition: all 0.3s ease;
    }

    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 0.25rem 2rem 0 rgba(58, 59, 69, 0.2);
    }

    .feature-icon {
        font-size: 2.5rem;
        color: #8E24AA;
        margin-bottom: 1rem;
    }

    .feature-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #212121;
    }

    .feature-description {
        color: #616161;
    }

    /* Metrics Display */
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #8E24AA;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #616161;
        margin-top: -0.5rem;
    }

    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        color: #212121;
    }

    /* File Upload Section */
    .upload-section {
        padding: 1.5rem;
        border-radius: 0.7rem;
        background: #f5f0ff;
        margin-bottom: 1.5rem;
        border: 1px dashed #8E24AA;
        text-align: center;
    }

    /* Customize File Upload Hover Effect */
    div.stFileUploader > label div:hover {
        background-color: #9C27B0 !important;
        color: white !important;
        transition: 0.3s;
    }

    /* Footer */
    .home-footer {
        margin-top: 2rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e0e0e0;
        color: #757575;
        font-size: 0.9rem;
        text-align: center;
    }

    .home-footer a {
        color: #8E24AA;
        text-decoration: none;
        font-weight: bold;
    }

    .home-footer a:hover {
        color: #5E35B1;
        text-decoration: underline;
    }
</style>

""", unsafe_allow_html=True)
