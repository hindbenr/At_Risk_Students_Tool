import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import base64
import os
import joblib
import time
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# ===== SETUP =====
st.set_page_config(
    page_title="At-Risk Student Prediction Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Montserrat font, blue theme, enhanced interactivity, and spacing
def inject_css():
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700&display=swap');
        
        * {{
            font-family: 'Montserrat', sans-serif !important;
        }}
        
        .main {{
            background-color: #f9f0ff;
        }}
        
        .stButton>button {{
            background-color: #9c6ade;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            transition: all 0.3s ease;
            position: relative;
            z-index: 1;
            margin-top: 1.5rem;
        }}
        
        .stButton>button:hover {{
            background-color: #7d48c1;
            transform: scale(1.05);
            box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        }}
        
        .stButton>button:active {{
            transform: scale(0.95);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .metric-card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
            margin-bottom: 2rem;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 15px rgba(0,0,0,0.2);
        }}
        
        .metric-card::after {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: 0.5s;
        }}
        
        .metric-card:hover::after {{
            left: 100%;
        }}
        
        .risk-badge {{
            padding: 8px 15px;
            border-radius: 20px;
            font-weight: bold;
            display: inline-block;
            transition: all 0.3s ease;
            position: relative;
            z-index: 2;
        }}
        
        .risk-badge:hover {{
            transform: scale(1.1);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        
        .at-risk {{
            background-color: #ff6b6b;
            color: white;
        }}
        
        .not-risk {{
            background-color: #51cf66;
            color: white;
        }}
        
        /* Scroll animations */
        [data-aos="fade-up"] {{
            opacity: 0;
            transition: all 0.6s ease;
        }}
        
        /* Tooltip styles */
        .tooltip {{
            position: relative;
            display: inline-block;
            cursor: pointer;
            margin-bottom: 1rem;
        }}
        
        .tooltip .tooltiptext {{
            visibility: hidden;
            width: 220px;
            background-color: #555;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 8px;
            position: absolute;
            z-index: 10;
            bottom: 125%;
            left: 50%;
            margin-left: -110px;
            opacity: 0;
            transition: all 0.3s ease;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }}
        
        .tooltip:hover .tooltiptext {{
            visibility: visible;
            opacity: 1;
            transform: translateY(-5px);
        }}
        
        /* Input field hover and focus effects */
        .stTextInput input, .stSlider div[role="slider"] {{
            transition: all 0.3s ease;
            border: 2px solid #E8E8E8;
            border-radius: 6px;
            margin-bottom: 1.5rem;
        }}
        
        .stTextInput input:hover, .stSlider div[role="slider"]:hover {{
            border-color: #E8E8E8;
            box-shadow: 0 0 8px rgba(125, 72, 193, 0.3);
        }}
        
        .stTextInput input:focus, .stSlider div[role="slider"]:focus {{
            border-color: #E8E8E8;
            box-shadow: 0 0 12px rgba(125, 72, 193, 0.5);
            outline: none;
        }}
        
        /* File uploader hover effect */
        .stFileUploader {{
            position: relative;
            transition: all 0.3s ease;
            margin-bottom: 2rem;
        }}
        
        .stFileUploader:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        
        /* Sidebar navigation hover effect */
        .stRadio > div > label {{
            transition: all 0.3s ease;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 0.5rem;
        }}
        
        .stRadio > div > label:hover {{
            background-color: #e6f0fa;
            transform: translateX(5px);
        }}
        
        /* Chart containers */
        .stPlotlyChart {{
            margin-bottom: 2rem;
        }}
        
        /* Column spacing */
        .stColumn > div {{
            padding: 0 1rem;
        }}
    </style>
    """, unsafe_allow_html=True)

inject_css()

# ===== NAVIGATION =====
def navigation():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["About", "Individual Analysis", "Batch Analysis"])
    return page

# ===== PAGES =====
def about_page():
    st.markdown("""
    <style>
        .project-title {
            font-family: 'Montserrat', sans-serif;
            font-weight: 700;
            font-size: 3rem;
            color: #1e3a8a;
            text-align: center;
            margin: 2rem 0;
            padding: 1rem;
            background: linear-gradient(135deg, #e6f0fa 0%, #f9f0ff 100%);
            border-radius: 15px;
            box-shadow: 0 8px 15px rgba(0,0,0,0.1);
            text-transform: uppercase;
            letter-spacing: 2px;
            transition: all 0.3s ease;
        }
        .project-title:hover {
            transform: scale(1.02);
            box-shadow: 0 10px 20px rgba(0,0,0,0.15);
        }
        .about-title {
            font-family: 'Montserrat', sans-serif;
            font-weight: 600;
            font-size: 2rem;
            color: #1e3a8a;
            margin-bottom: 2rem;
        }
        .section-title {
            font-family: 'Montserrat', sans-serif;
            font-weight: 500;
            font-size: 1.8rem;
            color: #1e3a8a;
            margin-top: 2rem;
            margin-bottom: 1.5rem;
        }
        .metric-circle {
            width: 150px;
            height: 150px;
            border-radius: 50%;
            background: #e6f0fa;
            border: 8px solid #4b6cb7;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            margin: 0 auto;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            position: relative;
            z-index: 1;
        }
        .metric-circle:hover {
            transform: scale(1.05) rotate(2deg);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        .metric-label {
            font-family: 'Montserrat', sans-serif;
            font-weight: 500;
            font-size: 1rem;
            color: #1e3a8a;
            margin-bottom: 0.5rem;
        }
        .metric-value {
            font-family: 'Montserrat', sans-serif;
            font-weight: 700;
            font-size: 2rem;
            color: #1e3a8a;
        }
        .about-container {
            margin-bottom: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Big Project Title
    st.markdown('<div class="project-title">At-Risk Student Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div style="margin-bottom: 2rem;"></div>', unsafe_allow_html=True)

    # Page Title
    st.markdown('<div class="about-title">About This Project</div>', unsafe_allow_html=True)

    # Project Overview
    st.markdown("""
        <div style="background-color: #e6f0fa; padding: 2rem; border-radius: 10px; 
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 2rem;" class="metric-card">
            <p style="font-family: 'Montserrat', sans-serif; font-size: 1.1rem; color: #333;">
                This project aims to identify students who may be at risk of academic difficulties 
                to enable timely intervention and support. It leverages a machine learning model 
                to predict the 'At-Risk' status based on various student data points including 
                academic performance, engagement metrics, and behavioral indicators.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Predictive Model Information
    st.markdown("""
    <div style="font-family: 'Montserrat', sans-serif; font-weight: 600; font-size: 2rem; color: #1e3a8a; margin-bottom: 2rem;">
    Predictive Model Information
    </div>
    """, unsafe_allow_html=True)

    # Metrics in Circle Format
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
            <div class="metric-circle">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value">89%</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="metric-circle">
                <div class="metric-label">Precision</div>
                <div class="metric-value">85%</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="metric-circle">
                <div class="metric-label">Recall</div>
                <div class="metric-value">82%</div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
            <div class="metric-circle">
                <div class="metric-label">F1-Score</div>
                <div class="metric-value">83%</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="margin-bottom: 2rem;"></div>', unsafe_allow_html=True)

    # Model Description
    st.markdown("""
        <div style="background-color: #e6f0fa; padding: 2rem; border-radius: 10px; 
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 2rem;" class="metric-card">
            <p style="font-family: 'Montserrat', sans-serif; font-size: 1.1rem; color: #333;">
                The model is a Logistic Regression trained on historical student performance data 
                from over 5,000 students across multiple academic years, including assignment scores, 
                attendance records, LMS engagement metrics, and behavioral indicators.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Project Creator & Supervisors
    st.markdown("""
    <div style="font-family: 'Montserrat', sans-serif; font-weight: 600; font-size: 2rem; color: #1e3a8a; margin-bottom: 2rem;">
    Project Creator & Supervisors
    </div>
    """, unsafe_allow_html=True)
    
    # Image paths (relative)
    image_files = {
        "hinda": "hinda.png",
        "souhaib": "souhaib.png",
        "tarik": "tarik.png",
        "ens": "ens.png",
        "aui": "aui.png"
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Creator card
        with st.container():
            col_img, col_text = st.columns([1, 2])
            with col_img:
                try:
                    st.image(image_files["hinda"], width=100)

                except FileNotFoundError:
                    st.error(f"Failed to load hinda.jpg")
            with col_text:
                st.markdown("**Hind Ben Rahmoun**")  
                st.markdown("Project Creator")  
                st.markdown("E-LSEI Student")  
                st.markdown("hind.benrahmoun@etu.uae.ac.ma")
            st.markdown('<div style="margin-bottom: 2rem;"></div>', unsafe_allow_html=True)
        
        # Supervisor card
        with st.container():
            col_img, col_text = st.columns([1, 2])
            with col_img:
                try:
                    st.image(image_files["souhaib"], width=100)
                except FileNotFoundError:
                    st.error(f"Failed to load souhaib.jpg")
            with col_text:
                st.markdown("**Souhaib Aammou**")  
                st.markdown("Project Supervisor")  
                st.markdown("Computer science professor at ENS")  
                st.markdown("aammou.souhaib@gmail.com")
            st.markdown('<div style="margin-bottom: 2rem;"></div>', unsafe_allow_html=True)
        
        # Co-Supervisor card
        with st.container():
            col_img, col_text = st.columns([1, 2])
            with col_img:
                try:
                    st.image(image_files["tarik"], width=100)
                except FileNotFoundError:
                    st.error(f"Failed to load tarik.jpg")
            with col_text:
                st.markdown("**Tarik Touis Ghmari**")  
                st.markdown("Co-Supervisor")  
                st.markdown("Data Analytics at Babel")  
                st.markdown("touistarik@gmail.com")
            st.markdown('<div style="margin-bottom: 2rem;"></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="font-family: 'Montserrat', sans-serif; font-weight: 550; font-size: 1.5rem; color: #1e3a8a; margin-bottom: 2rem;">
        Affiliated Institutions
        </div>
        """, unsafe_allow_html=True)
        
        # First institution with logo
        with st.container():
            col_logo, col_info = st.columns([1, 2])
            with col_logo:
                try:
                    st.image(image_files["ens"], width=100)
                except FileNotFoundError:
                    st.error(f"Failed to load ens.jpg")
            with col_info:
                st.markdown("**Ecole Normale Supérieure**")
                st.markdown("Department of Mathematics and Computer Science")
                st.markdown("Tetouan")
                st.markdown("*Home Institution*")
            st.markdown('<div style="margin-bottom: 2rem;"></div>', unsafe_allow_html=True)
        
        st.markdown("---")  # Divider
        st.markdown('<div style="margin-bottom: 2rem;"></div>', unsafe_allow_html=True)
        
        # Second institution with logo
        with st.container():
            col_logo, col_info = st.columns([1, 2])
            with col_logo:
                try:
                    st.image(image_files["aui"], width=100)
                except FileNotFoundError:
                    st.error(f"Failed to load aui.jpg")
            with col_info:
                st.markdown("**Al-Akhawayn University**")
                st.markdown("Center for Teaching and Learning")
                st.markdown("Ifrane")
                st.markdown("*Internship Host Institution*")
            st.markdown('<div style="margin-bottom: 2rem;"></div>', unsafe_allow_html=True)

# --- Load the Model ---
try:
    with open('lr.pkl', 'rb') as file:
        model = joblib.load(file)
except FileNotFoundError:
    st.error("Error: 'lr.pkl' not found. Please ensure the file is in the same directory.")
    model = None
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

def predict_risk(student_data):
    """
    Fully fixed prediction function that properly handles categorical and numerical features.
    """
    if model is None:
        st.error("Model not loaded. Please ensure 'lr.pkl' is available.")
        return {'risk_status': 'Error', 'risk_score': 0}
    
    try:
        # 1. Load the preprocessor
        with open('preprocessor.pkl', 'rb') as f:
            preprocessor = joblib.load(f)
        
        # 2. Define complete feature set with proper types
        input_features = {
            # Core features from user input (convert to float)
            'Average_assignment_score': float(student_data.get('avg_score', 0)),
            'Num_of_missing_assingnment': float(student_data.get('missing_assignments', 0)),
            'Total_LMS_Activity': float(student_data.get('lms_activity', 0)),
            'rate_Of_Globale_Attandence': float(student_data.get('attendance', 0)),
            
            # Numerical features with defaults
            'num_attandence_inPerson': 0.0,
            'Num_of_submitted_assignment': 0.0,
            'num_attendance_online': 0.0,
            'num_total_absence': 0.0,
            'Unmuted_mic_num': 0.0,
            'InMeeting_Duration': 0.0,
            'total_canvas_ressources': 0.0,
            'Camera_activation_num': 0.0,
            'Session_duration': 0.0,
            'Age': 0.0,
            'Q&A_teams_ participation': 0.0,
            'num_of viewed_ ressources': 0.0,
            'total_sessions_num': 0.0,
            'Total_assignments': 10.0,
            'Participation_online_absence_Flag': 0.0,
            
            # Categorical features with defaults
            'Gender': 'Female',
            'Course_id': '001',
            'Student_id': '1472',
            'Academic_level': 'Bachelor',
            'Prefere_Mode': 'Online',
            'Course_School': 'Science',
            'Majors': 'Computer Science'
        }

        # 3. Create DataFrame with correct dtypes
        input_df = pd.DataFrame([input_features])
        
        # Ensure categorical columns have string type
        categorical_cols = ['Gender', 'Course_id', 'Student_id', 'Academic_level',
                          'Prefere_Mode', 'Course_School', 'Majors']
        for col in categorical_cols:
            input_df[col] = input_df[col].astype('object')
        
        # 4. Apply preprocessing
        try:
            processed_data = preprocessor.transform(input_df)
        except Exception as e:
            st.error(f"Preprocessing failed. Please check: {str(e)}")
            st.error("Possible causes:")
            st.error("- Mismatch between input features and preprocessor expectations")
            st.error("- Incorrect data types for categorical/numerical features")
            return {'risk_status': 'Error', 'risk_score': 0}
        
        # 5. Make prediction
        try:
            prediction = model.predict(processed_data)
            risk_status = "At-Risk" if prediction[0] == 1 else "Not At-Risk"
            risk_score = model.predict_proba(processed_data)[0][1] * 100 if hasattr(model, 'predict_proba') else (100 if prediction[0] == 1 else 0)
            
            return {
                'risk_status': risk_status,
                'risk_score': round(risk_score, 2)
            }
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            return {'risk_status': 'Error', 'risk_score': 0}
            
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return {'risk_status': 'Error', 'risk_score': 0}
    
def individual_analysis():
    st.markdown("""
    <div style="font-family: 'Montserrat', sans-serif; font-weight: 700; font-size: 2.5rem; color: #1e3a8a; margin-bottom: 1.5rem;">
    Individual Student Analysis
    </div>
    <div class="metric-card">
        <p style="font-family: 'Montserrat', sans-serif; font-size: 1.1rem; color: #333;">
            On this page, instructors can assess the risk status of an individual student by entering their academic and engagement data. Provide the student's ID, average assignment score, number of missing assignments, total LMS activity (in hours), and attendance percentage. After submitting, the system will predict whether the student is at risk of academic difficulties and display a risk score along with visualizations of their performance metrics.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div style="margin-bottom: 2rem;"></div>', unsafe_allow_html=True)

    with st.form("student_form"):
        col1, col2 = st.columns(2)

        with col1:
            student_id = st.text_input("Student ID", placeholder="Enter Student ID")
            avg_score = st.slider("Average Assignment Score", 0, 100, 70,
                                    help="The student's average score across all assignments")
            missing_assignments = st.slider("Number of Missing Assignments", 0, 10, 2,
                                            help="Count of assignments not submitted")

        with col2:
            lms_activity = st.slider("Total LMS Activity (hours)", 0, 100, 30,
                                        help="Total time spent on Learning Management System")
            attendance = st.slider("Total Attendance (%)", 0, 100, 80,
                                    help="Percentage of classes attended")

        submitted = st.form_submit_button("Predict Risk")

    if submitted:
        student_data = {
            'avg_score': avg_score,
            'missing_assignments': missing_assignments,
            'lms_activity': lms_activity,
            'attendance': attendance
        }

        with st.spinner("Analyzing student data..."):
            prediction = predict_risk(student_data)
            time.sleep(1)  # Simulate processing

            st.markdown('<div style="margin-bottom: 2rem;"></div>', unsafe_allow_html=True)
            st.markdown(f"""
            <style>
                .risk-badge {{
                    padding: 0.5rem 1rem;
                    border-radius: 5px;
                    font-weight: bold;
                    color: white;
                    display: inline-block;
                    margin-top: 0.5rem;
                }}
                .at-risk {{
                    background-color: #ff6b6b; /* Red */
                }}
                .not-risk {{
                    background-color: #51cf66; /* Green */
                }}
                .metric-card {{
                    margin: 20px 0;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
            </style>
            <div style="margin: 20px 0; padding: 20px; border-radius: 10px;
                         background: #e6f0fa; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h3>Prediction Result</h3>
                <div class="risk-badge {prediction['risk_status'].lower().replace(' ', '-')}">
                    {prediction['risk_status']}
                </div>
                <p>Risk Score: {prediction['risk_score']:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div style="margin-bottom: 2rem;"></div>', unsafe_allow_html=True)

            # Performance Overview Chart
            fig1 = px.bar(
                x=["Avg Score", "Missing Assignments", "LMS Activity", "Attendance"],
                y=[avg_score, missing_assignments, lms_activity, attendance],
                title="Performance Overview"
            )
            st.plotly_chart(fig1, use_container_width=True)

            st.markdown('<div style="margin-bottom: 2rem;"></div>', unsafe_allow_html=True)

            # Engagement Level Chart
            total_assignments = 10
            completed_assignments = total_assignments - missing_assignments
            fig2 = px.pie(
                names=["Completed Assignments", "Missing Assignments"],
                values=[completed_assignments, missing_assignments],
                title="Assignment Completion"
            )
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown('<div style="margin-bottom: 2rem;"></div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-card">
                <h3>Analysis</h3>
                <p>Based on the provided data, the model predicts this student is <strong>{prediction['risk_status']}</strong>
                with a risk score of <strong>{prediction['risk_score']:.2f}%</strong>
                due to {f"a lower average score ({avg_score}%) and a higher number of missing assignments ({missing_assignments})" if prediction['risk_status'] == 'At-Risk' else "relatively consistent performance across the monitored metrics"}.</p>
            </div>
            """, unsafe_allow_html=True)

def batch_analysis():
    st.markdown("""
    <div style="font-family: 'Montserrat', sans-serif; font-weight: 700; font-size: 2.5rem; color: #1e3a8a; margin-bottom: 1.5rem;">
    Batch Student Analysis
    </div>
    <div class="metric-card">
        <p style="font-family: 'Montserrat', sans-serif; font-size: 1.1rem; color: #333;">
            On this page, instructors can upload a CSV file containing data for multiple students to perform batch predictions. The CSV must include a 'Student_id' column and all required features (e.g., Average_assignment_score, Num_of_missing_assingnment, Total_LMS_Activity, rate_Of_Globale_Attandence, and others). The system will predict the risk status for each student and display the results in two groups: At-Risk and Not At-Risk, along with visualizations of risk distribution and feature importance.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div style="margin-bottom: 2rem;"></div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Import Student Data (.csv file)", type="csv")
    
    if uploaded_file:
        try:
            # Load the data
            df = pd.read_csv(uploaded_file)
            st.success(f"Successfully uploaded {len(df)} student records!")
            st.markdown('<div style="margin-bottom: 2rem;"></div>', unsafe_allow_html=True)
            
            # Check if Student_id is present
            if 'Student_id' not in df.columns:
                st.error("The uploaded file must contain a 'Student_id' column.")
                return
            
            # Load the model
            with st.spinner("Loading model..."):
                try:
                    with open('lr.pkl', 'rb') as f:
                        model = joblib.load(f)
                except FileNotFoundError:
                    st.error("Model file (lr.pkl) not found.")
                    return
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
                    return
            
            # Define features
            categorical_features = [
                'Gender', 'Majors', 'Academic_level', 'Prefere_Mode', 'Course_School',
                'Participation_online_absence_Flag'
            ]
            numerical_features = [
                'Student_id','Course_id', 'Age', 'total_sessions_num', 'Session_duration', 'Total_assignments',
                'total_canvas_ressources', 'num_attandence_inPerson', 'num_total_absence',
                'num_attendance_online', 'rate_Of_Globale_Attandence', 'Camera_activation_num',
                'Unmuted_mic_num', 'Q&A_teams_ participation', 'InMeeting_Duration',
                'Num_of_missing_assingnment', 'Num_of_submitted_assignment',
                'Average_assignment_score', 'num_of viewed_ ressources', 'Total_LMS_Activity'
            ]
            all_features = numerical_features + categorical_features
            
            # Verify all required columns
            missing_cols = [col for col in all_features if col not in df.columns]
            if missing_cols:
                st.error(f"Missing required columns in data: {', '.join(missing_cols)}")
                st.write("CSV columns found:", df.columns.tolist())
                return
            
            # Preprocess and predict
            with st.spinner("Processing batch predictions..."):
                try:
                    # Separate student IDs and features
                    student_ids = df['Student_id']
                    features = df[all_features]
                    
                    # Load preprocessor.pkl
                    try:
                        with open('preprocessor.pkl', 'rb') as f:
                            preprocessor = joblib.load(f)
                        processed_features = preprocessor.transform(features)
                    except FileNotFoundError:
                        st.warning("preprocessor.pkl not found. Using in-code preprocessing.")
                        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                        numerical_transformer = StandardScaler()
                        preprocessor = ColumnTransformer(
                            transformers=[
                                ('num', numerical_transformer, numerical_features),
                                ('cat', categorical_transformer, categorical_features)
                            ])
                        processed_features = preprocessor.fit_transform(features)
                    
                    # Make predictions
                    predictions = model.predict(processed_features)
                    prediction_labels = ['At-Risk' if pred == 1 else 'Not At-Risk' for pred in predictions]
                    
                    # Create results dataframe
                    results_df = pd.DataFrame({
                        'Student_id': student_ids,
                        'Predicted Risk': prediction_labels
                    })
                    
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    return
            
            # Display results
            st.markdown('<div style="margin-bottom: 2rem;"></div>', unsafe_allow_html=True)
            st.subheader("Batch Prediction Results")
            
            # Display student IDs in two columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="tooltip"><strong>At-Risk Students</strong><span class="tooltiptext">Students predicted to be at risk of academic difficulties</span></div>', unsafe_allow_html=True)
                at_risk_ids = results_df[results_df['Predicted Risk'] == 'At-Risk']['Student_id']
                if len(at_risk_ids) > 0:
                    st.dataframe(at_risk_ids.reset_index(drop=True))
                else:
                    st.info("No at-risk students identified")
                st.markdown('<div style="margin-bottom: 2rem;"></div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="tooltip"><strong>Not At-Risk Students</strong><span class="tooltiptext">Students predicted to be performing adequately</span></div>', unsafe_allow_html=True)
                not_at_risk_ids = results_df[results_df['Predicted Risk'] == 'Not At-Risk']['Student_id']
                if len(not_at_risk_ids) > 0:
                    st.dataframe(not_at_risk_ids.reset_index(drop=True))
                else:
                    st.info("All students are at-risk")
                st.markdown('<div style="margin-bottom: 2rem;"></div>', unsafe_allow_html=True)
            
            # Risk Distribution Chart
            risk_counts = results_df['Predicted Risk'].value_counts()
            fig1 = px.pie(
                names=risk_counts.index,
                values=risk_counts.values,
                title="Risk Distribution"
            )
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown('<div style="margin-bottom: 2rem;"></div>', unsafe_allow_html=True)
            
            # Feature Importance
            if hasattr(model, 'coef_'):
                try:
                    feature_names = preprocessor.get_feature_names_out()
                    importance = np.abs(model.coef_[0]) / np.sum(np.abs(model.coef_[0]))
                    fig2 = px.bar(
                        x=feature_names,
                        y=importance,
                        title="Feature Importance (Normalized Coefficients)",
                        labels={'x': 'Feature', 'y': 'Importance'}
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                    st.markdown('<div style="margin-bottom: 2rem;"></div>', unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"Could not display feature importance: {str(e)}")
                    st.markdown('<div style="margin-bottom: 2rem;"></div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.markdown('<div style="margin-bottom: 2rem;"></div>', unsafe_allow_html=True)

# ===== MAIN APP =====
def main():
    page = navigation()
    
    if page == "About":
        about_page()
    elif page == "Individual Analysis":
        individual_analysis()
    elif page == "Batch Analysis":
        batch_analysis()

if __name__ == "__main__":
    main()