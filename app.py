# app.py

import streamlit as st
import cv2
import time
import os
import numpy as np
from datetime import datetime
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Import your detection module
from detect import detect_bikes_from_video, BikeDetector

st.set_page_config(
    page_title="Bike Detection App",
    page_icon="🚲",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.stProgress > div > div > div > div {
    background-color: #1f77b4;
}
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🚲 Bike Detection System</h1>', unsafe_allow_html=True)

# Sidebar for controls
st.sidebar.header("⚙️ Detection Settings")

# Confidence threshold slider
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.5, 
    step=0.05,
    help="Minimum confidence score for detections"
)

# Enable/disable tracking
use_tracking = st.sidebar.checkbox(
    "Enable Object Tracking", 
    value=True,
    help="Track bikes across frames to count unique bikes"
)

# ROI Selection
enable_roi = st.sidebar.checkbox(
    "Enable ROI Selection", 
    value=False,
    help="Select region of interest for detection"
)

# Alert thresholds
st.sidebar.subheader("📊 Alert Settings")
alert_threshold = st.sidebar.number_input(
    "Alert when bike count exceeds:", 
    min_value=1, 
    max_value=100, 
    value=10,
    help="Get notified when bike count exceeds this number"
)

# Processing options
st.sidebar.subheader("🎬 Processing Options")
process_every_nth_frame = st.sidebar.slider(
    "Process every Nth frame (for speed)", 
    min_value=1, 
    max_value=10, 
    value=1,
    help="Process every Nth frame to speed up processing"
)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Video Upload")
    uploaded_file = st.file_uploader(
        "Upload a video file", 
        type=["mp4", "mov", "avi", "mkv"],
        help="Supported formats: MP4, MOV, AVI, MKV"
    )

with col2:
    st.subheader("📊 Quick Stats")
    if 'detection_results' in st.session_state:
        results = st.session_state.detection_results
        st.metric("Unique Bikes", results.get('unique_bikes', 0))
        st.metric("Total Detections", results.get('total_detections', 0))
        st.metric("Processing Time", results.get('duration', 'N/A'))

# Video processing section
if uploaded_file:
    # Save uploaded file
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display original video
    st.subheader("📹 Original Video")
    st.video(file_path)
    
    # ROI Selection (simplified version)
    roi_coordinates = None
    if enable_roi:
        st.subheader("🎯 Region of Interest Selection")
        st.info("ROI selection will be applied during processing. Full implementation requires additional UI components.")
        
        # For now, provide manual input
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            roi_x1 = st.number_input("X1", min_value=0, value=0)
        with col2:
            roi_y1 = st.number_input("Y1", min_value=0, value=0)
        with col3:
            roi_x2 = st.number_input("X2", min_value=0, value=640)
        with col4:
            roi_y2 = st.number_input("Y2", min_value=0, value=480)
        
        roi_coordinates = (roi_x1, roi_y1, roi_x2, roi_y2)
    
    # Processing button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        process_button = st.button("🚀 Start Detection", type="primary", use_container_width=True)
    
    if process_button:
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Real-time metrics display
        metrics_container = st.container()
        with metrics_container:
            metric_cols = st.columns(4)
            metric_placeholders = [col.empty() for col in metric_cols]
        
        # Chart containers for real-time updates
        chart_container = st.container()
        
        try:
            status_text.text("🔄 Initializing detection system...")
            progress_bar.progress(0.1)
            
            # Process video with detection
            output_dir = "outputs"
            os.makedirs(output_dir, exist_ok=True)
            output_filename = f"processed_{uploaded_file.name}"
            output_path = os.path.join(output_dir, output_filename)
            
            status_text.text("🎬 Processing video...")
            progress_bar.progress(0.2)
            
            # Start processing
            start_time = time.time()
            
            # Custom progress callback function
            def progress_callback(current_frame, total_frames):
                progress = 0.2 + (current_frame / total_frames) * 0.7  # 20% to 90%
                progress_bar.progress(progress)
                status_text.text(f"🎬 Processing frame {current_frame}/{total_frames} ({progress*100:.1f}%)")
                
                # Update real-time metrics if available
                if hasattr(progress_callback, 'detector') and progress_callback.detector:
                    stats = progress_callback.detector.get_statistics()
                    metric_placeholders[0].metric("Current Detections", current_frame)
                    metric_placeholders[1].metric("Unique Bikes", stats.get('unique_bikes', 0))
                    metric_placeholders[2].metric("Total Detections", stats.get('total_detections', 0))
                    metric_placeholders[3].metric("Progress", f"{progress*100:.1f}%")
            
            results = detect_bikes_from_video(
                file_path, 
                output_path,
                confidence_threshold=confidence_threshold,
                use_tracking=use_tracking,
                progress_callback=progress_callback
            )
            
            # Store results in session state
            st.session_state.detection_results = results
            
            processing_time = time.time() - start_time
            progress_bar.progress(1.0)
            
            status_text.text(f"✅ Processing completed in {processing_time:.2f} seconds!")
            
            # Display results
            st.success("🎉 Detection completed successfully!")
            
            # Display processed video
            st.subheader("🎥 Processed Video with Detections")

            # Check if processed video exists and display it
            if os.path.exists(output_path):
                try:
                    # Get file info
                    file_size = os.path.getsize(output_path)
                    file_size_mb = file_size / (1024 * 1024)
                    
                    st.info(f"📁 Video file: `{os.path.basename(output_path)}` ({file_size_mb:.2f} MB)")
                    

                    st.video(output_path)
                    st.success("✅ Video loaded successfully!")
                                       
                    # Additional debugging info
                    with st.expander("🔍 Debug Information"):
                        st.write("**File Details:**")
                        st.write(f"- Path: `{output_path}`")
                        st.write(f"- Exists: {os.path.exists(output_path)}")
                        st.write(f"- Size: {file_size_mb:.2f} MB")
                        st.write(f"- Extension: {os.path.splitext(output_path)[1]}")
                        
                        # Try to get video properties
                        try:
                            import cv2
                            cap = cv2.VideoCapture(output_path)
                            if cap.isOpened():
                                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                fps = cap.get(cv2.CAP_PROP_FPS)
                                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                
                                st.write("**Video Properties:**")
                                st.write(f"- Resolution: {width}x{height}")
                                st.write(f"- FPS: {fps:.2f}")
                                st.write(f"- Frame count: {frame_count}")
                                st.write(f"- Duration: {frame_count/fps:.2f} seconds")
                                cap.release()
                            else:
                                st.write("❌ Could not read video properties")
                        except Exception as e:
                            st.write(f"❌ Error reading video properties: {str(e)}")
                    
                except Exception as e:
                    st.error(f"Error loading processed video: {str(e)}")
                    
                    # Fallback: provide download link
                    try:
                        with open(output_path, 'rb') as f:
                            st.download_button(
                                "📥 Download Processed Video (Fallback)",
                                f,
                                file_name=f"processed_{uploaded_file.name}",
                                mime="video/mp4"
                            )
                    except Exception as download_error:
                        st.error(f"Even download failed: {str(download_error)}")

            else:
                st.error(f"❌ Processed video not found at: `{output_path}`")
                
                # Debug: List files in outputs directory
                if os.path.exists("outputs"):
                    files = os.listdir("outputs")
                    st.info(f"📁 Files in outputs directory: {files}")
                    
                    # If there are files, try to display them
                    if files:
                        st.write("**Available processed videos:**")
                        for file in files:
                            if file.endswith(('.mp4', '.avi')):
                                file_path = os.path.join("outputs", file)
                                file_size = os.path.getsize(file_path) / (1024*1024)
                                
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.write(f"📹 {file} ({file_size:.2f} MB)")
                                with col2:
                                    with open(file_path, 'rb') as f:
                                        st.download_button(
                                            "Download",
                                            f,
                                            file_name=file,
                                            mime="video/mp4",
                                            key=f"download_{file}"
                                        )
                else:
                    st.error("❌ Outputs directory not found")
                        
            # Detailed statistics
            st.subheader("📊 Detection Statistics")
            
            # Metrics display
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Unique Bikes Tracked", 
                    results.get('unique_bikes', 0),
                    help="Number of unique bikes identified through tracking"
                )
            
            with col2:
                st.metric(
                    "Total Detections", 
                    results.get('total_detections', 0),
                    help="Total number of detection instances"
                )
            
            with col3:
                st.metric(
                    "Processing Time", 
                    f"{processing_time:.2f}s",
                    help="Time taken to process the video"
                )
            
            with col4:
                fps = results.get('fps', 0)
                if fps > 0:
                    st.metric(
                        "Processing Speed", 
                        f"{results.get('total_frames', 0) / processing_time:.1f} FPS",
                        help="Frames processed per second"
                    )
            
            # Brand distribution chart
            brand_counts = results.get('counts', {})
            if brand_counts:
                st.subheader("📈 Brand Distribution")
                
                # Create DataFrame for plotting
                brand_df = pd.DataFrame(
                    list(brand_counts.items()), 
                    columns=['Brand', 'Count']
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart
                    fig_pie = px.pie(
                        brand_df, 
                        values='Count', 
                        names='Brand',
                        title="Bike Brand Distribution",
                        color_discrete_map={
                            'Didi': '#FF6B35',
                            'HelloRide': '#4ECDC4', 
                            'Meituan': '#FFE66D'
                        }
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Bar chart
                    fig_bar = px.bar(
                        brand_df, 
                        x='Brand', 
                        y='Count',
                        title="Bike Count by Brand",
                        color='Brand',
                        color_discrete_map={
                            'Didi': '#FF6B35',
                            'HelloRide': '#4ECDC4', 
                            'Meituan': '#FFE66D'
                        }
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
            
            # Alert check
            total_bikes = results.get('unique_bikes', 0)
            if total_bikes > alert_threshold:
                st.warning(f"⚠️ Alert: Bike count ({total_bikes}) exceeds threshold ({alert_threshold})!")
            
            # Generate and offer report download
            st.subheader("📋 Download Report")
            
            # Create detailed report
            report_content = f"""
Bike Detection Report
============================

Processing Details:
- Video: {uploaded_file.name}
- Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Duration: {results.get('duration', 'N/A')}
- Confidence Threshold: {confidence_threshold}
- Object Tracking: {'Enabled' if use_tracking else 'Disabled'}

Detection Results:
- Unique Bikes Tracked: {results.get('unique_bikes', 0)}
- Total Detections: {results.get('total_detections', 0)}
- Processing Time: {processing_time:.2f} seconds

Brand Breakdown:
"""
            
            for brand, count in brand_counts.items():
                report_content += f"- {brand}: {count} bikes\n"
            
            report_content += f"""

Technical Details:
- Video FPS: {results.get('fps', 'N/A')}
- Total Frames: {results.get('total_frames', 'N/A')}
- Tracking Status: {results.get('tracking_enabled', 'N/A')}

Alerts:
- Threshold: {alert_threshold}
- Status: {'⚠️ EXCEEDED' if total_bikes > alert_threshold else '✅ Normal'}
"""
            
            # Save report
            report_filename = f"bike_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_filename, "w", encoding="utf-8") as f:
                f.write(report_content)
            
            # Download button
            with open(report_filename, "r", encoding="utf-8") as f:
                st.download_button(
                    "📥 Download Detailed Report",
                    f,
                    file_name=report_filename,
                    mime="text/plain",
                    type="primary"
                )
            
        except Exception as e:
            st.error(f"❌ Error during processing: {str(e)}")
            st.info("Please check your video file and model path.")
            # Add more detailed error information
            import traceback
            st.code(traceback.format_exc())

# Information section
with st.expander("ℹ️ About This System"):
    st.markdown("""
    ### Bike Detection Features:
    
    🎯 **Object Tracking**: Uses DeepSORT algorithm to track bikes across frames
    
    📊 **Confidence Scoring**: Displays detection confidence for each bike
    
    📈 **Real-time Analytics**: Live charts and statistics during processing
    
    🎮 **Interactive Controls**: Adjustable thresholds and processing options
    
    🚨 **Smart Alerts**: Notifications when bike counts exceed thresholds
    
    📋 **Detailed Reports**: Comprehensive analysis with download option
    
    ### Supported Bike Brands:
    - **Didi** (Orange boxes)
    - **HelloRide** (Green boxes)  
    - **Meituan** (Yellow boxes)
    
    ### How to Use:
    1. Upload your video file
    2. Adjust detection settings in the sidebar
    3. Click "Start Detection" to process
    4. View results and download report
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, YOLO, and DeepSORT")