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
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import re
import io
import plotly.io as pio

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
.email-container {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #dee2e6;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Email configuration
SMTP_SERVER = "smtp.gmail.com"  
SMTP_PORT = 587
SENDER_EMAIL = "***********@gmail.com"  # Change this to your email
SENDER_PASSWORD = "****************"  # Use app password for Gmail
SENDER_NAME = "Bike Detection App"

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def create_charts_for_pdf(brand_counts):
    """Create and save charts as images for PDF inclusion"""
    if not brand_counts:
        return None, None
    
    # Create DataFrame
    brand_df = pd.DataFrame(
        list(brand_counts.items()), 
        columns=['Brand', 'Count']
    )
    
    # Define colors for each brand
    color_map = {
        'Didi': "#38E8FF",
        'HelloRide': "#2A3EEC", 
        'Meituan': '#FFE66D'
    }
    
    # Create colors list in the same order as the DataFrame
    colors_list = [color_map.get(brand, '#808080') for brand in brand_df['Brand']]
    
    # Create pie chart with explicit color assignment
    fig_pie = px.pie(
        brand_df, 
        values='Count', 
        names='Brand',
        title="Bike Brand Distribution",
        color='Brand',
        color_discrete_map=color_map
    )
    
    # Update pie chart layout for better PDF rendering
    fig_pie.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        marker=dict(
            colors=colors_list,
            line=dict(color='#FFFFFF', width=2)
        )
    )
    
    fig_pie.update_layout(
        title_font_size=16,
        font=dict(size=12),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        ),
        margin=dict(l=20, r=80, t=50, b=20)
    )
    
    # Create bar chart
    fig_bar = px.bar(
        brand_df, 
        x='Brand', 
        y='Count',
        title="Bike Count by Brand",
        color='Brand',
        color_discrete_map=color_map
    )
    
    # Update bar chart layout
    fig_bar.update_layout(
        title_font_size=16,
        font=dict(size=12),
        showlegend=False,
        margin=dict(l=40, r=20, t=50, b=40)
    )
    
    # Save charts as images with higher DPI for better quality
    pie_img_path = "temp_pie_chart.png"
    bar_img_path = "temp_bar_chart.png"
    
    try:
        # Save with higher resolution and explicit engine
        fig_pie.write_image(
            pie_img_path, 
            width=500, 
            height=400, 
            scale=2,  # Higher resolution
            engine="kaleido"
        )
        fig_bar.write_image(
            bar_img_path, 
            width=500, 
            height=400, 
            scale=2,  # Higher resolution
            engine="kaleido"
        )
    except Exception as e:
        print(f"Error saving charts with kaleido: {e}")
        # Fallback to default engine
        try:
            fig_pie.write_image(pie_img_path, width=500, height=400)
            fig_bar.write_image(bar_img_path, width=500, height=400)
        except Exception as e2:
            print(f"Error saving charts with default engine: {e2}")
            return None, None
    
    return pie_img_path, bar_img_path

def send_email_report(recipient_email, report_file_path, video_name, detection_results):
    """Send email with PDF report attachment"""
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = f"{SENDER_NAME} <{SENDER_EMAIL}>"
        msg['To'] = recipient_email
        msg['Subject'] = f"Bike Detection Report - {video_name}"

        msg['Reply-To'] = f"{SENDER_NAME} <{SENDER_EMAIL}>"
        
        # Email body
        body = f"""
        Dear User,

        Your bike detection analysis has been completed successfully!

        Here's a summary of the results:
        • Video: {video_name}
        • Unique Bikes Detected: {detection_results.get('unique_bikes', 0)}
        • Total Detections: {detection_results.get('total_detections', 0)}
        • Processing Time: {detection_results.get('duration', 'N/A')}

        Brand Breakdown:
        """
        
        # Add brand counts to email body
        brand_counts = detection_results.get('counts', {})
        for brand, count in brand_counts.items():
            body += f"• {brand}: {count} bikes\n        "
        
        body += """

        Please find the detailed PDF report attached to this email.

        Thank you for using our Bike Detection System!

        Best regards,
        Bike Detection App Team
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach PDF report
        with open(report_file_path, "rb") as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
        
        encoders.encode_base64(part)
        part.add_header(
            'Content-Disposition',
            f'attachment; filename= {os.path.basename(report_file_path)}'
        )
        
        msg.attach(part)
        
        # Send email
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        text = msg.as_string()
        server.sendmail(SENDER_EMAIL, recipient_email, text)
        server.quit()
        
        return True, "Email sent successfully!"
        
    except Exception as e:
        return False, f"Failed to send email: {str(e)}"

# Main title
st.markdown('<h1 class="main-header">🚲 Shared Bike Detection System</h1>', unsafe_allow_html=True)

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
# enable_roi = st.sidebar.checkbox(
#     "Enable ROI Selection", 
#     value=False,
#     help="Select region of interest for detection"
# )

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

# Email input section
# st.markdown('<div class="email-container">', unsafe_allow_html=True)
with st.container():
    st.subheader("📧 Email Report Delivery")
    st.write("Enter your email address to receive the detection report automatically:")

    email_input = st.text_input(
        "Email Address",
        placeholder="your.email@example.com",
        help="We'll send you a detailed PDF report once processing is complete"
    )

    if email_input:
        if validate_email(email_input):
            st.success("✅ Valid email address")
        else:
            st.error("❌ Please enter a valid email address")

# st.markdown('</div>', unsafe_allow_html=True)

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
    # roi_coordinates = None
    # if enable_roi:
    #     st.subheader("🎯 Region of Interest Selection")
    #     st.info("ROI selection will be applied during processing. Full implementation requires additional UI components.")
        
    #     # For now, provide manual input
    #     col1, col2, col3, col4 = st.columns(4)
    #     with col1:
    #         roi_x1 = st.number_input("X1", min_value=0, value=0)
    #     with col2:
    #         roi_y1 = st.number_input("Y1", min_value=0, value=0)
    #     with col3:
    #         roi_x2 = st.number_input("X2", min_value=0, value=640)
    #     with col4:
    #         roi_y2 = st.number_input("Y2", min_value=0, value=480)
        
    #     roi_coordinates = (roi_x1, roi_y1, roi_x2, roi_y2)
    
    # Initialize processing state
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False

    # Processing button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Check if email is valid before enabling processing
        email_valid = email_input and validate_email(email_input)

        # Determine button state and text
        if st.session_state.is_processing:
            button_text = "🔄 Processing Video..."
            button_disabled = True
            button_type = "secondary"
        else:
            button_text = "🚀 Start Detection"
            button_disabled = not email_valid
            button_type = "primary"

        if not email_input and not st.session_state.is_processing:
            st.warning("⚠️ Please enter your email address to receive the report")
        elif not email_valid and not st.session_state.is_processing:
            st.error("❌ Please enter a valid email address")
        
        process_button = st.button(
            button_text,
            type=button_type,
            use_container_width=True,
            disabled=button_disabled
    )
    
    if process_button and email_valid and not st.session_state.is_processing:
        # Set processing state to True
        st.session_state.is_processing = True
        
        # Force UI update
        st.rerun()

    # Check if we're in processing state
    if st.session_state.is_processing:
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
                progress = 0.2 + (current_frame / total_frames) * 0.6  # 20% to 80%
                progress_bar.progress(progress)
                status_text.text(f"🎬 Video Processing Progress: {progress*100:.1f}%")
                
                # Update real-time metrics if available
                if hasattr(progress_callback, 'detector') and progress_callback.detector:
                    stats = progress_callback.detector.get_statistics()
                    metric_placeholders[0].metric("Processed Frames", f"{current_frame}/{total_frames}")
                    metric_placeholders[1].metric("Unique Bikes", stats.get('unique_bikes', 0))
                    metric_placeholders[2].metric("Total Detections", stats.get('total_detections', 0))
            
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
            progress_bar.progress(0.8)
            status_text.text("📋 Generating report...")
            
            # Generate PDF report
            report_filename = f"bike_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

            # Create charts for PDF inclusion
            brand_counts = results.get('counts', {})
            pie_chart_path, bar_chart_path = None, None
            
            if brand_counts:
                try:
                    pie_chart_path, bar_chart_path = create_charts_for_pdf(brand_counts)
                except Exception as e:
                    st.warning(f"Could not create charts for PDF: {str(e)}")

            # Create PDF document
            doc = SimpleDocTemplate(report_filename, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.blue,
                alignment=1  # Center alignment
            )
            story.append(Paragraph("🚲 Bike Detection Report", title_style))
            story.append(Spacer(1, 20))

            # Processing Details
            story.append(Paragraph("Processing Details", styles['Heading2']))
            processing_data = [
                ["Video:", uploaded_file.name],
                ["Processed:", datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                ["Duration:", str(results.get('duration', 'N/A'))],
                ["Confidence Threshold:", str(confidence_threshold)],
                ["Object Tracking:", 'Enabled' if use_tracking else 'Disabled']
            ]
            processing_table = Table(processing_data, colWidths=[150, 300])
            processing_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(processing_table)
            story.append(Spacer(1, 20))

            # Detection Results
            story.append(Paragraph("Detection Results", styles['Heading2']))
            results_data = [
                ["Unique Bikes Tracked:", str(results.get('unique_bikes', 0))],
                ["Total Detections:", str(results.get('total_detections', 0))],
                ["Processing Time:", f"{processing_time:.2f} seconds"]
            ]
            results_table = Table(results_data, colWidths=[150, 300])
            results_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(results_table)
            story.append(Spacer(1, 20))

            # Brand Breakdown
            if brand_counts:
                story.append(Paragraph("Brand Breakdown", styles['Heading2']))
                brand_data = [["Brand", "Count"]]
                for brand, count in brand_counts.items():
                    brand_data.append([brand, str(count)])
                
                brand_table = Table(brand_data, colWidths=[150, 100])
                brand_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey)
                ]))
                story.append(brand_table)
                story.append(Spacer(1, 20))

                # Add charts to PDF
                story.append(Paragraph("Brand Distribution Charts", styles['Heading2']))
                
                # Add pie chart
                if pie_chart_path and os.path.exists(pie_chart_path):
                    try:
                        story.append(Paragraph("Pie Chart - Brand Distribution", styles['Heading3']))
                        pie_image = ReportLabImage(pie_chart_path, width=400, height=300)
                        story.append(pie_image)
                        story.append(Spacer(1, 10))
                    except Exception as e:
                        story.append(Paragraph(f"Could not include pie chart: {str(e)}", styles['Normal']))
                
                # Add bar chart
                if bar_chart_path and os.path.exists(bar_chart_path):
                    try:
                        story.append(Paragraph("Bar Chart - Bike Count by Brand", styles['Heading3']))
                        bar_image = ReportLabImage(bar_chart_path, width=400, height=300)
                        story.append(bar_image)
                        story.append(Spacer(1, 20))
                    except Exception as e:
                        story.append(Paragraph(f"Could not include bar chart: {str(e)}", styles['Normal']))

            # Technical Details
            story.append(Paragraph("Technical Details", styles['Heading2']))
            technical_data = [
                ["Video FPS:", str(results.get('fps', 'N/A'))],
                ["Total Frames:", str(results.get('total_frames', 'N/A'))],
                ["Tracking Status:", str(results.get('tracking_enabled', 'N/A'))]
            ]
            technical_table = Table(technical_data, colWidths=[150, 300])
            technical_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(technical_table)
            story.append(Spacer(1, 20))

            # Alerts
            story.append(Paragraph("Alerts", styles['Heading2']))
            total_bikes = results.get('unique_bikes', 0)
            alert_status = '⚠️ EXCEEDED' if total_bikes > alert_threshold else '✅ Normal'
            alerts_data = [
                ["Threshold:", str(alert_threshold)],
                ["Status:", alert_status]
            ]
            alerts_table = Table(alerts_data, colWidths=[150, 300])
            alerts_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(alerts_table)

            # Build PDF
            doc.build(story)
            
            # Clean up temporary chart files
            if pie_chart_path and os.path.exists(pie_chart_path):
                os.remove(pie_chart_path)
            if bar_chart_path and os.path.exists(bar_chart_path):
                os.remove(bar_chart_path)
            
            progress_bar.progress(0.9)
            status_text.text("📧 Sending email report...")
            
            # Send email
            email_success, email_message = send_email_report(
                email_input, 
                report_filename, 
                uploaded_file.name, 
                results
            )
            
            progress_bar.progress(1.0)
            
            if email_success:
                status_text.text("✅ Processing completed and report sent successfully!")
                st.success(f"🎉 Detection completed and report sent to {email_input}!")
                st.balloons()
            else:
                status_text.text("⚠️ Processing completed but email failed to send")
                st.warning(f"Detection completed but failed to send email: {email_message}")
            
            # Reset processing state
            st.session_state.is_processing = False
            
            # Display results
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
                            'Didi': "#38E8FF",
                            'HelloRide': "#2A3EEC", 
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
                            'Didi': '#38E8FF',
                            'HelloRide': "#2A3EEC", 
                            'Meituan': '#FFE66D'
                        }
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
            
            # Alert check
            total_bikes = results.get('unique_bikes', 0)
            if total_bikes > alert_threshold:
                st.warning(f"⚠️ Alert: Bike count ({total_bikes}) exceeds threshold ({alert_threshold})!")
            
            # Offer manual download as backup
            st.subheader("📋 Manual Download")
            st.info("📧 Report has been sent to your email. You can also download it manually:")
            
            with open(report_filename, "rb") as f:
                st.download_button(
                    "📥 Download PDF Report",
                    f,
                    file_name=report_filename,
                    mime="application/pdf",
                    type="secondary"
                )
            
        except Exception as e:
            st.error(f"❌ Error during processing: {str(e)}")
            st.info("Please check your video file and model path.")

            # Reset processing state on error
            st.session_state.is_processing = False

            # Add more detailed error information
            import traceback
            st.code(traceback.format_exc())

        finally:
            # Ensure processing state is reset
            if st.session_state.is_processing:
                st.session_state.is_processing = False

# Information section
with st.expander("ℹ️ About This System"):
    st.markdown("""
    ### Bike Detection Features:
    
    🎯 **Object Tracking**: Uses DeepSORT algorithm to track bikes across frames
    
    📊 **Confidence Scoring**: Displays detection confidence for each bike
    
    📈 **Real-time Analytics**: Live charts and statistics during processing
    
    🎮 **Interactive Controls**: Adjustable thresholds and processing options
    
    🚨 **Smart Alerts**: Notifications when bike counts exceed thresholds
    
    📧 **Automatic Email Reports**: Receive detailed PDF reports via email
    
    📋 **Detailed Reports**: Comprehensive analysis with download option
    
    ### Supported Bike Brands:
    - **Didi** (Orange boxes)
    - **HelloRide** (Green boxes)  
    - **Meituan** (Yellow boxes)
    
    ### How to Use:
    1. Upload your video file
    2. Enter your email address
    3. Adjust detection settings in the sidebar
    4. Click "Start Detection" to process
    5. Receive report via email automatically
    """)

# Footer
st.markdown("---")
st.markdown("Built using Streamlit, YOLO, and DeepSORT | 📧 Email reports powered by SMTP")