"""
BTP/MTP Allocation Portal - Streamlit Cloud Interface
Optimized for cloud deployment with robust error handling and user experience.
"""

import streamlit as st
import pandas as pd
import logging
import sys
import traceback
from datetime import datetime
from io import StringIO
import re
from btp_mtp_allocation_engine import AllocationEngine

# -----------------------------
# Cloud-Optimized Configuration
# -----------------------------
st.set_page_config(
    page_title="BTP/MTP Allocation Portal",
    page_icon=":mortar_board:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Logging for cloud deployment
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Security constants
MAX_FILE_SIZE_MB = 10  # Maximum file size in MB
MAX_ROWS = 10000       # Maximum number of rows to process
ALLOWED_EXTENSIONS = ['csv']

# Custom CSS for better cloud UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)

def validate_file_security(uploaded_file) -> tuple[bool, str]:
    """File security validation."""
    try:
        # File size check
        if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            return False, f"File size ({uploaded_file.size / (1024*1024):.1f} MB) exceeds maximum allowed size ({MAX_FILE_SIZE_MB} MB)"
        
        # File extension validation
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            return False, f"File type '.{file_extension}' not allowed. Only {', '.join(ALLOWED_EXTENSIONS)} files are permitted."
        
        # Basic filename validation (prevent directory traversal)
        if '..' in uploaded_file.name or '/' in uploaded_file.name or '\\' in uploaded_file.name:
            return False, "Invalid filename detected. Filename contains prohibited characters."
        
        return True, "File validation passed"
        
    except Exception as e:
        logger.error(f"File security validation error: {e}")
        return False, f"File validation error: {str(e)}"

def validate_csv_content(data: pd.DataFrame) -> tuple[bool, str]:
    """Validate CSV content for security and data quality."""
    try:
        # Row count check
        if len(data) > MAX_ROWS:
            return False, f"Dataset too large ({len(data)} rows). Maximum allowed: {MAX_ROWS} rows."
        
        # Empty file check
        if len(data) == 0:
            return False, "Uploaded file is empty."
        
        # Column count check (reasonable limit)
        if len(data.columns) > 50:
            return False, f"Too many columns ({len(data.columns)}). Maximum allowed: 50 columns."
        
        # Check for suspicious content patterns
        for col in data.columns:
            # Check column names for suspicious patterns
            if re.search(r'[<>{}]|script|javascript|eval', str(col), re.IGNORECASE):
                return False, f"Suspicious content detected in column name: {col}"
        
        # Basic data type validation
        if not all(data.dtypes.apply(lambda x: x.name in ['object', 'int64', 'float64', 'bool'])):
            return False, "Unsupported data types detected in the dataset."
        
        return True, "Content validation passed"
        
    except Exception as e:
        logger.error(f"CSV content validation error: {e}")
        return False, f"Content validation error: {str(e)}"

def main():
    """Entry point for the Streamlit web application."""
    # Header with description
    st.markdown('<h1 class="main-header">BTP/MTP Allocation Portal</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to the Automated Project Allocation System
    
    This application automates the allocation of BTP (Bachelor Thesis Project) and MTP (Master Thesis Project) 
    assignments based on:
    - **Student CGPA** (merit-based ranking)
    - **Faculty preferences** (student choices)
    - **Round-robin algorithm** (fair distribution)
    
    ---
    """)
    
    # Sidebar information with security notice
    with st.sidebar:
        st.markdown("### Instructions")
        st.markdown("""
        1. **Upload CSV File**: Include Roll, Name, Email, CGPA columns
        2. **Faculty Columns**: Add preference columns after CGPA
        3. **Run Allocation**: Process the algorithm
        4. **Download Results**: Get allocation and statistics
        """)
        
        # Security information
        st.markdown("### Security & Limits")
        st.markdown(f"""
        - **Max file size**: {MAX_FILE_SIZE_MB} MB
        - **Max rows**: {MAX_ROWS:,} students
        - **Allowed formats**: CSV only
        - **Data privacy**: Files processed in memory only
        """)
        
        st.markdown("### Sample CSV Format")
        sample_data = {
            'Roll': ['2021001', '2021002'],
            'Name': ['John Doe', 'Jane Smith'],
            'Email': ['john@college.edu', 'jane@college.edu'],
            'CGPA': [8.5, 9.2],
            'Dr. Smith': [1, 2],
            'Dr. Johnson': [2, 1]
        }
        st.dataframe(pd.DataFrame(sample_data), use_container_width=True)
    
    handle_file_upload()

def handle_file_upload():
    """Handles file upload, validation, and triggers processing with security."""
    st.header("File Upload Section")
    
    # File uploader with security information
    st.markdown("### Upload Your Allocation Data")
    st.markdown(f"""
    <div class="info-box">
        <strong>Security Notice:</strong> Files are processed securely in memory and never stored permanently. 
        Maximum file size: {MAX_FILE_SIZE_MB} MB, Maximum rows: {MAX_ROWS:,}
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help=f"Upload a properly formatted CSV file (max {MAX_FILE_SIZE_MB} MB, {MAX_ROWS:,} rows)",
        accept_multiple_files=False  # Security: prevent multiple file uploads
    )

    if not uploaded_file:
        # Waiting state with example
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("""
            **Required Columns:**
            - `Roll`: Student roll number
            - `Name`: Student full name  
            - `Email`: Student email address
            - `CGPA`: Current CGPA (numeric)
            - `Faculty_Name`: Preference rank (1, 2, 3...)
            """)
        with col2:
            st.markdown("""
            **Example Data:**
            ```
            Roll,Name,Email,CGPA,Dr.Smith,Dr.Johnson
            2021001,John Doe,john@edu,8.5,1,2
            2021002,Jane Smith,jane@edu,9.2,2,1
            ```
            """)
        return

    try:
        # Security validation
        file_valid, file_msg = validate_file_security(uploaded_file)
        if not file_valid:
            st.markdown(f"""
            <div class="error-box">
                <strong>Security Validation Failed:</strong><br>
                {file_msg}
            </div>
            """, unsafe_allow_html=True)
            return

        # File processing with progress and error handling
        with st.spinner("Processing uploaded file..."):
            # Use robust CSV reading with error handling
            data = pd.read_csv(
                uploaded_file, 
                on_bad_lines='skip',  # Skip malformed lines
                skipinitialspace=True,
                encoding='utf-8',
                low_memory=False,
                nrows=MAX_ROWS + 1  # Read one extra to check if file exceeds limit
            )
        
        # Content validation
        content_valid, content_msg = validate_csv_content(data)
        if not content_valid:
            st.markdown(f"""
            <div class="error-box">
                <strong>Content Validation Failed:</strong><br>
                {content_msg}
            </div>
            """, unsafe_allow_html=True)
            return

        # Trim to max rows if needed
        if len(data) > MAX_ROWS:
            data = data.head(MAX_ROWS)
            st.markdown(f"""
            <div class="warning-box">
                <strong>Dataset Trimmed:</strong> File contained more than {MAX_ROWS:,} rows. 
                Only the first {MAX_ROWS:,} rows will be processed.
            </div>
            """, unsafe_allow_html=True)
        
        # Success message with file info
        st.markdown(f"""
        <div class="success-box">
            <strong>File uploaded and validated successfully!</strong><br>
            Rows: {len(data):,} | Columns: {len(data.columns)} | Size: {uploaded_file.size:,} bytes
        </div>
        """, unsafe_allow_html=True)

        # Data preview with data quality info
        with st.expander("Preview Uploaded Data", expanded=True):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.dataframe(data.head(10), use_container_width=True)
            with col2:
                st.markdown("**Data Quality Report:**")
                st.write(f"• Total rows: {len(data):,}")
                st.write(f"• Total columns: {len(data.columns)}")
                st.write(f"• Missing values: {data.isnull().sum().sum():,}")
                st.write(f"• Duplicate rows: {data.duplicated().sum():,}")
                st.markdown("**Column names:**")
                st.write(", ".join(data.columns.tolist()))

        # Validation with security checks
        required = {"Roll", "Name", "Email", "CGPA"}
        missing = required - set(data.columns)
        
        if missing:
            st.markdown(f"""
            <div class="error-box">
                <strong>Missing Required Columns:</strong> {', '.join(missing)}<br>
                Please ensure your CSV includes all required columns.
            </div>
            """, unsafe_allow_html=True)
            return

        # CGPA validation
        try:
            original_cgpa = data['CGPA'].copy()
            data['CGPA'] = pd.to_numeric(data['CGPA'], errors='coerce')
            
            invalid_cgpa_count = data['CGPA'].isnull().sum()
            if invalid_cgpa_count > 0:
                st.markdown(f"""
                <div class="warning-box">
                    <strong>CGPA Validation Warning:</strong> {invalid_cgpa_count} rows have invalid CGPA values. 
                    These rows will be excluded from processing.
                </div>
                """, unsafe_allow_html=True)
                # Remove rows with invalid CGPA
                data = data.dropna(subset=['CGPA'])
                
            # CGPA range validation
            invalid_range = data[(data['CGPA'] < 0) | (data['CGPA'] > 10)]
            if not invalid_range.empty:
                st.warning(f"Warning: {len(invalid_range)} students have CGPA values outside 0-10 range.")
                
        except Exception as e:
            st.error(f"Error validating CGPA column: {str(e)}")
            return

        # Validate faculty columns
        cgpa_index = data.columns.get_loc("CGPA")
        faculty_columns = list(data.columns[cgpa_index + 1:])
        
        if len(faculty_columns) == 0:
            st.error("No faculty preference columns found after CGPA column!")
            return
        
        if len(faculty_columns) > 20:  # Reasonable limit
            st.error(f"Too many faculty columns ({len(faculty_columns)}). Maximum allowed: 20 faculties.")
            return
            
        st.success(f"Found {len(faculty_columns)} faculty preference columns: {', '.join(faculty_columns)}")

        # Run button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Run Allocation Algorithm", type="primary", use_container_width=True):
                execute_allocation(data)

    except pd.errors.EmptyDataError:
        st.error("The uploaded file is empty or contains no valid data.")
    except pd.errors.ParserError as e:
        st.error(f"CSV parsing error: {str(e)}. Please check your file format.")
    except UnicodeDecodeError:
        st.error("File encoding error. Please ensure your CSV file is UTF-8 encoded.")
    except MemoryError:
        st.error("File too large to process. Please reduce file size and try again.")
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"File processing error: {e}\n{error_details}")
        
        st.markdown(f"""
        <div class="error-box">
            <strong>Failed to process uploaded file:</strong><br>
            {str(e)}<br><br>
            <details>
            <summary>Technical Details</summary>
            <pre>{error_details}</pre>
            </details>
        </div>
        """, unsafe_allow_html=True)

def execute_allocation(df: pd.DataFrame):
    """Allocation execution with detailed progress and error handling."""
    try:
        st.markdown("## Processing Allocation")
        
        # Create progress containers
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            step_info = st.empty()

        # Initialize allocation engine
        status_text.text("Initializing allocation engine...")
        step_info.info("Setting up the allocation system with your data")
        
        engine = AllocationEngine()
        engine.students_data = df.copy()

        # Identify faculty columns
        cgpa_index = df.columns.get_loc("CGPA")
        engine.faculties = list(df.columns[cgpa_index + 1:])
        
        progress_bar.progress(20)
        status_text.text(f"Detected {len(engine.faculties)} faculty options...")
        step_info.info(f"Faculty options: {', '.join(engine.faculties)}")

        # Execute allocation
        status_text.text("Running student-faculty allocation algorithm...")
        step_info.info("Processing CGPA-based ranking and preference matching")
        progress_bar.progress(50)
        
        allocation_result = engine.allocate_students()

        # Generate statistics
        status_text.text("Computing faculty preference statistics...")
        step_info.info("Analyzing preference patterns and satisfaction rates")
        progress_bar.progress(80)
        
        faculty_stats = engine.generate_preference_stats()
        summary = engine.get_allocation_summary()

        # Complete
        progress_bar.progress(100)
        status_text.text("Allocation process completed successfully!")
        step_info.success(f"Processed {len(allocation_result)} students across {len(engine.faculties)} faculties")

        st.balloons()  # Celebration animation
        
        # Display results
        display_summary(allocation_result, faculty_stats, summary)
        enable_downloads(allocation_result, faculty_stats)

    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Allocation process failed: {e}\n{error_details}")
        
        st.markdown(f"""
        <div class="error-box">
            <strong>Allocation Process Failed:</strong><br>
            {str(e)}<br><br>
            <details>
            <summary>Technical Details</summary>
            <pre>{error_details}</pre>
            </details>
        </div>
        """, unsafe_allow_html=True)

def display_summary(allocation: pd.DataFrame, stats: pd.DataFrame, summary: dict):
    """Results display with better visualization."""
    st.markdown("## Allocation Results Dashboard")

    # Key metrics with styling
    st.markdown("### Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Students", 
            summary.get("total_students", 0),
            help="Total number of students processed"
        )
    with col2:
        pref_1 = summary.get("preference_satisfaction", {}).get("pref_1", 0)
        total = summary.get("total_students", 1)
        st.metric(
            "First Choice", 
            pref_1,
            delta=f"{(pref_1/total)*100:.1f}%",
            help="Students who got their first preference"
        )
    with col3:
        pref_2 = summary.get("preference_satisfaction", {}).get("pref_2", 0)
        st.metric(
            "Second Choice", 
            pref_2,
            delta=f"{(pref_2/total)*100:.1f}%",
            help="Students who got their second preference"
        )
    with col4:
        other = summary.get("preference_satisfaction", {}).get("other", 0)
        st.metric(
            "Other/Unallocated", 
            other,
            delta=f"{(other/total)*100:.1f}%",
            help="Students with other preferences or unallocated"
        )

    # Faculty distribution
    st.markdown("### Faculty Allocation Distribution")
    fac_dist = summary.get("faculty_distribution", {})
    if fac_dist:
        col1, col2 = st.columns([2, 1])
        with col1:
            chart_df = pd.DataFrame(list(fac_dist.items()), columns=["Faculty", "Students"])
            st.bar_chart(chart_df.set_index("Faculty"), use_container_width=True)
        with col2:
            st.markdown("**Distribution Details:**")
            for faculty, count in fac_dist.items():
                percentage = (count / sum(fac_dist.values())) * 100
                st.write(f"• **{faculty}**: {count} ({percentage:.1f}%)")

    # Sample allocation results
    st.markdown("### Sample Allocation Results")
    sample_cols = ["Roll", "Name", "CGPA", "Allocated", "Preference_Rank"]
    
    # Table with filtering option
    col1, col2 = st.columns([3, 1])
    with col2:
        show_all = st.checkbox("Show all results", help="Uncheck to show only first 20 rows")
        filter_faculty = st.selectbox("Filter by faculty:", ["All"] + list(fac_dist.keys()))
    
    with col1:
        display_df = allocation[sample_cols].copy()
        if filter_faculty != "All":
            display_df = display_df[display_df["Allocated"] == filter_faculty]
        
        if not show_all:
            display_df = display_df.head(20)
            
        st.dataframe(display_df, use_container_width=True)

    # Faculty preference statistics
    st.markdown("### Faculty Preference Statistics")
    st.dataframe(stats, use_container_width=True)

def enable_downloads(allocation: pd.DataFrame, stats: pd.DataFrame):
    """Download section with multiple formats and better UX."""
    st.markdown("## Download Results")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare download data
    alloc_csv = allocation.to_csv(index=False)
    stats_csv = stats.to_csv(index=False)
    
    # Create summary report
    summary_report = f"""
BTP/MTP Allocation Report
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

SUMMARY STATISTICS:
- Total Students Processed: {len(allocation)}
- Total Faculty Options: {len(stats)}
- First Preference Satisfied: {(allocation['Preference_Rank'] == 1).sum()}
- Second Preference Satisfied: {(allocation['Preference_Rank'] == 2).sum()}
- Other/Unallocated: {(allocation['Preference_Rank'] > 2).sum() + (allocation['Preference_Rank'] == 'Unallocated').sum()}

FACULTY DISTRIBUTION:
{allocation['Allocated'].value_counts().to_string()}
"""

    st.divider()
    
    # Download buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Allocation Results**")
        st.download_button(
            label="Download Allocation CSV",
            data=alloc_csv,
            file_name=f"btp_mtp_allocation_{timestamp}.csv",
            mime="text/csv",
            help="Complete allocation results with all student assignments",
            use_container_width=True
        )
    
    with col2:
        st.markdown("**Faculty Statistics**")
        st.download_button(
            label="Download Statistics CSV",
            data=stats_csv,
            file_name=f"faculty_stats_{timestamp}.csv",
            mime="text/csv",
            help="Faculty preference analysis and statistics",
            use_container_width=True
        )
    
    with col3:
        st.markdown("**Summary Report**")
        st.download_button(
            label="Download Summary Report",
            data=summary_report,
            file_name=f"allocation_summary_{timestamp}.txt",
            mime="text/plain",
            help="Executive summary of allocation results",
            use_container_width=True
        )

    # Success message
    st.success("All files are ready for download! Click the buttons above to save your results.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        logger.error(f"Application failed: {e}")
        st.info("Please refresh the page and try again.")
