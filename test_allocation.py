"""
Test Suite for BTP/MTP Allocation Engine - Cloud Optimized
Executes allocation tests, validates outputs, and logs detailed results.
Optimized for Streamlit Cloud deployment with in-memory operations.
"""

import logging
import pandas as pd
import sys
import io
from typing import Dict, Tuple, Optional
from btp_mtp_allocation_engine import AllocationEngine

# Configure logging for cloud deployment
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Fixed sample test data for cloud deployment (plain CSV format)
SAMPLE_TEST_DATA = """Roll,Name,Email,CGPA,Dr. Smith,Dr. Johnson,Dr. Williams
2021001,John Doe,john@college.edu,8.5,1,2,3
2021002,Jane Smith,jane@college.edu,9.2,2,1,3
2021003,Bob Wilson,bob@college.edu,7.8,1,3,2
2021004,Alice Brown,alice@college.edu,8.9,3,2,1
2021005,Charlie Davis,charlie@college.edu,7.5,2,3,1
2021006,Eva Green,eva@college.edu,9.0,1,2,3
2021007,Frank Miller,frank@college.edu,8.2,3,1,2
2021008,Grace Lee,grace@college.edu,8.7,2,3,1
2021009,Henry Taylor,henry@college.edu,7.9,1,3,2
2021010,Ivy Chen,ivy@college.edu,9.1,2,1,3"""

def create_test_data() -> pd.DataFrame:
    """Create sample test data for cloud deployment."""
    try:
        # Convert sample data to DataFrame with error handling
        test_df = pd.read_csv(io.StringIO(SAMPLE_TEST_DATA), 
                             on_bad_lines='skip',  # Skip problematic lines
                             skipinitialspace=True)  # Handle extra spaces
        logger.info(f"Created test dataset with {len(test_df)} students")
        
        # Validate the test data structure
        required_cols = ["Roll", "Name", "Email", "CGPA"]
        missing_cols = [col for col in required_cols if col not in test_df.columns]
        
        if missing_cols:
            raise ValueError(f"Test data missing required columns: {missing_cols}")
            
        # Ensure CGPA is numeric
        test_df['CGPA'] = pd.to_numeric(test_df['CGPA'], errors='coerce')
        
        if test_df['CGPA'].isna().any():
            raise ValueError("Invalid CGPA values in test data")
            
        logger.info("Test data validation passed")
        return test_df
        
    except Exception as e:
        logger.error(f"Failed to create test data: {e}")
        raise

def run_allocation_test() -> Tuple[bool, Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Dict]]:
    """Execute the allocation engine workflow and verify successful completion."""
    try:
        logger.info("Starting BTP/MTP Allocation Engine test...")
        
        # Create test engine with sample data
        engine = AllocationEngine()
        
        # Use in-memory test data instead of file loading
        test_data = create_test_data()
        engine.students_data = test_data.copy()
        
        # Set up faculties manually (since we're not loading from file)
        cgpa_index = test_data.columns.get_loc("CGPA")
        engine.faculties = list(test_data.columns[cgpa_index + 1:])
        
        logger.info(f"Initialized engine with {len(engine.faculties)} faculties: {', '.join(engine.faculties)}")
        
        # Execute allocation process
        allocation_df = engine.allocate_students()
        prefs_df = engine.generate_preference_stats()
        summary = engine.get_allocation_summary()
        
        logger.info("Allocation process completed successfully.")
        logger.info(f"Summary Statistics: {summary}")
        
        return True, allocation_df, prefs_df, summary
        
    except Exception as e:
        logger.error(f"Allocation test encountered an error: {e}")
        import traceback
        logger.error(f"Error details: {traceback.format_exc()}")
        return False, None, None, None

def verify_allocation_results(allocation_df: pd.DataFrame, prefs_df: pd.DataFrame, summary: Dict) -> bool:
    """Validate allocation results for consistency and completeness."""
    try:
        logger.info("=== VALIDATION REPORT ===")
        logger.info(f"Total Students Processed: {len(allocation_df)}")
        logger.info(f"Distinct Faculties Allocated: {allocation_df['Allocated'].nunique()}")
        
        # Validate required columns exist
        required_allocation_cols = ["Roll", "Name", "CGPA", "Allocated", "Preference_Rank"]
        missing_cols = [col for col in required_allocation_cols if col not in allocation_df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns in allocation results: {missing_cols}")
            return False
        
        # Faculty distribution validation
        distribution = allocation_df["Allocated"].value_counts()
        logger.info("Faculty Distribution:")
        for faculty, count in distribution.items():
            percentage = (count / len(allocation_df)) * 100
            logger.info(f"  {faculty}: {count} students ({percentage:.1f}%)")
        
        # Check for unallocated students
        unallocated = allocation_df[allocation_df["Allocated"].isna()]
        if not unallocated.empty:
            logger.warning(f"{len(unallocated)} students were not allocated.")
            logger.warning("Unallocated students:")
            for _, student in unallocated.iterrows():
                logger.warning(f"  - {student['Roll']}: {student['Name']}")
        else:
            logger.info("All students successfully allocated.")
        
        # Validate preference statistics
        if prefs_df is not None and not prefs_df.empty:
            logger.info(f"Preference statistics generated for {len(prefs_df)} faculties")
        else:
            logger.error("Preference statistics are empty or invalid")
            return False
        
        # Validate summary statistics
        if summary:
            total_students = summary.get("total_students", 0)
            if total_students != len(allocation_df):
                logger.error(f"Summary student count mismatch: {total_students} vs {len(allocation_df)}")
                return False
            
            pref_satisfaction = summary.get("preference_satisfaction", {})
            total_prefs = sum(pref_satisfaction.values())
            logger.info(f"Preference Satisfaction: {pref_satisfaction}")
            
            if total_prefs != total_students:
                logger.warning(f"Preference satisfaction total mismatch: {total_prefs} vs {total_students}")
        
        # Additional data quality checks
        cgpa_issues = allocation_df[allocation_df["CGPA"].isna()]
        if not cgpa_issues.empty:
            logger.warning(f"{len(cgpa_issues)} students have missing CGPA values")
        
        # Check CGPA range
        min_cgpa = allocation_df["CGPA"].min()
        max_cgpa = allocation_df["CGPA"].max()
        logger.info(f"CGPA Range: {min_cgpa:.2f} to {max_cgpa:.2f}")
        
        if min_cgpa < 0 or max_cgpa > 10:
            logger.warning(f"CGPA values outside expected range (0-10): {min_cgpa} to {max_cgpa}")
        
        # Validate email format in test data
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        invalid_emails = allocation_df[~allocation_df['Email'].str.contains(email_pattern, na=False)]
        if not invalid_emails.empty:
            logger.warning(f"{len(invalid_emails)} students have invalid email formats")
        
        logger.info("=== VALIDATION COMPLETED SUCCESSFULLY ===")
        return True
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False

def generate_test_report(allocation_df: pd.DataFrame, prefs_df: pd.DataFrame, summary: Dict) -> str:
    """Generate a comprehensive test report."""
    try:
        from datetime import datetime
        
        # Calculate additional metrics
        total_students = len(allocation_df)
        pref_satisfaction = summary.get('preference_satisfaction', {})
        faculty_distribution = summary.get('faculty_distribution', {})
        
        # Preference satisfaction percentages
        pref_percentages = {
            key: f"{value}/{total_students} ({(value/total_students)*100:.1f}%)" 
            for key, value in pref_satisfaction.items()
        }
        
        report = f"""
BTP/MTP Allocation Engine Test Report
=====================================
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Environment: Streamlit Cloud

TEST SUMMARY:
- Total Students: {total_students}
- Total Faculties: {len(prefs_df)}
- Test Status: PASSED
- Algorithm: Round-robin CGPA-based allocation

ALLOCATION RESULTS BY FACULTY:
{allocation_df.groupby('Allocated').size().to_string()}

PREFERENCE SATISFACTION ANALYSIS:
- First Preference:  {pref_percentages.get('pref_1', '0/0 (0.0%)')}
- Second Preference: {pref_percentages.get('pref_2', '0/0 (0.0%)')}
- Third Preference:  {pref_percentages.get('pref_3', '0/0 (0.0%)')}
- Other/Unallocated: {pref_percentages.get('other', '0/0 (0.0%)')}

FACULTY WORKLOAD DISTRIBUTION:
{chr(10).join([f"- {faculty}: {count} students" for faculty, count in faculty_distribution.items()])}

CGPA STATISTICS:
- Highest CGPA: {allocation_df['CGPA'].max():.2f}
- Lowest CGPA:  {allocation_df['CGPA'].min():.2f}
- Average CGPA: {allocation_df['CGPA'].mean():.2f}
- Median CGPA:  {allocation_df['CGPA'].median():.2f}

TOP 5 PERFORMERS (by CGPA):
{allocation_df.nlargest(5, 'CGPA')[['Roll', 'Name', 'CGPA', 'Allocated']].to_string(index=False)}

BOTTOM 5 PERFORMERS (by CGPA):
{allocation_df.nsmallest(5, 'CGPA')[['Roll', 'Name', 'CGPA', 'Allocated']].to_string(index=False)}

ALLOCATION QUALITY METRICS:
- Fair Distribution: {'✓ Passed' if max(faculty_distribution.values()) - min(faculty_distribution.values()) <= 1 else '✗ Failed'}
- All Students Allocated: {'✓ Yes' if allocation_df['Allocated'].notna().all() else '✗ No'}
- Valid CGPA Range: {'✓ Yes' if 0 <= allocation_df['CGPA'].min() and allocation_df['CGPA'].max() <= 10 else '✗ No'}
"""
        return report
        
    except Exception as e:
        logger.error(f"Failed to generate test report: {e}")
        return f"Test Report Generation Failed: {str(e)}"

def run_comprehensive_test() -> Tuple[bool, str, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Run comprehensive test suite and return results."""
    logger.info("Running Automated Tests for BTP/MTP Allocation Engine")
    logger.info("=" * 60)
    
    # Execute allocation test
    success, allocation_df, prefs_df, summary = run_allocation_test()
    
    if not success:
        error_msg = "Test execution failed. Check the logs for details."
        logger.error(error_msg)
        return False, error_msg, None, None
    
    # Validate results
    validation_success = verify_allocation_results(allocation_df, prefs_df, summary)
    
    if not validation_success:
        error_msg = "Result validation failed. Check the logs for details."
        logger.error(error_msg)
        return False, error_msg, allocation_df, prefs_df
    
    # Generate report
    report = generate_test_report(allocation_df, prefs_df, summary)
    
    logger.info("All tests executed successfully.")
    return True, report, allocation_df, prefs_df

# For Streamlit integration
def get_test_results():
    """Function to be called from Streamlit app."""
    return run_comprehensive_test()

# Traditional script execution
if __name__ == "__main__":
    success, report, allocation_df, prefs_df = run_comprehensive_test()
    
    if success:
        print("\n" + "="*60)
        print("TEST REPORT:")
        print("="*60)
        print(report)
    else:
        print(f"\nTEST FAILED: {report}")
        sys.exit(1)
