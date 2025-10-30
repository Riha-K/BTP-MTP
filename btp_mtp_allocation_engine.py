"""
BTP/MTP Allocation Engine
Implements CGPA-based and cyclic preference-based allocation
Cloud-optimized version with enhanced error handling and validation.
"""

import pandas as pd
import logging
import numpy as np
from typing import Dict, Optional
import traceback

logger = logging.getLogger(__name__)

class AllocationEngine:
    """Performs automated student allocation to faculties with enhanced validation"""

    def __init__(self):
        self.faculties = []
        self.students_data = None
        self.allocation_results = None
        self.preference_stats = None
        self._validated = False

    def load_data(self, file_path: str) -> bool:
        """Load student dataset from CSV with enhanced validation"""
        try:
            logger.info(f"Loading dataset from: {file_path}")
            self.students_data = pd.read_csv(file_path)

            # Enhanced validation
            if not self._validate_data_structure():
                return False

            # Faculty columns appear after 'CGPA'
            cgpa_index = self.students_data.columns.get_loc("CGPA")
            self.faculties = list(self.students_data.columns[cgpa_index + 1:])

            logger.info(f"Loaded {len(self.students_data)} records and {len(self.faculties)} faculties")
            self._validated = True
            return True

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False

    def _validate_data_structure(self) -> bool:
        """Validate the structure and content of loaded data"""
        try:
            # Check if dataframe is empty
            if self.students_data is None or len(self.students_data) == 0:
                logger.error("Dataset is empty")
                return False

            # Check required columns
            required_cols = ["Roll", "Name", "Email", "CGPA"]
            missing_cols = [col for col in required_cols if col not in self.students_data.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return False

            # Validate CGPA column
            try:
                # Convert CGPA to numeric, handling errors gracefully
                self.students_data['CGPA'] = pd.to_numeric(self.students_data['CGPA'], errors='coerce')
                
                # Check for invalid CGPA values
                invalid_cgpa = self.students_data['CGPA'].isna()
                if invalid_cgpa.any():
                    logger.warning(f"Found {invalid_cgpa.sum()} rows with invalid CGPA values")
                    # Remove rows with invalid CGPA
                    self.students_data = self.students_data.dropna(subset=['CGPA'])
                
                # Validate CGPA range (0-10 is typical)
                out_of_range = (self.students_data['CGPA'] < 0) | (self.students_data['CGPA'] > 10)
                if out_of_range.any():
                    logger.warning(f"Found {out_of_range.sum()} CGPA values outside normal range (0-10)")
                    
            except Exception as e:
                logger.error(f"CGPA validation failed: {e}")
                return False

            # Check for faculty preference columns
            cgpa_index = self.students_data.columns.get_loc("CGPA")
            faculty_cols = list(self.students_data.columns[cgpa_index + 1:])
            
            if len(faculty_cols) == 0:
                logger.error("No faculty preference columns found after CGPA")
                return False

            if len(faculty_cols) > 20:  # Reasonable limit
                logger.error(f"Too many faculty columns ({len(faculty_cols)}). Maximum supported: 20")
                return False

            # Validate preference values
            for col in faculty_cols:
                # Convert to numeric and check range
                try:
                    self.students_data[col] = pd.to_numeric(self.students_data[col], errors='coerce')
                    
                    # Check preference range (should be 1 to n_faculties)
                    valid_range = (self.students_data[col] >= 1) & (self.students_data[col] <= len(faculty_cols))
                    invalid_prefs = ~valid_range & ~self.students_data[col].isna()
                    
                    if invalid_prefs.any():
                        logger.warning(f"Invalid preferences found in column {col}: {invalid_prefs.sum()} entries")
                        
                except Exception as e:
                    logger.warning(f"Could not validate preferences for {col}: {e}")

            logger.info("Data structure validation completed successfully")
            return True

        except Exception as e:
            logger.error(f"Data validation error: {e}")
            return False

    def sort_students_by_cgpa(self) -> pd.DataFrame:
        """Return a copy of the student data sorted by CGPA (descending) with validation"""
        if not self._validated:
            logger.error("Data not validated. Call load_data() first.")
            raise ValueError("Data not validated")
            
        try:
            # Handle potential NaN values in CGPA
            sorted_df = self.students_data.dropna(subset=['CGPA']).copy()
            sorted_df = sorted_df.sort_values(by="CGPA", ascending=False).reset_index(drop=True)
            
            logger.info(f"Sorted {len(sorted_df)} students by CGPA")
            return sorted_df
            
        except Exception as e:
            logger.error(f"Error sorting students: {e}")
            raise

    def allocate_students(self) -> pd.DataFrame:
        """Allocate students to faculties using a round-robin mod-n preference algorithm with enhanced error handling"""
        try:
            if not self._validated:
                raise ValueError("Data not validated. Call load_data() first.")

            sorted_df = self.sort_students_by_cgpa()
            faculty_count = len(self.faculties)
            
            if faculty_count == 0:
                raise ValueError("No faculties available for allocation")

            allocation_tracker = {f: 0 for f in self.faculties}
            results = []

            logger.info(f"Starting allocation for {len(sorted_df)} students across {faculty_count} faculties")

            for idx, student in sorted_df.iterrows():
                allocated = False

                # Try assigning faculty based on ranked preferences
                for pref_rank in range(1, faculty_count + 1):
                    try:
                        # Find faculty with this preference rank
                        faculty = None
                        for f in self.faculties:
                            if pd.notna(student[f]) and int(student[f]) == pref_rank:
                                faculty = f
                                break
                        
                        # Check if this faculty is available in the current rotation
                        if faculty and allocation_tracker[faculty] == idx // faculty_count:
                            results.append({
                                "Roll": student["Roll"],
                                "Name": student["Name"],
                                "Email": student["Email"],
                                "CGPA": float(student["CGPA"]),
                                "Allocated": faculty,
                                "Preference_Rank": pref_rank
                            })
                            allocation_tracker[faculty] += 1
                            allocated = True
                            break
                            
                    except (ValueError, KeyError) as e:
                        logger.warning(f"Error processing preference {pref_rank} for student {student.get('Roll', 'Unknown')}: {e}")
                        continue

                # If no preferred faculty fits the rotation, assign to least-loaded faculty
                if not allocated:
                    fallback = min(allocation_tracker, key=allocation_tracker.get)
                    results.append({
                        "Roll": student["Roll"],
                        "Name": student["Name"],
                        "Email": student["Email"],
                        "CGPA": float(student["CGPA"]),
                        "Allocated": fallback,
                        "Preference_Rank": "Unallocated"
                    })
                    allocation_tracker[fallback] += 1

            self.allocation_results = pd.DataFrame(results)
            logger.info(f"Completed allocation for {len(results)} students")
            
            # Log allocation statistics
            distribution = self.allocation_results['Allocated'].value_counts()
            logger.info(f"Faculty distribution: {distribution.to_dict()}")
            
            return self.allocation_results

        except Exception as e:
            logger.error(f"Error during allocation: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def generate_preference_stats(self) -> pd.DataFrame:
        """Generate preference frequency per faculty with enhanced error handling"""
        try:
            if not self._validated:
                raise ValueError("Data not validated. Call load_data() first.")

            stats = []
            for faculty in self.faculties:
                try:
                    pref_counts = {"Faculty": faculty}
                    
                    # Calculate preference counts for each rank
                    for r in range(1, len(self.faculties) + 1):
                        try:
                            # Handle NaN values properly
                            count = (self.students_data[faculty] == r).sum()
                            pref_counts[f"Pref {r}"] = int(count)
                        except Exception as e:
                            logger.warning(f"Error calculating preference {r} for {faculty}: {e}")
                            pref_counts[f"Pref {r}"] = 0
                    
                    stats.append(pref_counts)
                    
                except Exception as e:
                    logger.error(f"Error processing stats for faculty {faculty}: {e}")
                    # Add default stats for this faculty
                    default_stats = {"Faculty": faculty}
                    for r in range(1, len(self.faculties) + 1):
                        default_stats[f"Pref {r}"] = 0
                    stats.append(default_stats)

            self.preference_stats = pd.DataFrame(stats)
            logger.info("Preference statistics generated successfully")
            return self.preference_stats

        except Exception as e:
            logger.error(f"Failed to generate preference stats: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def save_csv(self, df: pd.DataFrame, path: str, label: str) -> bool:
        """Save a DataFrame to a CSV file with enhanced error handling"""
        try:
            if df is None or len(df) == 0:
                logger.warning(f"Cannot save empty DataFrame for {label}")
                return False
                
            df.to_csv(path, index=False, encoding='utf-8')
            logger.info(f"{label} saved at: {path} ({len(df)} rows)")
            return True
            
        except PermissionError:
            logger.error(f"Permission denied saving {label} to {path}")
            return False
        except Exception as e:
            logger.error(f"Error saving {label}: {e}")
            return False

    def save_allocation_results(self, path: str) -> bool:
        """Save allocation results to CSV with validation"""
        if self.allocation_results is None:
            logger.error("No allocation results to save. Run allocate_students() first.")
            return False
            
        # Select only essential columns for output
        output_cols = ["Roll", "Name", "Email", "CGPA", "Allocated"]
        available_cols = [col for col in output_cols if col in self.allocation_results.columns]
        
        return self.save_csv(
            self.allocation_results[available_cols], 
            path, 
            "Allocation Results"
        )

    def save_preference_stats(self, path: str) -> bool:
        """Save preference statistics to CSV with validation"""
        if self.preference_stats is None:
            logger.error("No preference stats to save. Run generate_preference_stats() first.")
            return False
            
        return self.save_csv(self.preference_stats, path, "Preference Statistics")

    def get_allocation_summary(self) -> Dict:
        """Generate and return summary statistics with enhanced error handling"""
        try:
            if self.allocation_results is None or len(self.allocation_results) == 0:
                logger.warning("No allocation results available for summary")
                return {}

            # Calculate preference satisfaction with error handling
            pref_satisfaction = {}
            try:
                pref_satisfaction = {
                    "pref_1": int((self.allocation_results["Preference_Rank"] == 1).sum()),
                    "pref_2": int((self.allocation_results["Preference_Rank"] == 2).sum()),
                    "pref_3": int((self.allocation_results["Preference_Rank"] == 3).sum()),
                    "other": int((self.allocation_results["Preference_Rank"] > 3).sum())
                }
                
                # Count unallocated students
                unallocated_count = int((self.allocation_results["Preference_Rank"] == "Unallocated").sum())
                pref_satisfaction["unallocated"] = unallocated_count
                
            except Exception as e:
                logger.error(f"Error calculating preference satisfaction: {e}")
                pref_satisfaction = {"pref_1": 0, "pref_2": 0, "pref_3": 0, "other": 0, "unallocated": 0}

            # Calculate faculty distribution
            faculty_distribution = {}
            try:
                faculty_distribution = self.allocation_results["Allocated"].value_counts().to_dict()
                # Ensure all values are Python integers (not numpy)
                faculty_distribution = {k: int(v) for k, v in faculty_distribution.items()}
            except Exception as e:
                logger.error(f"Error calculating faculty distribution: {e}")
                faculty_distribution = {}

            summary = {
                "total_students": int(len(self.allocation_results)),
                "total_faculties": len(self.faculties),
                "faculty_distribution": faculty_distribution,
                "preference_satisfaction": pref_satisfaction,
                "algorithm_version": "round-robin-mod-n",
                "validation_status": self._validated
            }
            
            return summary

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"error": f"Summary generation failed: {str(e)}"}

    def reset(self):
        """Reset the engine to initial state"""
        self.faculties = []
        self.students_data = None
        self.allocation_results = None
        self.preference_stats = None
        self._validated = False
        logger.info("Allocation engine reset successfully")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = AllocationEngine()
    
    try:
        if engine.load_data("input_btp_mtp_allocation.csv"):
            engine.allocate_students()
            engine.generate_preference_stats()
            engine.save_allocation_results("output_allocation.csv")
            engine.save_preference_stats("output_preference_stats.csv")
            
            summary = engine.get_allocation_summary()
            logger.info(f"Allocation Summary: {summary}")
        else:
            logger.error("Failed to load data")
    except Exception as e:
        logger.error(f"Application error: {e}")
