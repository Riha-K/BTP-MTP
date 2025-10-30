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
        
        # FIXED: Set up data directly instead of using load_data()
        engine.students_data = df.copy()
        
        # Identify faculty columns
        cgpa_index = df.columns.get_loc("CGPA")
        engine.faculties = list(df.columns[cgpa_index + 1:])
        
        # FIXED: Manually validate the data structure (bypass load_data validation)
        if not engine._validate_data_structure():
            st.error("Data validation failed. Please check your CSV format and data quality.")
            return
            
        # FIXED: Set validation flag to True
        engine._validated = True
        
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
