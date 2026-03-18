import pandas as pd
import streamlit as st

st.title("🎓 Student Feedback Dashboard")

file = st.file_uploader("Upload Student Feedback CSV", type=["csv"])

if file is not None:

    data = pd.read_csv(file)

    st.success("File loaded successfully!")

    st.subheader("📊 Data")
    st.dataframe(data)

    st.subheader("📈 Charts")

    # Select numeric columns
    numeric_data = data[["Age", "Rating"]]

    st.line_chart(numeric_data)
    st.area_chart(numeric_data)
    st.bar_chart(numeric_data)

    if len(numeric_data.columns) >= 2:
        st.scatter_chart(numeric_data)

    # Columns layout
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("Line Chart")
        st.line_chart(numeric_data)

    with col2:
        st.write("Bar Chart")
        st.bar_chart(numeric_data)

    with col3:
        st.write("Area Chart")
        st.area_chart(numeric_data)

    st.subheader("📌 Analysis")

    st.write("Total Feedbacks:", len(data))
    st.write("Average Rating:", round(data["Rating"].mean(), 2))

    st.write("Course Ratings")
    st.bar_chart(data.groupby("Course")["Rating"].mean())

    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "student_feedback.csv")

else:
    st.info("Upload CSV file to view dashboard")