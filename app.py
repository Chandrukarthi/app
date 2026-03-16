import streamlit as st
st.title("Student Course Feedback App")
st.header("Welcome to Course Feedback System")
st.write("Share your feedback about the course")
st.text("Please fill the details below")
st.markdown("### Course Feedback Form")
st.markdown("*Your feedback helps improve the course quality*")
st.image("course.png", caption="Learning Time")
name = st.text_input("Enter your Name")
course = st.text_input("Enter Course Name")
age = st.number_input("Enter your Age")
rating = st.slider("Rate the Course", 1, 10)
department = st.selectbox(
    "Select Department",
    ["ECE", "CSE", "IT", "EEE", "Mechanical"]
)

if st.button("Submit Feedback"):
    st.write("### Feedback Submitted ")
    st.write("Name:", name)
    st.write("Course:", course)
    st.write("Age:", age)
    st.write("Rating:", rating)
    st.write("Department:", department)
    st.write("Thank you for your feedback!")