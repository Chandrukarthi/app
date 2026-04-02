def get_eda_prompt(sample_data):
    return f"""
Analyze the given dataset sample and provide a short EDA summary.

Instructions:
- Keep the answer very concise
- Use simple language
- Maximum 5 bullet points

Include:
1. What the dataset is about (1 line)
2. Data quality issues (missing values, duplicates, outliers)
3. Most important issue
4. One simple suggestion

Dataset Sample:
{sample_data}
"""