from preprocess import clean_text

essay = """
Artificial Intelligence is transforming education.
Students are learning with AI tools every day.
"""

print("Original Essay:")
print(essay)

print("\nProcessed Essay:")
print(clean_text(essay))