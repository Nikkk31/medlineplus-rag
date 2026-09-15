
EVAL_SET = [
    {"question": "What is type 1 diabetes?", "expected_sources": ["MedlinePlus"]},
    {"question": "What is type 2 diabetes?", "expected_sources": ["MedlinePlus"]},
    {"question": "What are the symptoms of diabetes?", "expected_sources": ["MedlinePlus"]},
    {"question": "What causes hyperglycemia?", "expected_sources": ["MedlinePlus"]},
    {"question": "What are the signs of low blood sugar?", "expected_sources": ["MedlinePlus"]},
    {"question": "What is diabetic ketoacidosis?", "expected_sources": ["MedlinePlus"]},
    {"question": "How is type 2 diabetes diagnosed?", "expected_sources": ["MedlinePlus"]},
    {"question": "What is prediabetes?", "expected_sources": ["MedlinePlus"]},
    {"question": "How can diabetes be prevented?", "expected_sources": ["MedlinePlus"]},
    {"question": "What foods should someone with diabetes avoid?", "expected_sources": ["MedlinePlus"]},
    {"question": "What is the A1C test?", "expected_sources": ["MedlinePlus"]},
    {"question": "What is insulin resistance?", "expected_sources": ["MedlinePlus"]},
    # Out-of-scope controls - retrieval SHOULD return low-confidence / irrelevant results
    {"question": "What is the capital of France?", "expected_sources": [], "out_of_scope": True},
    {"question": "How do I fix a flat tire?", "expected_sources": [], "out_of_scope": True},
]