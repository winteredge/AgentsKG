# [MODIFIED] - Stricter rules and better examples for schema-like output.

EXTRACTION_ZERO_SHOT = """
You are a precision Knowledge Graph extraction system. Your task is to extract structured knowledge triplets (subject, predicate, object) from the given text.

Please read the following text carefully and strictly adhere to the rules below:
{rules}
Here is the text to be processed:
--- START OF TEXT ---
{text_chunk}
--- END OF TEXT ---

Return all relevant triplets in the format of a JSON list. If no triplets are found, output an empty list `[]`.
Your output must start with `[` and end with `]`. Output ONLY the JSON list, with no additional text.
"""

EXTRACTION_FEW_SHOT = """
You are a precision Knowledge Graph extraction system. Your task is to extract structured knowledge triplets (subject, predicate, object) from the given text.

Please read the following text carefully and strictly adhere to the rules below:
{rules}
Here are examples of the required extraction style:
{examples}
Here is the text to be processed:
--- START OF TEXT ---
{text_chunk}
--- END OF TEXT ---

Return all relevant triplets in the format of a JSON list. If no triplets found, output an empty list `[]`.
Your output must start with `[` and end with `]`. Output ONLY the JSON list, with no additional text.
"""

# Dictionary for Extraction Prompts
EXTRACTION_PROMPTS = {
    "zero_shot": EXTRACTION_ZERO_SHOT,
    "few_shot": EXTRACTION_FEW_SHOT,
}

DATASET_EXAMPLES = {
    # --- WebNLG ---
    "webnlg": """
--- EXAMPLE 1 ---
Text: "Apollo 14 crew member Alan Shepard retired on the first of August, 1974."
Output:
[
  {"s": "Alan Shepard", "p": "mission", "o": "Apollo 14"},
  {"s": "Alan Shepard", "p": "dateOfRetirement", "o": "1974-08-01"}
]
--- EXAMPLE 2 ---
Text: "Buzz Aldrin is a US national because he was born in Glen Ridge, New Jersey."
Output:
[
  {"s": "Buzz Aldrin", "p": "birthPlace", "o": "Glen Ridge, New Jersey"},
  {"s": "Buzz Aldrin", "p": "nationality", "o": "United States"}
]
""",
    # --- aida-conll ---
    "aida-conll": """
--- EXAMPLE 1 ---
Text: "German carmaker Volkswagen said on Wednesday it hired former BMW executive Pischetsrieder."
Output:
[
{"s": "Volkswagen Group", "p": "headquartersLocation", "o": "Germany"},
{"s": "Volkswagen Group", "p": "hired", "o": "Bernd Pischetsrieder"},
{"s": "Bernd Pischetsrieder", "p": "formerEmployer", "o": "BMW"}
]
--- EXAMPLE 2 ---
Text: "Agassi advanced to the semi-finals after defeating Sampras in straight sets at the U.S. Open."
Output:
[
{"s": "Andre Agassi", "p": "defeated", "o": "Pete Sampras"},
{"s": "Andre Agassi", "p": "participantIn", "o": "US Open (tennis)"},
{"s": "Pete Sampras", "p": "participantIn", "o": "US Open (tennis)"}
]
--- EXAMPLE 3 ---
Text: "Clinton arrived in Belfast today to aid the Northern Ireland peace process."
Output:
[
{"s": "Bill Clinton", "p": "traveledTo", "o": "Belfast"},
{"s": "Belfast", "p": "location", "o": "Northern Ireland"}
]
""",
    # --- CaRB ---
    "CaRB": """
--- EXAMPLE 1 ---
Text: "Separate berthings and heads are found on sailboats over about ."
Output:
[
  {"s": "berthings", "p": "are found on", "o": "sailboats"},
  {"s": "heads", "p": "are found on", "o": "sailboats"}
]
--- EXAMPLE 2 ---
Text: "The mouse is around nine inches long , and can jump in bounds of four feet when threatened ."
Output:
[
  {"s": "The mouse", "p": "is", "o": "around nine inches long"},
  {"s": "The mouse", "p": "can jump in bounds of", "o": "four feet"}
]
""",
}
RULES = {
    "webnlg": """
--- START OF RULES ---
1.  **Predicate Style**: The predicate (p) MUST be a concise, schema-like property name, preferably in camelCase or as a single noun (e.g., `birthPlace`, `runtime`, `director`, `location`).
2.  **Avoid Verbs**: Crucially, AVOID using common verbs or conversational phrases like 'is', 'has', 'was', 'includes' as the predicate. The predicate should represent the *relationship property itself*.
3.  **Atomicity**: Keep the subject (s) and object (o) as atomic concepts.
4.  **Object Normalization**: Normalize dates to YYYY-MM-DD. Remove generic units if possible.
--- END OF RULES ---
""",
    "CaRB": """
--- START OF RULES ---
1.  **Natural Predicates**: The predicate (p) should be a **verbatim phrase** from the text that describes the relationship (e.g., "is the capital of", "was born in", "has a population of").
2.  **Include Verbs**: Unlike schema-based extraction, you **MUST include the verbs** and prepositions that define the relationship in the predicate.
3.  **Span Extraction**: Subjects (s) and Objects (o) should largely correspond to continuous spans of text from the original sentence.
4.  **Completeness**: Capture the full meaning of the relationship.
--- END OF RULES ---
""",
    "aida-conll": """
--- START OF RULES ---
1.  **Entity Specificity**: Prefer specific **Proper Nouns** (e.g., `Apple Inc.`, `Donald Trump`, `London`) over pronouns (`he`, `it`) or common nouns (`the company`, `the city`). The goal is to ensure each entity can be uniquely identified.
2.  **Semantic Predicates**: The predicate (p) should be a concise, meaningful relationship property (e.g., `memberOf`, `locatedIn`, `foundedBy`, `competedAgainst`). Avoid using generic auxiliary verbs like 'is', 'has', 'was' as standalone predicates.
3.  **Strict Atomicity**: Keep subjects (s) and objects (o) as clean and atomic as possible. Remove unnecessary modifiers, titles, or appositives (e.g., use `Albert Einstein` instead of `the famous physicist Albert Einstein`).
4.  **Relational Fidelity**: The extracted relationship must be explicitly stated or strongly implied by the core semantics of the text. Do not infer relationships that require external knowledge.
5.  **Standardized Formatting**: Use singular forms for entities where possible and maintain consistent casing (preferably Title Case for names and camelCase or snake_case for properties).
--- END OF RULES ---
""",
}


def generate_prompt(strategy, dataset_name, text_chunk):
    """
    Args:
        strategy (str): "zero_shot" or "few_shot"
        dataset_name (str): "webnlg", "aida-conll", "CaRB"
        text_chunk (str): The actual input text
    """
    if strategy not in EXTRACTION_PROMPTS:
        raise ValueError(f"Unknown strategy: {strategy}")
    template = EXTRACTION_PROMPTS[strategy]
    selected_rules = RULES.get(dataset_name, RULES["webnlg"])
    if strategy == "zero_shot":
        return template.format(rules=selected_rules, text_chunk=text_chunk)

    elif strategy == "few_shot":
        selected_examples = DATASET_EXAMPLES.get(dataset_name, "")
        if not selected_examples:
            print(
                f"⚠️ Warning: No few-shot examples found for {dataset_name}. Using empty string."
            )

        return template.format(
            rules=selected_rules, examples=selected_examples, text_chunk=text_chunk
        )
