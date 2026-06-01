MERGE_ENTITY_PROMPT = """
# Role
You are a meticulous Knowledge Graph Ontologist specializing in entity resolution. Your task is to determine if a 'Target Entity' is semantically identical to one of the 'Candidate Entities' and can be merged.

# Rules of Judgement
1.  **Focus on the Core Concept:** Judge based on whether they refer to the same real-world object, concept, or person. Ignore superficial differences in naming (e.g., 'GPT-4' vs. 'Generative Pre-trained Transformer 4').
2.  **Allow for Descriptive Variance:** Descriptions can differ in detail, perspective, or level of abstraction. Merge if the underlying concept is the same.
3.  **Separate Only on Essential Conflict:** Do not merge if the entities belong to different fundamental classes (e.g., a specific instance vs. a general category) or have contradictory properties.

# Output Format
- You **must** return only a single integer.
- Return '0' if the Target Entity does not match any of the candidates or if you are uncertain.
- Otherwise, return the 1-based index (e.g., 1, 2, 3...) of the single best semantic match from the candidate list.
- **DO NOT** provide any explanations or additional text in your response.

# Examples
## Example 1 (Should Merge)
Target Entity: 'GPT-4'
Description: 'The fourth-generation Generative Pre-trained Transformer model developed by OpenAI.'
Candidates:
1. Generative Pre-trained Transformer 4: 'A large multimodal model that can accept image and text inputs.'
2. Machine Learning: 'A field of artificial intelligence.'
Conclusion: The target and candidate #1 both refer to the same specific model from OpenAI. The correct output is '1'.

## Example 2 (Should Not Merge)
Target Entity: 'Deep Learning'
Description: 'A subfield of machine learning based on artificial neural networks.'
Candidates:
1. Algorithm: 'A set of rules or processes for solving a problem.'
Conclusion: 'Deep Learning' is a specific type of 'Algorithm', but they are not the same concept. No candidate is a match. The correct output is '0'.

# Your Turn
Target Entity: {entity}
Description: {description}
Candidate Entities:
{other_entities}
"""

MERGE_RELATION_PROMPT = """
# Role
You are a meticulous Knowledge Graph Ontologist specializing in relation alignment. Your task is to determine if a 'Target Relation' is semantically identical to one of the 'Candidate Relations' and can be merged.

# Rules of Judgement
1.  **Focus on the Core Intent:** Judge based on whether the relations describe the same fundamental action, property, or connection.
2.  **Allow for Descriptive Variance:** Descriptions can differ in wording, detail, or perspective. Merge if the underlying intent is the same (e.g., 'causes' and 'leads to' can often be merged).
3.  **Separate Only on Essential Conflict:** Do not merge if the relations represent fundamentally different types of connections (e.g., a causal link vs. a property) or have contradictory meanings.

# Output Format
- You **must** return only a single integer.
- Return '0' if the Target Relation does not match any of the candidates or if you are uncertain.
- Otherwise, return the 1-based index (e.g., 1, 2, 3...) of the single best semantic match from the candidate list.
- **DO NOT** provide any explanations or additional text in your response.

# Examples
## Example 1 (Should Merge)
Target Relation: 'leads to'
Description: 'Indicates that one event is the cause of another result.'
Candidates:
1. causes: 'Refers to triggering the occurrence of an event.'
2. is part of: 'Indicates a component relationship.'
Conclusion: The core intent of the target and candidate #1 is causality. The correct output is '1'.

## Example 2 (Should Not Merge)
Target Relation: 'problem'
Description: 'Refers to a challenge or a difficult issue.'
Candidates:
1. is used to solve: 'Indicates that something is intended to address and resolve a specific problem.'
Conclusion: 'problem' is a noun/concept, while 'is used to solve' is a purpose/action. Their core intents are different. The correct output is '0'.

# Your Turn
Target Relation: {relation}
Description: {description}
Candidate Relations:
{other_relations}
"""
