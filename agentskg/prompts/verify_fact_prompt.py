VERIFY_OUTPUT_INSTRUCTION = """
## Output Format
You must output a single Python list of integers (0 or 1) corresponding strictly to the order of the input triplets.
- 1 = Pass / Valid / True
- 0 = Fail / Invalid / False

Example Output:
[1, 0, 1]
"""

SEMANTIC_BOOL_PROMPT = """
## Role
You are a pragmatic Knowledge Graph Engineer. Your task is to analyze the logical properties of a relation to determine database constraints.

## Important Strategy: "Relaxed Constraints"
- In Knowledge Graph extraction, data is often messy or incomplete.
- **When in doubt, output 0.** It is better to allow multiple values (0) than to incorrectly block valid data (1).
- Only mark a property as **Functional (1)** if it is **physically impossible** or **strictly forbidden** to have more than one value (e.g., `birthDate`, `atomicNumber`).
- Common relations like `author`, `parent`, `nationality`, `manufacturer`, `spouse` (due to remarriage histories) should usually be **0**.

## Logical Property Definitions
Evaluate the relation against these seven definitions:

1.  **Functional** (Is it STRICTLY unique?): For a single subject, can this relation hold **multiple different values** over time or context?
    -   If YES (e.g., `hasChild`, `hasAuthor`, `hasAlias`), output **0**.
    -   Only if it is STRICTLY limited to exactly one (e.g., `hasBirthDate`, `hasID`), output **1**.
2.  **Inverse Functional** (Is it a unique identifier?): Can distinct subjects share the same object value?
    -   If YES (e.g., `hasEmail` can be shared by a team, `hasName`), output **0**.
    -   Only if the value uniquely identifies the subject (e.g., `hasSocialSecurityNumber`), output **1**.
3.  **Transitive**: If `(A, relation, B)` and `(B, relation, C)` are true, then `(A, relation, C)` must also be true.
    -   *Example: `isLocatedIn`.*
4.  **Symmetric**: If `(A, relation, B)` is true, then `(B, relation, A)` must also be true.
    -   *Example: `isSpouseOf`, `isPartnerOf`.*
5.  **Asymmetric**: If `(A, relation, B)` is true, then `(B, relation, A)` must **never** be true.
    -   *Example: `isChildOf`, `isOlderThan`.*
6.  **Reflexive**: For **any** individual `A`, `(A, relation, A)` must be true.
    -   *Example: `isEqualTo`, `sameAs`.*
7.  **Irreflexive**: For **any** individual `A`, `(A, relation, A)` must **never** be true.
    -   *Example: `isParentOf`, `isCreatedBy`.*

## Task Instruction
I will provide a description of a specific relation. Determine the properties based on the "Relaxed Constraints" strategy to avoid data loss during insertion.

## Output Format
Your output must strictly adhere to the following format:
1.  Do not include any explanations.
2.  The result must be a Python list of 7 integers (0 or 1).
3.  Order: [Functional, Inverse Functional, Transitive, Symmetric, Asymmetric, Reflexive, Irreflexive]
4.  Enclose the list within a `<result>` XML tag.

### Example Output
<result>[0, 0, 1, 0, 0, 0, 1]</result>

## Input Relation Description:
"""
