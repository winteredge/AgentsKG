DESCRIBE_ENTITY_PROMPT = """## Role
You are a Knowledge Engineer building a formal ontology. Your task is to define the fundamental class or category of a given target entity.

## Context and Task
I will provide a triple (subject-predicate-object) to give you semantic context, and a target term which is one of the entities (subject or object) from that triple.
Your mission is to generate a concise, accurate, and universally applicable one-sentence definition for the **target term itself**.

## Crucial Rules
1.  The definition must describe the general **type or class** of the entity (e.g., "a type of fruit," "a government agency," "a geographical coding system").
2.  **DO NOT** include any specific details from the triple's context. You are defining the term in general, not its specific state in the example.
3.  The output must strictly follow the format: `Target Term:One-sentence definition`

## Examples
### Example 1
Context Triple: Apple is red
Target Term: Apple
Output: Apple:An edible fruit produced by an apple tree.
(Explanation: The context helps identify "Apple" as the fruit, not the company. The definition is for the fruit in general, not "a red fruit".)

### Example 2
Context Triple: Qingdao Petrochemical Inspection Center - Postal Code - 266071
Target Term: Qingdao Petrochemical Inspection Center
Output: Qingdao Petrochemical Inspection Center:An organization specializing in professional inspection and testing services.
(Explanation: The definition identifies its organizational type, not its specific postal code.)

### Example 3
Context Triple: Qingdao Petrochemical Inspection Center - Postal Code - 266071
Target Term: Postal Code
Output: Postal Code:A system of alphanumeric codes used by postal services to simplify mail sorting and delivery.
(Explanation: The definition describes the concept of a "Postal Code," not the specific number 266071.)

## Your Turn
"""

DESCRIBE_RELATION_PROMPT = """## Role
You are a Knowledge Engineer building a formal ontology. Your task is to define the semantics of a given predicate (relation).

## Context and Task
I will provide a triple (subject-predicate-object) to give you semantic context, and the target relation from that triple.

Your mission is to generate a concise, accurate, and universally applicable one-sentence definition for the **relation itself**.

## Crucial Rules
1.  The definition must explain the general meaning of the relation type, describing the kind of connection or property it represents.
2.  **DO NOT** include or refer to the specific subject or object from the triple. You are defining the relation as an abstract concept.
3.  The output must strictly follow the format: `Relation Name:One-sentence definition`

## Examples
### Example 1
Context Triple: Apple is red
Target Relation: is
Output: is:A copular verb that indicates an attribute, state, or identity relationship.
(Explanation: Here, "is" denotes the property of being red. The definition is for this type of attributive relationship, not the specific fact that "Apple is red.")

### Example 2
Context Triple: Shanghai Explosion-proof Electric Test Center - legal entity - Shanghai Industrial Automation Institute
Target Relation: legal entity
Output: legal entity:Refers to the legally established status of an organization that allows it to independently bear civil liability.
(Explanation: The definition describes the legal or organizational concept of a "legal entity," not the specific institute.)

### Example 3
Context Triple: Beijing Labor Protection Equipment Test Center - contact person - Yang Wenfen
Target Relation: contact person
Output: contact person:Refers to the role of an individual responsible for communication and liaison within a specific organization, event, or matter.
(Explanation: The definition describes the functional role of a "contact person," not the specific individual Yang Wenfen.)

## Your Turn
"""
