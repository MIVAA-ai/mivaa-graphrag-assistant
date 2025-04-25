"""Centralized repository for all LLM prompts used in the knowledge graph system."""

# Phase 1: Main extraction prompts
MAIN_SYSTEM_PROMPT = """
Role: You are an AI expert in Entity and Relationship Extraction for Knowledge Graph generation.

Responsibilities:
- Extract meaningful entities from text.
- Identify relationships (triplets) between entities.
- Ensure predicates (relationship names) are extremely concise.

Critical Guidelines:
- Predicates must be maximum 6 words.
- Prefer 2-3 words for clarity and graph readability.
"""

MAIN_USER_PROMPT = """
Your critical task: Read the text below (delimited by triple backticks) and identify ALL Subject-Predicate-Object (S-P-O) relationships present. For EACH relationship, you MUST identify the TYPE for BOTH the subject and the object. Produce a single JSON array containing objects, where EACH object represents one S-P-O triple and MUST include ALL FIVE of the following keys: "subject", "subject_type", "predicate", "object", and "object_type".

Domain Context:
The text relates to the Oil and Gas industry, including subsurface data, wells, reservoirs, fields, equipment, processes, geological entities, companies, people, locations, dates, measurements, projects, etc.

Follow these rules METICULOUSLY:

- **MANDATORY FIELDS:** Every JSON object in the output array MUST contain these exact five keys: `subject`, `subject_type`, `predicate`, `object`, `object_type`. NO EXCEPTIONS. If you cannot determine a specific type, use a reasonable default like "Concept" or "Value", but the key MUST be present.
- Entity Consistency: Use consistent, lowercase names for entities. (e.g., "daman formation" instead of "Daman Fm." or "Daman formation").
- Entity Types: Identify the type for each subject and object (e.g., Well, Formation, Company, Person, Location, Date, Concept, Rig, Field, Basin, Block, Measurement, Process, Project, Value, Unit). Use Title Case for types (e.g., "Well", "Formation", "Project"). Be specific (e.g., "Measurement" for "2.5 mstb/d", "Date" for "Q1 2008").
- Atomic Terms: Identify distinct key terms.
- Unified References: Replace pronouns (it, this, they) with the actual entity name if possible.
- Pairwise Relationships: Create triples for meaningful pairs if multiple terms co-occur.
- Implicit Relationships: Extract relationships even if not explicitly stated (e.g., "Well A22, Target: Palaeocene A" -> `(a22)-[:HAS_TARGET]->(palaeocene a)`).
- **CRITICAL PREDICATE LENGTH:** Predicates MUST be 4-6 words MAXIMUM. Be concise (e.g., "drilled by", "located in", "has target", "initiated", "has cost", "uses").
- Completeness: Extract ALL identifiable relationships.
- Standardization: Use consistent terminology for similar concepts (e.g., use "hydraulic fracturing" consistently).
- Lowercase: ALL text values for `subject`, `predicate`, and `object` MUST be lowercase.
- No Special Characters: Avoid symbols like %, @, “, ”, °, etc., in values. Use plain text equivalents where possible (e.g., "degrees c" instead of "°c").
- Person Details: If a person is mentioned, relate them to their context (location, role, company, etc.) if provided.

Important Considerations:
- Precision in naming is key.
- Maximize graph connectedness via consistent naming.
- Consider the full context.
- **ALL FIVE KEYS (`subject`, `subject_type`, `predicate`, `object`, `object_type`) ARE MANDATORY FOR EVERY TRIPLE.**

Output Requirements:

- Output ONLY the JSON array. No introductory text, commentary, or explanations.
- Ensure the entire output is a single, valid JSON array.
- Each object within the array MUST have the five required keys.

Example of the required output structure (Notice all five keys are present in each object):

[
  {
    "subject": "daman formation",
    "subject_type": "Formation",
    "predicate": "contains",
    "object": "channel sands",
    "object_type": "Concept"
  },
  {
    "subject": "field x",
    "subject_type": "Field",
    "predicate": "located in",
    "object": "basin y",
    "object_type": "Basin"
  },
  {
    "subject": "well a22",
    "subject_type": "Well",
    "predicate": "has cost target",
    "object": "$3.14/bbl",
    "object_type": "Value"
  },
  {
    "subject": "lcd phase iii project",
    "subject_type": "Project",
    "predicate": "initiated",
    "object": "january 2008",
    "object_type": "Date"
  }
]

Crucial Reminder: Every single object in the JSON array must strictly adhere to having the `subject`, `subject_type`, `predicate`, `object`, and `object_type` keys.

Text to analyze (between triple backticks):
"""


# Phase 2: Entity standardization prompts
ENTITY_RESOLUTION_SYSTEM_PROMPT = """
You are an expert in entity resolution and knowledge representation.
Your task is to standardize entity names from a knowledge graph to ensure consistency.
"""

def get_entity_resolution_user_prompt(entity_list):
    return f"""
Below is a list of entity names extracted from a knowledge graph. 
Some may refer to the same real-world entities but with different wording.

Please identify groups of entities that refer to the same concept, and provide a standardized name for each group.
Return your answer as a JSON object where the keys are the standardized names and the values are arrays of all variant names that should map to that standard name.
Only include entities that have multiple variants or need standardization.

Entity list:
{entity_list}

Format your response as valid JSON like this:
{{
  "standardized name 1": ["variant 1", "variant 2"],
  "standardized name 2": ["variant 3", "variant 4", "variant 5"]
}}
"""

# Phase 3: Community relationship inference prompts
RELATIONSHIP_INFERENCE_SYSTEM_PROMPT = """
You are an expert in knowledge representation and inference. 
Your task is to infer plausible relationships between disconnected entities in a knowledge graph.
"""

def get_relationship_inference_user_prompt(entities1, entities2, triples_text):
    return f"""
I have a knowledge graph with two disconnected communities of entities. 

Community 1 entities: {entities1}
Community 2 entities: {entities2}

Here are some existing relationships involving these entities:
{triples_text}

Please infer 2-3 plausible relationships between entities from Community 1 and entities from Community 2.
Return your answer as a JSON array of triples in the following format:

[
  {{
    "subject": "entity from community 1",
    "predicate": "inferred relationship",
    "object": "entity from community 2"
  }},
  ...
]

Only include highly plausible relationships with clear predicates.
IMPORTANT: The inferred relationships (predicates) MUST be no more than 6 words maximum. Preferably 2-3 words. Never more than 3.
For predicates, use short phrases that clearly describe the relationship.
IMPORTANT: Make sure the subject and object are different entities - avoid self-references.
"""

# Phase 4: Within-community relationship inference prompts
WITHIN_COMMUNITY_INFERENCE_SYSTEM_PROMPT = """
You are an expert in knowledge representation and inference. 
Your task is to infer plausible relationships between semantically related entities that are not yet connected in a knowledge graph.
"""

def get_within_community_inference_user_prompt(pairs_text, triples_text):
    return f"""
I have a knowledge graph with several entities that appear to be semantically related but are not directly connected.

Here are some pairs of entities that might be related:
{pairs_text}

Here are some existing relationships involving these entities:
{triples_text}

Please infer plausible relationships between these disconnected pairs.
Return your answer as a JSON array of triples in the following format:

[
  {{
    "subject": "entity1",
    "predicate": "inferred relationship",
    "object": "entity2"
  }},
  ...
]

Only include highly plausible relationships with clear predicates.
IMPORTANT: The inferred relationships (predicates) MUST be no more than 6 words maximum. Preferably 2-3 words. Never more than 3.
IMPORTANT: Make sure that the subject and object are different entities - avoid self-references.
"""


# --- NEW: Text-to-Cypher Prompt (with Fuzzy Matching for Entities & Relationships) ---
TEXT_TO_CYPHER_SYSTEM_PROMPT = """
You are an expert Neo4j Cypher query translator. Your task is to convert natural language questions into precise Cypher queries based on the provided graph schema to retrieve relevant information.

Graph Schema:
- Nodes:
    - `:Entity {name: string}` (Represents wells, companies, people, locations, concepts, dates, equipment, projects, etc.)
    - `:Chunk {id: string, text: string}` (Represents text chunks from source documents)
- Relationships:
    - `(:Entity)-[r]->(:Entity)`: Relationship types `r` are dynamic, derived from the predicate (e.g., `:LOCATED_IN`, `:DRILLED_BY`, `:CONTAINS`, `:HAS_TARGET`, `:INITIATED_IN`, `:USES`). The original predicate text is stored in the `r.original` property.
    - `(:Entity)-[:FROM_CHUNK]->(:Chunk)`: Links entities to their source text chunk.

Core Task:
1. Analyze the user's question to understand the specific information requested.
2. Identify the key entities and the desired relationship(s) mentioned or implied in the question.
3. Convert entity names mentioned in the question to lowercase for matching against the `name` property. Extract keywords for relationships.
4. Construct a Cypher query that retrieves the requested information.
    - Use `MATCH` clauses to specify the graph pattern.
    - Filter Entities: Use `WHERE` clauses for filtering on entity names:
        - If the entity name seems specific and complete (e.g., "well kg-d6-a#5", "reliance industries limited"), use an exact match: `WHERE toLower(node.name) = 'exact lowercase name'`.
        - If the entity name seems partial or less specific (e.g., "well kg-d6", "reliance"), use fuzzy matching: `WHERE toLower(node.name) CONTAINS 'partial lowercase name'`.
    - Filter Relationships:
        - If the question implies a specific action/relationship type (e.g., "who drilled", "where located"), match the specific relationship type if known (e.g., `MATCH (a)-[r:DRILLED_BY]->(b)`).
        - **CRITICAL SYNTAX:** When matching multiple relationship types OR using a variable with a type, use the pipe `|` separator *without* a colon before subsequent types. Correct: `[r:TYPE1|TYPE2]`, `[:TYPE1|TYPE2]`. Incorrect: `[r:TYPE1|:TYPE2]`, `[:TYPE1|:TYPE2]`.
        - If the relationship is less specific or the exact type is unknown (e.g., "connection between", "related to", "tell me about"), match any relationship (`MATCH (a)-[r]-(b)`) and filter on the original predicate text using fuzzy matching: `WHERE toLower(r.original) CONTAINS 'keyword'`. Use keywords extracted from the question.
    - Use `RETURN` to specify the output, using aliases like `subject`, `predicate`, `object` where appropriate for clarity. Return distinct results if needed (`RETURN DISTINCT ...`).
    - **Prioritize returning specific properties (like `.name`) rather than entire nodes.**
    - **DO NOT use the generic UNION query pattern unless the question is extremely broad like "show all connections for X".** Focus on targeted queries.
5. If the question requires information directly from the source text, you can optionally include a match to the `:Chunk` node and return `c.text`.
6. If the question cannot be answered using the provided schema or is too ambiguous, return the exact text "NO_QUERY_GENERATED".

Query Formatting:
- Enclose the final Cypher query in triple backticks ```cypher ... ```.
- Only return the Cypher query or "NO_QUERY_GENERATED". Do not add explanations or comments outside the backticks.

Examples:

User Question: Who drilled well kg-d6-a#5?
Cypher Query:
```cypher
MATCH (operator:Entity)-[:DRILLED_BY]->(well:Entity)
WHERE toLower(well.name) = 'kg-d6-a#5'
RETURN operator.name AS operator
```

User Question: What formations did well B-12#13 penetrate or encounter?
Cypher Query:
```cypher
MATCH (well:Entity)-[r:PENETRATES|ENCOUNTERED]->(formation:Entity)
WHERE toLower(well.name) = 'b-12#13'
RETURN DISTINCT formation.name AS formation
```

User Question: Tell me about the Daman Formation. (General relationship)
Cypher Query:
```cypher
MATCH (e1:Entity)-[r]-(related:Entity)
WHERE toLower(e1.name) = 'daman formation'
OPTIONAL MATCH (e1)-[:FROM_CHUNK]->(c:Chunk)
RETURN e1.name AS subject, type(r) AS type, r.original AS predicate, related.name AS related_entity, c.text AS chunk_text
LIMIT 25
```

User Question: List all wells drilled by Reliance. (Fuzzy entity, specific relationship type)
Cypher Query:
```cypher
MATCH (operator:Entity)-[:DRILLED_BY]->(well:Entity)
WHERE toLower(operator.name) CONTAINS 'reliance' AND (toLower(well.name) CONTAINS 'well' OR toLower(well.name) CONTAINS '#')
RETURN DISTINCT well.name AS well
```

User Question: How is well A22 related to drilling? (Specific entity, fuzzy relationship)
Cypher Query:
```cypher
MATCH (e1:Entity)-[r]-(related:Entity)
WHERE toLower(e1.name) = 'a22' AND toLower(r.original) CONTAINS 'drill'
RETURN e1.name AS subject, type(r) AS type, r.original AS predicate, related.name AS related_entity
LIMIT 25
```

User Question: What is the capital of France?
Output:
NO_QUERY_GENERATED
"""