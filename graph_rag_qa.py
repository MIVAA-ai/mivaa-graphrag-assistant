import logging
import re
import os # Added os import, might be needed indirectly
from typing import List, Dict, Optional, Any, Tuple # <<< Added Tuple
import json # Import json for formatting context

# *** ADDED MISSING IMPORTS FOR STANDALONE EXECUTION ***
from pathlib import Path
import configparser
import sys # Ensure sys is imported
import time # Import time for potential delays in fallback testing

# Use tomli if available (standardized as tomllib)
try:
    # Use tomllib for Python 3.11+
    import tomllib
except ImportError:
    try:
        # Fallback to tomli for older Python versions
        import tomli as tomllib
    except ImportError:
        # If neither is available, log an error or handle appropriately
        # For now, we'll assume one exists if config loading is attempted later
        pass


# Neo4j Imports
try: # <<< Added try/except for Neo4j import
    from neo4j import GraphDatabase, exceptions as neo4j_exceptions
except ImportError:
    raise ImportError("Neo4j Python driver not found. Please install: pip install neo4j")


# LLM Imports (Assuming you have a way to call your LLM)
# Replace with your actual LLM calling mechanism if different
try:
    # Example: Using a hypothetical llm_interface module
    from src.knowledge_graph.llm import call_llm, extract_json_from_text, QuotaError # <<< Added QuotaError import possibility
    print("[INFO] Using actual 'call_llm' function.")
except ImportError:
    print("[WARN] 'call_llm' not found. Using mock function for demonstration.")
    # Define QuotaError if not imported
    class QuotaError(Exception): pass
    # Fallback or error if your LLM interface isn't found
    def call_llm(*args, **kwargs):
        # Mock response for testing structure
        print("[WARN] Mock call_llm returning placeholder answer.")
        # Simulate different LLM responses for testing
        if "TEXT_TO_CYPHER_SYSTEM_PROMPT" in kwargs.get("system_prompt", ""): # Check if it's a cypher request
             question = kwargs.get("user_prompt", "")
             # Simulate Cypher generation failure sometimes
             if "depth" in question or "change" in question:
                 print("[WARN] Mock LLM: Simulating NO_QUERY_GENERATED")
                 return "NO_QUERY_GENERATED"
             elif "sidetracked" in question:
                  # Simulate correct query
                  print("[WARN] Mock LLM: Simulating specific Cypher query")
                  return """```cypher
                  MATCH (originalWell:Entity)-[r:SIDETRACKED_FROM]->(newWell:Entity)
                  WHERE toLower(newWell.name) CONTAINS '16/28-a22'
                  RETURN originalWell.name AS original_well
                  ```"""
             else:
                  # Simulate a generic query attempt (will likely fail syntax check later)
                  print("[WARN] Mock LLM: Simulating generic Cypher query")
                  return """```cypher
                  MATCH (e:Entity)-[r]-(t:Entity)
                  WHERE toLower(e.name) CONTAINS 'some_entity'
                  RETURN e.name as subject, r.original as predicate, t.name as object LIMIT 5
                  ```"""
        else:
             # Simulate Q&A synthesis
             print("[WARN] Mock LLM: Simulating Q&A synthesis")
             return "Placeholder answer based on provided context."

    def extract_json_from_text(text):
        # Basic fallback, replace with robust parsing if needed
        import json
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

# Vector DB Imports
try:
    import chromadb
    from chromadb.utils import embedding_functions
    from chromadb.api.types import EmbeddingFunction # Import type hint
except ImportError:
    raise ImportError("ChromaDB library not found. Please install: pip install chromadb")

# Embeddings Imports
try:
    # Using ChromaDB's helper is convenient if using SentenceTransformers
    # If using a different embedding provider, adjust accordingly
    # The actual SentenceTransformer import happens within embedding_functions if needed
    pass
except ImportError:
    # This catch might be redundant if SentenceTransformer is only used via embedding_functions
    raise ImportError("ERROR: 'sentence-transformers' library not found. Please install it: pip install sentence-transformers")

# Import Prompts (including the new Text-to-Cypher prompt)
try:
    # --- UPDATED: Import the corrected prompt constant ---
    from src.knowledge_graph.prompts import TEXT_TO_CYPHER_SYSTEM_PROMPT
except ImportError:
    print("[WARN] Could not import TEXT_TO_CYPHER_SYSTEM_PROMPT. Define it manually or check prompts.py.")
    # --- Use the corrected prompt definition ---
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


# --- Logger Setup ---
logger = logging.getLogger(__name__)
# Ensure logger is configured (basic setup if not configured elsewhere)
if not logger.handlers:
     # import sys # Make sure sys is imported for handler (already imported above)
     logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s')
     logger = logging.getLogger(__name__)
     # Add handler explicitly if basicConfig didn't add one (e.g., in some environments)
     if not logger.handlers:
          handler = logging.StreamHandler(sys.stdout)
          formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s')
          handler.setFormatter(formatter)
          logger.addHandler(handler)
          logger.propagate = False


# --- Main GraphRAG QA Class ---
class GraphRAGQA:
    """
    Handles GraphRAG Question Answering by combining Knowledge Graph context
    from Neo4j and Vector Search context from ChromaDB.
    Includes fallback to broader graph query if initial query fails.
    """

    def __init__(self, *, # Use keyword-only arguments for clarity
                 neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 llm_model: str, llm_api_key: str, llm_base_url: Optional[str] = None,
                 embedding_model_name: str,
                 chroma_path: str, collection_name: str,
                 llm_config_extra: Optional[Dict[str, Any]] = None):
        """ Initializes components. """
        logger.info(f"Initializing GraphRAGQA Engine...")
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url
        self.llm_config_extra = llm_config_extra or {}
        self.embedding_model_name = embedding_model_name

        # --- Initialize status flags ---
        self.is_neo4j_connected = False
        self.is_vector_search_enabled = False
        self.driver: Optional[GraphDatabase.driver] = None # Type hint for driver
        self.embedding_function: Optional[EmbeddingFunction] = None # Type hint
        self.chroma_client: Optional[chromadb.ClientAPI] = None # Type hint
        self.chroma_collection: Optional[chromadb.Collection] = None # Type hint

        # --- Neo4j Connection ---
        try:
            logger.info(f"GraphRAGQA: Connecting to Neo4j at {neo4j_uri}")
            self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            self.driver.verify_connectivity()
            self.is_neo4j_connected = True
            logger.info(f"GraphRAGQA: Successfully connected to Neo4j.")
        except Exception as e:
            logger.error(f"GraphRAGQA: Fatal - Could not connect to Neo4j: {e}", exc_info=True)
            self.is_neo4j_connected = False

        # --- Embedding Function Initialization ---
        try:
            logger.info(f"GraphRAGQA: Initializing embedding function for model: {self.embedding_model_name}")
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model_name
            )
            logger.info("GraphRAGQA: Embedding function initialized.")
        except Exception as e:
             logger.error(f"GraphRAGQA: Failed to initialize embedding function '{self.embedding_model_name}'. Vector search will be disabled. Error: {e}", exc_info=True)
             self.embedding_function = None
             self.is_vector_search_enabled = False

        # --- ChromaDB Connection & Collection ---
        if self.embedding_function:
            try:
                logger.info(f"GraphRAGQA: Initializing ChromaDB client at path: {chroma_path}")
                Path(chroma_path).mkdir(parents=True, exist_ok=True)
                self.chroma_client = chromadb.PersistentClient(path=chroma_path)
                logger.info("GraphRAGQA: ChromaDB client initialized.")
                try:
                    logger.info(f"GraphRAGQA: Getting or creating ChromaDB collection: {collection_name}")
                    self.chroma_collection = self.chroma_client.get_or_create_collection(
                        name=collection_name,
                        embedding_function=self.embedding_function
                    )
                    if self.chroma_collection:
                        self.is_vector_search_enabled = True
                        logger.info(f"GraphRAGQA: Successfully got/created ChromaDB collection '{collection_name}'. Count: {self.chroma_collection.count()}")
                    else:
                         logger.error(f"GraphRAGQA: Failed to get or create ChromaDB collection '{collection_name}' (returned None). Vector search disabled.")
                         self.is_vector_search_enabled = False
                except Exception as e:
                    logger.error(f"GraphRAGQA: Failed to get or create ChromaDB collection '{collection_name}'. Vector search will be disabled. Error: {e}", exc_info=True)
                    self.chroma_collection = None
                    self.is_vector_search_enabled = False
            except Exception as e:
                 logger.error(f"GraphRAGQA: Failed to initialize ChromaDB Persistent Client at path '{chroma_path}'. Vector search will be disabled. Error: {e}", exc_info=True)
                 self.chroma_client = None
                 self.chroma_collection = None
                 self.is_vector_search_enabled = False
        else:
             logger.warning("GraphRAGQA: Skipping ChromaDB initialization because embedding function failed to load.")
             self.is_vector_search_enabled = False

        self.llm_qna_config_base = {"model": llm_model, "api_key": llm_api_key, "base_url": llm_base_url}
        self.llm_qna_config_extra = llm_config_extra if llm_config_extra else {}

        logger.info(f"GraphRAGQA engine initialization complete. Neo4j Connected: {self.is_neo4j_connected}, Vector Search Enabled: {self.is_vector_search_enabled}")


    def is_ready(self) -> bool:
        """Check if the engine is ready for Q&A."""
        return self.is_neo4j_connected and bool(self.llm_qna_config_base.get('api_key'))

    def close(self):
        """Closes the Neo4j database driver connection."""
        if self.driver:
            try:
                self.driver.close()
                logger.info("GraphRAGQA: Closed Neo4j connection.")
                self.driver = None
            except Exception as e:
                logger.error("GraphRAGQA: Error closing Neo4j connection: %s", e, exc_info=True)

    def _generate_cypher_query(self, question: str) -> Optional[str]:
        """ Uses an LLM to generate a Cypher query based on the user's question. """
        logger.debug(f"Generating Cypher query for question: '{question}'")
        system_prompt = TEXT_TO_CYPHER_SYSTEM_PROMPT
        user_prompt = f"User Question: {question}"
        try:
            temp = self.llm_qna_config_extra.get("cypher_temperature", 0.0)
            max_tokens = self.llm_qna_config_extra.get("cypher_max_tokens", 500)
            response_text = call_llm(
                model=self.llm_qna_config_base['model'],
                user_prompt=user_prompt,
                api_key=self.llm_qna_config_base['api_key'],
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temp,
                base_url=self.llm_qna_config_base['base_url']
            )
            if not response_text: return None
            cypher_match = re.search(r"```cypher\s*([\s\S]+?)\s*```", response_text, re.IGNORECASE)
            if cypher_match:
                generated_query = cypher_match.group(1).strip()
                logger.info(f"LLM generated Cypher query:\n{generated_query}")
                # Basic validation: check for MATCH and RETURN
                if "MATCH" in generated_query.upper() and "RETURN" in generated_query.upper():
                    return generated_query
                else:
                    logger.warning(f"Generated query seems invalid (missing MATCH or RETURN): {generated_query}")
                    return None
            elif "NO_QUERY_GENERATED" in response_text:
                logger.info("LLM indicated no suitable Cypher query could be generated.")
                return None
            else:
                logger.warning(f"LLM response did not contain a valid Cypher code block or 'NO_QUERY_GENERATED'. Response: {response_text}")
                return None
        except Exception as e:
            logger.error(f"Error during LLM call for Cypher generation: {e}", exc_info=True)
            return None

    # --- ADDED: Simple Entity Extractor ---
    def _extract_main_entity(self, question: str) -> Optional[str]:
        """
        Very basic attempt to extract the main entity name from the question.
        Looks for patterns like "well XXX", "project YYY", or capitalized words.
        Returns the extracted entity name in lowercase or None.
        """
        # Try specific patterns first (more reliable)
        patterns = [
            r"well\s+([\w\d#/\-.:]+)",       # Matches "well 16/28-A22", "well B-12#13" etc.
            r"project\s+([\w\d\s\-]+)",     # Matches "project LCD Phase III"
            r"field\s+([\w\d\s\-]+)",       # Matches "field Andrew"
            r"formation\s+([\w\d\s\-]+)",   # Matches "formation Palaeocene Sands"
            r"platform\s+([\w\d\s\-]+)",    # Matches "platform Andrew"
            r"company\s+([\w\d\s\.\-]+)",   # Matches "company Reliance Industries Ltd."
        ]
        for pattern in patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                # Extract, lower, and strip extra whitespace
                entity = ' '.join(match.group(1).split()).lower()
                logger.info(f"Extracted entity using pattern '{pattern}': {entity}")
                return entity

        # Fallback: Look for quoted strings
        quoted_match = re.search(r'["\']([^"\']+)["\']', question)
        if quoted_match:
            entity = quoted_match.group(1).strip().lower()
            logger.info(f"Extracted entity using quotes: {entity}")
            return entity

        # Fallback: Look for capitalized words (less reliable)
        cap_words = re.findall(r'\b[A-Z][a-zA-Z0-9#/\-.:]+\b', question)
        cap_words = [w for w in cap_words if w not in ["What", "Who", "Where", "When", "Why", "How", "Is", "Are", "The", "List", "Tell"]]
        if cap_words:
             # Try to join consecutive capitalized words
             # Example: "Low Cost Drilling" -> "low cost drilling"
             potential_entities = []
             current_entity = []
             for word in cap_words:
                 # Check if the word starts with a capital letter (handles cases like Andrew)
                 if word[0].isupper():
                     current_entity.append(word)
                 else:
                     if current_entity:
                         potential_entities.append(" ".join(current_entity).lower())
                     current_entity = [] # Reset if non-capitalized word encountered
             if current_entity: # Add the last sequence if any
                 potential_entities.append(" ".join(current_entity).lower())

             # Return the longest potential entity found
             if potential_entities:
                  entity = max(potential_entities, key=len)
                  logger.info(f"Extracted entity using capitalization fallback: {entity}")
                  return entity

        logger.warning(f"Could not reliably extract main entity from question: '{question}'")
        return None
    # --- End Entity Extractor ---

    def _query_neo4j(self, cypher_query: str, params: Optional[Dict]=None) -> List[Dict[str, Any]]:
        """Executes a Cypher query against Neo4j, optionally with parameters."""
        if not self.is_neo4j_connected or not self.driver:
            logger.warning("GraphRAGQA: Skipping Neo4j query (Neo4j not connected).")
            return []
        if not cypher_query:
            logger.warning("GraphRAGQA: Skipping Neo4j query (no query provided).")
            return []

        logger.info("GraphRAGQA: Executing Cypher query...")
        logger.debug(f"GraphRAGQA: Query:\n{cypher_query}\nParams: {params}")
        try:
            # Use parameters argument in execute_query
            records, summary, keys = self.driver.execute_query(
                cypher_query,
                parameters_=params, # Pass parameters dict here
                database_="neo4j"
            )
            logger.info("GraphRAGQA: Cypher query returned %d records.", len(records))
            return [record.data() for record in records]
        except neo4j_exceptions.CypherSyntaxError as e:
             logger.error(f"GraphRAGQA: Cypher syntax error: {e}\nQuery:\n{cypher_query}\nParams: {params}", exc_info=True)
             return [] # Return empty on syntax error
        except Exception as e:
            logger.error(f"GraphRAGQA: Cypher query execution failed! Query:\n{cypher_query}\nParams: {params}\nError: {e}", exc_info=True)
            return [] # Return empty on other errors

    def _query_vector_db(self, question: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Queries the ChromaDB vector store."""
        if not self.is_vector_search_enabled or not self.embedding_function or not self.chroma_collection:
            logger.warning("GraphRAGQA: Vector search skipped (not enabled or components missing).")
            return []

        logger.info("GraphRAGQA: Performing vector search for question (top %d)...", top_k)
        try:
            results = self.chroma_collection.query(
                query_texts=[question],
                n_results=top_k,
                include=['documents', 'distances', 'metadatas']
            )
            logger.debug(f"GraphRAGQA: Raw ChromaDB query results: {results}")

            formatted_results = []
            if results and results.get('ids') and results['ids'][0]:
                ids = results['ids'][0]
                documents = results.get('documents', [[]])[0]
                distances = results.get('distances', [[]])[0]
                metadatas = results.get('metadatas', [[]])[0]

                num_results = len(ids)
                # Ensure lists have the expected length, padding with defaults if necessary
                documents = documents + ['[Error retrieving document]'] * (num_results - len(documents)) if len(documents) < num_results else documents[:num_results]
                distances = distances + [-1.0] * (num_results - len(distances)) if len(distances) < num_results else distances[:num_results]
                metadatas = metadatas + [{}] * (num_results - len(metadatas)) if len(metadatas) < num_results else metadatas[:num_results]


                for i in range(num_results):
                    formatted_results.append({
                        "text": documents[i],
                        "metadata": metadatas[i],
                        "distance": distances[i]
                    })
                logger.info("GraphRAGQA: Vector search returned %d formatted results.", len(formatted_results))
            else:
                logger.info("GraphRAGQA: Vector search returned no results.")
            return formatted_results
        except Exception as e:
            logger.error("GraphRAGQA: Error during vector database query: %s", e, exc_info=True)
            return []

    def _format_context(self, graph_results: List[Dict], vector_results: List[Dict]) -> str:
        """Formats graph and vector results into a combined context string for the LLM."""
        context_parts = []
        # Format Vector Results first (often more detailed)
        if vector_results:
            vector_context_str = "Relevant Text Snippets (Vector Search):\n---\n"
            vector_context_str += "\n---\n".join([
                 f"Source Document: {chunk.get('metadata', {}).get('source_document', 'Unknown')}\n"
                 f"Content: {chunk.get('text', '[No text found]')}"
                 for chunk in vector_results
            ])
            vector_context_str += "\n---"
            context_parts.append(vector_context_str)

        # Format Graph Results
        if graph_results:
            graph_context_str = "Knowledge Graph Facts:\n---\n"
            graph_facts_list = []
            max_graph_records = 15 # Limit number of facts shown
            seen_facts = set()
            for i, record in enumerate(graph_results):
                if len(graph_facts_list) >= max_graph_records:
                    logger.warning("GraphRAGQA: Truncating graph facts in context (%d max).", max_graph_records)
                    break
                # Try to format as S-P-O if possible from query aliases
                subj = record.get('subject', record.get('e1.name', record.get('e.name', '?')))
                pred_type = record.get('type', '?') # Get relationship type if returned
                pred_orig = record.get('predicate', '?') # Get original predicate if returned
                pred = pred_orig if pred_orig != '?' else pred_type # Prefer original predicate text
                obj = record.get('object', record.get('related.name', record.get('related_entity', '?')))

                # Handle single value returns
                if len(record) == 1:
                     key, value = list(record.items())[0]
                     fact_str = f"- {key}: {value}"
                # Handle S-P-O returns (prefer original predicate)
                elif subj != '?' and pred != '?' and obj != '?':
                     fact_str = f"- {subj} -[{pred}]-> {obj}"
                # Fallback for other structures
                else:
                     fact_str = "- " + ", ".join([f"{k}: {v}" for k, v in record.items()])

                fact_tuple = tuple(sorted(record.items())) # Deduplicate based on full record content
                if fact_tuple in seen_facts: continue
                seen_facts.add(fact_tuple)
                graph_facts_list.append(fact_str)

            if graph_facts_list:
                 graph_context_str += "\n".join(graph_facts_list)
            else:
                 graph_context_str += "No specific facts found." # Message if list is empty after loop/dedup
            graph_context_str += "\n---"
            context_parts.append(graph_context_str)
        # --- MODIFIED: Only add this section if graph search was actually attempted ---
        elif graph_results is not None: # Check if graph_results is not None (meaning search was attempted)
             context_parts.append("Knowledge Graph Facts:\n---\nNo relevant facts found or query failed/not generated.\n---")


        if not context_parts or all(("No relevant" in part or "No specific facts found" in part) and "Knowledge Graph Facts" in part for part in context_parts):
            # Check if the *only* context is the negative graph context message
            if len(context_parts) == 1 and "Knowledge Graph Facts" in context_parts[0]:
                 return "No relevant context found in knowledge graph or vector search."
            # If vector results exist, use them even if graph failed
            elif any("Relevant Text Snippets" in part for part in context_parts):
                pass # Allow synthesis from vector context alone
            else: # Should not happen if vector search always runs, but as safety
                 return "No relevant context found in knowledge graph or vector search."


        return "\n\n".join(context_parts).strip()

    def _synthesize_answer(self, query: str, context: str, context_chunks: List[Dict]) -> Dict[str, Any]:
        """ Generates the final answer using an LLM, based on the query and formatted context. """
        logger.info("GraphRAGQA: Generating final answer using LLM with combined context...")
        final_system_prompt = """You are a helpful assistant answering questions based *only* on the provided context.
        The context may include 'Knowledge Graph Context' (structured facts) and 'Relevant Text Snippets' (unstructured text).
        1. First, check if the 'Knowledge Graph Context' directly answers the user's question about the entities mentioned. If they do, synthesize your answer primarily from these facts.
        2. Use the 'Relevant Text Snippets' ONLY to add supporting details or context IF they are clearly related to the entities and relationships already identified from the graph facts or the user's question.
        3. **CRITICAL:** Ignore any information in the 'Relevant Text Snippets' that seems unrelated to the core entities or the specific question asked, even if the snippets were retrieved. Do not merge unrelated topics.
        4. If the answer cannot be confidently determined from the relevant parts of the context, state that you don't know or cannot answer based on the provided information. Do not make up information."""
        final_user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        try:
            temp = self.llm_qna_config_extra.get("qna_temperature", 0.1)
            max_tokens = self.llm_qna_config_extra.get("qna_max_tokens", 500)
            answer_text = call_llm(
                model=self.llm_qna_config_base['model'],
                user_prompt=final_user_prompt,
                api_key=self.llm_qna_config_base['api_key'],
                system_prompt=final_system_prompt,
                temperature=temp,
                max_tokens=max_tokens,
                base_url=self.llm_qna_config_base['base_url']
            )
            logger.info("GraphRAGQA: Successfully generated final answer.")
            return {
                "answer": answer_text.strip() if answer_text else "Sorry, I could not generate an answer.",
                "sources": context_chunks
            }
        except Exception as e:
            logger.error(f"GraphRAGQA: Error during final LLM call for answer generation: {e}", exc_info=True)
            return {
                 "answer": f"Sorry, I encountered an error generating the answer: {e}",
                 "sources": context_chunks
            }

    # --- MODIFIED: answer_question with Fallback Logic ---
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answers a user's question using a combination of Graph and Vector retrieval,
        with a fallback to a broader graph query if the initial one fails.

        Returns:
            A dictionary containing the answer and the sources used.
        """
        logger.info(f"--- GraphRAGQA: Starting RAG pipeline for question: {question} ---")
        if not self.is_ready():
            logger.error("GraphRAGQA: Engine not ready (Neo4j disconnected or LLM key missing).")
            return {"answer": "Error: Backend connection not ready. Please check system status.", "sources": []}

        # 1. Generate Specific Cypher Query
        cypher_query = self._generate_cypher_query(question)
        graph_results = None # Initialize as None to distinguish from empty list []
        graph_query_failed = True # Assume failure initially

        if cypher_query:
            # 2. Execute Specific Cypher Query
            graph_results = self._query_neo4j(cypher_query) # Returns [] on error or no results
            if graph_results: # Check if list is not empty
                 graph_query_failed = False # Mark as success
                 logger.info("Specific Cypher query yielded results.")
            else:
                 logger.warning("Specific Cypher query returned 0 records.")
                 # Keep graph_query_failed = True
        else:
            logger.warning("GraphRAGQA: Could not generate specific Cypher query.")
            # Keep graph_query_failed = True
            graph_results = [] # Ensure graph_results is an empty list if query generation failed

        # --- ADDED: Fallback Graph Query Logic ---
        if graph_query_failed:
            logger.info("Attempting fallback broad graph query...")
            # Use the simple entity extractor
            entity_name = self._extract_main_entity(question)
            if entity_name:
                # Construct a broad query for the extracted entity
                # Using CONTAINS for robustness, limiting results
                # Using parameters to avoid injection issues
                fallback_query = """
                MATCH (e:Entity)-[r]-(t:Entity)
                WHERE toLower(e.name) CONTAINS $entity_name_lc
                RETURN e.name AS subject, type(r) AS type, r.original AS predicate, t.name AS object
                LIMIT 25
                """
                fallback_params = {"entity_name_lc": entity_name.lower()} # Pass entity name as parameter
                logger.info(f"Executing fallback query for entity '{entity_name}'...")
                # Execute using parameters
                graph_results = self._query_neo4j(fallback_query, params=fallback_params) # Overwrite graph_results
                if graph_results:
                    logger.info(f"Fallback graph query returned {len(graph_results)} results.")
                else:
                    logger.info(f"Fallback graph query for entity '{entity_name}' returned 0 results.")
                    # Keep graph_results as empty list []
            else:
                logger.warning("Could not extract entity for fallback query.")
                graph_results = [] # Ensure graph_results is empty list if no entity found
        # --- End Fallback Logic ---


        # 3. Perform Vector Search (Always perform this)
        vector_search_top_k = self.llm_qna_config_extra.get("vector_search_top_k", 5)
        similar_chunks = self._query_vector_db(question, top_k=vector_search_top_k)

        # 4. Format Combined Context (using potentially updated graph_results)
        # graph_results will be None only if Neo4j connection failed entirely at init
        # Otherwise it will be [] if specific and fallback queries failed/returned nothing
        context_str = self._format_context(graph_results if graph_results is not None else [], similar_chunks)
        logger.debug("GraphRAGQA: Combined Context for LLM:\n%s", context_str)

        # 5. Handle No Context Found
        if not context_str or context_str == "No relevant context found in knowledge graph or vector search.":
            logger.warning("GraphRAGQA: No context retrieved from graph or vector store for the question.")
            return {
                "answer": "Sorry, I could not find relevant information in the processed documents to answer your question.",
                "sources": []
            }

        # 6. Generate Final Answer with LLM
        # Pass the original similar_chunks list for source tracking in the final answer
        answer_dict = self._synthesize_answer(question, context_str, similar_chunks)

        return answer_dict


# --- Example Usage (main block) ---
# (Keep the original __main__ block for standalone testing)
if __name__ == "__main__":
    print("--- GraphRAGQA Script Start (Combined Vector/Graph) ---")

    # --- Configuration Loading ---
    llm_extra_params = {}
    embedding_model_name = "all-MiniLM-L6-v2" # Default
    chroma_path = "./chroma_db_embeddings" # Default
    collection_name = "doc_pipeline_embeddings" # Default
    try:
        # Load Neo4j config
        config_path = Path("graph_config.ini")
        if not config_path.is_file(): raise FileNotFoundError(f"Neo4j config not found: {config_path}")
        neo4j_config = configparser.ConfigParser()
        neo4j_config.read(config_path)
        uri = neo4j_config.get("neo4j", "uri", fallback=None)
        user = neo4j_config.get("neo4j", "user", fallback=None)
        password = neo4j_config.get("neo4j", "password", fallback=None)
        chroma_path = neo4j_config.get("vector_db", "chroma_path", fallback=chroma_path)
        collection_name = neo4j_config.get("vector_db", "collection_name", fallback=collection_name)

        # Load LLM config
        toml_config_path = Path("config.toml")
        if not toml_config_path.is_file(): raise FileNotFoundError(f"LLM config not found: {toml_config_path}")
        with open(toml_config_path, 'rb') as f:
             try:
                 import tomllib
                 config_toml = tomllib.load(f)
             except ImportError:
                 with open(toml_config_path, 'r') as f_text:
                     import toml
                     config_toml = toml.load(f_text)

        llm_config = config_toml.get("llm", {})
        llm_model = llm_config.get("model", None) # QnA model
        llm_api_key = llm_config.get("api_key", None)
        llm_base_url = llm_config.get("base_url", None)
        llm_extra_params = llm_config.get("parameters", {})
        embedding_model_name = config_toml.get("embeddings", {}).get("model_name", embedding_model_name)

        if not all([uri, user, password, llm_model, llm_api_key]):
             missing = [k for k,v in locals().items() if v is None and k in ['uri','user','password','llm_model','llm_api_key']]
             raise ValueError(f"Missing required configuration values: {missing}")

        print("[INFO] Configuration loaded successfully.")
        print(f"[INFO] Using Embedding Model: {embedding_model_name}")
        print(f"[INFO] Using Chroma Path: {chroma_path}")
        print(f"[INFO] Using Chroma Collection: {collection_name}")

    except Exception as e:
        print(f"ERROR loading configuration: {e}")
        logger.exception("Fatal error loading configuration.")
        sys.exit(1)

    # --- Initialize and Run QA Engine ---
    print("\n--- Initializing GraphRAG QA Engine (with Vector Search) ---")
    qa_engine_instance = None
    try:
        qa_engine_instance = GraphRAGQA(
            neo4j_uri=uri, neo4j_user=user, neo4j_password=password,
            llm_model=llm_model, llm_api_key=llm_api_key, llm_base_url=llm_base_url,
            embedding_model_name=embedding_model_name,
            chroma_path=chroma_path,
            collection_name=collection_name,
            llm_config_extra=llm_extra_params
        )

        if not qa_engine_instance.is_ready():
             print("\nFATAL: QA Engine failed to initialize correctly (check logs). Exiting.")
             sys.exit(1)

        print("✅ QA engine ready.")
        print("\n🔍 Ask questions (type 'exit' or 'quit' to stop):")

        while True:
            try:
                question = input("❓ Your Question: ").strip()
                if not question: continue
                if question.lower() in {"exit", "quit"}: break
                print("Processing...")
                # --- MODIFIED: Handle dictionary response ---
                response_dict = qa_engine_instance.answer_question(question)
                print(f"\n💡 Answer:\n{response_dict['answer']}\n")
                if response_dict.get("sources"):
                    print("--- Sources ---")
                    for i, src in enumerate(response_dict["sources"]):
                        src_doc = src.get("metadata", {}).get("source_document", "Unknown")
                        print(f"[{i+1}] From: {src_doc}")
                        print(f"   Text: {src.get('text', '')[:150]}...") # Show snippet
                    print("---------------\n")

            except EOFError: break
            except KeyboardInterrupt: break
        print("\nExiting interactive session.")

    except ConnectionError as e:
         print(f"\nFATAL: Could not initialize QA Engine (Neo4j Connection Failed): {e}")
         sys.exit(1)
    except Exception as e:
        logger.exception("❌ An unexpected error occurred during QA engine operation.")
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)
    finally:
        # --- Ensure Neo4j driver is closed ---
        if qa_engine_instance:
            qa_engine_instance.close()

    print("\n--- GraphRAGQA Script End ---")
