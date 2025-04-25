import streamlit as st
import base64
import tempfile
import os
import io
import logging
import re
import json
from pathlib import Path
import configparser
import sys
import time # Import time for delays and timing
import shutil # Import shutil for copying files
import hashlib # <<< IMPORTED FOR HASHING CHUNK IDs & FILES
import streamlit.components.v1 as components # <<< IMPORT FOR HTML DISPLAY
from datetime import date # <<< IMPORT FOR DAILY LOGGING (Though logging function will be commented)
import requests # <<< IMPORT requests library

# Use tomli if available (standardized as tomllib)
try:
    # Use tomllib for Python 3.11+
    import tomllib
except ImportError:
    try:
        # Fallback to tomli for older Python versions
        import tomli as tomllib
    except ImportError:
        st.error("No TOML parser found. Please install 'tomli' or use Python 3.11+.")
        sys.exit(1)

from typing import List, Dict, Optional, Any, Set, Tuple

# --- Core Logic Imports ---
# Assuming these files are in the same directory or accessible via PYTHONPATH
try:
    # Use the fixed exporter
    from neo4j_exporter import Neo4jExporter # Assuming neo4j_exporter_fixed.py is saved as neo4j_exporter.py
    from graph_rag_qa import GraphRAGQA
    # --- Import the UPDATED text_utils ---
    from src.knowledge_graph.text_utils import chunk_text
    # --- Import the MODIFIED llm.py functions AND the custom Exception ---
    from src.knowledge_graph.llm import call_llm, extract_json_from_text, QuotaError # <<< IMPORT QuotaError
    # --- FIX: Corrected import path for entity_standardization ---
    from src.knowledge_graph.entity_standardization import standardize_entities, infer_relationships, limit_predicate_length # Assuming entity_standardization_fixed.py is saved as entity_standardization.py
    # Import the specific prompts
    from src.knowledge_graph.prompts import MAIN_SYSTEM_PROMPT # Import system prompt

    # --- REVISED MAIN_USER_PROMPT ---
    # Emphasizing mandatory fields and adding clarity
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
    # --- FIX: Corrected import path for visualization ---
    from src.knowledge_graph.visualization import visualize_knowledge_graph # Assuming visualization.py is accessible within src.knowledge_graph
except ImportError as e:
    st.error(
        f"Import Error: Failed to import core modules (e.g., Neo4jExporter, GraphRAGQA, src.*, entity_standardization, visualization). " # Added visualization
        f"Ensure the script is run from the project root directory or that the modules are in the Python path. Error: {e}"
    )
    # Basic logging for import errors
    logging.basicConfig(level=logging.ERROR)
    logging.error(f"Import Error: {e}", exc_info=True)
    # Attempt path adjustment (same as before)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if current_dir not in sys.path: sys.path.insert(0, current_dir)
    if parent_dir not in sys.path: sys.path.insert(0, parent_dir)
    st.warning("Attempted to adjust Python path. Please ensure project structure is correct if errors persist.")
    try:
        from neo4j_exporter import Neo4jExporter
        from graph_rag_qa import GraphRAGQA
        # --- Import the UPDATED text_utils ---
        from src.knowledge_graph.text_utils import chunk_text
        # --- Import the MODIFIED llm.py functions AND the custom Exception ---
        from src.knowledge_graph.llm import call_llm, extract_json_from_text, QuotaError # <<< IMPORT QuotaError
        # --- FIX: Corrected import path for entity_standardization (in retry block too) ---
        from src.knowledge_graph.entity_standardization import standardize_entities, infer_relationships, limit_predicate_length
        # Re-define MAIN_USER_PROMPT here if prompts.py is not found
        if 'MAIN_USER_PROMPT' not in locals():
             # --- REVISED MAIN_USER_PROMPT (Fallback) ---
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
        from src.knowledge_graph.prompts import MAIN_SYSTEM_PROMPT
        # --- FIX: Corrected import path for visualization (in retry block too) ---
        from src.knowledge_graph.visualization import visualize_knowledge_graph
        st.info("Retried imports successfully after adjusting path.")
    except ImportError as e2:
        st.error(f"Still failed to import modules after path adjustment. Error: {e2}")
        sys.exit(1)


# --- UI & OCR Imports ---
from PIL import Image

try:
    from mistralai import Mistral
    # from mistralai.models import OCRResponse # This specific model might not be needed directly if using client.ocr.process
except ImportError:
    st.error("MistralAI library not found. Please install: pip install mistralai")
    sys.exit(1)

# --- Vector DB & Embeddings Imports ---
try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    st.error("ChromaDB library not found. Please install: pip install chromadb")
    sys.exit(1)
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    st.error("ERROR: 'sentence-transformers' library not found. Please install it: pip install sentence-transformers")
    sys.exit(1)

# --- Neo4j Driver Import --- ADDED
try:
    import neo4j
except ImportError:
    st.error("Neo4j Python driver not found. Please install: pip install neo4j")
    sys.exit(1)

# --- Logger Setup ---
# Ensure logger is configured (using basic config for simplicity here if needed)
# Set level to DEBUG to see the new log messages
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s')
logger = logging.getLogger(__name__)
# Prevent duplicate handlers if run multiple times in some environments
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
# Ensure Streamlit doesn't interfere with logging level
st_logger = logging.getLogger('streamlit')
st_logger.setLevel(logging.WARNING)

# --- Constants ---
CACHE_DIR = Path("./graphrag_cache")
TRIPLE_CACHE_DIR = CACHE_DIR / "triples"
GRAPH_HTML_FILENAME = "graph_visualization.html" # Define filename for graph output
# TOKEN_LOG_FILE = Path("./token_usage_log.json") # Define token log filename # <<< COMMENTED OUT
CHAT_HISTORY_FILE = Path("./chat_history.json") # <<< ADDED: Path for chat history

# --- Configuration Loading Function (Copied from user upload) ---
@st.cache_data # Cache the loaded configuration dictionary
def load_config():
    """
    Loads configuration from files and environment variables with validation.
    Order of precedence: Environment Variables > config.toml > graph_config.ini > Defaults
    Returns the configuration as a dictionary, or None if there's an error.
    """
    config = {}
    logger.info("Loading configuration...")
    try:
        # 1. Load from graph_config.ini (primarily for Neo4j, fallback for Chroma)
        config_path = Path("graph_config.ini")
        if config_path.is_file():
            neo4j_config_parser = configparser.ConfigParser()
            neo4j_config_parser.read(config_path)
            if "neo4j" in neo4j_config_parser:
                config['NEO4J_URI'] = neo4j_config_parser.get("neo4j", "uri", fallback=None)
                config['NEO4J_USER'] = neo4j_config_parser.get("neo4j", "user", fallback=None)
                config['NEO4J_PASSWORD'] = neo4j_config_parser.get("neo4j", "password", fallback=None)
            if "vector_db" in neo4j_config_parser:
                config.setdefault('CHROMA_PERSIST_PATH', neo4j_config_parser.get("vector_db", "chroma_path", fallback=None))
                config.setdefault('COLLECTION_NAME', neo4j_config_parser.get("vector_db", "collection_name", fallback=None))
            logger.info("Loaded config from graph_config.ini")
        else:
            logger.warning("graph_config.ini not found.")

        # 2. Load from config.toml (overrides .ini if keys overlap, except Neo4j)
        toml_config_path = Path("config.toml")
        if toml_config_path.is_file():
            try:
                with open(toml_config_path, "rb") as f:
                    config_toml_data = dict(tomllib.load(f))
                logger.info("Loaded config from config.toml")

                llm_config = config_toml_data.get("llm", {})
                config['LLM_MODEL'] = llm_config.get("qna", {}).get("model", llm_config.get("model", config.get('LLM_MODEL')))
                config['LLM_API_KEY'] = llm_config.get("qna", {}).get("api_key", llm_config.get("api_key", config.get('LLM_API_KEY')))
                config['LLM_BASE_URL'] = llm_config.get("qna", {}).get("base_url", llm_config.get("base_url", config.get('LLM_BASE_URL')))
                config['LLM_EXTRA_PARAMS'] = llm_config.get("qna", {}).get("parameters", llm_config.get("parameters", config.get('LLM_EXTRA_PARAMS')))
                config['TRIPLE_EXTRACTION_LLM_MODEL'] = llm_config.get("triple_extraction", {}).get("model", config.get('LLM_MODEL'))
                config['TRIPLE_EXTRACTION_API_KEY'] = llm_config.get("triple_extraction", {}).get("api_key", config.get('LLM_API_KEY'))
                config['TRIPLE_EXTRACTION_BASE_URL'] = llm_config.get("triple_extraction", {}).get("base_url", config.get('LLM_BASE_URL'))
                config['TRIPLE_EXTRACTION_MAX_TOKENS'] = llm_config.get("triple_extraction", {}).get("max_tokens", 1500)
                config['TRIPLE_EXTRACTION_TEMPERATURE'] = llm_config.get("triple_extraction", {}).get("temperature", 0.2)
                config['MISTRAL_API_KEY'] = llm_config.get("ocr", {}).get("mistral_api_key", config.get('MISTRAL_API_KEY'))
                config['EMBEDDING_MODEL'] = config_toml_data.get("embeddings", {}).get("model_name", config.get('EMBEDDING_MODEL'))
                # --- CHUNKING CONFIG (Now using character-based) ---
                chunking_config = config_toml_data.get("chunking", {})
                config['CHUNK_SIZE'] = chunking_config.get("chunk_size_chars", 1000) # Default to 1000 chars
                config['CHUNK_OVERLAP'] = chunking_config.get("overlap_chars", 100) # Default to 100 chars overlap
                # --- End Chunking Config ---
                vector_db_config = config_toml_data.get("vector_db", {})
                config['CHROMA_PERSIST_PATH'] = vector_db_config.get('persist_directory', config.get('CHROMA_PERSIST_PATH'))
                config['COLLECTION_NAME'] = vector_db_config.get('collection_name', config.get('COLLECTION_NAME'))
                config['STANDARDIZATION_ENABLED'] = config_toml_data.get("standardization", {}).get("enabled", config.get('STANDARDIZATION_ENABLED'))
                config['INFERENCE_ENABLED'] = config_toml_data.get("inference", {}).get("enabled", config.get('INFERENCE_ENABLED'))
                # --- Caching Config ---
                config['CACHE_ENABLED'] = config_toml_data.get("caching", {}).get("enabled", True) # Default cache to True

                config['standardization'] = config_toml_data.get("standardization", {})
                config['inference'] = config_toml_data.get("inference", {})
                config['llm'] = llm_config

            except Exception as e:
                logger.error(f"Error processing config.toml: {e}", exc_info=True)
        else:
            logger.warning("config.toml not found.")

        # 3. Set defaults if still missing
        config.setdefault('NEO4J_URI', None)
        config.setdefault('NEO4J_USER', None)
        config.setdefault('NEO4J_PASSWORD', None)
        config.setdefault('LLM_MODEL', None)
        config.setdefault('LLM_API_KEY', None)
        config.setdefault('MISTRAL_API_KEY', None)
        config.setdefault('TRIPLE_EXTRACTION_LLM_MODEL', config.get('LLM_MODEL'))
        config.setdefault('TRIPLE_EXTRACTION_API_KEY', config.get('LLM_API_KEY'))
        config.setdefault('CHROMA_PERSIST_PATH', "./chroma_db_pipeline")
        config.setdefault('COLLECTION_NAME', "doc_pipeline_embeddings")
        config.setdefault('EMBEDDING_MODEL', "all-MiniLM-L6-v2")
        # --- Default Chunking (Chars) ---
        config.setdefault('CHUNK_SIZE', 1000)
        config.setdefault('CHUNK_OVERLAP', 100)
        # --- End Default Chunking ---
        config.setdefault('STANDARDIZATION_ENABLED', False)
        config.setdefault('INFERENCE_ENABLED', False)
        config.setdefault('CACHE_ENABLED', True) # Default cache to True
        config.setdefault('LLM_BASE_URL', None)
        config.setdefault('TRIPLE_EXTRACTION_BASE_URL', config.get('LLM_BASE_URL'))
        config.setdefault('LLM_EXTRA_PARAMS', {})
        config.setdefault('standardization', {})
        config.setdefault('inference', {})
        config.setdefault('llm', {})

        # 4. Override with Environment Variables
        config['NEO4J_URI'] = os.getenv('NEO4J_URI', config.get('NEO4J_URI'))
        config['NEO4J_USER'] = os.getenv('NEO4J_USER', config.get('NEO4J_USER'))
        config['NEO4J_PASSWORD'] = os.getenv('NEO4J_PASSWORD', config.get('NEO4J_PASSWORD'))
        config['LLM_API_KEY'] = os.getenv('LLM_API_KEY', os.getenv('GOOGLE_API_KEY', config.get('LLM_API_KEY')))
        config['TRIPLE_EXTRACTION_API_KEY'] = os.getenv('TRIPLE_EXTRACTION_API_KEY', config.get('TRIPLE_EXTRACTION_API_KEY', config.get('LLM_API_KEY')))
        config['MISTRAL_API_KEY'] = os.getenv('MISTRAL_API_KEY', config.get('MISTRAL_API_KEY'))
        config['LLM_MODEL'] = os.getenv('LLM_MODEL', config.get('LLM_MODEL'))
        config['TRIPLE_EXTRACTION_LLM_MODEL'] = os.getenv('TRIPLE_EXTRACTION_LLM_MODEL', config.get('TRIPLE_EXTRACTION_LLM_MODEL'))
        config['EMBEDDING_MODEL'] = os.getenv('EMBEDDING_MODEL', config.get('EMBEDDING_MODEL'))
        config['CHROMA_PERSIST_PATH'] = os.getenv('CHROMA_PERSIST_PATH', config.get('CHROMA_PERSIST_PATH'))
        config['COLLECTION_NAME'] = os.getenv('COLLECTION_NAME', config.get('COLLECTION_NAME'))
        config['LLM_BASE_URL'] = os.getenv('LLM_BASE_URL', config.get('LLM_BASE_URL'))
        config['TRIPLE_EXTRACTION_BASE_URL'] = os.getenv('TRIPLE_EXTRACTION_BASE_URL', config.get('TRIPLE_EXTRACTION_BASE_URL'))
        config['STANDARDIZATION_ENABLED'] = os.getenv('STANDARDIZATION_ENABLED', str(config.get('STANDARDIZATION_ENABLED', False))).lower() == 'true'
        config['INFERENCE_ENABLED'] = os.getenv('INFERENCE_ENABLED', str(config.get('INFERENCE_ENABLED', False))).lower() == 'true'
        config['CACHE_ENABLED'] = os.getenv('CACHE_ENABLED', str(config.get('CACHE_ENABLED', True))).lower() == 'true'

        # --- Convert CHUNK_SIZE and CHUNK_OVERLAP from env vars (which are strings) to int ---
        config['CHUNK_SIZE'] = int(os.getenv('CHUNK_SIZE_CHARS', config.get('CHUNK_SIZE', 1000))) # Use updated name if needed
        config['CHUNK_OVERLAP'] = int(os.getenv('CHUNK_OVERLAP_CHARS', config.get('CHUNK_OVERLAP', 100))) # Use updated name if needed
        # --- End Chunking Env Var Override ---


        # 5. Final Validation
        required_keys = [
            'NEO4J_URI', 'NEO4J_USER', 'NEO4J_PASSWORD',
            'LLM_MODEL', 'LLM_API_KEY',
            'TRIPLE_EXTRACTION_LLM_MODEL', 'TRIPLE_EXTRACTION_API_KEY',
            'EMBEDDING_MODEL', 'CHROMA_PERSIST_PATH', 'COLLECTION_NAME'
        ]
        missing_or_invalid_keys = []
        for k in required_keys:
            val = config.get(k)
            # Check for None OR empty string for critical string values
            if k in ['NEO4J_URI', 'NEO4J_USER', 'NEO4J_PASSWORD', 'LLM_API_KEY', 'TRIPLE_EXTRACTION_API_KEY', 'CHROMA_PERSIST_PATH', 'COLLECTION_NAME', 'LLM_MODEL', 'TRIPLE_EXTRACTION_LLM_MODEL', 'EMBEDDING_MODEL']:
                is_missing = val is None or (isinstance(val, str) and not val.strip())
            else: # For other types like numbers or booleans, None check might suffice
                 is_missing = val is None
            if is_missing:
                missing_or_invalid_keys.append(k)

        if missing_or_invalid_keys:
            unique_missing = sorted(list(set(missing_or_invalid_keys)))
            error_message = f"Missing or invalid required configuration/secrets: {', '.join(unique_missing)}"
            logger.error(error_message)
            return None

        logger.info(f"NEO4J_URI before return: '{config.get('NEO4J_URI')}' (Type: {type(config.get('NEO4J_URI'))})")
        logger.info("Configuration loaded and validated successfully.")
        logger.debug(f"Final Config - NEO4J_URI: {config.get('NEO4J_URI')}")
        logger.debug(f"Final Config - LLM_MODEL: {config.get('LLM_MODEL')}")
        logger.debug(f"Final Config - EMBEDDING_MODEL: {config.get('EMBEDDING_MODEL')}")
        logger.debug(f"Final Config - CHROMA_PERSIST_PATH: {config.get('CHROMA_PERSIST_PATH')}")
        logger.debug(f"Final Config - CACHE_ENABLED: {config.get('CACHE_ENABLED')}")
        logger.debug(f"Final Config - CHUNK_SIZE: {config.get('CHUNK_SIZE')}")
        logger.debug(f"Final Config - CHUNK_OVERLAP: {config.get('CHUNK_OVERLAP')}")


        return config

    except Exception as e:
        logger.exception("Critical error during configuration loading.")
        return None


# --- Resource Initialization (Cached) ---

# --- ADDED: Cached requests.Session ---
@st.cache_resource
def get_requests_session():
    """Creates and returns a requests.Session object."""
    logger.info("Initializing requests.Session resource...")
    session = requests.Session()
    # You could configure adapters here if needed (e.g., for retries)
    # adapter = requests.adapters.HTTPAdapter(pool_connections=50, pool_maxsize=50)
    # session.mount('http://', adapter)
    # session.mount('https://', adapter)
    logger.info("requests.Session resource initialized.")
    return session

@st.cache_resource
def get_mistral_client(api_key):
    """Initializes and returns a Mistral client."""
    if not api_key:
        logger.warning("Mistral API Key not provided. OCR will be disabled for PDF/Image files.")
        return None
    logger.info("Initializing Mistral client...")
    try:
        client = Mistral(api_key=api_key)
        logger.info("Mistral client initialized.")
        return client
    except Exception as e:
        logger.exception("Mistral client initialization failed.")
        st.error(f"Failed to initialize Mistral client: {e}") # Show error in UI
        return None

@st.cache_resource
def get_embedding_model(model_name):
    """Loads and returns a SentenceTransformer embedding model."""
    if not model_name:
        logger.error("Embedding model name not provided.")
        return None
    logger.info(f"Loading embedding model: {model_name}")
    try:
        model = SentenceTransformer(model_name, device=None)
        logger.info("Embedding model loaded.")
        return model
    except Exception as e:
        logger.exception(f"Failed to load embedding model '{model_name}'")
        st.error(f"Failed to load embedding model '{model_name}': {e}")
        return None

@st.cache_resource
def get_chroma_collection(chroma_path, collection_name, embedding_model_name):
    """Connects to ChromaDB, gets/creates collection, returns collection object."""
    if not all([chroma_path, collection_name, embedding_model_name]):
        logger.error("Missing ChromaDB path, collection name, or embedding model name.")
        return None
    logger.info(f"Initializing ChromaDB connection at {chroma_path} for collection: {collection_name}")
    try:
        Path(chroma_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Attempting to initialize PersistentClient at path: {chroma_path}")
        client = chromadb.PersistentClient(path=chroma_path)
        logger.info("PersistentClient initialized.")
        logger.info(f"Getting embedding function for model: {embedding_model_name}")
        chroma_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model_name)
        logger.info("Embedding function obtained.")
        logger.info(f"Getting or creating collection: {collection_name}")
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=chroma_ef,
            metadata={"hnsw:space": "cosine"} # Example metadata: Use cosine distance
        )
        logger.info(f"ChromaDB collection '{collection_name}' ready. Current items: {collection.count()}")
        return collection
    except Exception as e:
        logger.exception(f"Failed to initialize/get ChromaDB collection '{collection_name}' at path '{chroma_path}'")
        st.error(f"Failed to connect to or get/create ChromaDB collection '{collection_name}': {e}")
        return None

@st.cache_resource
def init_neo4j_exporter(uri, user, password):
    """Initializes and returns a Neo4jExporter instance."""
    if not all([uri, user, password]):
        logger.error("Missing Neo4j URI, user, or password for exporter.")
        return None
    logger.info("Initializing Neo4jExporter resource...")
    try:
        # Use the context manager for initialization if Neo4jExporter supports it
        # If not, standard instantiation is fine. Assuming standard here.
        exporter = Neo4jExporter(uri=uri, user=user, password=password)
        # Add a verification step if exporter doesn't do it internally
        # exporter.driver.verify_connectivity() # Example verification
        logger.info("Neo4jExporter resource initialized.")
        return exporter
    except ConnectionError as ce:
        logger.error(f"Neo4jExporter connection failed: {ce}")
        st.error(f"Failed to connect Neo4j Exporter: {ce}")
        return None
    except Exception as e:
        logger.exception("Neo4jExporter initialization failed.")
        st.error(f"Failed to initialize Neo4j Exporter: {e}")
        return None

@st.cache_resource
def load_qa_engine(config):
    """Initializes and returns the GraphRAGQA engine."""
    logger.info("Initializing GraphRAGQA Engine resource...")
    if not config:
         logger.error("Configuration is missing, cannot initialize GraphRAGQA engine.")
         return None
    try:
        engine = GraphRAGQA(
            neo4j_uri=config['NEO4J_URI'], neo4j_user=config['NEO4J_USER'], neo4j_password=config['NEO4J_PASSWORD'],
            llm_model=config['LLM_MODEL'], llm_api_key=config['LLM_API_KEY'], llm_base_url=config.get('LLM_BASE_URL'),
            embedding_model_name=config['EMBEDDING_MODEL'], chroma_path=config['CHROMA_PERSIST_PATH'], collection_name=config['COLLECTION_NAME'],
            llm_config_extra=config.get('LLM_EXTRA_PARAMS', {})
        )
        # The GraphRAGQA __init__ should handle internal checks now
        logger.info(f"GraphRAGQA Engine resource initialized. Neo4j Connected: {engine.is_neo4j_connected}, Vector Search Enabled: {engine.is_vector_search_enabled}")
        return engine
    except Exception as e:
        logger.exception("GraphRAGQA Engine initialization failed during instantiation.")
        st.error(f"Failed to initialize Q&A Engine: {e}")
        return None


# --- Helper Functions ---

# --- ADDED: Chat History Persistence Functions ---
def load_chat_history() -> List[Dict]:
    """Loads chat history from the JSON file."""
    if CHAT_HISTORY_FILE.is_file():
        try:
            with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
            if isinstance(history, list): # Basic validation
                logger.info(f"Loaded {len(history)} messages from {CHAT_HISTORY_FILE}")
                return history
            else:
                logger.warning(f"Chat history file {CHAT_HISTORY_FILE} does not contain a valid list. Starting fresh.")
                return []
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Error loading chat history file {CHAT_HISTORY_FILE}: {e}. Starting fresh.")
            return []
    return [] # Return empty list if file doesn't exist

def save_chat_history(messages: List[Dict]):
    """Saves the current chat history to the JSON file."""
    try:
        # Ensure complex objects (like sources) are serializable if they exist
        serializable_messages = []
        for msg in messages:
            new_msg = msg.copy()
            if "sources" in new_msg:
                # Assuming sources is a list of dicts with simple types or strings
                # If sources contain complex objects, more sophisticated serialization is needed
                pass # Keep sources as is if they are already JSON-serializable
            serializable_messages.append(new_msg)

        with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(serializable_messages, f, indent=2) # Use indent for readability
        logger.debug(f"Saved {len(serializable_messages)} messages to {CHAT_HISTORY_FILE}")
    except TypeError as te:
         logger.error(f"Error serializing chat history to JSON: {te}. History might not be fully saved.")
         st.warning(f"Could not save complete chat history due to unserializable data: {te}")
    except OSError as e:
        logger.error(f"Error saving chat history to {CHAT_HISTORY_FILE}: {e}")
        st.error(f"Could not save chat history: {e}") # Notify user
# --- End Chat History Functions ---


def display_pdf(file_path):
    """Displays PDF in Streamlit iframe."""
    try:
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
        # Ensure height is reasonable, consider making it dynamic or configurable
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except FileNotFoundError:
         logger.error(f"PDF file not found for display: {file_path}")
         st.error("Could not find the PDF file to display.")
    except Exception as e:
        logger.exception("Error displaying PDF.")
        st.error(f"Error displaying PDF: {e}")

# --- MODIFIED process_uploaded_file_ocr to return path ---
def process_uploaded_file_ocr(uploaded_file, mistral_client) -> Tuple[Optional[str], Optional[str]]:
    """
    Handles OCR processing for PDF or Image using Mistral.
    Returns tuple: (text_content, pdf_display_path)
    """
    ocr_text_content = None
    file_type = uploaded_file.type
    pdf_display_path_for_this_file = None # Path for this specific file

    # Using standard tempfile for simplicity here
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / uploaded_file.name
        try:
            # Write the uploaded file content to the temporary file
            with open(temp_path, "wb") as tmp:
                tmp.write(uploaded_file.getvalue())
            logger.info(f"Saved uploaded file to temporary path: {temp_path}")

            # Upload to Mistral
            logger.info(f"Uploading {file_type} ({temp_path.stat().st_size} bytes) to Mistral for OCR...")
            with open(temp_path, "rb") as file_obj:
                file_upload = mistral_client.files.upload(file={"file_name": uploaded_file.name, "content": file_obj}, purpose="ocr")
            signed_url = mistral_client.files.get_signed_url(file_id=file_upload.id).url
            logger.info(f"File uploaded (ID: {file_upload.id}), signed URL obtained.")

            # Prepare OCR input
            if file_type == "application/pdf":
                # Copy file for potential display later
                persistent_temp_dir = Path(tempfile.gettempdir()) / "graphrag_pdfs"
                persistent_temp_dir.mkdir(parents=True, exist_ok=True)
                persistent_pdf_path = persistent_temp_dir / uploaded_file.name
                shutil.copy(temp_path, persistent_pdf_path)
                pdf_display_path_for_this_file = str(persistent_pdf_path) # Store path
                ocr_input = {"document_url": signed_url}
            elif file_type in ["image/png", "image/jpeg", "image/jpg"]:
                 ocr_input = {"image_url": signed_url}
            else:
                st.error(f"Unsupported file type for OCR: {file_type}")
                return None, None # Return None for text and path

            # Perform OCR
            logger.info("Performing OCR via Mistral API...")
            ocr_response = mistral_client.ocr.process(document=ocr_input, model="mistral-ocr-latest")

            # Process response
            if ocr_response and ocr_response.pages:
                all_pages_text = [page.markdown for page in ocr_response.pages if page.markdown]
                ocr_text_content = "\n\n".join(all_pages_text).strip()
                logger.info(f"OCR successful. Extracted ~{len(ocr_text_content)} characters for {uploaded_file.name}.")
                if not ocr_text_content: st.warning(f"OCR completed for {uploaded_file.name}, but no text was extracted.")
            else:
                st.warning(f"OCR processing for {uploaded_file.name} did not return any content or pages.")

        except Exception as e:
            logger.error(f"Error during file processing or OCR for {uploaded_file.name}: {e}", exc_info=True)
            st.error(f"An error occurred during OCR for {uploaded_file.name}: {e}")
            ocr_text_content = None
            pdf_display_path_for_this_file = None # Ensure path is None on error

    # Return text content and the path to the (potentially) saved PDF
    return ocr_text_content, pdf_display_path_for_this_file

# --- Knowledge Graph Construction ---
# --- MODIFIED: Accept status_container AND requests_session ---
def extract_knowledge_graph(
    text_content: str,
    config: Dict,
    status_container,
    requests_session: requests.Session # ADDED session argument
) -> Tuple[List[Dict], List[str]]:
    """
    Extracts, standardizes, and infers knowledge graph triples from text.
    Uses a requests.Session for efficient LLM calls. Handles QuotaError.

    Args:
        text_content: The text content of the document.
        config: Configuration dictionary.
        status_container: The st.status object from the calling function.
        requests_session: The requests.Session object for API calls.

    Returns:
        Tuple containing the list of final triples and the list of original text chunks.
    """
    initial_triples = [] # Store initially extracted triples with chunk info
    text_chunks = []
    if not text_content:
        logger.warning("No text content provided for KG extraction.")
        return initial_triples, text_chunks

    # --- UPDATED: Get chunking parameters from config ---
    chunk_size_chars = config.get('CHUNK_SIZE', 1000) # Default 1000 chars
    overlap_chars = config.get('CHUNK_OVERLAP', 100) # Default 100 chars
    try:
        # --- UPDATED: Call chunk_text with character-based parameters ---
        text_chunks = chunk_text(
            text_content,
            chunk_size=chunk_size_chars,
            chunk_overlap=overlap_chars
            # separators=None, # Use default separators from text_utils
            # keep_separator=True # Default is True in text_utils
        )
        # --- UPDATED: Log message reflects character parameters ---
        logger.info(f"Split text into {len(text_chunks)} chunks (size={chunk_size_chars} chars, overlap={overlap_chars} chars).")
    except Exception as e:
        logger.error(f"Error during text chunking: {e}", exc_info=True)
        status_container.write("⚠️ Error during text chunking.")
        st.error(f"Failed to chunk text: {e}")
        return [], []

    # --- Use the passed-in status_container ---
    status_container.write("⚙️ Extracting Knowledge Graph...")


    extraction_model = config.get('TRIPLE_EXTRACTION_LLM_MODEL')
    extraction_api_key = config.get('TRIPLE_EXTRACTION_API_KEY')
    extraction_base_url = config.get('TRIPLE_EXTRACTION_BASE_URL')
    extraction_max_tokens = config.get('TRIPLE_EXTRACTION_MAX_TOKENS', 1500)
    extraction_temperature = config.get('TRIPLE_EXTRACTION_TEMPERATURE', 0.2)

    if not all([extraction_model, extraction_api_key, extraction_base_url]): # Added base_url check
        # Update the *outer* status container
        status_container.update(label="KG Extraction Error!", state="error", expanded=True)
        st.error("Missing configuration for Triple Extraction LLM (Model/API Key/Base URL). Cannot proceed.")
        logger.error("Missing config for Triple Extraction LLM (model/API key/Base URL).")
        return [], text_chunks

    # Use imported system prompt
    system_prompt = MAIN_SYSTEM_PROMPT
    total_extracted = 0
    max_retries = 2
    # --- MODIFIED: Different delays for different errors ---
    default_retry_delay = 5 # seconds for general errors
    # --- INCREASED QUOTA DELAY ---
    quota_retry_delay = 70 # seconds for quota errors (increased from 60) <<< INCREASED

    num_chunks = len(text_chunks)
    if num_chunks == 0:
        logger.warning("No chunks generated from text content.")
        status_container.write("⚠️ No text chunks found.")
        return [], []

    # --- LLM Extraction Loop ---
    status_container.write(f"⚙️ 1. Preparing to extract triples from {num_chunks} chunk(s)...") # Initial message
    overall_start_time = time.time() # Time the whole extraction process

    for i, chunk in enumerate(text_chunks):
        chunk_start_time = time.time() # Time each chunk
        chunk_label = f"⚙️ 1. Extracting triples from chunk {i + 1}/{num_chunks}..."
        status_container.write(chunk_label) # Update status in UI
        logger.info(f"Processing chunk {i + 1}/{num_chunks}...")

        # Construct user prompt using the imported base and the current chunk
        # Use the REVISED MAIN_USER_PROMPT defined globally
        user_prompt = MAIN_USER_PROMPT + f"\n```text\n{chunk}\n```\n" # Append chunk to base prompt

        attempt = 0
        success = False
        response_text = None # Initialize response_text
        while attempt < max_retries and not success:
            try:
                # --- Time the LLM call specifically ---
                llm_call_start_time = time.time()
                # --- MODIFIED: Pass the requests_session to call_llm ---
                response_text = call_llm(
                    model=extraction_model,
                    user_prompt=user_prompt,
                    api_key=extraction_api_key,
                    system_prompt=system_prompt,
                    max_tokens=extraction_max_tokens,
                    temperature=extraction_temperature,
                    base_url=extraction_base_url,
                    session=requests_session # Pass the session object
                )
                llm_call_duration = time.time() - llm_call_start_time
                logger.info(f"Chunk {i + 1}/{num_chunks}: call_llm duration: {llm_call_duration:.2f} seconds.")
                # --- (Conceptual token counting would go here) ---

                # --- Time the JSON extraction ---
                json_extract_start_time = time.time()
                chunk_results = extract_json_from_text(response_text)
                json_extract_duration = time.time() - json_extract_start_time
                # logger.debug(f"Chunk {i + 1}/{num_chunks}: extract_json_from_text duration: {json_extract_duration:.4f} seconds.") # Optional: log JSON time

                # --- Validate JSON structure ---
                if chunk_results and isinstance(chunk_results, list):
                    valid_chunk_triples = []
                    required_keys = {"subject", "subject_type", "predicate", "object", "object_type"}
                    for item_idx, item in enumerate(chunk_results):
                        if isinstance(item, dict):
                            missing_keys = required_keys - item.keys()
                            invalid_values = {k: item[k] for k in required_keys.intersection(item.keys()) if not isinstance(item[k], str) or not item[k].strip()}

                            if not missing_keys and not invalid_values:
                                # All keys present and have valid string values
                                chunk_hash = hashlib.sha256(chunk.encode('utf-8')).hexdigest()
                                item["chunk_id"] = chunk_hash
                                item["chunk_text"] = chunk.strip() # Store original chunk text
                                item["predicate"] = limit_predicate_length(item["predicate"]) # Enforce predicate length
                                valid_chunk_triples.append(item)
                            else:
                                # --- IMPROVED LOGGING ---
                                reason = []
                                if missing_keys: reason.append(f"missing keys: {missing_keys}")
                                if invalid_values: reason.append(f"invalid/empty values: {invalid_values}")
                                logger.warning(f"Invalid triple structure in chunk {i + 1}, item {item_idx+1} ({'; '.join(reason)}): {item}")
                        else:
                            logger.warning(f"Invalid item type (expected dict) in chunk {i + 1}, item {item_idx+1}: {item}")

                    initial_triples.extend(valid_chunk_triples)
                    total_extracted += len(valid_chunk_triples)
                    success = True # Mark chunk processing as successful (even if some triples were invalid)
                    logger.info(f"Chunk {i + 1}/{num_chunks}: Extracted {len(valid_chunk_triples)} valid triples.")

                elif chunk_results is None:
                    logger.warning(f"LLM response for chunk {i+1} did not contain valid JSON.")
                    success = True # Treat as success (no triples extracted), move to next chunk
                else:
                     logger.warning(f"No valid list of triples extracted from chunk {i + 1}. Response was: {response_text[:200]}...") # Log start of response
                     success = True # Treat as success (no triples extracted)

            # --- ADDED: Specific handling for QuotaError ---
            except QuotaError as qe:
                attempt += 1
                logger.error(f"Attempt {attempt}/{max_retries}: Quota/Rate Limit Error processing chunk {i + 1}: {qe}")
                status_container.write(f"⏳ Quota/Rate limit hit on chunk {i+1}. Waiting {quota_retry_delay}s before retry {attempt+1}/{max_retries}...")
                if attempt >= max_retries:
                    logger.error(f"Max retries reached for chunk {i + 1} due to quota limits. Skipping.")
                    status_container.write(f"⚠️ Failed to process chunk {i+1} after {max_retries} attempts (Quota Limit).")
                    break # Skip this chunk
                else:
                    time.sleep(quota_retry_delay) # Wait longer for quota errors

            except TimeoutError as e: # Catch specific timeout error from call_llm
                 attempt += 1
                 logger.error(f"Attempt {attempt}/{max_retries}: Timeout processing chunk {i + 1}: {e}")
                 status_container.write(f"⏳ Timeout on chunk {i+1}. Waiting {default_retry_delay}s before retry {attempt+1}/{max_retries}...")
                 if attempt >= max_retries:
                     logger.error(f"Max retries reached for chunk {i + 1} due to timeout. Skipping.")
                     status_container.write(f"⚠️ Failed to process chunk {i+1} after {max_retries} attempts (Timeout).")
                     break # Skip this chunk
                 else:
                     time.sleep(default_retry_delay) # Default shorter delay

            except Exception as e: # Catch other general errors
                attempt += 1
                logger.error(f"Attempt {attempt}/{max_retries}: Failed to process chunk {i + 1} with LLM: {e}", exc_info=True)
                status_container.write(f"⏳ Error on chunk {i+1}. Waiting {default_retry_delay}s before retry {attempt+1}/{max_retries}...")
                if attempt >= max_retries:
                    logger.error(f"Max retries reached for chunk {i + 1}. Skipping.")
                    status_container.write(f"⚠️ Failed to process chunk {i+1} after {max_retries} attempts (Error).")
                    break # Skip this chunk
                else:
                    time.sleep(default_retry_delay) # Default shorter delay
        # --- End of retry loop for a chunk ---
        chunk_duration = time.time() - chunk_start_time
        logger.info(f"Chunk {i + 1}/{num_chunks}: Total processing time: {chunk_duration:.2f} seconds.")

    # --- After processing all chunks ---
    overall_duration = time.time() - overall_start_time
    logger.info(f"Finished extracting triples from all {num_chunks} chunks. Total time: {overall_duration:.2f} seconds.")
    logger.info(f"Initial extraction complete. Total valid triples extracted: {total_extracted}")
    status_container.write(f"✅ 1. Extracted {total_extracted} initial triples.")

    # --- Post-Processing ---
    processed_triples = initial_triples

    # --- Standardization (Optional) ---
    if config.get('STANDARDIZATION_ENABLED', False) and processed_triples:
        status_container.write("⚙️ 2. Standardizing entities...")
        logger.info("Applying entity standardization...")
        try:
            # --- Pass session to standardization if it also calls LLM ---
            # Modify standardize_entities if needed to accept and use the session
            standardized_result = standardize_entities(processed_triples, config) # Potentially add: session=requests_session
            if standardized_result is not None: processed_triples = standardized_result
            status_container.write("✅ 2. Standardization complete.")
            logger.info("Entity standardization complete.")
        except Exception as e:
            logger.error(f"Error during entity standardization: {e}", exc_info=True)
            status_container.write("⚠️ 2. Standardization error.")
            st.warning("An error occurred during entity standardization.")
    else:
        status_container.write("ℹ️ 2. Skipping standardization.")
        logger.info("Skipping entity standardization (disabled or no triples).")

    # --- Inference (Optional) ---
    if config.get('INFERENCE_ENABLED', False) and processed_triples:
        status_container.write("⚙️ 3. Inferring relationships...")
        logger.info("Applying relationship inference...")
        try:
             # --- Pass session to inference if it also calls LLM ---
             # Modify infer_relationships if needed to accept and use the session
            inferred_result = infer_relationships(processed_triples, config) # Potentially add: session=requests_session
            if inferred_result is not None: processed_triples = inferred_result
            status_container.write("✅ 3. Inference complete.")
            logger.info("Relationship inference complete.")
        except Exception as e:
            logger.error(f"Error during relationship inference: {e}", exc_info=True)
            status_container.write("⚠️ 3. Inference error.")
            st.warning("An error occurred during relationship inference.")
    else:
        status_container.write("ℹ️ 3. Skipping inference.")
        logger.info("Skipping relationship inference (disabled or no triples).")

    final_triples = processed_triples
    logger.info(f"Final processed triple count: {len(final_triples)}")
    if final_triples: logger.debug(f"Final triple example before returning (check chunk_id): {final_triples[0]}")

    return final_triples, text_chunks # Return the final list


# --- MODIFIED store_chunks_and_embeddings signature ---
def store_chunks_and_embeddings(text_chunks: List[str], config, current_doc_name: str):
    """Generates embeddings for text chunks and stores/updates them in ChromaDB."""
    if not text_chunks:
        logger.warning(f"No text chunks provided for embedding for doc '{current_doc_name}'.")
        return False

    embedding_model = get_embedding_model(config['EMBEDDING_MODEL'])
    chroma_collection = get_chroma_collection(
        config['CHROMA_PERSIST_PATH'], config['COLLECTION_NAME'], config['EMBEDDING_MODEL']
    )

    if not embedding_model or not chroma_collection:
        st.error(f"Embedding model or ChromaDB collection not available. Cannot store embeddings for '{current_doc_name}'.")
        logger.error(f"Cannot store embeddings for {current_doc_name} - embedding model or Chroma collection missing.")
        return False

    chunk_ids = [hashlib.sha256(chunk.encode('utf-8')).hexdigest() for chunk in text_chunks]
    logger.debug(f"Generated chunk IDs (hashes) for embedding: {chunk_ids}") # Log the generated hashes

    logger.info(f"Generating embeddings for {len(text_chunks)} chunks from '{current_doc_name}'...")
    # Use st.status for embedding progress reporting
    with st.spinner(f"⚙️ Generating embeddings for {len(text_chunks)} chunks..."):
        try:
            embeddings = embedding_model.encode(
                text_chunks,
                show_progress_bar=False # Use Streamlit progress bar instead
                # batch_size=32 # Consider adding batch_size
            )
            embeddings_list = embeddings.tolist() if hasattr(embeddings, 'tolist') else list(embeddings)
        except Exception as e:
             logger.error(f"Failed to generate embeddings for {current_doc_name}: {e}", exc_info=True)
             st.error(f"Embedding generation failed for {current_doc_name}.")
             return False # Exit if embedding fails

    logger.info("Embeddings generated.")
    logger.info(f"Upserting {len(chunk_ids)} embeddings to ChromaDB collection '{config['COLLECTION_NAME']}'...")
    with st.spinner(f"💾 Storing {len(chunk_ids)} embeddings..."):
        try:
            metadatas = [{"source_document": current_doc_name, "original_chunk_index": i} for i in range(len(text_chunks))]
            chroma_collection.upsert(
                ids=chunk_ids,
                embeddings=embeddings_list,
                documents=text_chunks,
                metadatas=metadatas
            )
            logger.info(f"Successfully upserted embeddings for '{current_doc_name}' to ChromaDB.")
            count = chroma_collection.count()
            logger.info(f"Collection '{config['COLLECTION_NAME']}' now contains {count} items.")
            return True
        except Exception as e:
            st.error(f"Failed to store embeddings for '{current_doc_name}' in ChromaDB: {e}")
            logger.exception(f"Failed to store embeddings for {current_doc_name}")
            return False


# --- Caching Helper Functions ---
def get_file_hash(file_content: bytes) -> str:
    """Calculates the SHA256 hash of file content."""
    return hashlib.sha256(file_content).hexdigest()

def load_triples_from_cache(file_hash: str) -> Optional[Tuple[List[Dict], List[str]]]:
    """Loads triples and chunks from a cache file."""
    cache_file = TRIPLE_CACHE_DIR / f"{file_hash}.json"
    if cache_file.is_file():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Basic validation
            if isinstance(data, dict) and "triples" in data and "chunks" in data:
                 logger.info(f"Cache hit: Loaded triples from {cache_file}")
                 return data["triples"], data["chunks"]
            else:
                 logger.warning(f"Invalid cache file format: {cache_file}. Reprocessing.")
                 # Optionally delete invalid cache file: cache_file.unlink(missing_ok=True)
                 return None
        except Exception as e:
            logger.warning(f"Failed to load cache file {cache_file}: {e}. Reprocessing.")
             # Optionally delete invalid cache file: cache_file.unlink(missing_ok=True)
            return None
    return None

def save_triples_to_cache(file_hash: str, triples: List[Dict], chunks: List[str]):
    """Saves triples and chunks to a cache file."""
    try:
        TRIPLE_CACHE_DIR.mkdir(parents=True, exist_ok=True) # Ensure cache dir exists
        cache_file = TRIPLE_CACHE_DIR / f"{file_hash}.json"
        data_to_save = {"triples": triples, "chunks": chunks}
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2) # Use indent for readability
        logger.info(f"Saved triples to cache file: {cache_file}")
    except Exception as e:
        logger.error(f"Failed to save triples to cache file {cache_file}: {e}", exc_info=True)


# --- Token Logging Functions --- # <<< SECTION COMMENTED OUT START
# def load_token_log() -> Dict:
#     """Loads the token log from the JSON file."""
#     if TOKEN_LOG_FILE.is_file():
#         try:
#             with open(TOKEN_LOG_FILE, 'r', encoding='utf-8') as f:
#                 return json.load(f)
#         except (json.JSONDecodeError, OSError) as e:
#             logger.error(f"Error loading token log file {TOKEN_LOG_FILE}: {e}")
#             return {}
#     return {}

# def save_token_log(log_data: Dict):
#     """Saves the token log to the JSON file."""
#     try:
#         with open(TOKEN_LOG_FILE, 'w', encoding='utf-8') as f:
#             json.dump(log_data, f, indent=2)
#     except OSError as e:
#         logger.error(f"Error saving token log file {TOKEN_LOG_FILE}: {e}")

# def update_daily_token_log():
#     """Adds current session token counts to the daily log."""
#     today_str = str(date.today())
#     log_data = load_token_log()

#     # Get today's entry or initialize it
#     today_data = log_data.get(today_str, {"prompt_tokens": 0, "completion_tokens": 0})

#     # Add session totals to today's totals
#     today_data["prompt_tokens"] += st.session_state.get("session_prompt_tokens", 0)
#     today_data["completion_tokens"] += st.session_state.get("session_completion_tokens", 0)

#     # Update the log data and save
#     log_data[today_str] = today_data
#     save_token_log(log_data)
#     logger.info(f"Updated token log for {today_str}: Prompt={today_data['prompt_tokens']}, Completion={today_data['completion_tokens']}")

#     # Reset session counters after logging
#     st.session_state.session_prompt_tokens = 0
#     st.session_state.session_completion_tokens = 0
# --- Token Logging Functions --- # <<< SECTION COMMENTED OUT END


# --- Streamlit App Main Logic ---
def main():
    st.set_page_config(page_title="GraphRAG Assistant", page_icon="📄", layout="wide") # Added page icon

    # --- MOVED & CENTERED TITLE ---
    st.markdown("<h1 style='text-align: center;'>📄 Document GraphRAG Q&A Assistant</h1>", unsafe_allow_html=True)
    # st.title("📄 Document GraphRAG Q&A Assistant") # Removed original title call

    config = load_config()
    if not config:
        st.error("Fatal: Failed to load configuration. Application cannot start.")
        st.stop()

    # --- Ensure Neo4j config is loaded before initializing QA engine --- ADDED CHECK
    if not all([config.get('NEO4J_URI'), config.get('NEO4J_USER'), config.get('NEO4J_PASSWORD')]):
         st.error("Fatal: Missing Neo4j connection details in configuration. Cannot start.")
         st.stop()

    # --- Initialize resources ---
    mistral_client = get_mistral_client(config.get('MISTRAL_API_KEY'))
    qa_engine = load_qa_engine(config) # This should be loaded AFTER config validation
    requests_session = get_requests_session() # <<< GET CACHED SESSION >>>

    # --- Initialize session state ---
    # Load existing chat history if available
    if "messages" not in st.session_state:
        st.session_state.messages = load_chat_history()
    # Ensure other states are initialized
    st.session_state.setdefault("processing_complete", False)
    st.session_state.setdefault("processed_files", set())
    st.session_state.setdefault("pdf_paths", {})
    st.session_state.setdefault("graph_html_path", None)


    # --- Sidebar UI ---
    with st.sidebar:
        st.header("Document Processing")

        file_types = ["pdf", "png", "jpg", "jpeg", "txt"]
        if not mistral_client:
            st.warning("Mistral API Key missing. OCR for PDF/Image disabled. Only TXT upload available.")
            file_types = ["txt"]

        # Allow multiple files
        uploaded_files = st.file_uploader(
            "Upload Document(s)", type=file_types,
            key="file_uploader",
            accept_multiple_files=True, # <<< ALLOW MULTIPLE FILES
            help="Upload one or more documents to build the knowledge graph and enable Q&A."
        )

        # --- Processing Options Expander ---
        with st.expander("Processing Options", expanded=False):
            # Cache Toggle
            default_cache_state = config.get('CACHE_ENABLED', True)
            use_cache = st.toggle("Use Triple Cache", value=default_cache_state, key="use_cache_toggle",
                                  help="If enabled, saves extracted triples to disk and reuses them if the same file content is uploaded again.")
            # Add other options here if needed (e.g., chunk size override)


        # Process Button
        process_button_disabled = not uploaded_files # Disable if list is empty
        if st.button("Process Uploaded Document(s)", disabled=process_button_disabled, key="process_button",
                     use_container_width=True):
            if uploaded_files:
                # Reset state partially for this batch run
                st.session_state.processing_complete = False
                # Keep pdf_paths from previous runs unless explicitly cleared
                # st.session_state.pdf_paths = {} # Uncomment to clear PDF paths on each run
                # st.session_state.vector_inspection_results = None # Removed
                st.session_state.graph_html_path = None # Clear previous graph viz

                num_files = len(uploaded_files)
                st.info(f"Processing {num_files} file(s)... Cache {'ENABLED' if use_cache else 'DISABLED'}")
                any_file_succeeded = False
                all_files_succeeded = True
                all_processed_triples = [] # --- Collect triples across files ---

                # --- Overall Progress Bar ---
                overall_progress_bar = st.progress(0.0, text="Starting batch processing...")

                # --- Loop through uploaded files ---
                for i, uploaded_file in enumerate(uploaded_files):
                    file_name = uploaded_file.name
                    # Update overall progress
                    overall_progress = (i + 1) / num_files
                    overall_progress_bar.progress(overall_progress, text=f"Processing file {i+1}/{num_files}: {file_name}")

                    st.markdown(f"---")
                    # st.subheader(f"File {i+1}/{num_files}: `{file_name}`") # Subheader removed, using status label

                    # Use st.status to manage progress display for this file
                    with st.status(f"Processing `{file_name}`...", expanded=True) as status: # Outer status
                        file_success = True
                        text_content = None
                        original_text_chunks = []
                        extracted_triples = []
                        pdf_path_this_run = None
                        cache_hit = False

                        # === Pipeline Execution for EACH file ===

                        # Step 1: Get Text Content (and file hash for caching)
                        try:
                            status.update(label=f"`{file_name}`: 1/4 Reading file content...")
                            file_content_bytes = uploaded_file.getvalue() # Read content once
                            file_hash = get_file_hash(file_content_bytes) if use_cache else None
                            logger.info(f"Processing file '{file_name}' (Type: {uploaded_file.type}), Hash: {file_hash if file_hash else 'N/A'}")

                            status.update(label=f"`{file_name}`: 1/4 Extracting text...")
                            if uploaded_file.type == "text/plain":
                                text_content = file_content_bytes.decode('utf-8', errors='replace')
                                if not text_content: status.update(label=f"`{file_name}`: ⚠️ Text file empty/undecodable.", state="warning"); file_success = False
                                else: status.write("✅ Text loaded directly.")
                            elif uploaded_file.type in ["application/pdf", "image/png", "image/jpeg", "image/jpg"]:
                                if not mistral_client: status.update(label=f"`{file_name}`: ❌ Mistral client needed for OCR.", state="error"); file_success = False
                                else:
                                    text_content, pdf_path_this_run = process_uploaded_file_ocr(uploaded_file, mistral_client)
                                    if text_content: status.write("✅ OCR Complete.")
                                    else: status.update(label=f"`{file_name}`: ⚠️ OCR returned no text.", state="warning"); file_success = False
                                    # --- Store PDF Path ---
                                    if pdf_path_this_run: st.session_state.pdf_paths[file_name] = pdf_path_this_run
                            else: status.update(label=f"`{file_name}`: ❌ Unsupported type - {uploaded_file.type}", state="error"); file_success = False
                        except Exception as e:
                            logger.error(f"Error during text extraction for '{file_name}': {e}", exc_info=True)
                            status.update(label=f"`{file_name}`: ❌ Error reading/OCR.", state="error"); file_success = False

                        # Step 2: Knowledge Graph Construction (Check Cache First)
                        if file_success and text_content:
                            status.update(label=f"`{file_name}`: 2/4 Extracting Knowledge Graph...")
                            cached_data = None
                            if use_cache and file_hash: cached_data = load_triples_from_cache(file_hash)

                            if cached_data:
                                extracted_triples, original_text_chunks = cached_data
                                status.write(f"✅ KG Loaded from Cache ({len(extracted_triples)} triples).")
                                cache_hit = True
                            else:
                                try:
                                    # --- Pass the outer status container AND the session ---
                                    extracted_triples, original_text_chunks = extract_knowledge_graph(
                                        text_content,
                                        config,
                                        status,
                                        requests_session # Pass the cached session
                                    )
                                    if extracted_triples is not None:
                                        # Status update is now handled inside extract_knowledge_graph
                                        if use_cache and file_hash: save_triples_to_cache(file_hash, extracted_triples, original_text_chunks)
                                    else:
                                        # Handle case where extraction function returned None (major error)
                                        status.update(label=f"`{file_name}`: ❌ KG Extraction Failed (returned None).", state="error"); file_success = False
                                    extracted_triples = extracted_triples or [] # Ensure it's a list
                                except Exception as e:
                                    status.update(label=f"`{file_name}`: ❌ KG Extraction Error.", state="error"); logger.exception(f"KG Extraction Failed for {file_name}"); file_success = False
                        elif file_success and not text_content: status.update(label=f"`{file_name}`: ⚠️ No text content found.", state="warning"); file_success = False

                        # Step 3: Export Triples to Neo4j
                        if file_success and extracted_triples:
                            status.update(label=f"`{file_name}`: 3/4 Storing in Neo4j...")
                            try:
                                exporter = init_neo4j_exporter(config['NEO4J_URI'], config['NEO4J_USER'], config['NEO4J_PASSWORD'])
                                if exporter:
                                    if isinstance(extracted_triples, list) and extracted_triples: # Check if list is not empty
                                         # --- Pass entity types to exporter ---
                                         exporter.store_triples(extracted_triples) # Assumes exporter handles types
                                         status.write(f"✅ Stored/Updated {len(extracted_triples)} triples in Neo4j.")
                                    else: status.write("ℹ️ No valid triples extracted/found to store in Neo4j.")
                                else: status.update(label=f"`{file_name}`: ❌ Neo4j exporter init failed.", state="error"); file_success = False
                            except Exception as e:
                                status.update(label=f"`{file_name}`: ❌ Neo4j Export Error.", state="error"); logger.exception(f"Neo4j export failed for {file_name}"); file_success = False
                        elif file_success and not extracted_triples and cache_hit: status.write("ℹ️ No triples found in cache to store in Neo4j.")
                        elif file_success and not extracted_triples: status.write("ℹ️ No triples extracted to store in Neo4j.")

                        # Step 4: Create and Store Embeddings
                        if file_success and original_text_chunks:
                            status.update(label=f"`{file_name}`: 4/4 Storing Embeddings...")
                            try:
                                embedding_success = store_chunks_and_embeddings(original_text_chunks, config, file_name)
                                if embedding_success: status.write("✅ Embeddings Stored/Updated.")
                                else: status.update(label=f"`{file_name}`: ❌ Embedding Storage Failed.", state="error"); file_success = False
                            except Exception as e:
                                status.update(label=f"`{file_name}`: ❌ Embedding Storage Error.", state="error"); logger.exception(f"Vector storage failed for {file_name}"); file_success = False
                        elif file_success and not original_text_chunks:
                            status.write("ℹ️ No text chunks available to embed.")

                        # --- Update Final Status for this file ---
                        if file_success:
                            final_label = f"✅ Processing Complete for `{file_name}` {'(from cache)' if cache_hit else ''}"
                            status.update(label=final_label, state="complete", expanded=False)
                            st.session_state.processed_files.add(file_name)
                            # --- Collect triples for final visualization ---
                            if extracted_triples:
                                all_processed_triples.extend(extracted_triples)
                            any_file_succeeded = True
                        else:
                            status.update(label=f"❌ Processing Failed for `{file_name}`", state="error", expanded=True)
                            all_files_succeeded = False

                # --- Final Status Update for the Batch ---
                overall_progress_bar.empty() # Remove progress bar after loop
                st.markdown("---")
                final_status_placeholder = st.empty()
                if any_file_succeeded:
                    st.session_state.processing_complete = True # Mark processing as having run successfully at least once
                    if all_files_succeeded:
                        final_status_placeholder.success(f"✅ All {num_files} file(s) processed successfully. Ready for Q&A.")
                    else:
                        final_status_placeholder.warning(f"⚠️ Processed {num_files} file(s) with some errors. Q&A enabled based on successfully processed files.")

                    # --- Generate Graph Visualization ---
                    if all_processed_triples:
                         st.info("Generating graph visualization...")
                         try:
                             logger.info(f"Generating visualization for {len(all_processed_triples)} total triples.")
                             viz_stats = visualize_knowledge_graph(
                                 all_processed_triples,
                                 output_file=GRAPH_HTML_FILENAME,
                                 config=config
                             )
                             st.session_state.graph_html_path = GRAPH_HTML_FILENAME
                             logger.info(f"Graph visualization generated: {viz_stats}")
                             st.success("Graph visualization generated successfully.")
                         except Exception as e:
                             logger.error(f"Failed to generate graph visualization: {e}", exc_info=True)
                             st.error("Failed to generate graph visualization.")
                             st.session_state.graph_html_path = None
                    else:
                        logger.warning("No triples were extracted from any file, skipping visualization.")
                        st.session_state.graph_html_path = None

                else:
                    final_status_placeholder.error(f"❌ Failed to process any of the {num_files} file(s). Check status messages above and logs.")
                    st.session_state.processing_complete = False # Ensure this is False if nothing succeeded
                    st.session_state.pdf_paths = {} # Clear PDF paths if all failed
                    st.session_state.graph_html_path = None


        # --- Footer Expander for Status & Debug ---
        st.divider()
        with st.expander("System Status & Debugging", expanded=False):
            st.subheader("System Status")
            status_ok = True
            # Use the temporary connection check logic here as well for consistency
            neo4j_conn_status = False
            try:
                # Use the imported neo4j library
                driver = neo4j.GraphDatabase.driver(
                    config['NEO4J_URI'],
                    auth=(config['NEO4J_USER'], config['NEO4J_PASSWORD'])
                )
                driver.verify_connectivity()
                neo4j_conn_status = True
                driver.close() # Close the temporary driver
            except Exception as e:
                 logger.error(f"Sidebar Neo4j connection check failed: {e}")
                 neo4j_conn_status = False

            if neo4j_conn_status: st.success("Neo4j Connected")
            else: st.error("Neo4j Disconnected"); status_ok = False

            # Check vector search readiness (assuming qa_engine handles this check internally)
            if qa_engine and qa_engine.is_vector_search_enabled: st.success("Vector Search Ready")
            else: st.warning("Vector Search Disabled/Unavailable")

            if config.get('LLM_API_KEY'): st.success("Q&A LLM Key Loaded")
            else: st.error("Q&A LLM Key Missing"); status_ok = False
            if config.get('MISTRAL_API_KEY'): st.success("Mistral Key Loaded")
            else: st.warning("Mistral Key Missing (Needed for PDF/Image OCR)")
            if config.get('TRIPLE_EXTRACTION_API_KEY'): st.success("Triple Extract LLM Key Loaded")
            else: st.error("Triple Extract LLM Key Missing")
            if not status_ok: st.warning("One or more critical components failed to initialize or configuration is missing.")

            # --- Token Usage Display --- # <<< SECTION COMMENTED OUT START
            # st.subheader("Token Usage")
            # st.metric(label="Session Prompt Tokens", value=f"{st.session_state.get('session_prompt_tokens', 0):,}")
            # st.metric(label="Session Completion Tokens", value=f"{st.session_state.get('session_completion_tokens', 0):,}")
            # st.metric(label="Today's Total Prompt Tokens", value=f"{st.session_state.get('today_prompt_tokens', 0):,}")
            # st.metric(label="Today's Total Completion Tokens", value=f"{st.session_state.get('today_completion_tokens', 0):,}")
            # st.caption("Session tokens reset on page reload. Today's totals update when context is cleared.")
            # --- Token Usage Display --- # <<< SECTION COMMENTED OUT END


            st.subheader("Debugging Tools")
            # --- REMOVED: Inspect Vector Store Button ---
            # if st.button("Inspect Vector Store", key="inspect_vector_button_expander"):
            #     st.session_state.vector_inspection_results = "pending"
            #     st.rerun()

            # --- Clear Context Button ---
            if st.button("Clear Processed Context", key="clear_context_button", type="secondary",
                         help="Removes processed file context and clears chat history."):
                # --- Save current session tokens before clearing --- # <<< COMMENTED OUT
                # update_daily_token_log()
                # --- Clear state ---
                st.session_state.processed_files = set()
                st.session_state.messages = [] # Clear in-memory messages
                st.session_state.pdf_paths = {}
                st.session_state.processing_complete = False
                st.session_state.graph_html_path = None # Clear graph path
                # --- Clear Triple Cache ---
                if TRIPLE_CACHE_DIR.exists():
                     try:
                         shutil.rmtree(TRIPLE_CACHE_DIR)
                         st.success("Cleared triple cache.")
                         logger.info("Cleared triple cache directory.")
                     except Exception as e:
                         st.error(f"Could not clear triple cache: {e}")
                         logger.error(f"Error clearing triple cache: {e}", exc_info=True)
                # --- ADDED: Clear Chat History File ---
                if CHAT_HISTORY_FILE.exists():
                    try:
                         CHAT_HISTORY_FILE.unlink()
                         st.success("Cleared chat history file.")
                         logger.info("Cleared chat history file.")
                    except Exception as e:
                         st.error(f"Could not clear chat history file: {e}")
                         logger.error(f"Error clearing chat history file: {e}", exc_info=True)

                st.success("Context and chat history cleared.")
                time.sleep(1) # Give time to see message
                st.rerun() # Rerun to update UI


    # --- Main Area ---
    # Redesigned Layout: Left column for Chat, Right column for Context

    main_col, right_col = st.columns([2, 1]) # Keep 2:1 ratio or adjust as needed

    # --- Left Column: Chat Interface ---
    with main_col:
        st.header("Chat Interface") # Header for the chat column

        # --- Define chat readiness state *within this column's scope* ---
        is_engine_ready = qa_engine and qa_engine.is_ready()
        chat_disabled = not is_engine_ready # Define chat_disabled based on engine readiness

        # --- Display Context Message ---
        if is_engine_ready:
            if st.session_state.processed_files: # Files processed in *this* session
                processed_files_str = ", ".join(f"`{f}`" for f in sorted(list(st.session_state.processed_files)))
                st.info(f"**Context:** Q&A based on document(s) processed this session: {processed_files_str}")
            else: # Engine is ready, but no files processed this session
                # --- FIXED: Use temporary connection for DB counts check ---
                neo4j_count = 0
                chroma_count = 0
                driver = None # Initialize driver variable
                try:
                    # Check Neo4j count using a temporary connection
                    if config.get('NEO4J_URI') and config.get('NEO4J_USER') and config.get('NEO4J_PASSWORD'):
                        try:
                            # Use the imported neo4j library
                            driver = neo4j.GraphDatabase.driver(
                                config['NEO4J_URI'],
                                auth=(config['NEO4J_USER'], config['NEO4J_PASSWORD'])
                            )
                            with driver.session(database="neo4j") as session: # Specify database if needed
                                result = session.run("MATCH (n:Entity) RETURN count(n) as count") # Check for :Entity nodes
                                record = result.single()
                                if record:
                                    neo4j_count = record["count"]
                        except Exception as neo_e:
                             logger.warning(f"Context check: Neo4j query failed: {neo_e}") # Log specific Neo4j error
                        finally:
                            if driver:
                                driver.close() # Ensure driver is closed

                    # Check Chroma count (assuming qa_engine handles Chroma connection internally)
                    if qa_engine and qa_engine.is_vector_search_enabled:
                        try:
                            chroma_count = qa_engine.chroma_collection.count()
                        except Exception as chroma_e:
                            logger.warning(f"Context check: Chroma query failed: {chroma_e}")

                    # Display message based on counts
                    if neo4j_count > 0 or chroma_count > 0:
                        st.info("**Context:** Q&A based on previously processed documents stored in the database.")
                    else:
                        st.info("Upload and process document(s) using the sidebar to begin.")

                except Exception as e:
                    # Catch any broader errors during the check
                    logger.warning(f"Could not check database counts for context message: {e}", exc_info=True) # Log full traceback
                    st.info("Upload and process document(s) using the sidebar to begin.")
        else: # Engine not ready
             st.info("Initializing... Upload and process document(s) using the sidebar to begin.")

        # --- Chat History Display ---
        # This loop renders all messages from the state directly into the main column
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                # Display the main content (answer)
                st.markdown(message["content"])
                # --- Display sources if available ---
                if message["role"] == "assistant" and "sources" in message and message["sources"]:
                    with st.expander("Show Sources"):
                        for i, source in enumerate(message["sources"]):
                            # Try to get source document name from metadata
                            source_doc = "Unknown Document"
                            source_text = ""
                            if isinstance(source, dict):
                                source_text = source.get("text", "Source text not available.")
                                metadata = source.get("metadata", {})
                                source_doc = metadata.get("source_document", source_doc)
                            elif isinstance(source, str): # Handle case where sources are just strings
                                source_text = source

                            st.markdown(f"**Source {i+1} ({source_doc}):**")
                            st.markdown(f"> {source_text.replace('\n', '\n> ')}") # Blockquote format
                            st.markdown("---")
                # --- END: Display sources ---

        # --- Spacer to push input to the bottom (optional) ---
        # st.container() # Or use st.markdown("<br>"*N, unsafe_allow_html=True) if needed

        # --- Chat Input Widget (Placed at the end of the main column) ---
        # Use the chat_disabled variable defined at the start of this 'with' block
        chat_placeholder = "Ask a question..." if not chat_disabled else "System initializing or not ready..."
        if prompt := st.chat_input(chat_placeholder, disabled=chat_disabled, key="chat_input"):
            # 1. Append user message to state IMMEDIATELY
            st.session_state.messages.append({"role": "user", "content": prompt})
            # --- ADDED: Save history after user message ---
            save_chat_history(st.session_state.messages)

            # --- Check for generic greetings ---
            normalized_prompt = prompt.strip().lower()
            # List of common greetings/phrases to bypass RAG
            generic_greetings = [
                "hi", "hello", "hey",
                "how are you", "how are you?",
                "what's up", "whats up", "sup",
                "good morning", "good afternoon", "good evening",
                "thanks", "thank you" # Added thanks as well
            ]

            if normalized_prompt in generic_greetings:
                # Handle generic greeting directly without RAG
                if "how are you" in normalized_prompt:
                    response_content = "I'm just a program, but I'm ready to help! Ask me about your documents."
                elif "thank" in normalized_prompt:
                     response_content = "You're welcome!"
                else: # Default greeting response
                    response_content = "Hello! I'm the Document GraphRAG Assistant. How can I help you with the processed documents?"
                logger.info(f"Handling generic input directly: '{prompt}'")
                # --- MODIFIED: Add message without sources ---
                assistant_message = {"role": "assistant", "content": response_content}
                st.session_state.messages.append(assistant_message)
                save_chat_history(st.session_state.messages)
                st.rerun() # Rerun immediately for predefined responses
            else:
                # --- Proceed with RAG for non-generic questions ---
                # 2. Generate assistant response using RAG
                response_content = None
                response_sources = None # <<< Initialize sources variable
                # Show spinner *while generating response*
                with st.spinner("Thinking..."):
                    # Check engine readiness again just in case, though chat_disabled should prevent this if not ready
                    if qa_engine and qa_engine.is_ready():
                        try:
                            logger.info(f"Performing RAG query for: '{prompt}'")
                            # --- MODIFIED: Expect dict response ---
                            response_dict = qa_engine.answer_question(prompt)
                            if isinstance(response_dict, dict):
                                response_content = response_dict.get("answer")
                                response_sources = response_dict.get("sources", []) # Get sources list
                            else: # Handle case where old format might be returned
                                response_content = str(response_dict)
                                logger.warning("qa_engine.answer_question did not return the expected dictionary format.")

                            # --- Conceptual token tracking (commented out) ---
                        except Exception as e:
                            error_msg = f"Sorry, I encountered an error processing your question: {e}"
                            logger.exception("Error during qa_engine.answer_question")
                            response_content = f"Error: {e}"
                            response_sources = None # Ensure sources are None on error
                    else:
                        # This case might be redundant if chat_disabled works correctly, but good as a fallback
                        response_content = "Error: Q&A Engine is not available or not ready."
                # --- END OF CONDITIONAL LOGIC ---

                # 3. Append assistant response (RAG result or error) to state AFTER generation
                if response_content:
                    # --- MODIFIED: Include sources in the message dict ---
                    assistant_message = {"role": "assistant", "content": response_content}
                    if response_sources: # Only add sources key if sources were returned
                        assistant_message["sources"] = response_sources
                    st.session_state.messages.append(assistant_message)
                    # --- ADDED: Save history after assistant message ---
                    save_chat_history(st.session_state.messages)

                # 4. Rerun the script ONCE to redraw the UI with the new messages
                # This single rerun should update the chat history display cleanly.
                st.rerun()


    # --- Right Column: Context Viewer ---
    with right_col:
        st.header("Context Viewer") # Header for the context column

        # --- Use Tabs for different views ---
        tab1, tab2 = st.tabs(["PDF Viewer", "Graph Visualization"])

        with tab1:
            st.subheader("Processed PDF Viewer")
            if st.session_state.pdf_paths: # Check if there are any PDF paths stored
                 pdf_files = list(st.session_state.pdf_paths.keys())
                 # Add a default "Select PDF" option
                 options = ["Select a PDF..."] + pdf_files
                 selected_pdf_file = st.selectbox("Select a processed PDF to view:", options=options, index=0, key="pdf_select")
                 if selected_pdf_file != "Select a PDF...":
                     pdf_path_to_display = st.session_state.pdf_paths[selected_pdf_file]
                     if os.path.exists(pdf_path_to_display):
                         display_pdf(pdf_path_to_display)
                     else:
                         st.warning(f"Selected PDF '{selected_pdf_file}' view unavailable (temporary file might have been removed).")
            else:
                 st.info("No PDFs have been processed in this session, or processed PDFs were not saved.")


        with tab2: # Now this is the second tab again
            st.subheader("Knowledge Graph Visualization")
            graph_html_file = st.session_state.get("graph_html_path")
            if graph_html_file and Path(graph_html_file).is_file():
                # --- Add Download Button within the tab ---
                try:
                    with open(graph_html_file, "rb") as fp:
                        st.download_button(
                             label="Download Graph HTML",
                             data=fp,
                             file_name=GRAPH_HTML_FILENAME, # Use constant
                             mime="text/html",
                             key="download_graph_tab_button"
                         )
                except Exception as download_e:
                    st.error(f"Could not prepare graph for download: {download_e}")

                # --- Display the graph ---
                try:
                    with open(graph_html_file, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    # Use a reasonable height, but allow scrolling
                    components.html(html_content, height=800, scrolling=True)
                except Exception as e:
                    st.error(f"Error reading or displaying graph visualization: {e}")
                    logger.error(f"Error reading/displaying graph HTML: {e}", exc_info=True)
            else:
                st.info("Process one or more documents to generate the knowledge graph visualization.")


# --- Entry Point ---
# (The rest of your file remains the same)
if __name__ == "__main__":
    # Basic logging setup if not already configured
    if not logging.getLogger().hasHandlers():
         logging.basicConfig(level=logging.DEBUG, format='%(asctime)s | %(levelname)s | %(module)s:%(lineno)d | %(message)s') # Ensure DEBUG for main
    main()

