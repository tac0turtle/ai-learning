"""LLM prompt templates for memory extraction and consolidation."""

EXTRACT_FACTS_SYSTEM = """You are a memory extraction system. Given a piece of conversation text, extract discrete, self-contained facts that are worth remembering long-term.

Rules:
- Write each fact in third person, present tense
- Each fact must be self-contained (understandable without the original context)
- Extract only meaningful facts, preferences, relationships, or events
- Assign an importance score from 0.0 to 1.0 (1.0 = critical life fact, 0.5 = moderately useful, 0.1 = trivial)
- Classify each as: fact, preference, event, or relationship
- Identify the main topic and any named entities
- If there are no meaningful facts to extract, return an empty list"""

EXTRACT_FACTS_TOOL = {
    "name": "store_memories",
    "description": "Store extracted facts as memories",
    "input_schema": {
        "type": "object",
        "properties": {
            "memories": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The self-contained fact in third person",
                        },
                        "memory_type": {
                            "type": "string",
                            "enum": ["fact", "preference", "event", "relationship"],
                        },
                        "importance": {
                            "type": "number",
                            "description": "0.0 to 1.0 importance score",
                        },
                        "topic": {
                            "type": "string",
                            "description": "Main topic category",
                        },
                        "entities": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Named entities mentioned",
                        },
                    },
                    "required": ["text", "memory_type", "importance", "topic", "entities"],
                },
            }
        },
        "required": ["memories"],
    },
}

CONSOLIDATE_SYSTEM = """You are a memory consolidation system. Given a batch of related memories on the same topic, produce a single concise summary that preserves the key information.

Rules:
- Write in third person, present tense
- Preserve specific names, dates, and numbers
- Resolve any contradictions by keeping the most recent information
- The summary should be self-contained
- Keep it concise but complete"""

CONSOLIDATE_TOOL = {
    "name": "store_summary",
    "description": "Store a consolidated memory summary",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary_text": {
                "type": "string",
                "description": "The consolidated summary",
            },
            "importance": {
                "type": "number",
                "description": "0.0 to 1.0 importance of the consolidated memory",
            },
            "entities": {
                "type": "array",
                "items": {"type": "string"},
                "description": "All named entities from the source memories",
            },
        },
        "required": ["summary_text", "importance", "entities"],
    },
}
