#!/usr/bin/env python3
"""
Configuration file for NanoLM Legal Tokenizer
"""

# Data paths
DATA_CONFIG = {
    'constitution_path': "/home/swadhin/text/experiments2/data/constitution.txt",
    'legal_acts_path': "/home/swadhin/text/experiments2/data/Contextualized_Bangladesh_Legal_Acts.json",
    'output_dir': "./nanolm_legal_tokenizer",
    'dataset_output': "./nanolm_legal_dataset.txt"
}

# Tokenizer configuration
TOKENIZER_CONFIG = {
    'vocab_size': 32000,
    'min_frequency': 2,
    'batch_size': 1500,  # Adjust based on GPU memory
    'max_sequence_length': 512,
    'test_corpus_size': 200
}

# Special tokens for legal domain
LEGAL_SPECIAL_TOKENS = [
    # Base tokens
    "[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]",

    # Legal document structure
    "[DOC_START]", "[DOC_END]", "[SECTION_START]", "[SECTION_END]",
    "[ARTICLE_START]", "[ARTICLE_END]", "[CLAUSE_START]", "[CLAUSE_END]",
    "[SUBSECTION_START]", "[SUBSECTION_END]", "[PARAGRAPH_START]", "[PARAGRAPH_END]",

    # Legal content markers
    "[LEGAL_START]", "[LEGAL_END]", "[CONSTITUTION_START]", "[CONSTITUTION_END]",
    "[ACT_START]", "[ACT_END]", "[AMENDMENT_START]", "[AMENDMENT_END]",
    "[PROVISION_START]", "[PROVISION_END]", "[REGULATION_START]", "[REGULATION_END]",

    # Headers and titles
    "[TITLE_START]", "[TITLE_END]", "[SUBTITLE_START]", "[SUBTITLE_END]",
    "[HEADER_START]", "[HEADER_END]", "[FOOTER_START]", "[FOOTER_END]",

    # Legal entities and references
    "[COURT_START]", "[COURT_END]", "[JUDGE_START]", "[JUDGE_END]",
    "[PARLIAMENT_START]", "[PARLIAMENT_END]", "[GOVERNMENT_START]", "[GOVERNMENT_END]",
    "[CITIZEN_START]", "[CITIZEN_END]", "[AUTHORITY_START]", "[AUTHORITY_END]",

    # Legal procedures
    "[PROCEDURE_START]", "[PROCEDURE_END]", "[PROCESS_START]", "[PROCESS_END]",
    "[REQUIREMENT_START]", "[REQUIREMENT_END]", "[CONDITION_START]", "[CONDITION_END]",

    # Rights and obligations
    "[RIGHT_START]", "[RIGHT_END]", "[DUTY_START]", "[DUTY_END]",
    "[OBLIGATION_START]", "[OBLIGATION_END]", "[PRIVILEGE_START]", "[PRIVILEGE_END]",

    # Legal reasoning (for NanoLM)
    "[THINK]", "[REASON]", "[ANALYZE]", "[CONCLUDE]", "[INTERPRET]",
    "[PRECEDENT]", "[PRINCIPLE]", "[EXCEPTION]", "[CLARIFICATION]",

    # Dates and references
    "[DATE_START]", "[DATE_END]", "[YEAR_START]", "[YEAR_END]",
    "[REF_START]", "[REF_END]", "[CITATION_START]", "[CITATION_END]",
    "[REFERENCE_ID]", "[CASE_REF]", "[ACT_REF]", "[ARTICLE_REF]",

    # Identification
    "[ID_START]", "[ID_END]", "[NUMBER_START]", "[NUMBER_END]",
    "[CODE_START]", "[CODE_END]", "[SERIAL_START]", "[SERIAL_END]",

    # Legal status markers
    "[ACTIVE]", "[REPEALED]", "[AMENDED]", "[SUPERSEDED]", "[PENDING]",
    "[ENFORCED]", "[SUSPENDED]", "[PROVISIONAL]", "[FINAL]",

    # Multimodal tokens (for future NanoLM extensions)
    "[IMG_START]", "[IMG_END]", "[TABLE_START]", "[TABLE_END]",
    "[CHART_START]", "[CHART_END]", "[DIAGRAM_START]", "[DIAGRAM_END]",

    # Language and encoding
    "[LANG_BN]", "[LANG_EN]", "[TRANSLATION_START]", "[TRANSLATION_END]",

    # Special formatting
    "[BOLD_START]", "[BOLD_END]", "[ITALIC_START]", "[ITALIC_END]",
    "[UNDERLINE_START]", "[UNDERLINE_END]", "[EMPHASIS_START]", "[EMPHASIS_END]"
]

# Training configuration
TRAINING_CONFIG = {
    'end_of_word_suffix': "</w>",
    'continuing_subword_prefix': "▁",
    'show_progress': True,
    'normalize_text': True,
    'lowercase': False,  # Keep case for legal terms
    'strip_accents': True
}

# GPU configuration
GPU_CONFIG = {
    'enable_gpu': True,
    'memory_fraction': 0.8,
    'allow_growth': True,
    'benchmark_mode': True
}

# Evaluation configuration
EVAL_CONFIG = {
    'test_samples': 50,
    'legal_terms': [
        'constitution', 'article', 'section', 'clause', 'amendment', 'law', 'act',
        'parliament', 'government', 'court', 'judge', 'justice', 'legal', 'rights',
        'fundamental', 'citizen', 'republic', 'bangladesh', 'provision', 'authority',
        'jurisdiction', 'tribunal', 'legislation', 'statute', 'ordinance', 'regulation',
        'enforcement', 'compliance', 'violation', 'penalty', 'procedure', 'process'
    ],
    'compression_target': 3.5,  # Target chars per token
    'coverage_threshold': 0.8   # Minimum coverage for legal terms
}

# Dataset creation configuration
DATASET_CONFIG = {
    'sequence_length': 512,
    'overlap_ratio': 0.5,
    'min_chunk_length': 50,
    'include_special_sequences': True,
    'add_reasoning_examples': True
}