"""
Dataset mapping functions for different benchmark datasets.

This module provides dataset-specific mapping functions that convert
raw dataset samples into BenchmarkSample format.
"""
from typing import Dict, Any, List, Optional
import logging
from data_loader import BenchmarkSample


def map_squad(data: Dict[str, Any], idx: int = 0) -> BenchmarkSample:
    """Map SQuAD dataset sample to BenchmarkSample.
    
    SQuAD structure:
    - context: str (the passage)
    - question: str
    - answers: dict with "text" (list of strings) and "answer_start" (list of ints)
    - id: str
    
    Args:
        data: Raw dataset sample
        idx: Sample index (for logging)
        
    Returns:
        BenchmarkSample with input=context+question, output=first answer text
        Note: Instruction formatting is handled by the fine-tuning pipeline
    """
    context = data.get("context", "")
    question = data.get("question", "")
    answers = data.get("answers", {})
    sample_id = data.get("id", "")
    
    # Extract answer text from SQuAD format
    if isinstance(answers, dict) and "text" in answers:
        answer_list = answers["text"]
        if isinstance(answer_list, list) and len(answer_list) > 0:
            output_text = answer_list[0]  # Use first answer
        elif isinstance(answer_list, str):
            output_text = answer_list
        else:
            logging.warning(f"SQuAD sample {idx}: Unexpected answer format, using empty string")
            output_text = ""
    else:
        logging.warning(f"SQuAD sample {idx}: Missing or invalid answers field")
        output_text = ""
    
    # Combine context and question as input
    # The fine-tuning pipeline will add system prompts and format instructions
    input_text = f"Context: {context}\n\nQuestion: {question}"
    
    return BenchmarkSample(
        input=input_text,
        output=output_text,
        metadata={
            "question": question,
            "context": context,
            "id": sample_id,
            "all_answers": answers.get("text", []) if isinstance(answers, dict) else []
        }
    )


def map_mmlu(data: Dict[str, Any], idx: int = 0) -> Optional[BenchmarkSample]:
    """Map MMLU dataset sample to BenchmarkSample.
    
    MMLU structure:
    - input: str (the question)
    - target: int (index of correct answer, 0-3)
    - answer: int (alternative field name for target)
    - choices: list of 4 strings (answer options)
    - question: str (same as input usually)
    
    Args:
        data: Raw dataset sample
        idx: Sample index (for logging)
        
    Returns:
        BenchmarkSample with input=question, output=correct choice text
        None if target index is invalid (skip the sample)
    """
    question = data.get("input", data.get("question", ""))
    target = data.get("answer", data.get("target", -1))
    choices = data.get("choices", [])
    
    # Convert target to int if it's a string representation
    if isinstance(target, str):
        try:
            target = int(target)
        except (ValueError, TypeError):
            target = -1
    
    # Map target index to actual choice text
    if isinstance(target, int) and 0 <= target < len(choices):
        output_text = choices[target]
    elif isinstance(target, str):
        # Sometimes target is already the text
        output_text = target
    else:
        # Invalid target index - skip this sample
        if idx < 3:
            logging.debug(f"MMLU sample {idx}: Invalid target index {target}, skipping sample. Choices available: {len(choices)}")
        return None
    
    question = f"Question: {question}\nChoices\n" + "\n".join([f"{chr(65+i)}: {choice}" for i, choice in enumerate(choices)])
    return BenchmarkSample(
        input=question,
        output=output_text,
        metadata={
            "question": question,
            "target_index": target,
            "choices": choices
        }
    )


def map_mmlu_cf(data: Dict[str, Any], idx: int = 0) -> BenchmarkSample:
    """Map MMLU-CF dataset sample to BenchmarkSample.

    MMLU-CF structure:
    - Question: str (the question)
    - Answer: int (index of correct answer, 0-3)
    - A: str (answer option A)
    - B: str (answer option B)
    - C: str (answer option C)
    - D: str (answer option D)
    Args:
        data: Raw dataset sample
        idx: Sample index (for logging)
    Returns:
        BenchmarkSample with input=question, output=correct choice text
    """
    question = data.get("Question", "")
    target = data.get("Answer", -1)
    choices = []
    answer_a = data.get("A", "")
    answer_b = data.get("B", "")
    answer_c = data.get("C", "")
    answer_d = data.get("D", "")

    if answer_a: choices.append(answer_a)
    if answer_b: choices.append(answer_b)
    if answer_c: choices.append(answer_c)
    if answer_d: choices.append(answer_d)
    # Map target index to actual choice text
    if isinstance(target, int) and 0 <= target < len(choices):
        output_text = choices[target]
    elif isinstance(target, str):
        # Sometimes target is already the text
        output_text = target
    else:
        logging.warning(f"MMLU sample {idx}: Invalid target index {target}, using first choice")
        output_text = choices[0] if choices else ""

    question = f"Question: {question}\nChoices\n" + "\n".join([f"{chr(65+i)}: {choice}" for i, choice in enumerate(choices)])

    return BenchmarkSample(
        input=question,
        output=output_text,
        metadata={
            "question": question,
            "target_index": target,
            "choices": choices
        }
    )

def map_humaneval(data: Dict[str, Any], idx: int = 0) -> BenchmarkSample:
    """Map HumanEval dataset sample to BenchmarkSample.
    
    HumanEval structure:
    - prompt: str (the function signature and docstring)
    - canonical_solution: str (the solution code)
    - task_id: str
    - test: str (test cases)
    
    Args:
        data: Raw dataset sample
        idx: Sample index (for logging)
        
    Returns:
        BenchmarkSample with input=prompt, output=canonical_solution
    """
    prompt = data.get("prompt", "")
    solution = data.get("canonical_solution", "")
    task_id = data.get("task_id", "")
    test = data.get("test", "")
    
    return BenchmarkSample(
        input=prompt,
        output=solution,
        metadata={
            "task_id": task_id,
            "test": test
        }
    )


def map_mbpp(data: Dict[str, Any], idx: int = 0) -> BenchmarkSample:
    """Map MBPP dataset sample to BenchmarkSample.
    
    MBPP structure:
    - text: str (the problem description)
    - code: str (the solution code)
    - task_id: int or str
    - test_list: list of test cases
    
    Args:
        data: Raw dataset sample
        idx: Sample index (for logging)
        
    Returns:
        BenchmarkSample with input=text, output=code
    """
    text = data.get("text", "")
    code = data.get("code", "")
    task_id = data.get("task_id", "")
    test_list = data.get("test_list", [])
    
    return BenchmarkSample(
        input=text,
        output=code,
        metadata={
            "task_id": task_id,
            "test_list": test_list
        }
    )


def map_wikitext(data: Dict[str, Any], idx: int = 0) -> BenchmarkSample:
    """Map Wikitext dataset sample to BenchmarkSample.
    
    Wikitext structure:
    - text: str (the text passage)
    
    For text completion, we split the text into input (first part) and output (remaining part).
    
    Args:
        data: Raw dataset sample
        idx: Sample index (for logging)
        
    Returns:
        BenchmarkSample with input=first part of text, output=remaining part
    """
    text = data.get("text", "").strip()
    
    # Handle empty or very short text
    if not text or len(text) < 10:
        logging.warning(f"Wikitext sample {idx}: Text too short or empty, using full text as both input and output")
        return BenchmarkSample(
            input=text,
            output=text,
            metadata={"full_text": text, "note": "text_too_short"}
        )
    
    # Simple split: first 80% as input, last 20% as output
    split_index = int(len(text) * 0.8)
    input_text = text[:split_index].strip()
    output_text = text[split_index:].strip()
    
    # Ensure we have both input and output
    if not output_text:
        # If split resulted in empty output, use last 20% of text
        output_text = text[-int(len(text) * 0.2):].strip()
    
    return BenchmarkSample(
        input=input_text,
        output=output_text,
        metadata={
            "full_text": text
        }
    )
    

def map_latest_eval(data: Dict[str, Any], idx: int = 0) -> Optional[BenchmarkSample]:
    """Map LatestEval dataset sample to BenchmarkSample.
    
    LatestEval structure (may vary):
    - passage: str (the passage)
    - query: str (the question)
    - answer: dict with "text" (list of strings) and "answer_start" (list of ints)
    - answers: alternative field name (plural)
    - output: alternative field name for direct answer
    - doc_id: str
    
    Args:
        data: Raw dataset sample
        idx: Sample index (for logging)
        
    Returns:
        BenchmarkSample with input=passage+question, output=answer text
        None if no valid answer found (skip the sample)
    """
    context = data.get("passage", data.get("context", ""))
    question = data.get("query", data.get("question", ""))
    sample_id = data.get("doc_id", data.get("id", ""))
    
    # Try multiple possible field names for answers
    answers = data.get("answer", data.get("answers", {}))
    output_text = data.get("output", "")  # Try direct output field
    
    # Extract answer text from LatestEval format
    if output_text:
        # Direct output field found
        pass
    elif isinstance(answers, dict) and "text" in answers:
        answer_list = answers["text"]
        if isinstance(answer_list, list) and len(answer_list) > 0:
            output_text = answer_list[0]  # Use first answer
        elif isinstance(answer_list, str):
            output_text = answer_list
    elif isinstance(answers, list) and len(answers) > 0:
        # Answers might be a direct list
        if isinstance(answers[0], str):
            output_text = answers[0]
        elif isinstance(answers[0], dict) and "text" in answers[0]:
            output_text = answers[0]["text"]
    elif isinstance(answers, str):
        # Answers might be a direct string
        output_text = answers
    
    # If still no answer found, log available fields for debugging (only first few samples)
    if not output_text:
        if idx < 3:
            available_keys = list(data.keys())
            logging.debug(f"LatestEval sample {idx}: No answer found. Available fields: {available_keys}")
        # Skip samples without answers
        return None
    
    # Combine context and question as input
    input_text = f"Passage: {context}\n\nQuestion: {question}"
    
    # Collect all answers for metadata
    all_answers = []
    if isinstance(answers, dict) and "text" in answers:
        all_answers = answers["text"] if isinstance(answers["text"], list) else [answers["text"]]
    elif isinstance(answers, list):
        all_answers = [a if isinstance(a, str) else a.get("text", "") for a in answers]
    
    return BenchmarkSample(
        input=input_text,
        output=output_text,
        metadata={
            "question": question,
            "context": context,
            "id": sample_id,
            "all_answers": all_answers
        }
    )

def map_opencodeinstruct(data: Dict[str, Any], idx: int = 0) -> Optional[BenchmarkSample]:
    """Map OpenCodeInstruct dataset sample to BenchmarkSample.
    
    OpenCodeInstruct structure:
    - instruction: str (the instruction for code generation)
    - input: str (additional input context)
    - output: str (the solution code)
    - id: str
    - average_test_score: float or str (test score, should be 1.0 for valid samples)
    
    Args:
        data: Raw dataset sample
        idx: Sample index (for logging)
        
    Returns:
        BenchmarkSample with input=instruction+input, output=output if valid,
        None if average_test_score < 1.0 (invalid sample)
    """
    instruction = data.get("instruction", "")
    input_context = data.get("input", "")
    output_code = data.get("output", "")
    sample_id = data.get("id", "")
    average_test_score = data.get("average_test_score", 0)
    tests = data.get("unit_tests", "")

    # Convert average_test_score to float if it's a string
    if isinstance(average_test_score, str):
        try:
            average_test_score = float(average_test_score)
        except (ValueError, TypeError):
            # If conversion fails, treat as invalid
            if idx < 3:
                logging.debug(f"OpenCodeInstruct sample {idx}: Cannot convert average_test_score '{average_test_score}' to float, skipping")
            return None
    elif not isinstance(average_test_score, (int, float)):
        # If it's neither string nor number, treat as invalid
        if idx < 3:
            logging.debug(f"OpenCodeInstruct sample {idx}: average_test_score has invalid type {type(average_test_score)}, skipping")
        return None

    # Only use samples with perfect test score (1.0)
    if average_test_score < 1.0:
        logging.debug(f"OpenCodeInstruct sample {idx}: average_test_score {average_test_score} < 1.0, skipping invalid sample")
        return None  # Skip invalid samples

    # Combine instruction and input context as input
    if input_context:
        input_text = f"Instruction: {instruction}\n\nInput: {input_context}"
    else:
        input_text = instruction
    
    return BenchmarkSample(
        input=input_text,
        output=output_code,
        metadata={
            "instruction": instruction,
            "input_context": input_context,
            "id": sample_id,
            "average_test_score": average_test_score,
            "tests": tests
        }
    )

def map_generic(data: Dict[str, Any], mapping_config: Dict[str, Any], idx: int = 0) -> BenchmarkSample:
    """Generic mapper that uses mapping configuration.
    
    This is a fallback mapper that uses the old key-based mapping approach.
    
    Args:
        data: Raw dataset sample
        mapping_config: Mapping configuration with input_key, output_key, metadata_keys
        idx: Sample index (for logging)
        
    Returns:
        BenchmarkSample
    """
    input_key = mapping_config.get('input_key', 'input')
    output_key = mapping_config.get('output_key', 'output')
    metadata_keys = mapping_config.get('metadata_keys', [])
    
    input_text = data.get(input_key, "")
    output_text = data.get(output_key, "")
    
    # Convert to strings
    if not isinstance(input_text, str):
        input_text = str(input_text)
    
    if not isinstance(output_text, str):
        # Handle dictionary outputs (like SQuAD)
        if isinstance(output_text, dict) and "text" in output_text:
            answer_list = output_text["text"]
            if isinstance(answer_list, list) and len(answer_list) > 0:
                output_text = answer_list[0]
            elif isinstance(answer_list, str):
                output_text = answer_list
            else:
                output_text = str(output_text)
        # Handle integer targets (like MMLU)
        elif isinstance(output_text, int) and "choices" in data:
            choices = data.get("choices", [])
            if 0 <= output_text < len(choices):
                output_text = choices[output_text]
            else:
                output_text = str(output_text)
        else:
            output_text = str(output_text)
    
    # Extract metadata
    metadata = None
    if metadata_keys:
        metadata = {}
        for key in metadata_keys:
            if key in data:
                metadata[key] = data[key]
        if not metadata:
            metadata = None
    
    return BenchmarkSample(
        input=input_text,
        output=output_text,
        metadata=metadata
    )


# Mapper registry
_MAPPERS = {
    "squad": map_squad,
    "mmlu": map_mmlu,
    "humaneval": map_humaneval,
    "mbpp": map_mbpp,
    "wikitext": map_wikitext,
    "generic": map_generic,
    "latesteval": map_latest_eval,
    "mmlu_cf": map_mmlu_cf,
    "mmlu-cf": map_mmlu_cf,  # Support hyphenated version
    "opencodeinstruct": map_opencodeinstruct,
}


def get_mapper(mapper_name: Optional[str] = None) -> callable:
    """Get a mapper function by name.
    
    Args:
        mapper_name: Name of the mapper (e.g., "squad", "mmlu", "generic")
                    If None, returns generic mapper
        
    Returns:
        Mapper function
    """
    if mapper_name is None:
        mapper_name = "generic"
    
    mapper_name_lower = mapper_name.lower()
    if mapper_name_lower not in _MAPPERS:
        logging.warning(f"Unknown mapper '{mapper_name}', using 'generic' mapper")
        mapper_name_lower = "generic"
    
    return _MAPPERS[mapper_name_lower]


def map_dataset_samples(
    dataset_list: List[Dict[str, Any]],
    dataset_config: Dict[str, Any]
) -> List[BenchmarkSample]:
    """Map a list of dataset samples using the specified mapper.
    
    Args:
        dataset_list: List of raw dataset samples
        dataset_config: Dataset configuration with mapping settings
        
    Returns:
        List of BenchmarkSample instances
    """
    mapping_config = dataset_config.get('mapping', {})
    mapper_name = mapping_config.get('mapper', 'generic')
    
    # Get the appropriate mapper
    mapper_func = get_mapper(mapper_name)
    
    samples = []
    skipped_count = 0
    for idx, data in enumerate(dataset_list):
        if not isinstance(data, dict):
            logging.warning(f"Sample at index {idx} is not a dictionary, skipping.")
            continue
        
        try:
            if mapper_name == "generic":
                # Generic mapper needs the mapping config
                sample = mapper_func(data, mapping_config, idx)
            else:
                # Dataset-specific mappers only need data and idx
                sample = mapper_func(data, idx)
            
            # Skip None samples (e.g., invalid OpenCodeInstruct samples)
            if sample is None:
                skipped_count += 1
                continue
                
            samples.append(sample)
        except Exception as e:
            logging.error(f"Error mapping sample at index {idx}: {e}")
            continue
    
    if skipped_count > 0:
        logging.info(f"Skipped {skipped_count} invalid samples (e.g., OpenCodeInstruct with average_test_score < 1.0)")
    
    return samples

