import random
import numpy as np

def generate_text_conditions(batch_lol, dropout_rate=0.5):
    """
    Converts a list of lists into augmented semantic prompts.
    Handles potential contradictions and ensures valid string outputs.
    """
    text_conditions = []

    for row_parts in batch_lol:
        current_parts = [str(p).strip() for p in row_parts if str(p).strip()]

        if not current_parts:
            text_conditions.append("")
            continue

        if random.random() < dropout_rate and len(current_parts) > 1:
            keep_n = np.random.randint(1, len(current_parts) + 1)
            current_parts = list(np.random.choice(current_parts, size=keep_n, replace=False))

        random.shuffle(current_parts)

        details = ", ".join(current_parts)
        prompt = f"A patient with {details}" if details else ""
        text_conditions.append(prompt)

    return text_conditions