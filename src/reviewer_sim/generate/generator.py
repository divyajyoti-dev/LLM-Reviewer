from typing import Dict

from reviewer_sim.generate.providers import BaseGenerator


def generate_review(example: Dict, generator: BaseGenerator) -> Dict:
    """Generate a review for the given example using the provided generator.
    
    This function is intentionally thin - all logic is in the generator.
    """
    return generator.generate(example)
