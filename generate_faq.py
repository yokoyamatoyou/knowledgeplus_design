#!/usr/bin/env python
"""CLI to generate FAQ files from an existing knowledge base."""
import sys
from shared import faq_utils


def main():
    if len(sys.argv) != 2:
        print("Usage: python generate_faq.py <kb_name>")
        sys.exit(1)
    kb_name = sys.argv[1]
    created = faq_utils.generate_faqs_from_chunks(kb_name)
    print(f"Generated {len(created)} FAQ files in knowledge_base/{kb_name}/faq")


if __name__ == "__main__":
    main()
