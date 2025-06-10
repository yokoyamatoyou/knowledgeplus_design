# Instructions for Codex: Phase 4 User Experience Improvements

The repository now includes initial FAQ generation support. The command-line script `generate_faq.py` can produce FAQ entries from existing chunks, and the unified Streamlit app exposes a sidebar button for this feature. The search engine loads `faqs.json` so that question–answer pairs are included in results.

Use this document as guidance for implementing further user experience enhancements in a follow-up conversation.

## Current Status
- FAQ generation runs from the CLI and Streamlit.
- Generated FAQs are stored under each knowledge base and loaded by `HybridSearchEngine`.
- The UI shows a spinner while FAQs are generated and reports the number created.

## Next Steps
1. **Customisable Parameters**
   - Allow users to set the maximum token limit and the number of Q&A pairs when generating FAQs.
   - Add optional arguments in `generate_faqs_from_chunks()` and expose them both in `generate_faq.py` and the Streamlit sidebar.

2. **Search Experience**
   - Consider a dedicated tab or a blended display so FAQ answers are clearly distinguished from normal chunk results.
   - Ensure the search engine can filter or highlight FAQ content as appropriate.

3. **Testing Enhancements**
   - Extend `tests/test_generate_faq.py` to verify the content of `faqs.json` and to cover the new parameters.
   - Where possible, add basic Streamlit tests or document how to manually check the FAQ sidebar features.

4. **Documentation Updates**
   - Keep `docs/integration_plan.md` in sync, especially the new `Phase 4 Implementation` section.
   - Provide short usage examples for FAQ generation in both CLI and UI contexts.

Follow these notes to continue improving the user experience in the next coding session.
