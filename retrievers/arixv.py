# Copyright (c) 2026 Bishnuprasad Sahoo (GitHub: Bishnu9078)
# Licensed under the MIT License. See LICENSE in the project root.

import arxiv

# Create an arXiv client
client = arxiv.Client()

# Search for research papers
search = arxiv.Search(
    query="large language models",
    max_results=2,
    sort_by=arxiv.SortCriterion.Relevance
)

# Fetch results
papers = client.results(search)

# Display papers
for i, paper in enumerate(papers, start=1):
    print(f"\nResult {i}")
    print("Title:", paper.title)
    print("Authors:", ", ".join(a.name for a in paper.authors))
    print("Summary:", paper.summary[:500])
    print("PDF URL:", paper.pdf_url)
    print("-" * 60)
