# Paper discovery

The scholarly landing page is
[paper.html](https://atilavahedian.github.io/cotangent/paper.html).
It presents the complete manuscript abstract in static HTML and links directly
to the searchable PDF in the same directory. The project homepage links to it.

The page provides `citation_title`, `citation_author`,
`citation_publication_date`, `citation_pdf_url`, and `citation_language` metadata.
The publication date is the public report's original release date, October 4,
2026. The title and author agree with the manuscript and PDF metadata.

This follows [Google Scholar's inclusion guidance for individual authors](https://scholar.google.com/intl/en/scholar/inclusion.html).
Google Scholar decides inclusion and crawl timing. Publishing the page does
not confirm indexing, acceptance, or peer review. The paper remains a technical
report. No journal, conference, or institutional affiliation is asserted.

When revising the manuscript, update the complete abstract and metadata in
`docs/paper.html` and refresh its hash in `artifacts/v6/publication.json`.
Preserve the original release date for this report version. The homepage link
is maintained in `analysis/v6/site.template.html`.
