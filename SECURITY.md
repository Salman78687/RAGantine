# Security

RAGantine is a research benchmark and defense layer. Its poisoned documents are deliberately
harmless: wrong answers and canary codes, aimed at a local benchmark corpus.

- **Found a bypass of a RAGantine defense?** That is a welcome result. Open an issue or PR with
  a reproducible benchmark run, and keep the payload harmless (wrong answer or canary).
- **Found a vulnerability in RAGantine's own code** (the API, the quarantine release path,
  parsing)? Please report it privately through GitHub's "Report a vulnerability" (Security tab)
  rather than a public issue.
- Never test RAGantine attacks against systems you do not own.
