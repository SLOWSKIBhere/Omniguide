# Security Policy

## Supported version

Security fixes are applied to the current `main` branch.

## Reporting a vulnerability

Please do not post credentials, API keys, tokens, private URLs, or exploit details in a public issue or pull request.

If GitHub private vulnerability reporting is enabled for this repository, use it for security reports. Otherwise, open a minimal public issue that says you need a private security contact, without including sensitive details.

For an exposed credential:

1. Revoke or rotate the credential at the provider immediately.
2. Remove the credential from the current branch.
3. Treat the old value as compromised even after deletion, because Git history may still contain it.
4. Review provider logs and billing for unexpected use.

## Repository safeguards

- Local environment files and key material belong outside Git.
- Use `.env.example` only for variable names and non-secret example values.
- Pull requests and pushes to `main` are scanned for likely secrets.
- Dependency updates should arrive through Dependabot pull requests and pass the normal test workflow before merge.
