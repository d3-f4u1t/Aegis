# Security

## Supported versions

This project is pre-release research software. No versions are currently
supported for production use, and no security patches are provided.

## Reporting a vulnerability

If you discover a security issue, please contact the project maintainer
privately. **Do not open a public GitHub issue** for security vulnerabilities.

Include:

* A description of the vulnerability
* Steps to reproduce it
* The affected component(s) and version/commit
* Any suggested mitigation

## Security considerations for this project

* This is a neural-network engine and research prototype, not a security
  product. Do not rely on it for real attack detection or prevention.
* The intended deployment is passive and out of band: it must never sit
  synchronously in a website's request path, and it must never block, modify,
  delay, or interfere with live requests.
* Do not collect passwords, session tokens, authorisation headers, cookies, or
  unnecessary personal information. Redact sensitive query parameters and
  request fields.
* Raw telemetry and alert records must be access-controlled and subject to
  defined retention periods.
* Model output is an investigation lead, not proof of malicious activity.

## Dependency security

The runtime engine uses only the Python standard library (`nn/`) or NumPy
(`nn_numpy/`). Development dependencies are `pytest` and `numpy`. Review their
advisories before use.
