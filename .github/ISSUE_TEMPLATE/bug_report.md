---
name: Bug Report
description: File a bug report about the neural-network engine or test suite
title: "[Bug] "
labels: ["bug"]
body:
  - type: markdown
    attributes:
      value: |
        ## Before you report

        Check that the issue is not already reported in open or closed issues.
        This repository contains a from-scratch neural-network engine; website
        traffic detection and telemetry are not yet implemented.

  - type: textarea
    attributes:
      label: Description
      description: A clear and concise description of the bug.
    validations:
      required: true

  - type: textarea
    attributes:
      label: To reproduce
      description: |
        Steps to reproduce the behaviour, including commands and expected vs.
        actual output.
    validations:
      required: true

  - type: textarea
    attributes:
      label: Environment
      description: |
        - OS: [e.g. Windows 11, Ubuntu 24.04]
        - Python version: [e.g. 3.11.9]
        - How you installed dependencies: [e.g. `pip install -r requirements-dev.txt`]
    validations:
      required: true

  - type: textarea
    attributes:
      label: Test output
      description: |
        Paste the full `python -m pytest` output if the failure is in the test
        suite. Use a code block.
    validations:
      required: false

  - type: checkboxes
    attributes:
      label: Confirmation
      options:
        - label: I have searched existing issues for this problem
          required: true
