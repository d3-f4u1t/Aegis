---
name: Feature Request
description: Suggest an idea or enhancement for the project
title: "[Feature] "
labels: ["enhancement"]
body:
  - type: textarea
    attributes:
      label: Description
      description: A clear and concise description of the proposed feature.
    validations:
      required: true

  - type: textarea
    attributes:
      label: Motivation
      description: |
        Why is this needed? How does it fit the project's from-scratch,
        passive-monitoring scope? See docs/PROJECT_DIRECTION.md for scope and
        non-goals.
    validations:
      required: true

  - type: textarea
    attributes:
      label: Alternatives considered
      description: Any other approaches you considered.
    validations:
      required: false

  - type: checkboxes
    attributes:
      label: Scope check
      options:
        - label: I have read the project direction and non-goals in docs/PROJECT_DIRECTION.md
          required: true
        - label: This is not a request for a deep-learning framework (PyTorch/TensorFlow/JAX)
          required: true
