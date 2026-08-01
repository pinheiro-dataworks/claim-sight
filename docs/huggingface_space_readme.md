<!--
This file is the README.md a Hugging Face Space (Docker SDK) needs at
its repo root — Spaces require this exact YAML frontmatter to build.
It is kept here (not at the project root, which already has its own
README) so it can be copied into the separate Space git repo when you
deploy: see "Deployment" in the main README.md for the full steps.
-->
---
title: ClaimSight
emoji: 🚗
colorFrom: orange
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# ClaimSight — Live Demo

Vehicle damage triage classifier (ResNet50, 94.3% validation accuracy,
ROC-AUC 0.986) with Grad-CAM explainability, served via FastAPI.

Upload a vehicle photo at the root URL to see a live prediction and its
Grad-CAM overlay, or call the JSON API directly — see `/docs` for the
full OpenAPI schema.

**This is a decision-support prototype for insurance claims triage. It
does not issue, deny, or finalize any claim — every prediction requires
human review.**

Full project, training code, and results: https://github.com/<your-username>/claimsight
