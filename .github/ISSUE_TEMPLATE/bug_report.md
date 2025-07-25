---
name: 🐛 Bug report
about: Create a report to help us improve
title: '[BUG] '
labels: ['bug', 'needs-triage']
assignees: ['@your-username']

---

## 🐛 Bug Description
A clear and concise description of what the bug is.

## 🔄 Steps to Reproduce
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## ✅ Expected Behavior
A clear and concise description of what you expected to happen.

## ❌ Actual Behavior
A clear and concise description of what actually happened.

## 📸 Screenshots
If applicable, add screenshots to help explain your problem.

## 🖥️ Environment
- **OS**: [e.g. Ubuntu 20.04, Windows 10, macOS 12]
- **Python Version**: [e.g. 3.9.7]
- **Pipeline Version**: [e.g. 1.0.0]
- **Processing Engine**: [e.g. pandas, spark, ray]

## 📋 Configuration
```yaml
# Relevant part of your config.yaml
data_source:
  type: "synthetic"
processing:
  engine: "pandas"
```

## 📊 Logs
```
# Paste relevant logs here
```

## 🔧 Additional Context
Add any other context about the problem here.

## 🧪 Reproduction Code
```python
# Minimal code to reproduce the issue
from src.pcc_pipeline import PCCDataPipeline

pipeline = PCCDataPipeline("config.yaml")
# ... your code here
```