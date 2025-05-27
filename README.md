# EmergenCease

This project builds a pipeline to generate multilingual emergency alerts using FCC Alert Templates in 13 languages. It fetches alerts, classifies the emergency type, extracts key information, and fills out multilingual templates in the Common Alerting Protocol format. The goal is to improve alert accessibility for non-native speakers.

```
conda create -n EmergenCease python=3.11 -y
conda activate EmergenCease
pip install -r requirements.txt
```