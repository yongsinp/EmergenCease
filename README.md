# EmergenCease

This project builds a pipeline to generate multilingual emergency alerts using FCC Wireless Emergency Alert Templates in
14 languages. It classifies the emergency type, extracts key information, and fills out multilingual templates in the
Common Alerting Protocol (CAP) format. The goal is to improve alert accessibility for non-native speakers.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Hugging Face account and authentication token for model access
- NVIDIA GPU (optional, for fine-tuning)

### Installation

Clone the repository:

```bash
git clone https://github.com/yongsinp/EmergenCease.git
cd EmergenCease
```

Create a virtual environment and install the required dependencies:

```bash
# Create and activate virtual environment
conda create -n EmergenCease python=3.11 -y
conda activate EmergenCease

# Install dependencies
pip install -r requirements.txt
```

Add the following to `.bashrc` or run the command to set up Hugging Face authentication token:

```bash
export HF_TOKEN="YOUR_HF_TOKEN"  # Replace with your actual Hugging Face token
```

## Usage

The project's main code is located in the `src` directory. Use this directory as the working directory to run scripts.

```bash
export PYTHONPATH=/PATH_TO_PROJECT/EmergenCease:$PYTHONPATH
```

### Running the Pipeline

Run the following command with at least one of the `--headline`, `--description`, or `--instruction` arguments to
generate a multilingual CAP alert.
You can also choose to run the command with the `--cap` argument which expects a JSON string conforming to the Common
Alerting Protocol (CAP) format.

By default, the pipeline uses the **Llama 3.2 1B Instruct** model with a LoRA adapter trained on a small set of IPAWS
Archived Alerts data to generate the alert.

```bash
# Running the command without any arguments will generate a sample Tornado Warning alert
python -m src.cap_translator.translate --headline ALERT_HEADLINE --description ALERT_DESCRIPTION --instruction ALERT_INSTRUCTION
```

Run `python -m src.cap_translator.translate -h` for help.

### Downloading Models

The **Llama 3.2 1B Instruct** model is downloaded automatically from Hugging Face to the `models` directory.
Models may require authentication due to licensing restrictions.
To use these models, you need to have a Hugging Face account, set up your authentication token, and request access to
the
models if necessary. Refer to [Additional Resources](#additional-resources) for more information on Hugging Face
authentication.

You can pass the token with the model name, or set a `HF_TOKEN` environment variable. To set the enviroment variable,
add the following to `.bashrc` or run the command to set up the Hugging Face authentication token:

```bash
export HF_TOKEN="YOUR_HF_TOKEN"  # Replace with your actual Hugging Face token
```

You can use the following command to manually download the models of your choice:

```bash
python -m src.utils.model  # --model "meta-llama/Llama-3.2-1B-Instruct" --hf-token "YOUR_HF_TOKEN" 
```

Run `python -m src.utils.model -h` for help.

### Fine-tuning

You can fine-tune Llama 3 (`3.1-8B`, `3.2-1B`, and `3.2-3B`) models for better performance. If the model does not
already exist in the `models` directory, it will be downloaded automatically. The trained LoRA adpaters will be saved in the
`models` directory.

```bash
python -m src.finetune.finetune  # --model 3.2-1B --epochs 3 --batch-size 4 --log-level INFO
```

Run `python -m src.finetune.finetune -h` for help.

### Evaluation

To evaluate the performance of the extraction model, you can run the following command:

```bash
python -m src.eval.eval  # --model meta-llama/Llama-3.2-1B-Instruct --adapter LoRA-Llama-3.2-1B-Instruct --test-data ./data/finetune/finetune_test.csv --runs 5
```

Run `python -m src.eval.eval -h` for help.

### (Optional) Downloading the Dataset

A small set of tagged data is provided in the `data/finetune` directory. The full IPAWS Archived Alerts dataset can be
downloaded by running the following command:

```bash
python -m src.data.download
```

### (Optional) Preprocessing

The full IPAWS Archived Alerts dataset will be preprocessed, split, and saved in the `data` directory. The dataset will
be automatically downloaded if it does not already exist. The sum of the train, validation, and test splits must equal
1.0.

```bash
python -m src.preprocess.preprocess  # --train 0.8 --val 0.1 --test 0.1 --random-seed 575 --sample-per-class 2
```

Run `python -m src.preprocess.preprocess -h` for help.

Converting the split dataset in CSV format to the JSON format can be done by running the following command:

```bash
python -m src.data.convert
```

## Miscellaneous

### Common Alerting Protocol (CAP) Templates

The CAP templates are stored at `src/cap_translator/cap_templtes.json`, in the following format:

```json
{
  "EVENT1": {
    "LANGUAGE1": "EVENT1 template in LANGUAGE1",
    "LANGUAGE2": "EVENT1 template in LANGUAGE2",
    ...
  },
  "EVENT2": {
    "LANGUAGE1": "EVENT2 template in LANGUAGE1",
    "LANGUAGE2": "EVENT2 template in LANGUAGE2",
    ...
  }
}
```

### Common Alerting Protocol (CAP) JSON Schema

The Common Alerting Protocol (CAP) JSON schema is stored at `src.data.cap.SCHEMA`. This schema defines the structure and
constraints for CAP messages, ensuring that generated alerts conform to the expected format.

Possible values for `status`, `msgType`, `scope`, `urgency`, `category`, `severity`, `certainty`, `responseType`, and
others, including language and region codes, are defined in `src/data/enums.py`

### Emergency (Event) and Nested Field Mapping

The event types found in the IPAWS Archived Alerts dataset are mapped to the FCC Wireless Emergency Alert Template
events using the file at `src/preprocess/event_map.yaml`, in the following format:

```yaml
FCC_EVENT_TYPE_1:
  - IPAWS_EVENT_TYPE_1
  - IPAWS_EVENT_TYPE_2
FCC_EVENT_TYPE_2:
  - IPAWS_EVENT_TYPE_3
  - IPAWS_EVENT_TYPE_4
...
```

The nested field names found in the CAP templates are mapped to the internal field names using the file at
`src/preprocess/ner_config.yaml`, in the following format:

```yaml
nested.field.names1: internal_field_name1
nested.field.names2: internal_field_name2
```

## Additional Resources

- [Llama 3.2 1B Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- [Hugging Face User Access Tokens](https://huggingface.co/docs/hub/en/security-tokens)
- [Manage Your Hugging Face Access Tokens](https://huggingface.co/settings/tokens)
- [OpenFEMA Dataset: IPAWS Archived Alerts - v1](https://www.fema.gov/openfema-data-page/ipaws-archived-alerts-v1)
- [Common Alerting Protocol Version 1.2](https://docs.oasis-open.org/emergency/cap/v1.2/CAP-v1.2-os.pdf)
- [FCC Multilingual Wireless Emergency Alert Templates](https://www.fcc.gov/multilingual-wireless-emergency-alerts)

## Acknowledgments

Parts of this work were done on the University of Washington’s high-performance computing cluster, Hyak, which is funded
by the Student Technology Fee.