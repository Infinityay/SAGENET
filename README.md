# SAGE
A protocol reverse engineering (PRE) tool that accurately accomplishes format inference for protocol messages using advanced deeping learning techniques.

## Running Environment
We summarize the major setup instructions as follows:

```bash
# Install dependencies
pip install -r requirements.txt

# Data preparation
python 1_pcap2protocol.py
python 2_protocol2json.py
python 3_json2formatted.py
python preparation.py

# Training
python train.py

# Ablation study
python comprehensive_ablation.py
```


## Code Availability
The complete source code is available in this repository. Experimental data will be published after the review process.

## Contact
For any questions regarding this work, please contact the authors through the submission system.
