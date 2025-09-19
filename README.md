# SAGE
A protocol reverse engineering (PRE) tool that accurately accomplishes format inference for protocol messages using advanced deeping learning techniques.

## Running Environment
We summarize the major setup instructions as follows:

```bash
# Install dependencies
pip install -r requirements.txt

# Data preparation
python 1_pcap2protocol.py    # Extract protocol-specific pcaps from mixed pcap files
                              # → Generates: ./data/protocol/tcp.pcap, dns.pcap, etc.

python 2_protocol2json.py    # Convert pcap files to JSON and JSONraw formats
                              # → Generates: ./data/processed/json/*.json, ./data/processed/jsonraw/*_raw.json

python 3_json2formatted.py   # Transform JSON data into formatted training data
                              # → Generates: ./data/formatted/*.json with bit matrices and labels

python preparation.py        # Generate cross-validation datasets for training
                              # → Generates: ./dataset/

# Training
python train.py              # Train the model with cross-validation
                              

# Ablation study
python comprehensive_ablation.py  # Run ablation experiments
                                   
```


## Code Availability
The complete source code is available in this repository. Experimental data will be published after the review process.

## Contact
For any questions regarding this work, please contact the authors through the submission system.
