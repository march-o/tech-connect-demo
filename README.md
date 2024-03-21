## Setup

### Run in a terminal:
```bash
pip install torch transformers numpy json plotly
```
### [Optional] Modify [data.py](data.py) to use your own profiles

# To generate the looking_for and can_give data:
```bash
python gen_info.py
```

# Than to match profiles based on embeddings:
```bash
python match_profiles.py
```

# Results on demo data
![alt text](match_heatmap.png)