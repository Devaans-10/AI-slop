# Deployment Checklist

- [ ] `checkpoints/run1/adapter/` exists and contains `adapter_config.json` and `adapter_model.safetensors` (or `adapter_model.bin`)
- [ ] `requirements.txt` includes `gradio`, `requests`, `peft`, `transformers`, `torch`, and `google-genai`
- [ ] `README.md` starts with the Hugging Face header (`title`, `emoji`, `sdk: gradio`, `app_file: app.py`)
- [ ] `TINYFISH_API_KEY` is set as a Space secret, not in the code
- [ ] `app.py` runs locally with `python app.py`
