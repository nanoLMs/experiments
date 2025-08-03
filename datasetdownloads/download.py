from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="camel-ai/chemistry", repo_type="dataset", filename="chemistry.zip",
                local_dir="datasets/", local_dir_use_symlinks=False)
