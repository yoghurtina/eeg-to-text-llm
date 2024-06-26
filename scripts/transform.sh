pip install llama-recipes transformers datasets accelerate sentencepiece protobuf==3.20 py7zr scipy peft bitsandbytes fire torch_tb_profiler ipywidgets
TRANSFORM=$(python -c "import transformers;print('/'.join(transformers.__file__.split('/')[:-1])+'/models/llama/convert_llama_weights_to_hf.py')")
model_dir='./llama3'
model_size='8B'
hf_model_dir='./llama3-hf'
version='llama_version 3'
python $TRANSFORM --input_dir $model_dir --model_size $model_size --output_dir $hf_model_dir --llama_version 3
