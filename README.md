# BLRQ-BV
（1）For English, we adopt the bert-base-uncased model. For Chinese, we use the bert-base-chinese model. 
（2）Meanwhile, different tokenizers are employed for the BiLSTM model in Chinese and English scenarios respectively.
（3）"unified config.py" is the configuration file, and readers need to modify it according to the input directory of the actual dataset. Meanwhile, readers need to specify the output directories of the process file and the result file according to the actual path.
（4）After completing the above configuration, you can directly execute the main.py file to run the BLRQ-BV model.
（5）T_SNE  instruction illustration :python T_SNE.py \
  --model-path /data/coding/best_model.pth \
  --image-dir /data/coding/DateSet-Twitter/twitter16txt-image \
  --npy1-dir /data/coding/saving/Qcnn_twitter16txt/bert_twitter16txt_CNN1D_Quantum_Features \
  --npy2-dir /data/coding/saving/bilstm_twitter16txt/twitter_bilstm_features \
  --output-dir /data/coding/tsne_visualizations \
  --max-samples 300
