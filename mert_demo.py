"""Demo that visualizes MERT features for some speech audio samples. Taken from:
https://huggingface.co/m-a-p/MERT-v0
"""


from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
from torch import nn
import torchaudio.transforms as T
from datasets import load_dataset
import matplotlib.pyplot as plt


# loading our model weights
model = AutoModel.from_pretrained("m-a-p/MERT-v0", trust_remote_code=True)
# loading the corresponding preprocessor config
processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v0",trust_remote_code=True)

# load demo audio and set processor
dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

resample_rate = processor.sampling_rate
print("samping_rate: ", sampling_rate, "resample rate", resample_rate)
# make sure the sample_rate aligned
if resample_rate != sampling_rate:
    print(f'setting rate from {sampling_rate} to {resample_rate}')
    resampler = T.Resample(sampling_rate, resample_rate)
else:
    resampler = None

for i in range(100):
    # audio file is decoded on the fly
    if resampler is None:
        input_audio = dataset[i]["audio"]["array"]
    else:
        input_audio = resampler(torch.from_numpy(dataset[0]["audio"]["array"]))

    print(input_audio.shape)
    plt.plot(input_audio)
    plt.show()

    inputs = processor(input_audio, sampling_rate=resample_rate, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # take a look at the output shape, there are 13 layers of representation
    # each layer performs differently in different downstream tasks, you should choose empirically
    all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
    print(all_layer_hidden_states.shape) # [13 layer, Time steps, 768 feature_dim]
    plt.imshow(all_layer_hidden_states[-1, :, :].T)
    #plt.plot(all_layer_hidden_states[-1, :, 20:30])
    plt.show()

    # for utterance level classification tasks, you can simply reduce the representation in time
    time_reduced_hidden_states = all_layer_hidden_states.mean(-2)
    print(time_reduced_hidden_states.shape) # [13, 768]
    
    plt.plot(time_reduced_hidden_states[:, :].T)
    plt.show()
    # you can even use a learnable weighted average representation
    aggregator = nn.Conv1d(in_channels=13, out_channels=1, kernel_size=1)
    weighted_avg_hidden_states = aggregator(time_reduced_hidden_states.unsqueeze(0)).squeeze()
    print(weighted_avg_hidden_states.shape) # [768]
