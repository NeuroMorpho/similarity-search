# Similarity search module
Please note: cloning this repo will not result in a working solution, since connection with NeuroMorpho.Org database is not provided due to security reasons.
However, the full solution, including data saved as .pkl files is available as a Docker image at Docker hub: https://hub.docker.com/repository/docker/neuromorpho/sis
It is also deployed live with https://neuromorpho.org/similarity/ as API root.

## API description
The UI (available at neuromorpho.org, neuron detailed view) uses an underlying API which may be accessed with the following URL as a base URL: [https://neuromorpho.org/similarity](https://neuromorpho.org/similarity). 
By adding strings to the base URL, the different methods of the API may be accessed, please see table below. For example:
[https://neuromorpho.org/similarity/similarNeurons/0/1/10/1](https://neuromorpho.org/similarity/similarNeurons/0/1/10/1).

| Method | Method URL | Parameters |
| --- | --- | --- |
| Persistence vector (pvec) | similarNeurons/ | [usePCA, 0 or 1]/[neuron ID]/[result size]/[search type] |
| summary measurements | similarNeuronsPvec/ | As above |
| Pvec + summary | similarNeuronsSum/ | As above |
| Detailed morphometrics | similarNeuronsDetailed/ | As above |
| Metadata | similarNeuronsMeta | As Above |

The neuron ID parameter is an integer as listed at NeuroMorpho.Org. The resulting length parameter is an integer describing the desired data length of the result. It will be provided as a JSON (JavaScript Object Notation) array with fields &quot;neuron ID&quot; and &quot;similarity&quot; with the latter being a score from 0 to 1, where 0 is not similar at all and 1 identical. The &quot;search type&quot; parameter indicates desired use of (0) FAISS, (1) Pearson correlation and (2) the product of the two.
