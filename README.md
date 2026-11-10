# groovi: a graph learning framework for complex structural variant discovery

##### Table of Contents   
[Install](#install)   
[Models](#models)  
[User Guide](#guide)   

<a name="install"></a>
### Install

```bash
git clone git@github.com:PopicLab/groovi.git
cd groovi
conda env create -f install/conda_env.yml
conda activate groovi
pip install .
```

<a name="models"></a>
### Groovi models
Pre-trained Groovi models are available in the `models/` directory.

The latest model is: `models/groovi.v1.pt`.

<a name="guide"></a>
### User guide

#### Execution

* To call structural variants: ```$> groovi call --config </path/to/config>```
* To generate a graph dataset for training: ```$> groovi generate --config </path/to/config>```
* To train a new model: ```$> groovi train --config </path/to/config>```

Each ```groovi``` command accepts a YAML file with configuration parameters. 
All outputs are written to the directory that contains the config file.

#### SV calling (`groovi call`)

Example `groovi.yaml` config:

```yaml
bam: /path/to/sample.bam
fa: /path/to/reference.fa
model: /path/to/groovi_model.pt
gpu_ids: [0]
n_cpus: 8
```

* `bam` [*required*]: path to the alignments file (must be sorted and indexed; 
aligned with soft clipping only and `=/X` CIGAR operation encoding)
* `fa` [*required*] path to the reference FASTA file (must be indexed)
* `model` [*required*] path to the pretrained `groovi` model 
* `gpu_ids` [*optional*] list of GPU ids to use for calling (default: CPU(s) will be used if empty)
* `n_cpus`  [*optional*] number of CPUs to use for BAM processing (and calling if no GPUs are listed) (default: 8)

#### Graph generation (`groovi generate`)
`generate` takes the same bam, fa and n_cpus parameters as call and writes the graph to `graphs/graph_full.pkl`.

### Training

Example `config.yaml`:

```yaml
graphs: [/path/to/sample1/graphs/graph_full.pkl, /path/to/sample2/graphs/graph_full.pkl]
graphs_info: [/path/to/sample1_truth.vcf, /path/to/sample2_truth.vcf]
graphs_validation:
  sample3: [/path/to/sample3/graphs/graph_full.pkl]
graphs_info_validation:
  sample3: [/path/to/sample3_truth.vcf]
gpu_ids: [0]
```

* `graphs` [*required*]: list of training graphs produced by `groovi generate`
* `graphs_info` [*required*]: list of ground-truth VCF files used to label `graphs` (must be in the same order)
* `graphs_validation` [*required*]: validation graphs (grouped by sample name)
* `graphs_info_validation` [*required*]: ground-truth VCF files for `graphs_validation` (must be in same names and order)
* `gpu_ids` [*optional*]: GPU id to use for training; a CPU is used if empty (default: `[]`)
* `pretrained_model` [*optional*]: model to initialize training from (default: none)
* `num_epochs` [*optional*]: number of training epochs (default: `100`)

The best model is saved as `best_model.pt` (model checkpoints are stored in the `models/` folder).

### Authors
Tony Cui, Sayan Ghosal, Bert Huang, and Victoria Popic (contact: vpopic@broadinstitute.org).