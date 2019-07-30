# NuclearSegmentationPipeline

This repository contains the code for the paper

**Deep Learning architectures for generalized immunofluorescence based nuclear image segmentation**
<br>
[Florian Kromp](http://science.ccri.at/contact-us/contact-details/), [Lukas Fischer](https://www.scch.at/en/team/person_id/207), [Eva Bozsaky](http://science.ccri.at/contact-us/contact-details/), [Inge Ambros](http://science.ccri.at/contact-us/contact-details/), Wolfgang Doerr, [Sabine Taschner-Mandl](http://science.ccri.at/contact-us/contact-details/), [Peter Ambros](http://science.ccri.at/contact-us/contact-details/), Allan Hanbury

This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible.

## Citing this work

If you use the Nuclear Segmentation Pipeline in your research, please cite the following BibTeX entry:

```
@article{kromp2019,
    author={Kromp, Florian and Fischer, Lukas and Bozsaky, Eva and Ambros, Inge and Doerr, Wolfgang and Taschner-Mandl, Sabine and Ambros, Peter and Hanbury, Allen},
    title={Deep Learning architectures for generalized immunofluorescence based nuclear image segmentation},
    journal = {arXiv},
    volume = {},
    number = {},
    pages = {},
    year = {2019},
    doi = {},
    note ={},
    URL = {},
    eprint = {}
}
```

## Setup
The Nuclear Segmentation Pipeline was developed using a windows batch script. The batch script is necessary as we used multiple frameworks, each utilizing different python environments.
Therefore, the python environments must be set using the requirement scripts provided. 
The following three environments have to be set: 1. DataGenerator 2. pix2pix image translation 3. U-net architecture + lasagne wrapper for the U-net architecture
We use multiple environments and link them using a batchscript, as we want to include any framework no matter of the required environment.
After setting up the environments using the requirement files (DataGenerator\requirements.txt, UnetPure\requirements.txt, pix2pix-tensorflow-master\requirements.txt), 
the path to the python environments and the local folders has to be set in the pipeline-batch-script (run_pipeline_unet.bat).
Subsequently, the lasagne wrapper has to be built: change to the environment for the U-Net (UnetPure) and run these commands:
```
python setup.py build
python setup.py install
```

Please dont't forget to [cite](#citing-this-work)!