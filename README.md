# Project Elenchus

This project provides tools and libraries to train deep learning models, accessing data directly from a SQL table.


## Getting-Started

In this Getting-Started section, we will demonstrate how to fine tune a model on a Question Natural Language Inference (QNLI) tasks.

As described on [Huggingface](https://huggingface.co/tasks/text-classification#question-natural-language-inference-qnli), "QNLI is the task of determining if the answer to a certain question can be found in a given document. If the answer can be found the label is 'entailment'. If the answer cannot be found the label is 'not entailment'."

For example:
```
Question: What percentage of marine life died during the extinction?
Sentence: It is also known as the “Great Dying” because it is considered the largest mass extinction in the Earth’s history.
Label: not entailment

Question: Who was the London Weekend Television’s Managing Director?
Sentence: The managing director of London Weekend Television (LWT), Greg Dyke, met with the representatives of the "big five" football clubs in England in 1990.
Label: entailment
```

**Key steps:**
1. Clone this repository
1. Create Azure SQL Managed Instance (or similar if you know what you are doing)
1. Install ODBC driver for SQL Server
1. Generate Conda environment
1. Create table and insert tutorial data
1. Fine-tune model

We will describe this in the following subsections. The training script is heavily influenced by the Hugging Face fine-tuning tutorial, which can be found [here](https://huggingface.co/course/chapter3/1?fw=pt).

### Create Azure SQL Managed Instance (MI)

Go to the Azure Portal to create a Azure SQL Managed Instance on the Azure Marketplace: [link](https://ms.portal.azure.com/#view/Microsoft_Azure_Marketplace/GalleryItemDetailsBladeNopdl/id/Microsoft.SQLManagedInstance)

For the purposes of this tutorial, you can select default options, except that you may want to make some changes to `Compute + storage` settings:
- Reduce the number of vCores to the minumum
- Reduce amount of storage to minimum
- Select `Locally-redundant backup storage`

Detailed information on Azure SQL Managed Instance can be found [here](https://azure.microsoft.com/en-us/products/azure-sql/managed-instance).

### Install ODBC driver for SQL

You can find the driver [here](https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server).

Please install version 18 of the driver for your operating system.

### Generate Conda Environment

We provide a Conda environment definition (`environment.yml`), to enable you to insetall all software dependencies with one command: `conda env create -f environment.yml`.

> Prerequisite: If you don't have the conda package manager installed yet, we recommend installing miniconda from [here](https://docs.conda.io/en/latest/miniconda.html)

### Create table and insert data

Please update the provided template file `config_template.json` with the required information about your SQL server, and store the file under `config.json`.

Then execute the script `convert_dataset.py`.

### Fine-Tune the model

This can be done by running the script `train.py`.

For reference, when fine-tuning the model with the default settings on a Azure `Standard NC6` (NVIDIA Tesla K80), this should take about 15 minutes.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
