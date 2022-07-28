# Project Elenchus

This project provides tools and libraries to train deep learning models, accessing data directly from a SQL table.


## Getting-Started

In this Getting-Started section, we will demonstrate how to fine tune a model on a Question Natural Language Inference (QNLI) tasks. However, the code can easily be adapted for fine-tuning transfomer models for other variants of text classification.

As described on [Hugging Face](https://huggingface.co/tasks/text-classification#question-natural-language-inference-qnli), "QNLI is the task of determining if the answer to a certain question can be found in a given document. If the answer can be found the label is 'entailment'. If the answer cannot be found the label is 'not entailment'."

For example:
```
Question: What percentage of marine life died during the extinction?
Sentence: It is also known as the “Great Dying” because it is considered the largest mass extinction in the Earth’s history.
Label: 0 (not entailment)

Question: Who was the London Weekend Television’s Managing Director?
Sentence: The managing director of London Weekend Television (LWT), Greg Dyke, met with the representatives of the "big five" football clubs in England in 1990.
Label: 1 (entailment)
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

Go to the Azure Portal, and use the information below to create an Azure SQL Managed Instance on the Azure Marketplace: 

- Go to this [link](https://ms.portal.azure.com/#view/Microsoft_Azure_Marketplace/GalleryItemDetailsBladeNopdl/id/Microsoft.SQLManagedInstance) and press "Create" to go to "Basics" setting

- Subscription: Select your Azure Subscription. If you do not have an Azure Subscription, you can create one on this [link](https://azure.microsoft.com/en-us/free/).

- Resource group: If you have an existing resource-groups in your subscription for this project, you can select that. Otherwise, press "Create new" and select a name for your new resource-group. 

- Manage Instance name: Select a name for your Azure SQL Managed Instance

- Region: Select a region. Your database will reside in the datacenter in this region. Note that for some regions you may get error for not having availablity for your subscription. Try other regions!

- Compute + storage: Click `Configure Managed Instance`. For the purposes of this tutorial, you can select default options, except that you may want to make some changes to `Compute + storage` settings:
    - Reduce the number of vCores to the minumum
    - Reduce amount of storage to minimum
    - Select `Locally-redundant backup storage`

Detailed information on Azure SQL Managed Instance can be found [here](https://azure.microsoft.com/en-us/products/azure-sql/managed-instance).

- Authentication: Select `Use SQL authentication` and pick a secure username and password for database admin on the corresponding spaces. Keep the username/password somewhere safe; you will need that on the next steps.

- Click `Review + create` and click `Create`. 

Leave this tab open as deployment may take some time! You can do the next two subsections as you are waiting.

### Install ODBC driver for SQL

You can find the driver [here](https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server). 

Please install version 18 of the driver for your operating system.


### Generate Conda Environment

We provide a Conda environment definition (`environment.yml`), to enable you to install all software dependencies with one command: `conda env create -f environment.yml`.

> Prerequisite: If you don't have the conda package manager installed yet, we recommend installing miniconda from [here](https://docs.conda.io/en/latest/miniconda.html)


Run `conda activate elenchus`

### Create Database
If the Azure SQL Managed Instance deployment is still in progress, you need to wait for the completion.

After the deployment is completed, click on `Go to resource` button. This takes you to the the resource profile. You can see the `Managed instance admin` name you selected and the `Host` URL on the Essentials section of the page. Save these information for the next subsection. 

On top left side, press + to add `New database`. Keep all the field as default and just enter a name for your database on `Database name` field. Press `Review and Create`. Then press `Create`.

### Create table and insert data
Please update the provided template file `config_template.json` with the required information about your SQL server, listed below, and store the file under `config.json`:

- username: admin username you selected on Step 2 
- password: admin password you selected on Step 2
- driver: the default value is "ODBC Driver 18 for SQL Server", but if you installed a different ODBC driver, you can modify this field accordingly.
- server: Go to the Azure SQL resource profile, under `Server name` you see your database server URL. Copy the URL and paste it here.
- database: pick a name for your database
- table_prefix: select a prefix for the tables name. For example, if your prefix would be "glue_", the table names would 'gule_train', 'glue_validation' and 'glue_test'.

Run `python convert_dataset.py`.

The script will download The Microsoft Research Paraphrase Corpus glue dataset and upload it to your database. You can find more info about the dataset on this [link](https://www.tensorflow.org/datasets/catalog/glue#gluemrpc).

### Fine-Tune the model

At this step you can fine-tune an example transformer with your data from database by running: `python train.py`.

Note that the performance improvement after each epoch.

For reference, when fine-tuning the model with the default settings on a Azure `Standard NC6` (NVIDIA Tesla K80), this should take about 15 minutes.

If you run into memory capacity issues such as `RuntimeError: CUDA out of memory.`, you can decrease the `batch-size`.

### Delete and Cleanup
If you need to delete the tables, you can run `python delete_dataset.py -tables`

If you need to delete the whole database, you can run `python delete_dataset.py -db`

If you are done with the experiment, you can also go to the Azure Portal and delete the Azure SQL Managed Instance and/or the Resource Group.

### Switching Classification Task Variant

You can switch to a different classification task (aka. glue subset), by editing the `data` section in the `config.json` configuration file. For example, you can set `glue_subset` to "mnli". Be careful to also update the number of labels `num_labels`, as well as the names of the `train` and `validation` splits for those task variants.

Don't forget to also run `python convert_dataset.py` before you try to fine-tune the model.

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
