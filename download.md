# Download Instructions

1. [Create an AWS account](https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/)
2. [Install the AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)
3. Create a `root` folder to store the dataset, example: `/path/to/data/boreas/` Each sequence will then be a folder under `root`.
4. Use the AWS CLI to download a sequence:

```Bash
root=/path/to/data/boreas/
sequence=boreas-2021-09-02-11-42
aws s3 sync s3://boreas/$sequence $root$sequence
```

The [Download page](https://www.boreas.utias.utoronto.ca/#/download) on our website can be used to browse through sequences. The website will output a bash script that can be run to download the desired sequences.

Each sequence is approximately 100GB.

Note that each benchmark comes with a predefined train/test split contained in `pyboreas/data/splits.py`

`download_task.py` can also be used to execute AWS CLI commands to download the sequences for a specific task (odometry, localization, detection).

Note that the Boreas-Objects-V1 3D object dataset has its own special sequence named `boreas-objects-v1`.

It can be downloaded with: `aws s3 sync s3://boreas/boreas-objects-v1 .`
