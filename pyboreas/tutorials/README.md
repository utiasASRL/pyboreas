# Tutorial Overview
Before starting any tutorial, you'll need to have at least one sequence from the Boreas or Boreas Road Trip (Boreas-RT) datasets downloaded.
If a tutorial requires a specific dataset, it will state it at the top.
If you're working on a local machine, follow these steps to download a sequence:
1. [Create an AWS account](https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/)
2. [Install the AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)
3. Create a `root` folder to store the dataset, example: `/path/to/data/boreas/` Each sequence will then be a folder under `root`.
4. Use the AWS CLI to download a sequence:
```
root=/path/to/data/boreas/
sequence=boreas-2024-12-03-12-54
aws s3 sync s3://boreas/$sequence $root$sequence
```

You will also need to run `pip install ipykernel` to run the notebooks.