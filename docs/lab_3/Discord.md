Matthew  — Yesterday at 1:23 PM
Hey! Two tips for Task 1: 

To be able to create the MLflow app, you have to first run terraform apply and deploy your infrastructure stack. You also should make sure that the Lab 2 Feature Group is populated.
Task 1 is primarily a Python implementation task, and you'll be writing code in churn_training_skeleton.py . The TODOs will guide you through what you have to implement.

Finally, as you're wor on lab 3, please post any questions here in the channel! 
Matthew  — Yesterday at 1:58 PM
Also, here's a suggested project repo structure to incorporate the new Lab3 files:

churn_training_skeleton.py                   → models/churn/train.py (or keep its original name)
evaluation_harness.py                        → evaluation/evaluation_harness.py
prompt_templates/offer_generation_prompts.md → prompt_templates/offer_generation_prompts.md
northstar-policy-docs/                       → northstar-policy-docs/
scripts/teardown-lab3.sh                     → scripts/teardown-lab3.sh
Tom  — 1:47 PM
thanks Matthew for the guide! I just was about to ask if we needed to terraform apply
Tom  — 3:58 PM
In my terraform destroy from the last lab I think i somehow delted the data that the glue job processed. Will that occur every time I do terraform destroy forcing me to run the glue job each time or is there a better way to do this.
Sara  — 4:12 PM
is your northstar-dev-data bucket empty or completely gone?
Tom  — 4:22 PM
its gone i believe, the teardown script wiped it
when I ran the query there was nothing returned
Sara  — 4:23 PM
Okay, so I checked the teardown script and I dont think it tears it down.  But I do remember tearing down something myself because it kept getting flagged.
either way, I'm here re running lab 2 to get my data.
Im just saying because maybe if we kept the bucket empty, we could just restore the data somehow?
Tom  — 4:28 PM
I think we can download a snapshot of the processed data locally and just upload that to the processed bucket so we dont have to wait the 15 minutes to run the glue job
Sara  — 4:29 PM
will do for sure
Matthew  — 5:54 PM
Tom, I think that sounds like a pretty good idea. AWS can be weird about how it stores the processed data from a glue job, using lots of parquet objects, but if that works, I think it would be helpful.
Since this lab is pretty long, I imagine lots of students will do the work over several sessions, and a simple manual one-time upload would be easier than waiting for a 15 minute glue job each time
Matthew  — 8:25 PM
Correction to my earlier message! Restoring/uploading processed/ data alone is not enough for Lab 3. The Lab 2 teardown deletes the bucket and Feature Group, and terraform apply recreates only empty resources. After a teardown, you will have to rerun the feature-engineering Glue job to repopulate the Feature Store and its Athena offline store. A saved processed/ snapshot can save the transform-job time, but feature engineering still has to run.
Sara  — 8:27 PM
Mmm okay good to know, also once task 1 is done we should run tear down for lab 3 and 2 right. Teardown 3 seems to not be super aggressive
Matthew  — 8:48 PM
If it's gonna be a while til you continue on task 3, you'll probably want to run tear down for both lab 3 and 2. BUT the caveat is that the models will be gone from your artifact bucket, and you'd have to re-train your models after doing terraform apply again. The way to save you from having to do that is by keeping lab 2 up which would cost up to 2$/day.
Sara  — 8:49 PM
Oh okay, good to know from now on
Matthew  — 8:50 PM
Yeah I hadn’t thought of that, really good catch