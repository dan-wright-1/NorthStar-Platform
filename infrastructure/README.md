# Infrastructure

Terraform for the NorthStar platform. Started from the Lab 1 Part B module template and
extended each lab; the root [README](../README.md) describes what each lab added.

## Layout

```
modules/
  vpc/            VPC, public subnet + internet gateway, private subnet + NAT gateway (Lab 2),
                  route tables, shared security group (self-referencing ingress for Glue)
  storage/        data bucket: public access block, versioning, encryption, lifecycle rules,
                  raw/ processed/ features/ artifacts/ prefixes
  iam/            MLEngineer, DataEngineer and ModelMonitor roles and their policies
  sagemaker/      SageMaker domain (VpcOnly, private subnet) and the MLEngineer user profile
  glue/           catalog database, network connection, raw crawler, transform and
                  feature-engineer jobs (scripts uploaded from ../glue-scripts/)
  feature_store/  customer Feature Group (online + offline store)
environments/
  dev/            the real deployment of the modules (aws provider ~> 5.0; S3 state + DynamoDB lock)
  local/          the same modules against LocalStack (no SageMaker, NAT or lifecycle rules)
  agent/          Lab 3 Track C AgentCore stack: tools Lambda, gateway + target, Cedar policy
                  engine and policies, harness. Separate root because AgentCore needs the
                  aws provider 6.x (pinned 6.67.0); same state bucket and lock table,
                  its own state key (agent/terraform.tfstate)
```

## Rules the modules keep

- **No hardcoded names in modules.** `grep -rn '"northstar-dev"' infrastructure/modules/` returns
  nothing; names are built from `var.project` and `var.environment`.
- Every variable has a `description`.
- Each module contains only its own resources.

```bash
terraform -chdir=infrastructure/environments/dev init
terraform fmt -check -recursive infrastructure      # no output = pass
terraform -chdir=infrastructure/environments/dev validate
```

Always `plan -out=tfplan` and apply the saved plan. Plan files and `*.tfstate` are gitignored
(both can contain state).
