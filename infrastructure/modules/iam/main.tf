# ── modules/iam ──────────────────────────────────────────────────────────────
# Required resources (Task B1). Exactly one of each:
#
#   aws_iam_role                     MLEngineer, trusted by sagemaker.amazonaws.com
#   aws_iam_policy
#   aws_iam_role_policy_attachment
#
# Lab 2 adds the same trio for DataEngineer and ModelMonitor (bottom of file).
#
# Least privilege is graded in later labs, so start narrow: grant only the S3
# prefixes and SageMaker actions this role actually needs. A wildcard policy
# here will cost you points in Lab 2.

resource "aws_iam_role" "ml_engineer" {
  name = "${var.project}-${var.environment}-MLEngineer"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = {
    Name = "${var.project}-${var.environment}-MLEngineer"
  }
}

resource "aws_iam_policy" "ml_engineer" {
  name        = "${var.project}-${var.environment}-NorthStarMLEngineerPolicy"
  description = "Least-privilege policy for the MLEngineer SageMaker execution role"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "SageMakerCore"
        Effect = "Allow"
        Action = [
          "sagemaker:CreateTrainingJob", "sagemaker:DescribeTrainingJob", "sagemaker:StopTrainingJob",
          "sagemaker:CreateEndpoint", "sagemaker:DescribeEndpoint", "sagemaker:DeleteEndpoint",
          "sagemaker:CreateEndpointConfig", "sagemaker:DeleteEndpointConfig",
          "sagemaker:CreateMlflowApp", "sagemaker:DescribeMlflowApp", "sagemaker:ListMlflowApps",
          "sagemaker:CreatePresignedMlflowAppUrl",
          "sagemaker:RegisterModel", "sagemaker:DescribeModelPackage", "sagemaker:ListModelPackages",
          # Lab 4: the SageMaker Pipeline runs as this role. Its RegisterModel
          # step calls CreateModelPackage, and pipelines tag every job they create.
          "sagemaker:CreateModelPackage", "sagemaker:AddTags", "sagemaker:ListTags",
          # RegisterModel also creates the package group when it is missing.
          "sagemaker:CreateModelPackageGroup", "sagemaker:DescribeModelPackageGroup",
        ]
        Resource = "*"
      },
      {
        # Lab 4: the pipeline starts a training job and hands it this same role,
        # which is an iam:PassRole call. Scoped to this role and to SageMaker.
        Sid      = "PassSelfToSageMaker"
        Effect   = "Allow"
        Action   = ["iam:PassRole"]
        Resource = aws_iam_role.ml_engineer.arn
        Condition = {
          StringEquals = { "iam:PassedToService" = "sagemaker.amazonaws.com" }
        }
      },
      {
        # ListBucket is a bucket-level action -- it has to target the bare
        # bucket ARN, not an object path. Kept in its own statement on
        # purpose: folding it into S3ArtifactsAndFeatures's Resource list
        # would put "...-data-*" (no path) in an array that also grants
        # GetObject/PutObject/DeleteObject, and IAM's ARN wildcard matches
        # across "/" -- so that single bucket-level entry would silently
        # grant object writes to raw/ and processed/ too.
        Sid      = "S3ListBucket"
        Effect   = "Allow"
        Action   = "s3:ListBucket"
        Resource = "arn:aws:s3:::${var.project}-${var.environment}-data-*"
      },
      {
        Sid    = "S3ArtifactsAndFeatures"
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
        Resource = [
          "arn:aws:s3:::${var.project}-${var.environment}-data-*/artifacts/*",
          "arn:aws:s3:::${var.project}-${var.environment}-data-*/features/*",
        ]
      },
      {
        Sid      = "CloudWatchLogs"
        Effect   = "Allow"
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "arn:aws:logs:*:*:log-group:/aws/sagemaker/*"
      },
      {
        Sid      = "ECRRead"
        Effect   = "Allow"
        Action   = ["ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage", "ecr:GetAuthorizationToken"]
        Resource = "*"
      },
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ml_engineer" {
  role       = aws_iam_role.ml_engineer.name
  policy_arn = aws_iam_policy.ml_engineer.arn
}

# ── Lab 2: DataEngineer and ModelMonitor ─────────────────────────────────────
# The new roles name the bucket exactly instead of using the "-data-*" pattern
# above. IAM's ARN wildcard matches across "/", so ".../data-*/raw/*" would also
# match ".../data-<acct>/artifacts/raw/x"; an exact bucket name closes that gap.
# The name is derived the same way modules/storage builds it.

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  region     = data.aws_region.current.name
  bucket_arn = "arn:aws:s3:::${var.project}-${var.environment}-data-${local.account_id}"
}

# ── DataEngineer ─────────────────────────────────────────────────────────────
# Data-plane identity for the Glue crawler and both ETL jobs, and the execution
# role of the Feature Group. Writes raw/, processed/, features/; reads its own
# job scripts from artifacts/glue/; cannot write artifacts/ or train models.

resource "aws_iam_role" "data_engineer" {
  name = "${var.project}-${var.environment}-DataEngineer"

  # sagemaker is required in Task 3: CreateFeatureGroup rejects an execution
  # role that does not trust it ("The execution role ARN is invalid").
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = ["glue.amazonaws.com", "lambda.amazonaws.com", "sagemaker.amazonaws.com"]
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = {
    Name = "${var.project}-${var.environment}-DataEngineer"
  }
}

resource "aws_iam_policy" "data_engineer" {
  name        = "${var.project}-${var.environment}-NorthStarDataEngineerPolicy"
  description = "Least-privilege policy for the DataEngineer Glue and Feature Store role"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        # Full Glue access per the spec, but only inside this account and
        # region. Includes glue:GetConnection, which Glue resolves before the
        # script ever runs when a job uses a NETWORK connection.
        Sid      = "GlueFullAccessThisAccount"
        Effect   = "Allow"
        Action   = "glue:*"
        Resource = "arn:aws:glue:${local.region}:${local.account_id}:*"
      },
      {
        # Glue places an ENI in the private subnet for every job run. The
        # Create/Describe calls do not support resource-level restriction.
        Sid    = "GlueVpcNetworkInterfaces"
        Effect = "Allow"
        Action = [
          "ec2:CreateNetworkInterface",
          "ec2:DeleteNetworkInterface",
          "ec2:DescribeNetworkInterfaces",
          "ec2:DescribeVpcs",
          "ec2:DescribeSubnets",
          "ec2:DescribeSecurityGroups",
          "ec2:DescribeVpcEndpoints",
          "ec2:DescribeRouteTables",
          "ec2:DescribeVpcAttribute",
        ]
        Resource = "*"
      },
      {
        # Glue tags every ENI it creates; without this the job fails with
        # "The specified role does not have a permission to create a tag for
        # your elastic network interface".
        Sid      = "GlueEniTags"
        Effect   = "Allow"
        Action   = ["ec2:CreateTags", "ec2:DeleteTags"]
        Resource = "arn:aws:ec2:*:*:network-interface/*"
      },
      {
        # Bucket-level actions on the bare bucket ARN only. GetBucketAcl is
        # what Feature Store checks before accepting the bucket as an offline
        # store target ("Invalid S3Uri provided" without it).
        Sid      = "S3BucketLevel"
        Effect   = "Allow"
        Action   = ["s3:ListBucket", "s3:GetBucketLocation", "s3:GetBucketAcl"]
        Resource = local.bucket_arn
      },
      {
        Sid    = "S3DataZonesReadWrite"
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
        Resource = [
          "${local.bucket_arn}/raw/*",
          "${local.bucket_arn}/processed/*",
          "${local.bucket_arn}/features/*",
          # Glue's S3 layer writes this directory marker when a job overwrites
          # features/customers/. A re-run fails with AccessDenied without it
          # (found re-running the feature job for Lab 4).
          "${local.bucket_arn}/features_$folder$",
        ]
      },
      {
        # The offline store writes its objects with an ACL; plain PutObject
        # is not enough.
        Sid      = "S3FeatureStoreOfflineAcl"
        Effect   = "Allow"
        Action   = "s3:PutObjectAcl"
        Resource = "${local.bucket_arn}/features/*"
      },
      {
        # Read-only: Glue fetches its own job scripts from here.
        Sid      = "S3GlueScriptsReadOnly"
        Effect   = "Allow"
        Action   = "s3:GetObject"
        Resource = "${local.bucket_arn}/artifacts/glue/*"
      },
      {
        Sid    = "FeatureStoreWrite"
        Effect = "Allow"
        Action = [
          "sagemaker:PutRecord",
          "sagemaker:CreateFeatureGroup",
          "sagemaker:DescribeFeatureGroup",
        ]
        Resource = "arn:aws:sagemaker:${local.region}:${local.account_id}:feature-group/${var.project}-${var.environment}-*"
      },
      {
        Sid    = "CloudWatchLogsWrite"
        Effect = "Allow"
        Action = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = [
          "arn:aws:logs:${local.region}:${local.account_id}:log-group:/aws-glue/*",
          "arn:aws:logs:${local.region}:${local.account_id}:log-group:/aws/sagemaker/*",
        ]
      },
    ]
  })
}

resource "aws_iam_role_policy_attachment" "data_engineer" {
  role       = aws_iam_role.data_engineer.name
  policy_arn = aws_iam_policy.data_engineer.arn
}

# ── ModelMonitor ─────────────────────────────────────────────────────────────
# Observes, does not act. Reads artifacts/ and drift-job status, writes only
# CloudWatch metrics, alarms, and logs. Starting a processing job belongs to
# ModelMonitorExecution in Lab 6, not to this role.

resource "aws_iam_role" "model_monitor" {
  name = "${var.project}-${var.environment}-ModelMonitor"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = {
    Name = "${var.project}-${var.environment}-ModelMonitor"
  }
}

resource "aws_iam_policy" "model_monitor" {
  name        = "${var.project}-${var.environment}-NorthStarModelMonitorPolicy"
  description = "Read-only observer policy for the ModelMonitor role"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        # PutMetricData and DescribeAlarms have no resource-level support.
        Sid    = "CloudWatchMetricsAndAlarms"
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "cloudwatch:GetMetricStatistics",
          "cloudwatch:PutMetricAlarm",
          "cloudwatch:DescribeAlarms",
        ]
        Resource = "*"
      },
      {
        # List has no resource-level support; Describe is scoped to jobs.
        Sid      = "SageMakerListProcessingJobs"
        Effect   = "Allow"
        Action   = "sagemaker:ListProcessingJobs"
        Resource = "*"
      },
      {
        Sid      = "SageMakerDescribeProcessingJob"
        Effect   = "Allow"
        Action   = "sagemaker:DescribeProcessingJob"
        Resource = "arn:aws:sagemaker:${local.region}:${local.account_id}:processing-job/*"
      },
      {
        Sid      = "S3ListArtifacts"
        Effect   = "Allow"
        Action   = "s3:ListBucket"
        Resource = local.bucket_arn
        Condition = {
          StringLike = { "s3:prefix" = ["artifacts/*"] }
        }
      },
      {
        Sid      = "S3ArtifactsReadOnly"
        Effect   = "Allow"
        Action   = "s3:GetObject"
        Resource = "${local.bucket_arn}/artifacts/*"
      },
      {
        Sid      = "CloudWatchLogsWrite"
        Effect   = "Allow"
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "arn:aws:logs:${local.region}:${local.account_id}:log-group:/aws/sagemaker/*"
      },
    ]
  })
}

resource "aws_iam_role_policy_attachment" "model_monitor" {
  role       = aws_iam_role.model_monitor.name
  policy_arn = aws_iam_policy.model_monitor.arn
}
