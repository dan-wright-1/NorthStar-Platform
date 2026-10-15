terraform {
  backend "s3" {
    bucket         = "northstar-tfstate-767397742230"
    key            = "agent/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "northstar-tfstate-lock" # same lock table as Labs 1-2
    encrypt        = true
  }
}
